#!/usr/bin/env python3
"""
Compute token-level effective rank (Option A only) for Qwen2.5-VL image tokens,
and VERIFY whether the number of image tokens T is constant across scripts.

Setup:
- Scale-invariance pair: image1 = original (scale=1.0), image2 = scaled down (scale=scale_factor),
  both rendered on a fixed image_size×image_size padded canvas.
- We compute eRank over the image-token matrix R_img ∈ R^{T×d} for EACH image (img1/img2),
  at EACH transformer layer.
- We aggregate across characters per script (mean/std) and save CSVs + plots.

Outputs:
  - CSVs:
      erank_token_per_image.csv
      erank_token_summary_by_script_layer.csv
      image_token_counts.csv   (this is the "Check if T is same" artifact)
  - Plots:
      group_erank_token_img2_by_layer.png
      english_vs_selected_erank_token_img2.png

Notes:
- We explicitly record T1 and T2 (token counts for image1 and image2 spans).
- If T varies across samples, consider also reporting normalized eRank = eRank/min(T,d).
  (Not enabled by default here; you can easily add it.)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Your existing helper
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qwen_vl_utils import process_vision_info


DEFAULT_PROMPT = (
    "Compare the two images and decide if they show the same character. "
    "Ignore differences in scale, size, or resolution. Answer with exactly YES or NO."
)

OMNIGLOT_GROUP_EXCLUDE = {"hand_digits", "hand_english"}  # exclude from "Omniglot group" plots


# -----------------------------
# Utilities
# -----------------------------
def resize_with_padding_pil(image_path: str, scale_factor: float, image_size: int = 336):
    """Return a PIL.Image resized to image_size×image_size, then scaled and center-padded."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)

    new_size = max(1, int(image_size * float(scale_factor)))
    img_small = img.resize((new_size, new_size), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    px = (image_size - new_size) // 2
    py = (image_size - new_size) // 2
    canvas.paste(img_small, (px, py))
    return canvas


def prepare_inputs(image1_pil, image2_pil, prompt: str, processor):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image1_pil},
                {"type": "image", "image": image2_pil},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    return inputs


def find_image_spans(input_ids: torch.Tensor, processor) -> List[Tuple[int, int]]:
    """Find token spans for image1 and image2 inside the packed sequence."""
    tok = processor.tokenizer
    start_tokens = ["<|image_start|>", "<|vision_start|>"]
    end_tokens = ["<|image_end|>", "<|vision_end|>"]

    start_ids, end_ids = [], []
    for t in start_tokens:
        tid = tok.convert_tokens_to_ids(t)
        if tid is not None and tid != tok.unk_token_id:
            start_ids.append(tid)
    for t in end_tokens:
        tid = tok.convert_tokens_to_ids(t)
        if tid is not None and tid != tok.unk_token_id:
            end_ids.append(tid)

    ids = input_ids[0].tolist()
    spans = []
    i = 0
    while i < len(ids):
        if ids[i] in start_ids:
            j = i + 1
            while j < len(ids) and ids[j] not in end_ids:
                j += 1
            if j >= len(ids):
                break
            spans.append((i + 1, j))  # content tokens only
            i = j + 1
        else:
            i += 1

    if len(spans) < 2:
        raise RuntimeError("Could not find two image spans in input_ids.")
    return spans[:2]


def effective_rank(R: torch.Tensor) -> float:
    """
    eRank(R) = exp( H(p) ), where p are normalized singular values of centered R.
    R: [T, d]
    """
    if R is None or R.numel() == 0 or R.shape[0] < 2:
        return float("nan")

    X = R - R.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(X.float())
        p = s / (s.sum() + 1e-12)
        er = torch.exp(-torch.sum(p * torch.log(p + 1e-12)))
        return float(er.detach().cpu())
    except Exception:
        return float("nan")


def plot_group_means(df: pd.DataFrame, value_col: str, title: str, ylabel: str, out_path: Path):
    """
    df must have columns: group, layer, value_col (and script_name,char_id for pivot)
    Plots mean±std across characters within each group.
    """
    plt.figure(figsize=(8.5, 4.5))
    for group, gdf in df.groupby("group"):
        piv = gdf.pivot_table(index=["script_name", "char_id"], columns="layer", values=value_col)
        mat = piv.to_numpy()
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        x = np.arange(len(mean))
        plt.plot(x, mean, marker="o", label=group)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_english_vs_selected(
    df: pd.DataFrame,
    value_col: str,
    scripts: List[str],
    title: str,
    ylabel: str,
    out_path: Path,
):
    """
    Line plot for mean±std vs layer for English and selected scripts.
    df columns: script_name, char_id, layer, value_col
    """
    plt.figure(figsize=(9.5, 4.8))
    for script in scripts:
        sdf = df[df["script_name"] == script]
        if sdf.empty:
            continue
        piv = sdf.pivot_table(index=["script_name", "char_id"], columns="layer", values=value_col)
        mat = piv.to_numpy()
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        x = np.arange(len(mean))
        plt.plot(x, mean, marker="o", label=script)
        plt.fill_between(x, mean - std, mean + std, alpha=0.15)

    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# Dataset loading
# -----------------------------
# def load_alphabet_images(alphabet_dir: str, num_chars: int) -> List[Tuple[str, str]]:
#     base = Path(alphabet_dir) / "times_new_roman"
#     pairs = []
#     for char_dir in sorted(base.glob("character*")):
#         img = char_dir / "image.png"
#         if img.exists():
#             pairs.append((char_dir.name, str(img)))
#     return pairs


# def load_omniglot_images(omniglot_dir: str, script_name: str, num_chars: int) -> List[Tuple[str, str]]:
#     base = Path(omniglot_dir) / "omniglot" / "omniglot-master" / "python" / "images_all" / script_name
#     pairs = []
#     for char_dir in sorted(base.glob("character*")):
#         image_files = sorted(char_dir.glob("*.png"))
#         if image_files:
#             pairs.append((char_dir.name, str(image_files[0])))
#     return pairs
def load_alphabet_images(alphabet_dir: str, num_chars: int) -> List[Tuple[str, str]]:
    base = Path(alphabet_dir) / "times_new_roman"
    pairs = []

    for char_dir in sorted(base.glob("character*")):
        image_files = sorted(char_dir.glob("*.png"))
        for img in image_files:
            pairs.append((char_dir.name, str(img)))

    return pairs


def load_omniglot_images(omniglot_dir: str, script_name: str, num_chars: int) -> List[Tuple[str, str]]:
    base = Path(omniglot_dir) / "omniglot" / "omniglot-master" / "python" / "images_all" / script_name
    pairs = []

    for char_dir in sorted(base.glob("character*")):
        image_files = sorted(char_dir.glob("*.png"))
        for img in image_files:
            pairs.append((char_dir.name, str(img)))

    return pairs


# -----------------------------
# Core computation (Option A only)
# -----------------------------
def compute_eranks_for_pair(
    model,
    processor,
    image1_pil,
    image2_pil,
    prompt: str,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Returns:
      - erank_token_img1: token-level eRank for image1 span [L]
      - erank_token_img2: token-level eRank for image2 span [L]
      - T1, T2: token counts for image1/image2 spans (should be constant if grid fixed)
    """
    inputs = prepare_inputs(image1_pil, image2_pil, prompt, processor)
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    spans = find_image_spans(inputs["input_ids"], processor)
    (s1, e1), (s2, e2) = spans

    T1 = int(e1 - s1)
    T2 = int(e2 - s2)

    # hidden_states[0] is embeddings; use [1..L] to align with transformer blocks.
    hidden_states = outputs.hidden_states
    L = len(hidden_states) - 1

    erank_token_img1 = np.zeros(L, dtype=np.float32)
    erank_token_img2 = np.zeros(L, dtype=np.float32)

    for layer in range(L):
        h = hidden_states[layer + 1][0]  # [seq, d]
        tok1 = h[s1:e1]  # [T1, d]
        tok2 = h[s2:e2]  # [T2, d]

        erank_token_img1[layer] = effective_rank(tok1)
        erank_token_img2[layer] = effective_rank(tok2)

    return {
        "erank_token_img1": erank_token_img1,
        "erank_token_img2": erank_token_img2,
        "T1": T1,
        "T2": T2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./erank_out")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--scale_factor", type=float, default=0.3)

    parser.add_argument("--alphabet_dir", type=str, default=None, help="Path containing times_new_roman/")
    parser.add_argument("--omniglot_dir", type=str, default=None, help="Path containing omniglot/omniglot-master/...")
    parser.add_argument(
        "--omniglot_scripts",
        type=str,
        nargs="*",
        default=None,
        help="Omniglot scripts to process. Default: all scripts under images_all.",
    )
    parser.add_argument("--num_chars", type=int, default=50)

    parser.add_argument(
        "--focus_scripts",
        type=str,
        nargs="*",
        default=None,
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model.eval()
    device = next(model.parameters()).device

    token_rows = []         # per-character per-layer
    token_count_rows = []   # per-character (T1, T2)

    # -----------------------------
    # English (Times New Roman) as "English"
    # -----------------------------
    if args.alphabet_dir:
        english_imgs = load_alphabet_images(args.alphabet_dir, args.num_chars)
        for char_id, img_path in tqdm(english_imgs, desc="English (Times New Roman)", unit="char"):
            img1 = resize_with_padding_pil(img_path, 1.0, image_size=args.image_size)
            img2 = resize_with_padding_pil(img_path, args.scale_factor, image_size=args.image_size)

            res = compute_eranks_for_pair(model, processor, img1, img2, args.prompt, device)

            token_count_rows.append(
                {"script_name": "English", "char_id": char_id, "T1": res["T1"], "T2": res["T2"]}
            )

            L = len(res["erank_token_img2"])
            for layer in range(L):
                token_rows.append(
                    {
                        "script_name": "English",
                        "char_id": char_id,
                        "layer": layer,
                        "erank_token_img1": float(res["erank_token_img1"][layer]),
                        "erank_token_img2": float(res["erank_token_img2"][layer]),
                    }
                )

    # -----------------------------
    # Omniglot scripts
    # -----------------------------
    if args.omniglot_dir:
        scripts = args.omniglot_scripts
        if scripts is None:
            base = Path(args.omniglot_dir) / "omniglot" / "omniglot-master" / "python" / "images_all"
            scripts = sorted([p.name for p in base.iterdir() if p.is_dir()])

        for script_name in tqdm(scripts, desc="Omniglot scripts", unit="script"):
            imgs = load_omniglot_images(args.omniglot_dir, script_name, args.num_chars)
            for char_id, img_path in tqdm(imgs, desc=f"{script_name}", unit="char", leave=False):
                img1 = resize_with_padding_pil(img_path, 1.0, image_size=args.image_size)
                img2 = resize_with_padding_pil(img_path, args.scale_factor, image_size=args.image_size)

                res = compute_eranks_for_pair(model, processor, img1, img2, args.prompt, device)

                token_count_rows.append(
                    {"script_name": script_name, "char_id": char_id, "T1": res["T1"], "T2": res["T2"]}
                )

                L = len(res["erank_token_img2"])
                for layer in range(L):
                    token_rows.append(
                        {
                            "script_name": script_name,
                            "char_id": char_id,
                            "layer": layer,
                            "erank_token_img1": float(res["erank_token_img1"][layer]),
                            "erank_token_img2": float(res["erank_token_img2"][layer]),
                        }
                    )

    if not token_rows:
        raise RuntimeError("No data processed. Provide --alphabet_dir and/or --omniglot_dir.")

    # -----------------------------
    # Save per-image per-layer Option A CSV
    # -----------------------------
    token_df = pd.DataFrame(token_rows)
    token_df.to_csv(out_dir / "erank_token_per_image.csv", index=False)

    # Summary by script & layer (mean/std across characters)
    token_summary = (
        token_df.groupby(["script_name", "layer"], as_index=False)
        .agg(
            erank_token_img1_mean=("erank_token_img1", "mean"),
            erank_token_img1_std=("erank_token_img1", "std"),
            erank_token_img2_mean=("erank_token_img2", "mean"),
            erank_token_img2_std=("erank_token_img2", "std"),
            n_chars=("char_id", "nunique"),
        )
    )
    token_summary.to_csv(out_dir / "erank_token_summary_by_script_layer.csv", index=False)

    # -----------------------------
    # Save token-count diagnostics (Check if T is same)
    # -----------------------------
    tc_df = pd.DataFrame(token_count_rows)
    tc_df.to_csv(out_dir / "image_token_counts.csv", index=False)

    print("\n[TOKEN COUNT CHECK]")
    print("Unique T1 counts:", sorted(tc_df["T1"].unique().tolist()))
    print("Unique T2 counts:", sorted(tc_df["T2"].unique().tolist()))
    print("\nT2 nunique by script (top 20):")
    print(tc_df.groupby("script_name")["T2"].nunique().sort_values(ascending=False).head(20).to_string())

    # -----------------------------
    # Plots
    # -----------------------------
    # Group plot (English vs Omniglot) using per-character values
    token_plot_df = token_df.copy()
    token_plot_df["group"] = np.where(token_plot_df["script_name"] == "English", "English", "Omniglot")
    token_plot_df = token_plot_df[
        ~((token_plot_df["group"] == "Omniglot") & (token_plot_df["script_name"].isin(OMNIGLOT_GROUP_EXCLUDE)))
    ]

    plot_group_means(
        token_plot_df,
        value_col="erank_token_img2",
        title=f"Token-level eRank (image2) by layer (scale={args.scale_factor})",
        ylabel="eRank (tokens within image)",
        out_path=out_dir / "group_erank_token_img2_by_layer.png",
    )

    # English vs selected scripts
    focus_scripts = args.focus_scripts
    if not focus_scripts:
        focus_scripts = sorted(token_df["script_name"].unique().tolist())
    focusA = token_df[token_df["script_name"].isin(focus_scripts)].copy()
    if not focusA.empty:
        plot_english_vs_selected(
            focusA,
            value_col="erank_token_img2",
            scripts=[s for s in focus_scripts if s in focusA["script_name"].unique()],
            title=f"Token-level eRank (image2) — English vs selected (scale={args.scale_factor})",
            ylabel="eRank (tokens within image)",
            out_path=out_dir / "english_vs_selected_erank_token_img2.png",
        )

    print(f"\n[OK] Saved CSVs + plots to: {out_dir.resolve()}")
    print("  - erank_token_per_image.csv")
    print("  - erank_token_summary_by_script_layer.csv")
    print("  - image_token_counts.csv")
    print("  - group_erank_token_img2_by_layer.png")
    print("  - english_vs_selected_erank_token_img2.png")


if __name__ == "__main__":
    main()