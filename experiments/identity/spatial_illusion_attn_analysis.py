#!/usr/bin/env python3
"""
Analyze Qwen2.5-VL internals for Spatial Illusion (rotation) pairs.

Same-character (positive) vs different-character (negative) pairs:
- image1: character at 0°
- image2: same or different character rotated by θ

Outputs:
- attention mass to image1 vs image2 per layer
- cosine similarity of pooled image token features per layer
- optional angle sweep
- group plots: English (alphabet) vs Omniglot (all scripts)
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qwen_vl_utils import process_vision_info

# Rotation angles (subset for attention analysis; can override via CLI)
DEFAULT_ROTATION_ANGLES = [30, 45, 90, 135, 180, 270]

OMNIGLOT_SCRIPTS_DEFAULT = [
    "English",
    "Greek",
    "Latin",
    "Braille",
    "Mongolian",
    "Keble",
    "Malayalam",
]

DEFAULT_PROMPT = (
    "You are given two images, each containing a single character. Decide whether they depict "
    "the same underlying character, allowing for rotation and orientation changes. Answer with exactly YES or NO."
)


def rotate_image(image_path: str, angle: float, output_path: str, image_size: int = 336) -> str:
    """Rotate image by angle (degrees). Uses OpenCV; output same size as input."""
    import cv2
    img_cv = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_cv is None:
        from PIL import Image
        img_pil = Image.open(image_path).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    height, width = img_cv.shape[0], img_cv.shape[1]
    if (width, height) != (image_size, image_size):
        img_cv = cv2.resize(img_cv, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        height, width = image_size, image_size
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated_cv = cv2.warpAffine(
        img_cv, rotation_matrix, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, rotated_cv)
    return output_path


def find_image_spans(input_ids: torch.Tensor, processor) -> List[Tuple[int, int]]:
    tok = processor.tokenizer
    start_tokens = ["<|image_start|>", "<|vision_start|>"]
    end_tokens = ["<|image_end|>", "<|vision_end|>"]
    start_ids = []
    end_ids = []
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
            spans.append((i + 1, j))
            i = j + 1
        else:
            i += 1
    if len(spans) < 2:
        raise RuntimeError("Could not find two image spans in input_ids.")
    return spans[:2]


def prepare_inputs(image1: str, image2: str, prompt: str, processor):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image1},
                {"type": "image", "image": image2},
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


def attention_mass_to_spans(attn: torch.Tensor, spans: List[Tuple[int, int]]) -> Tuple[float, float]:
    attn = attn.mean(dim=1)[0]
    query_idx = attn.shape[0] - 1
    src_attn = attn[query_idx]
    (s1, e1), (s2, e2) = spans
    a1 = float(src_attn[s1:e1].sum().cpu())
    a2 = float(src_attn[s2:e2].sum().cpu())
    total = float(src_attn.sum().cpu())
    if total > 0:
        a1 /= total
        a2 /= total
    return a1, a2


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / (a.norm() + 1e-8)
    b = b / (b.norm() + 1e-8)
    return float((a * b).sum().cpu())


def attention_entropy(attn_vec: torch.Tensor) -> float:
    v = attn_vec.detach().to(torch.float32).cpu().numpy()
    v = v / (v.sum() + 1e-9)
    return float(-np.sum(v * np.log(v + 1e-9)))


def effective_rank(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0 or tensor.shape[0] < 2:
        return 0.0
    t = tensor - tensor.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(t.float())
        p = s / (s.sum() + 1e-9)
        return float(torch.exp(-torch.sum(p * torch.log(p + 1e-9))).cpu())
    except Exception:
        return 0.0


def get_image_grids(inputs) -> Optional[List[Tuple[int, int, int]]]:
    grid = inputs.get("image_grid_thw")
    if grid is None:
        return None
    grid = grid.detach().cpu().numpy()
    if grid.ndim == 3:
        grid = grid[0]
    if grid.ndim == 2 and grid.shape[1] == 3:
        return [tuple(map(int, row.tolist())) for row in grid]
    if grid.ndim == 1 and grid.shape[0] == 3:
        return [tuple(map(int, grid.tolist()))]
    return None


def infer_effective_grid(num_tokens: int, H_raw: int, W_raw: int) -> Tuple[int, int]:
    raw = H_raw * W_raw
    if num_tokens == raw:
        return H_raw, W_raw
    k = int(round(math.sqrt(num_tokens)))
    if k * k == num_tokens:
        return k, k
    target_aspect = W_raw / H_raw
    best = None
    best_err = 1e9
    for h in range(1, num_tokens + 1):
        if num_tokens % h != 0:
            continue
        w = num_tokens // h
        err = abs((w / h) - target_aspect)
        if err < best_err:
            best_err = err
            best = (h, w)
    return best if best else (k, k)


def plot_attention_heatmap(
    attn_vec: torch.Tensor,
    span: Tuple[int, int],
    grid: Tuple[int, int, int],
    image_path: str,
    out_path: str,
    title: str,
):
    from PIL import Image
    start, end = span
    _, H_raw, W_raw = grid
    tokens = attn_vec[start:end].detach().to(torch.float32).cpu().numpy()
    num_tokens = tokens.size
    H_eff, W_eff = infer_effective_grid(num_tokens, H_raw, W_raw)
    heat = tokens.reshape(H_eff, W_eff)
    img = Image.open(image_path).convert("RGB")
    heat_img = Image.fromarray((heat / (heat.max() + 1e-8) * 255).astype(np.uint8))
    heat_img = heat_img.resize(img.size, resample=Image.Resampling.NEAREST)
    plt.figure(figsize=(5.5, 5.5))
    plt.imshow(img)
    plt.imshow(heat_img, cmap="inferno", alpha=0.45)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_pair(model, processor, image1: str, image2: str, prompt: str, device: torch.device):
    inputs = prepare_inputs(image1, image2, prompt, processor)
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

    spans = find_image_spans(inputs["input_ids"], processor)
    grids = get_image_grids(inputs)
    attn_list = outputs.attentions
    hidden_states = outputs.hidden_states

    attn_mass_1 = []
    attn_mass_2 = []
    cos_sims = []
    img1_layer_feats = []
    img2_layer_feats = []
    img2_attn_dists = []
    layer_metrics = []

    for layer_idx, attn in enumerate(attn_list):
        a1, a2 = attention_mass_to_spans(attn, spans)
        attn_mass_1.append(a1)
        attn_mass_2.append(a2)

        h = hidden_states[layer_idx]
        (s1, e1), (s2, e2) = spans
        img1_feat = h[0, s1:e1].mean(dim=0)
        img2_feat = h[0, s2:e2].mean(dim=0)
        cos_sims.append(cosine_similarity(img1_feat, img2_feat))
        img1_layer_feats.append(img1_feat.detach().to(torch.float32).cpu().numpy())
        img2_layer_feats.append(img2_feat.detach().to(torch.float32).cpu().numpy())

        feat1 = h[0, s1:e1]
        feat2 = h[0, s2:e2]
        norm1 = float(feat1.norm(dim=-1).mean().cpu()) if feat1.numel() else 0.0
        norm2 = float(feat2.norm(dim=-1).mean().cpu()) if feat2.numel() else 0.0
        rank1 = effective_rank(feat1)
        rank2 = effective_rank(feat2)

        attn_heads = attn[0]
        query_idx = -1
        attn_vecs = attn_heads[:, query_idx, :]
        head_ent = []
        head_mass_2 = []
        for h_idx in range(attn_vecs.shape[0]):
            vec = attn_vecs[h_idx]
            vec = vec / (vec.sum() + 1e-9)
            head_ent.append(attention_entropy(vec))
            head_mass_2.append(float(vec[s2:e2].sum().cpu()))

        attn_mean = attn_vecs.mean(dim=0)
        attn_mean = attn_mean / (attn_mean.sum() + 1e-9)
        img2_attn = attn_mean[s2:e2].detach().to(torch.float32).cpu().numpy()
        img2_attn = img2_attn / (img2_attn.sum() + 1e-9)
        img2_attn_dists.append(img2_attn)

        layer_metrics.append({
            "layer": layer_idx,
            "norm_img1": norm1,
            "norm_img2": norm2,
            "rank_img1": rank1,
            "rank_img2": rank2,
            "attn_entropy": float(np.mean(head_ent)) if head_ent else 0.0,
            "head_mass_2_std": float(np.std(head_mass_2)) if head_mass_2 else 0.0,
        })

    last_attn = attn_list[-1].mean(dim=1)[0]
    last_query = last_attn.shape[0] - 1
    last_attn_vec = last_attn[last_query]

    attn_ratio = np.array(attn_mass_2) / (np.array(attn_mass_1) + np.array(attn_mass_2) + 1e-8)
    return {
        "attn_mass_1": np.array(attn_mass_1),
        "attn_mass_2": np.array(attn_mass_2),
        "attn_ratio": attn_ratio,
        "cos_sims": np.array(cos_sims),
        "last_attn_vec": last_attn_vec,
        "spans": spans,
        "grids": grids,
        "layer_metrics": layer_metrics,
        "img1_layer_feats": np.array(img1_layer_feats),
        "img2_layer_feats": np.array(img2_layer_feats),
        "img2_attn_dists": img2_attn_dists,
    }


def load_alphabet_images(alphabet_dir: str) -> List[Tuple[str, str]]:
    """List of (char_id, image_path)."""
    base = Path(alphabet_dir) / "times_new_roman"
    if not base.exists():
        raise FileNotFoundError(f"Alphabet directory not found: {base}")
    images = []
    for char_dir in sorted(base.glob("character*")):
        img = char_dir / "image.png"
        if img.exists():
            images.append((char_dir.name, str(img)))
    return images


def load_omniglot_images(
    omniglot_dir: str,
    scripts: Optional[List[str]] = None,
    max_chars_per_script: Optional[int] = None,
) -> List[Tuple[str, str, str]]:
    """List of (script_name, char_id, image_path) for ALL Omniglot character folders.

    Notes:
    - We take a single representative PNG per `character*` folder (the first one) to keep
      output paths and downstream feature-keying consistent.
    """
    base = Path(omniglot_dir) / "omniglot" / "omniglot-master" / "python" / "images_all"
    if not base.exists():
        raise FileNotFoundError(f"Omniglot images_all directory not found: {base}")

    images: List[Tuple[str, str, str]] = []
    # Each subfolder is one Omniglot "script" (e.g. English, Greek, Braille, ...).
    script_dirs: List[Path]
    if scripts is None:
        script_dirs = sorted([p for p in base.iterdir() if p.is_dir()])
    else:
        # Preserve the script order requested by the caller (matches scale_illusion_attn_analysis.py default).
        script_dirs = []
        for script_name in scripts:
            sd = base / script_name
            if sd.is_dir():
                script_dirs.append(sd)
            else:
                print(f"[WARN] Omniglot script directory not found, skipping: {sd}")

    for script_dir in script_dirs:
        script_name = script_dir.name
        char_dirs = sorted(script_dir.glob("character*"))
        if max_chars_per_script is not None:
            char_dirs = char_dirs[: max_chars_per_script]
        for char_dir in char_dirs:
            imgs = sorted(char_dir.glob("*.png"))
            if imgs:
                images.append((script_name, char_dir.name, str(imgs[0])))
    return images


def build_negative_pairs_alphabet(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
    neg = []
    if len(pairs) < 2:
        return neg
    for i, (char_id, img1) in enumerate(pairs):
        j = (i + 1) % len(pairs)
        neg_char_id, neg_img1 = pairs[j]
        neg.append((char_id, img1, neg_img1))
    return neg


def build_negative_pairs_omniglot(pairs: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str, str, str]]:
    """(script_name, char_id, img_path, other_char_id, other_img_path)."""
    by_script = {}
    for script_name, char_id, img_path in pairs:
        if script_name not in by_script:
            by_script[script_name] = []
        by_script[script_name].append((char_id, img_path))
    neg = []
    for script_name, char_id, img_path in pairs:
        others = [(c, p) for c, p in by_script.get(script_name, []) if c != char_id]
        if not others:
            continue
        other_char_id, other_img_path = others[0]
        neg.append((script_name, char_id, img_path, other_char_id, other_img_path))
    return neg


def plot_lines(x, ys, labels, title, y_label, out_path):
    plt.figure(figsize=(9, 5))
    for y, label in zip(ys, labels):
        plt.plot(x, y, marker="o", label=label)
    plt.xlabel("Layer")
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap(matrix, x_labels, y_labels, title, out_path):
    plt.figure(figsize=(9, 5))
    plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(label="Cosine similarity")
    plt.xticks(range(len(x_labels)), x_labels)
    plt.yticks(range(len(y_labels)), y_labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_group_layer_means(data, labels: List[str], title: str, y_label: str, out_path: str):
    plt.figure(figsize=(9, 5))
    for arr, label in zip(data, labels):
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        x = np.arange(len(mean))
        plt.plot(x, mean, marker="o", label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Layer")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_group_erank_means(
    df,
    value_col: str,
    title: str,
    ylabel: str,
    out_path: str,
    group_col: str = "group",
):
    """Mean±std across (script_name,char_id) within each group."""
    plt.figure(figsize=(8.5, 4.5))
    for group, gdf in df.groupby(group_col):
        piv = gdf.pivot_table(index=["script_name", "char_id"], columns="layer", values=value_col)
        mat = piv.to_numpy()
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        x = np.arange(len(mean))
        plt.plot(x, mean, marker="o", label=str(group))
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_english_vs_selected_erank(
    df,
    value_col: str,
    scripts: List[str],
    title: str,
    ylabel: str,
    out_path: str,
):
    """Line plot (mean±std) for each script in scripts."""
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
        plt.plot(x, mean, marker="o", label=str(script))
        plt.fill_between(x, mean - std, mean + std, alpha=0.15)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Spatial Illusion (rotation) attention analysis (Qwen2.5-VL).")
    parser.add_argument("--image1", type=str, default=None, help="Reference image (0 deg) path")
    parser.add_argument("--image2", type=str, default=None, help="Rotated image path")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Text prompt")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name")
    parser.add_argument("--output_dir", type=str, default="./attn_out_spatial", help="Output directory")
    parser.add_argument("--rotation_angles", type=float, nargs="*", default=None,
                        help="Angles for angle sweep (e.g. 30 45 90 180)")
    parser.add_argument("--image_size", type=int, default=336, help="Image size")
    parser.add_argument("--angle", type=float, default=90.0, help="Rotation angle for single pair (degrees)")
    parser.add_argument("--alphabet_dir", type=str, default=None, help="Path containing times_new_roman")
    parser.add_argument("--omniglot_dir", type=str, default=None, help="Path containing omniglot")
    parser.add_argument(
        "--omniglot_scripts",
        type=str,
        nargs="*",
        default=None,
        help="Omniglot script subfolders to use (e.g. English Greek Latin ...). Default: all scripts.",
    )
    parser.add_argument(
        "--omniglot_char_limit",
        type=int,
        default=None,
        help="Optional limit of character folders per Omniglot script (None = all).",
    )
    parser.add_argument(
        "--focus_scripts",
        type=str,
        nargs="*",
        default=None,
        help="Scripts to plot for English-vs-others rank curves (Default: English + all Omniglot scripts).",
    )
    parser.add_argument(
        "--num_chars",
        type=int,
        default=None,
        help="Optional limit of Times New Roman `character*` folders for English. Default None = all characters.",
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

    metrics_rows = []
    pos_features = {}
    neg_features = {}
    token_rows = []        # per-positive-pair eRank: (script_name, char_id, angle, layer, erank_token_img1/img2)
    token_count_rows = []  # per-positive-pair token span lengths: (script_name, char_id, angle, T1, T2)

    def run_pair(image1: str, image2: str, save_dir: Path, script_name: str, char_id: str, pair_type: str, angle_val: float = None):
        save_dir.mkdir(parents=True, exist_ok=True)
        res = analyze_pair(model, processor, image1, image2, args.prompt, device)
        layers = list(range(len(res["cos_sims"])))

        plot_lines(
            layers,
            [res["attn_mass_1"], res["attn_mass_2"]],
            ["Attention to image1 (0 deg)", "Attention to image2 (rotated)"],
            "Attention Mass per Layer",
            "Attention mass (normalized)",
            str(save_dir / "attn_mass_per_layer.png"),
        )
        plot_lines(
            layers,
            [res["cos_sims"]],
            ["Cosine similarity"],
            "Image Feature Similarity per Layer",
            "Cosine similarity",
            str(save_dir / "feature_similarity_per_layer.png"),
        )
        if res.get("grids") and len(res["grids"]) >= 2:
            plot_attention_heatmap(
                res["last_attn_vec"], res["spans"][0], res["grids"][0],
                image1, str(save_dir / "attn_heatmap_image1_last_layer.png"),
                "Attention to image1 (0 deg) tokens (last layer)",
            )
            plot_attention_heatmap(
                res["last_attn_vec"], res["spans"][1], res["grids"][1],
                image2, str(save_dir / "attn_heatmap_image2_last_layer.png"),
                "Attention to image2 (rotated) tokens (last layer)",
            )

        for layer_idx in range(len(res["cos_sims"])):
            metrics_rows.append({
                "script_name": script_name,
                "char_id": char_id,
                "pair_type": pair_type,
                "angle": angle_val,
                "layer": layer_idx,
                "cos_sim": float(res["cos_sims"][layer_idx]),
                "attn_ratio": float(res["attn_ratio"][layer_idx]),
                "attn_mass_img1": float(res["attn_mass_1"][layer_idx]),
                "attn_mass_img2": float(res["attn_mass_2"][layer_idx]),
            })
        for row in res.get("layer_metrics", []):
            metrics_rows.append({
                "script_name": script_name,
                "char_id": char_id,
                "pair_type": pair_type,
                "angle": angle_val,
                "layer": row["layer"],
                "norm_img1": row["norm_img1"],
                "norm_img2": row["norm_img2"],
                "rank_img1": row["rank_img1"],
                "rank_img2": row["rank_img2"],
                "attn_entropy": row["attn_entropy"],
                "head_mass_2_std": row["head_mass_2_std"],
            })

        if pair_type == "positive":
            pos_features[(script_name, char_id)] = (res["img1_layer_feats"], res["img2_layer_feats"])
            # Token-level effective rank analysis (mirrors scale_illusion_attn_analysis.py).
            (s1, e1), (s2, e2) = res["spans"]
            T1 = int(e1 - s1)
            T2 = int(e2 - s2)
            token_count_rows.append(
                {"script_name": script_name, "char_id": char_id, "angle": angle_val, "T1": T1, "T2": T2}
            )
            for row in res.get("layer_metrics", []):
                token_rows.append(
                    {
                        "script_name": script_name,
                        "char_id": char_id,
                        "angle": angle_val,
                        "layer": row["layer"],
                        "erank_token_img1": float(row["rank_img1"]),
                        "erank_token_img2": float(row["rank_img2"]),
                    }
                )
        else:
            neg_features[(script_name, char_id)] = res["img2_layer_feats"]

    if args.alphabet_dir or args.omniglot_dir:
        angles = args.rotation_angles if args.rotation_angles is not None else DEFAULT_ROTATION_ANGLES
        if args.alphabet_dir:
            pairs = load_alphabet_images(args.alphabet_dir)
            if args.num_chars is not None:
                pairs = pairs[: args.num_chars]
            for char_id, img_path in pairs:
                img1_path = out_dir / "English" / char_id / "positive" / "img_0deg.png"
                img1_path.parent.mkdir(parents=True, exist_ok=True)
                rotate_image(img_path, 0.0, str(img1_path), image_size=args.image_size)
                for angle in angles:
                    img2_path = out_dir / "English" / char_id / "positive" / f"img_rot_{int(angle)}.png"
                    rotate_image(img_path, angle, str(img2_path), image_size=args.image_size)
                    run_pair(
                        str(img1_path), str(img2_path),
                        out_dir / "English" / char_id / "positive" / f"angle_{int(angle)}",
                        "English", char_id, "positive", angle,
                    )
            neg_pairs = build_negative_pairs_alphabet(pairs)
            for char_id, img_path, neg_img_path in neg_pairs:
                img1_path = out_dir / "English" / char_id / "negative" / "img_0deg.png"
                img1_path.parent.mkdir(parents=True, exist_ok=True)
                rotate_image(img_path, 0.0, str(img1_path), image_size=args.image_size)
                for angle in angles:
                    img2_path = out_dir / "English" / char_id / "negative" / f"img_rot_{int(angle)}.png"
                    rotate_image(neg_img_path, angle, str(img2_path), image_size=args.image_size)
                    run_pair(
                        str(img1_path), str(img2_path),
                        out_dir / "English" / char_id / "negative" / f"angle_{int(angle)}",
                        "English", char_id, "negative", angle,
                    )

        if args.omniglot_dir:
            # Use all Omniglot character folders (across scripts), like scale_illusion_attn_analysis.py.
            omni_pairs = load_omniglot_images(
                args.omniglot_dir,
                scripts=args.omniglot_scripts if args.omniglot_scripts else None,
                max_chars_per_script=args.omniglot_char_limit,
            )
            for script_name, char_id, img_path in omni_pairs:
                img1_path = out_dir / script_name / char_id / "positive" / "img_0deg.png"
                img1_path.parent.mkdir(parents=True, exist_ok=True)
                rotate_image(img_path, 0.0, str(img1_path), image_size=args.image_size)
                for angle in angles:
                    img2_path = out_dir / script_name / char_id / "positive" / f"img_rot_{int(angle)}.png"
                    rotate_image(img_path, angle, str(img2_path), image_size=args.image_size)
                    run_pair(
                        str(img1_path), str(img2_path),
                        out_dir / script_name / char_id / "positive" / f"angle_{int(angle)}",
                        script_name, char_id, "positive", angle,
                    )
            neg_omni = build_negative_pairs_omniglot(omni_pairs)
            for script_name, char_id, img_path, other_char_id, other_img_path in neg_omni:
                img1_path = out_dir / script_name / char_id / "negative" / "img_0deg.png"
                img1_path.parent.mkdir(parents=True, exist_ok=True)
                rotate_image(img_path, 0.0, str(img1_path), image_size=args.image_size)
                for angle in angles:
                    img2_path = out_dir / script_name / char_id / "negative" / f"img_rot_{int(angle)}.png"
                    rotate_image(other_img_path, angle, str(img2_path), image_size=args.image_size)
                    run_pair(
                        str(img1_path), str(img2_path),
                        out_dir / script_name / char_id / "negative" / f"angle_{int(angle)}",
                        script_name, char_id, "negative", angle,
                    )
    else:
        if not args.image1 or not args.image2:
            raise ValueError("Provide --image1 and --image2 for single-pair analysis.")
        run_pair(
            args.image1, args.image2,
            out_dir / "single" / "positive",
            "single", "single", "positive", args.angle,
        )

    if metrics_rows:
        import pandas as pd
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df["group"] = np.where(metrics_df["script_name"] == "English", "English", "Omniglot")
        metrics_df.to_csv(out_dir / "attn_metrics_by_layer.csv", index=False)

        def to_layer_matrix(df, col):
            piv = df.pivot_table(index=["script_name", "char_id", "angle"], columns="layer", values=col)
            return piv.to_numpy()

        english = metrics_df[(metrics_df["group"] == "English") & (metrics_df["pair_type"] == "positive")]
        omni = metrics_df[(metrics_df["group"] == "Omniglot") & (metrics_df["pair_type"] == "positive")]
        english_neg = metrics_df[(metrics_df["group"] == "English") & (metrics_df["pair_type"] == "negative")]
        omni_neg = metrics_df[(metrics_df["group"] == "Omniglot") & (metrics_df["pair_type"] == "negative")]

        if not english.empty and not omni.empty:
            eng_cos = to_layer_matrix(english, "cos_sim")
            omni_cos = to_layer_matrix(omni, "cos_sim")
            eng_attn = to_layer_matrix(english, "attn_ratio")
            omni_attn = to_layer_matrix(omni, "attn_ratio")

            plot_group_layer_means(
                [eng_cos, omni_cos],
                ["English", "Omniglot"],
                "Feature Similarity by Layer (Spatial Illusion, positive pairs)",
                "Cosine similarity",
                str(out_dir / "group_feature_similarity_per_layer.png"),
            )
            plot_group_layer_means(
                [eng_attn, omni_attn],
                ["English", "Omniglot"],
                "Attention to Image2 (rotated) by Layer (Spatial Illusion, positive pairs)",
                "Attention ratio to image2",
                str(out_dir / "group_attention_ratio_per_layer.png"),
            )

            for col, title, ylab, fname in [
                ("attn_entropy", "Attention Entropy by Layer (positive)", "Entropy", "group_attention_entropy_per_layer.png"),
                ("norm_img2", "Activation Norm Image2 by Layer (positive)", "L2 norm", "group_activation_norm_per_layer.png"),
                ("rank_img2", "Effective Rank Image2 by Layer (positive)", "Effective rank", "group_effective_rank_per_layer.png"),
                ("head_mass_2_std", "Head-wise Attention Variability Image2 by Layer (positive)", "Std of head mass", "group_head_variability_per_layer.png"),
            ]:
                if col in english.columns and col in omni.columns:
                    plot_group_layer_means(
                        [to_layer_matrix(english, col), to_layer_matrix(omni, col)],
                        ["English", "Omniglot"],
                        title,
                        ylab,
                        str(out_dir / fname),
                    )

        if not english_neg.empty and not omni_neg.empty:
            eng_cos_n = to_layer_matrix(english_neg, "cos_sim")
            omni_cos_n = to_layer_matrix(omni_neg, "cos_sim")
            eng_attn_n = to_layer_matrix(english_neg, "attn_ratio")
            omni_attn_n = to_layer_matrix(omni_neg, "attn_ratio")
            plot_group_layer_means(
                [eng_cos_n, omni_cos_n],
                ["English (neg)", "Omniglot (neg)"],
                "Feature Similarity by Layer (Spatial Illusion, negative pairs)",
                "Cosine similarity",
                str(out_dir / "group_feature_similarity_per_layer_negative.png"),
            )
            plot_group_layer_means(
                [eng_attn_n, omni_attn_n],
                ["English (neg)", "Omniglot (neg)"],
                "Attention to Image2 by Layer (negative pairs)",
                "Attention ratio to image2",
                str(out_dir / "group_attention_ratio_per_layer_negative.png"),
            )

        ratio_rows = []
        for key, (ref_feats, rot_feats) in pos_features.items():
            if key not in neg_features:
                continue
            neg_feats = neg_features[key]
            for layer_idx in range(ref_feats.shape[0]):
                ref_f = torch.tensor(ref_feats[layer_idx])
                rot_f = torch.tensor(rot_feats[layer_idx])
                neg_f = torch.tensor(neg_feats[layer_idx])
                d_same = torch.norm(ref_f - rot_f, p=2).item()
                d_diff = torch.norm(ref_f - neg_f, p=2).item()
                ratio = d_same / (d_diff + 1e-9)
                ratio_rows.append({
                    "script_name": key[0],
                    "char_id": key[1],
                    "layer": layer_idx,
                    "ratio": ratio,
                })
        if ratio_rows:
            ratio_df = pd.DataFrame(ratio_rows)
            ratio_df["group"] = np.where(ratio_df["script_name"] == "English", "English", "Omniglot")
            ratio_df.to_csv(out_dir / "cluster_ratio_by_layer.csv", index=False)
            _ratio_piv = lambda d: d.pivot_table(index=["script_name", "char_id"], columns="layer", values="ratio").to_numpy()
            eng_ratio = _ratio_piv(ratio_df[ratio_df["group"] == "English"])
            omni_ratio_df = ratio_df[ratio_df["group"] == "Omniglot"]
            if not omni_ratio_df.empty:
                omni_ratio = _ratio_piv(omni_ratio_df)
                plot_group_layer_means(
                    [eng_ratio, omni_ratio],
                    ["English", "Omniglot"],
                    "Cluster Tightness Ratio by Layer (d_same/d_diff, rotation)",
                    "Distance ratio",
                    str(out_dir / "group_cluster_ratio_per_layer.png"),
                )

    # -----------------------------
    # Rotation eRank (token-level effective rank) analysis
    # -----------------------------
    if token_rows:
        import pandas as pd

        token_df = pd.DataFrame(token_rows)
        token_df.to_csv(out_dir / "erank_token_per_image.csv", index=False)

        token_summary = (
            token_df.groupby(["script_name", "angle", "layer"], as_index=False)
            .agg(
                erank_token_img1_mean=("erank_token_img1", "mean"),
                erank_token_img1_std=("erank_token_img1", "std"),
                erank_token_img2_mean=("erank_token_img2", "mean"),
                erank_token_img2_std=("erank_token_img2", "std"),
                n_chars=("char_id", "nunique"),
            )
        )
        token_summary.to_csv(out_dir / "erank_token_summary_by_script_layer.csv", index=False)

        tc_df = pd.DataFrame(token_count_rows)
        tc_df.to_csv(out_dir / "image_token_counts.csv", index=False)

        unique_angles = sorted([a for a in tc_df["angle"].unique().tolist() if a is not None])

        scripts_all = sorted([s for s in token_df["script_name"].unique().tolist() if s != "English"])
        if args.focus_scripts is not None and len(args.focus_scripts) > 0:
            scripts_focus = list(args.focus_scripts)
            if "English" not in scripts_focus:
                scripts_focus = ["English"] + scripts_focus
        else:
            # English vs all scripts
            scripts_focus = ["English"] + scripts_all

        for angle in unique_angles:
            # Float angles like 90.0 -> 90 for filenames/titles.
            angle_i = int(round(float(angle)))
            dfA = token_df[token_df["angle"] == angle].copy()
            dfA["group"] = np.where(dfA["script_name"] == "English", "English", "Omniglot")

            plot_group_erank_means(
                dfA,
                value_col="erank_token_img2",
                title=f"Token-level eRank (rotation={angle_i}deg) by layer",
                ylabel="eRank (tokens within image)",
                out_path=str(out_dir / f"group_erank_token_img2_by_layer_rotation_{angle_i}deg.png"),
            )
            plot_english_vs_selected_erank(
                dfA,
                value_col="erank_token_img2",
                scripts=scripts_focus,
                title=f"Token-level eRank (rotation={angle_i}deg) — English vs scripts",
                ylabel="eRank (tokens within image)",
                out_path=str(out_dir / f"english_vs_selected_erank_token_img2_rotation_{angle_i}deg.png"),
            )

        print(f"\n[OK] Saved rotation eRank (ranks) analysis to: {out_dir.resolve()}")
        print("  - erank_token_per_image.csv")
        print("  - erank_token_summary_by_script_layer.csv")
        print("  - image_token_counts.csv")

    print(f"Saved analysis to: {out_dir}")


if __name__ == "__main__":
    main()
