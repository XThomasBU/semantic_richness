#!/usr/bin/env python3
"""
Effective rank (eRank) for Qwen2.5-VL image-token hidden states, plus paper-style
analyses when --results_csv (and image roots) are provided:

  • Perimetric complexity P²/(4πA) per script (mean over characters), same as
    analyze_scale_illusion.py; Pearson r vs mean transformation accuracy per script.
  • Pearson r between mean eRank (scaled image / image2) and mean accuracy, by layer.

Accuracy is computed from the results CSV via groupby(script_name)["is_correct"].mean(),
matching analyze_scale_illusion.accuracy_rate aggregation.

Outputs (under --output_dir):
  - erank_token_per_image.csv
  - erank_token_summary_by_script_layer.csv
  - image_token_counts.csv
  - script_accuracy_from_results.csv          (if --results_csv)
  - script_complexity.csv                     (if --results_csv and alphabet/omniglot dirs)
  - merged_script_metrics.csv                 (if --results_csv)
  - correlations_summary.txt                  (if --results_csv)
  - erank_pearson_accuracy_by_layer.csv     (if --results_csv)
  - run_metadata.json                       (full runs: CLI snapshot for --plots_only)

Use --plot to also write eRank line plots; with complexity data, adds accuracy vs perimetric scatter.

After a full run, use --plots_only to regenerate plots and correlation tables from saved CSVs
(no model load). run_metadata.json stores scale_factor and focus_scripts for reproducible figures.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qwen_vl_utils import process_vision_info

DEFAULT_PROMPT = (
    "Compare the two images and decide if they show the same character. "
    "Ignore differences in scale, size, or resolution. Answer with exactly YES or NO."
)

OMNIGLOT_GROUP_EXCLUDE = {"hand_digits", "hand_english"}

# Dropped from all eRank merges, accuracy, correlations, and plots (never analyzed).
EXCLUDE_FROM_ANALYSIS = frozenset({"hand_digits"})


def drop_excluded_scripts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "script_name" not in df.columns:
        return df
    return df[~df["script_name"].isin(EXCLUDE_FROM_ANALYSIS)].copy()


def accuracy_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return np.nan
    return float(series.sum()) / float(len(series))


def pearson_r_xy(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def load_script_image_paths(
    alphabet_dir: Optional[str],
    omniglot_dir: Optional[str],
    scripts_in_df: List[str],
) -> Dict[str, List[str]]:
    script_paths: Dict[str, List[str]] = {}
    scripts_set = {s for s in scripts_in_df}
    if alphabet_dir:
        alphabet_path = Path(alphabet_dir) / "times_new_roman"
        if alphabet_path.exists():
            for char_dir in sorted(alphabet_path.glob("character*")):
                image_file = char_dir / "image.png"
                if image_file.exists():
                    script_paths.setdefault("English", []).append(str(image_file))
        else:
            print(f"[WARN] Alphabet directory not found: {alphabet_path}")

    if omniglot_dir:
        omniglot_path = Path(omniglot_dir) / "omniglot" / "omniglot-master" / "python" / "images_all"
        if omniglot_path.exists():
            allowed_set = {s.lower() for s in scripts_set if s != "English"}
            for script_dir in sorted([p for p in omniglot_path.iterdir() if p.is_dir()]):
                script_name = script_dir.name
                if allowed_set and script_name.lower() not in allowed_set:
                    continue
                for char_dir in sorted(script_dir.glob("character*")):
                    image_files = sorted(char_dir.glob("*.png"))
                    if image_files:
                        script_paths.setdefault(script_name, []).append(str(image_files[0]))
        else:
            print(f"[WARN] Omniglot directory not found: {omniglot_path}")

    return script_paths


def compute_ink_ratio(image_path: str, threshold: int = 200) -> float:
    img = Image.open(image_path).convert("L")
    arr = np.asarray(img)
    if arr.size == 0:
        return np.nan
    ink = arr < threshold
    return float(ink.mean())


def compute_perimetric_complexity(image_path: str, threshold: int = 200) -> float:
    """P²/(4πA) on binarized character (same convention as analyze_scale_illusion.py)."""
    img = Image.open(image_path).convert("L")
    arr = np.asarray(img)
    if arr.size == 0:
        return np.nan
    mask = arr < threshold
    area = float(mask.sum())
    if area == 0:
        return np.nan
    up = np.zeros_like(mask, dtype=bool)
    down = np.zeros_like(mask, dtype=bool)
    left = np.zeros_like(mask, dtype=bool)
    right = np.zeros_like(mask, dtype=bool)
    up[1:, :] = mask[:-1, :]
    down[:-1, :] = mask[1:, :]
    left[:, 1:] = mask[:, :-1]
    right[:, :-1] = mask[:, 1:]
    boundary = mask & ~(up & down & left & right)
    perimeter = float(boundary.sum())
    return (perimeter**2) / (4 * np.pi * area)


def compute_script_complexity(
    script_paths: Dict[str, List[str]],
    threshold: int = 200,
) -> pd.DataFrame:
    rows = []
    for script_name, paths in script_paths.items():
        ink_vals = []
        perim_vals = []
        for path in paths:
            try:
                ink_vals.append(compute_ink_ratio(path, threshold=threshold))
                perim_vals.append(compute_perimetric_complexity(path, threshold=threshold))
            except Exception as exc:
                print(f"[WARN] Failed complexity for {path}: {exc}")
        if len(ink_vals) == 0:
            continue
        rows.append(
            {
                "script_name": script_name,
                "ink_ratio": float(np.nanmean(ink_vals)),
                "perimetric_complexity": float(np.nanmean(perim_vals)),
                "n_images": len(ink_vals),
            }
        )
    return pd.DataFrame(rows)


def script_accuracy_from_results_df(df: pd.DataFrame) -> pd.DataFrame:
    """Mean is_correct per script (same as analyze_scale_illusion groupby)."""
    if "is_correct" not in df.columns or "script_name" not in df.columns:
        raise ValueError("Results CSV must include script_name and is_correct.")
    g = df.groupby("script_name", as_index=False).agg(
        accuracy=("is_correct", accuracy_rate),
        n_rows=("is_correct", "count"),
    )
    g["accuracy_pct"] = g["accuracy"] * 100.0
    return g


def resize_with_padding_pil(image_path: str, scale_factor: float, image_size: int = 336):
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
            spans.append((i + 1, j))
            i = j + 1
        else:
            i += 1

    if len(spans) < 2:
        raise RuntimeError("Could not find two image spans in input_ids.")
    return spans[:2]


def effective_rank(R: torch.Tensor) -> float:
    """eRank(R) = exp(H(p)), p = normalized singular values of centered R. R: [T, d]."""
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


def compute_eranks_for_pair(
    model,
    processor,
    image1_pil,
    image2_pil,
    prompt: str,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    inputs = prepare_inputs(image1_pil, image2_pil, prompt, processor)
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    spans = find_image_spans(inputs["input_ids"], processor)
    (s1, e1), (s2, e2) = spans

    T1 = int(e1 - s1)
    T2 = int(e2 - s2)

    hidden_states = outputs.hidden_states
    L = len(hidden_states) - 1

    erank_token_img1 = np.zeros(L, dtype=np.float32)
    erank_token_img2 = np.zeros(L, dtype=np.float32)

    for layer in range(L):
        h = hidden_states[layer + 1][0]
        tok1 = h[s1:e1]
        tok2 = h[s2:e2]

        erank_token_img1[layer] = effective_rank(tok1)
        erank_token_img2[layer] = effective_rank(tok2)

    return {
        "erank_token_img1": erank_token_img1,
        "erank_token_img2": erank_token_img2,
        "T1": T1,
        "T2": T2,
    }


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


def plot_group_means(df: pd.DataFrame, value_col: str, title: str, ylabel: str, out_path: Path):
    import matplotlib.pyplot as plt

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
    accuracy_map: Optional[Dict[str, float]] = None,
):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import seaborn as sns

    # Match the paper-style aesthetics from final_plots.py.
    mpl.rcParams.update(
        {
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "0.90",
            "grid.linewidth": 0.8,
            "axes.edgecolor": "0.55",
            "axes.linewidth": 0.8,
            "xtick.color": "0.20",
            "ytick.color": "0.20",
            "text.color": "0.15",
            "axes.labelcolor": "0.15",
            "axes.titlecolor": "0.15",
            "legend.frameon": True,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams.update(
        {
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    english_name = "English"
    english_df = df[df["script_name"] == english_name]
    if english_df.empty:
        raise ValueError("plot_english_vs_selected expected 'English' rows in df.")

    # Selected scripts (one subplot per script).
    # Omniglot scripts: red shades; handwritten English: green.
    desired_scripts = ["hand_english", "Greek", "Oriya", "Braille"]
    scripts_set = set(scripts)
    available = set(df["script_name"].unique().tolist())
    selected_scripts = [s for s in desired_scripts if s in scripts_set and s in available and not df[df["script_name"] == s].empty]
    if not selected_scripts:
        return

    # Order subplots by decreasing accuracy (when available).
    if accuracy_map is not None:
        selected_scripts = sorted(
            selected_scripts,
            key=lambda s: accuracy_map.get(s, float("-inf")),
            reverse=True,
        )

    # Colors pulled to match final_plots.py helpers.
    en_color = "#1f77b4"  # Times New Roman
    hand_green = "#2ca02c"  # Handwritten English
    omni_red = "#d62728"  # Omniglot

    # Use the same omniglot red, but vary markers across scripts.
    omni_markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "+"]

    # Precompute English stats once.
    piv_en = english_df.pivot_table(
        index=["script_name", "char_id"], columns="layer", values=value_col
    )
    mat_en = piv_en.to_numpy()
    mean_en = np.nanmean(mat_en, axis=0)
    std_en = np.nanstd(mat_en, axis=0)
    x = np.arange(len(mean_en))

    fig, ax = plt.subplots(figsize=(14.0, 6.2))

    # Plot English first (Times New Roman).
    times_new_roman_label = "Times New Roman"
    if accuracy_map is not None and "English" in accuracy_map and accuracy_map["English"] is not None:
        try:
            en_acc = float(accuracy_map["English"])
            if np.isfinite(en_acc):
                times_new_roman_label = f"Times New Roman (Acc={en_acc * 100.0:.1f}%)"
        except Exception:
            pass

    ax.plot(
        x,
        mean_en,
        marker="o",
        markersize=10,
        color=en_color,
        linewidth=2.2,
        label=times_new_roman_label,
    )

    omni_idx = 0
    for script in selected_scripts:
        sdf = df[df["script_name"] == script]
        piv = sdf.pivot_table(index=["script_name", "char_id"], columns="layer", values=value_col)
        mat = piv.to_numpy()
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)

        if script == "hand_english":
            other_color = hand_green
            other_marker = "o"
        else:
            other_color = omni_red
            other_marker = omni_markers[omni_idx % len(omni_markers)]
            omni_idx += 1

        # Accuracy in legend label when available (display-friendly names).
        if script == "hand_english":
            display_name = "Handwritten English"
        else:
            display_name = str(script)
        legend_label = display_name
        acc_pct = None
        if accuracy_map is not None and script in accuracy_map and accuracy_map[script] is not None:
            try:
                acc_val = float(accuracy_map[script])
                if np.isfinite(acc_val):
                    acc_pct = acc_val * 100.0
            except Exception:
                acc_pct = None
        if acc_pct is not None:
            legend_label = f"{display_name} (Acc={acc_pct:.1f}%)"

        ax.plot(
            x,
            mean,
            marker=other_marker,
            markersize=10,
            color=other_color,
            linewidth=2.2,
            label=legend_label,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="0.90", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("0.55")
    ax.spines["bottom"].set_color("0.55")

    ax.legend(
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor="0.80",
        ncol=len(selected_scripts) + 1,
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        handlelength=3.2,
    )
    ax.tick_params(axis="both", which="major", labelsize=10, width=0.8, length=5)
    plt.tight_layout(pad=1.0, rect=[0.0, 0.07, 1.0, 1.0])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _erank_pearson_by_layer(
    token_summary: pd.DataFrame,
    acc_df: pd.DataFrame,
    omniglot_only: bool,
) -> pd.DataFrame:
    m = token_summary.merge(acc_df[["script_name", "accuracy"]], on="script_name", how="inner")
    if omniglot_only:
        m = m[(m["script_name"] != "English") & (~m["script_name"].isin(OMNIGLOT_GROUP_EXCLUDE))]
    rows = []
    for layer, g in m.groupby("layer"):
        r = pearson_r_xy(g["erank_token_img2_mean"].values, g["accuracy"].values)
        rows.append({"layer": int(layer), "pearson_r": r, "n_scripts": len(g)})
    return pd.DataFrame(rows).sort_values("layer")


def build_erank_accuracy_correlation_table(
    token_summary: pd.DataFrame,
    acc_df: pd.DataFrame,
) -> pd.DataFrame:
    df_all = _erank_pearson_by_layer(token_summary, acc_df, omniglot_only=False)
    df_omni = _erank_pearson_by_layer(token_summary, acc_df, omniglot_only=True)
    out = df_all.rename(columns={"pearson_r": "pearson_r_all", "n_scripts": "n_scripts_all"})
    if df_omni.empty:
        out["pearson_r_omniglot"] = np.nan
        out["n_scripts_omniglot"] = np.nan
        return out.sort_values("layer")
    out = out.merge(
        df_omni.rename(columns={"pearson_r": "pearson_r_omniglot", "n_scripts": "n_scripts_omniglot"}),
        on="layer",
        how="outer",
    )
    return out.sort_values("layer")


def plot_accuracy_vs_perimetric(merged: pd.DataFrame, out_path: Path) -> None:
    """Scatter mean accuracy vs mean perimetric complexity per script (expects accuracy in [0,1])."""
    import matplotlib.pyplot as plt

    sub = merged[merged["perimetric_complexity"].notna()].copy()
    if sub.empty or len(sub) < 3:
        return
    r = pearson_r_xy(sub["perimetric_complexity"].to_numpy(), sub["accuracy"].to_numpy())
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.scatter(sub["perimetric_complexity"], sub["accuracy"] * 100.0, s=45, alpha=0.65, edgecolors="none")
    ax.set_xlabel("Perimetric complexity")
    ax.set_ylabel("Mean transformation accuracy (%)")
    ax.set_title("Accuracy vs perimetric complexity (per script)")
    ax.text(
        0.98,
        0.98,
        f"Pearson r = {r:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.85", alpha=0.9),
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_posthoc_outputs(
    out_dir: Path,
    token_summary: pd.DataFrame,
    acc_df: pd.DataFrame,
    complexity_df: Optional[pd.DataFrame],
    plot: bool,
) -> None:
    """Merge eRank summary with accuracy (and optional complexity); write correlation artifacts."""
    token_summary = drop_excluded_scripts(token_summary)
    acc_df = drop_excluded_scripts(acc_df)
    merged = token_summary.merge(acc_df, on="script_name", how="inner")
    if complexity_df is not None and not complexity_df.empty:
        merged = merged.merge(complexity_df, on="script_name", how="left")

    merged.to_csv(out_dir / "merged_script_metrics.csv", index=False)

    erank_corr = build_erank_accuracy_correlation_table(token_summary, acc_df)
    erank_corr.to_csv(out_dir / "erank_pearson_accuracy_by_layer.csv", index=False)

    lines: List[str] = []
    lines.append("Post-hoc correlations (Pearson r)\n")
    lines.append("=" * 48 + "\n")

    has_perim = complexity_df is not None and not complexity_df.empty and merged["perimetric_complexity"].notna().any()
    if has_perim:
        mc = merged[
            (merged["script_name"] != "English")
            & (~merged["script_name"].isin(OMNIGLOT_GROUP_EXCLUDE))
            & merged["perimetric_complexity"].notna()
        ]
        if len(mc) >= 3:
            r_perim = pearson_r_xy(mc["perimetric_complexity"].values, mc["accuracy"].values)
            lines.append(
                f"Perimetric complexity vs accuracy (Omniglot, excl. hand_english; hand_digits omitted): "
                f"r = {r_perim:.4f} (n={len(mc)})\n"
            )
        mall = merged[merged["perimetric_complexity"].notna()]
        if len(mall) >= 3:
            r_all = pearson_r_xy(mall["perimetric_complexity"].values, mall["accuracy"].values)
            lines.append(
                f"Perimetric complexity vs accuracy (all scripts with complexity): r = {r_all:.4f} (n={len(mall)})\n"
            )

    lines.append("\neRank (mean, image2) vs transformation accuracy — by layer:\n")
    lines.append(erank_corr.to_string(index=False) + "\n")

    with open(out_dir / "correlations_summary.txt", "w") as f:
        f.writelines(lines)

    print("\n[POST-HOC]")
    print("".join(lines))

    if plot and has_perim:
        mc_plot = merged[merged["perimetric_complexity"].notna()]
        if len(mc_plot) >= 3:
            plot_accuracy_vs_perimetric(mc_plot, out_dir / "accuracy_vs_perimetric_complexity.png")


def run_posthoc_metrics(
    out_dir: Path,
    results_df: pd.DataFrame,
    token_summary: pd.DataFrame,
    *,
    alphabet_dir: Optional[str],
    omniglot_dir: Optional[str],
    ink_threshold: int,
    plot: bool,
) -> None:
    results_df = drop_excluded_scripts(results_df)
    acc_df = script_accuracy_from_results_df(results_df)
    acc_df.to_csv(out_dir / "script_accuracy_from_results.csv", index=False)

    complexity_df: Optional[pd.DataFrame] = None
    if alphabet_dir or omniglot_dir:
        scripts = sorted(results_df["script_name"].unique().tolist())
        script_paths = load_script_image_paths(alphabet_dir, omniglot_dir, scripts)
        complexity_df = compute_script_complexity(script_paths, threshold=ink_threshold)
        if complexity_df is not None and not complexity_df.empty:
            complexity_df.to_csv(out_dir / "script_complexity.csv", index=False)

    write_posthoc_outputs(out_dir, token_summary, acc_df, complexity_df, plot)


def save_run_metadata(out_dir: Path, args: argparse.Namespace) -> None:
    """Snapshot CLI so --plots_only can reuse scale_factor / focus_scripts without re-inference."""
    meta = {
        "version": 1,
        "model": args.model,
        "scale_factor": args.scale_factor,
        "image_size": args.image_size,
        "prompt": args.prompt,
        "focus_scripts": list(args.focus_scripts) if args.focus_scripts else None,
        "omniglot_scripts": list(args.omniglot_scripts) if args.omniglot_scripts else None,
        "num_chars": args.num_chars,
        "results_csv": str(args.results_csv) if getattr(args, "results_csv", None) else None,
        "ink_threshold": getattr(args, "ink_threshold", 200),
        "alphabet_dir": args.alphabet_dir,
        "omniglot_dir": args.omniglot_dir,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_run_metadata(out_dir: Path) -> Dict:
    p = out_dir / "run_metadata.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def run_plots_only(args: argparse.Namespace) -> Path:
    """
    Regenerate plots and post-hoc tables from erank_token_per_image.csv (and optional saved accuracy/complexity).
    Does not load the VLM. Requires a prior full run into --output_dir.
    """
    out_dir = Path(args.output_dir)
    per_image = out_dir / "erank_token_per_image.csv"
    if not per_image.is_file():
        raise FileNotFoundError(
            f"Missing {per_image}. Run once without --plots_only to produce cached CSVs."
        )

    token_df = drop_excluded_scripts(pd.read_csv(per_image))
    summary_path = out_dir / "erank_token_summary_by_script_layer.csv"
    if summary_path.is_file():
        token_summary = drop_excluded_scripts(pd.read_csv(summary_path))
    else:
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

    meta = load_run_metadata(out_dir)
    scale_factor = float(meta.get("scale_factor", args.scale_factor))
    focus_scripts = args.focus_scripts
    if focus_scripts is None and meta.get("focus_scripts") is not None:
        focus_scripts = list(meta["focus_scripts"])

    acc_path = out_dir / "script_accuracy_from_results.csv"
    accuracy_map: Optional[Dict[str, float]] = None
    comp_path = out_dir / "script_complexity.csv"
    if acc_path.is_file():
        acc_df = drop_excluded_scripts(pd.read_csv(acc_path))
        if "accuracy" not in acc_df.columns or "script_name" not in acc_df.columns:
            raise ValueError(f"{acc_path} must contain script_name and accuracy.")
        accuracy_map = dict(zip(acc_df["script_name"].astype(str), acc_df["accuracy"].astype(float)))
        complexity_df: Optional[pd.DataFrame] = None
        if comp_path.is_file():
            complexity_df = pd.read_csv(comp_path)
        write_posthoc_outputs(out_dir, token_summary, acc_df, complexity_df, plot=args.plot)
    else:
        print(f"[plots_only] No {acc_path.name}; skipping merged metrics and correlations.")

    if args.plot:
        token_plot_df = token_df.copy()
        token_plot_df["group"] = np.where(token_plot_df["script_name"] == "English", "English", "Omniglot")
        token_plot_df = token_plot_df[
            ~((token_plot_df["group"] == "Omniglot") & (token_plot_df["script_name"].isin(OMNIGLOT_GROUP_EXCLUDE)))
        ]

        plot_group_means(
            token_plot_df,
            value_col="erank_token_img2",
            title=f"Token-level eRank (image2) by layer (scale={scale_factor})",
            ylabel="eRank (tokens within image)",
            out_path=out_dir / "group_erank_token_img2_by_layer.png",
        )

        if not focus_scripts:
            focus_scripts = sorted(token_df["script_name"].unique().tolist())
        focusA = token_df[token_df["script_name"].isin(focus_scripts)].copy()
        if not focusA.empty:
            plot_english_vs_selected(
                focusA,
                value_col="erank_token_img2",
                scripts=[s for s in focus_scripts if s in focusA["script_name"].unique()],
                title=f"Token-level eRank (image2) — English vs selected (scale={scale_factor})",
                ylabel="eRank (tokens within image)",
                out_path=out_dir / "english_vs_selected_erank_token_img2.png",
                accuracy_map=accuracy_map,
            )

    print(f"\n[plots_only] Done. Outputs updated under {out_dir.resolve()}")
    return out_dir


def run_effective_rank_analysis(args: argparse.Namespace) -> Path:
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

    token_rows = []
    token_count_rows = []

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

    if args.omniglot_dir:
        scripts = args.omniglot_scripts
        if scripts is None:
            base = Path(args.omniglot_dir) / "omniglot" / "omniglot-master" / "python" / "images_all"
            scripts = sorted([p.name for p in base.iterdir() if p.is_dir()])

        for script_name in tqdm(scripts, desc="Omniglot scripts", unit="script"):
            if script_name in EXCLUDE_FROM_ANALYSIS:
                continue
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

    token_df = drop_excluded_scripts(pd.DataFrame(token_rows))
    token_df.to_csv(out_dir / "erank_token_per_image.csv", index=False)

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

    tc_df = pd.DataFrame(token_count_rows)
    tc_df.to_csv(out_dir / "image_token_counts.csv", index=False)

    print("\n[TOKEN COUNT CHECK]")
    print("Unique T1 counts:", sorted(tc_df["T1"].unique().tolist()))
    print("Unique T2 counts:", sorted(tc_df["T2"].unique().tolist()))
    print("\nT2 nunique by script (top 20):")
    print(tc_df.groupby("script_name")["T2"].nunique().sort_values(ascending=False).head(20).to_string())

    if getattr(args, "results_csv", None):
        rp = Path(args.results_csv)
        if not rp.is_file():
            print(f"\n[WARN] --results_csv not found: {rp}")
        else:
            results_df = pd.read_csv(rp)
            run_posthoc_metrics(
                out_dir,
                results_df,
                token_summary,
                alphabet_dir=args.alphabet_dir,
                omniglot_dir=args.omniglot_dir,
                ink_threshold=getattr(args, "ink_threshold", 200),
                plot=args.plot,
            )

    if args.plot:
        # Accuracy values are saved by run_posthoc_metrics when --results_csv is provided.
        accuracy_map: Optional[Dict[str, float]] = None
        acc_path = out_dir / "script_accuracy_from_results.csv"
        if acc_path.is_file():
            acc_df_plot = drop_excluded_scripts(pd.read_csv(acc_path))
            if "accuracy" in acc_df_plot.columns and "script_name" in acc_df_plot.columns:
                accuracy_map = dict(zip(acc_df_plot["script_name"].astype(str), acc_df_plot["accuracy"].astype(float)))

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
                accuracy_map=accuracy_map,
            )

    print(f"\n[OK] Saved CSVs to: {out_dir.resolve()}")
    print("  - erank_token_per_image.csv")
    print("  - erank_token_summary_by_script_layer.csv")
    print("  - image_token_counts.csv")
    if getattr(args, "results_csv", None) and Path(args.results_csv).is_file():
        print("  - script_accuracy_from_results.csv")
        print("  - merged_script_metrics.csv")
        print("  - erank_pearson_accuracy_by_layer.csv")
        print("  - correlations_summary.txt")
        if args.alphabet_dir or args.omniglot_dir:
            print("  - script_complexity.csv (if image roots found)")
            if args.plot:
                print("  - accuracy_vs_perimetric_complexity.png")
    if args.plot:
        print("  - group_erank_token_img2_by_layer.png")
        print("  - english_vs_selected_erank_token_img2.png")

    save_run_metadata(out_dir, args)

    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL image-token effective rank (per layer).")
    parser.add_argument(
        "--plots_only",
        action="store_true",
        help=(
            "Skip the model: load erank_token_per_image.csv from --output_dir, regenerate plots "
            "(and correlations if script_accuracy_from_results.csv exists). "
            "Use after a full run; reads scale_factor/focus_scripts from run_metadata.json when present."
        ),
    )
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
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also write PNG plots (matplotlib). Default: CSV-only.",
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default=None,
        help=(
            "Scale-illusion results CSV with script_name and is_correct. "
            "Exports script-level accuracy, perimetric complexity (if --alphabet_dir / --omniglot_dir), "
            "and Pearson correlations vs eRank by layer."
        ),
    )
    parser.add_argument(
        "--ink_threshold",
        type=int,
        default=200,
        help="Grayscale threshold for ink mask when computing perimetric complexity.",
    )
    args = parser.parse_args()
    if args.plots_only:
        run_plots_only(args)
    else:
        run_effective_rank_analysis(args)


if __name__ == "__main__":
    main()
