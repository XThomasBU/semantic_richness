#!/usr/bin/env python3
"""
Comprehensive analysis for Scale Illusion CSV results.

Focus: high vs low visual prior (Alphabet/English vs Omniglot scripts).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from matplotlib.lines import Line2D
try:
    import umap  # type: ignore
except Exception:
    umap = None

torch = None
Qwen2_5_VLForConditionalGeneration = None
AutoProcessor = None
AutoTokenizer = None
process_vision_info = None
_QWEN_IMPORT_ERROR = None
try:
    from scipy.stats import norm
except ImportError:  # optional dependency for SDT
    norm = None

# Base blue for recall/specificity (--ag): same as model_category_perf.PALETTE and bar style
try:
    from experiments.analysis.model_category_perf import PALETTE as _MODEL_CAT_PALETTE
    _AG_BLUE_BASE = _MODEL_CAT_PALETTE["Qwen2.5-VL-7B"]
    _AG_BLUE_LIGHT = "#7AA5D9"  # lighter shade of same blue
except ImportError:
    _AG_BLUE_BASE = "#4C78A8"
    _AG_BLUE_LIGHT = "#7AA5D9"
_AG_BAR_ALPHA = 0.92

# Global plot styling for CVPR publication quality
sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Leave space on right for legend when placed outside
_LEGEND_RIGHT = 0.78
_COLOR_ENGLISH = "#1f77b4"
_COLOR_HAND = "#2ca02c"
_COLOR_OMNI = "#d62728"


def _script_label_color(name: str) -> str:
    if name in ("English", "Times New Roman"):
        return _COLOR_ENGLISH
    if name in {"hand_digits", "hand_english", "Handwritten English"}:
        return _COLOR_HAND
    return _COLOR_OMNI


def _script_display_name(name: str) -> str:
    """Display name for plotting: English -> Times New Roman, hand_english -> Handwritten English."""
    if name == "English":
        return "Times New Roman"
    if name == "hand_english":
        return "Handwritten English"
    return name


def _apply_script_display_names(ax: plt.Axes) -> None:
    """Replace English -> Times New Roman, hand_english -> Handwritten English on axis tick labels."""
    ylabels = [_script_display_name(lbl.get_text()) for lbl in ax.get_yticklabels()]
    xlabels = [_script_display_name(lbl.get_text()) for lbl in ax.get_xticklabels()]
    if ylabels:
        ax.set_yticklabels(ylabels)
    if xlabels:
        ax.set_xticklabels(xlabels)


def _apply_script_label_colors(ax: plt.Axes) -> None:
    for tick in ax.get_yticklabels():
        tick.set_color(_script_label_color(tick.get_text()))


def _legend_outside(ax: plt.Axes, **kwargs: object) -> None:
    """Place legend outside plot area, to the right, no overlap."""
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=9,
        frameon=True,
        framealpha=1.0,
        edgecolor="0.8",
        **kwargs,
    )


def _legend_bottom(ax: plt.Axes, **kwargs: object) -> None:
    """Place legend below plot area, centered."""
    ax.legend(
        bbox_to_anchor=(0.5, -0.18),
        loc="upper center",
        ncol=2,
        fontsize=9,
        frameon=True,
        framealpha=1.0,
        edgecolor="0.8",
        **kwargs,
    )


def _apply_clean_style(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("0.2")


def save_fig(fig: plt.Figure, path: Path, legend_right: bool = False, legend_bottom: bool = False, also_pdf: bool = False) -> None:
    """Save figure with publication-ready layout."""
    if legend_right:
        fig.tight_layout(rect=[0, 0, _LEGEND_RIGHT, 1], pad=0.3)
    elif legend_bottom:
        fig.tight_layout(rect=[0, 0.08, 1, 1], pad=0.3)
    else:
        fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        a_sample = rng.choice(a, size=len(a), replace=True)
        b_sample = rng.choice(b, size=len(b), replace=True)
        diffs.append(a_sample.mean() - b_sample.mean())
    diffs = np.array(diffs)
    return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_std = np.sqrt(((a.var(ddof=1) + b.var(ddof=1)) / 2.0))
    if pooled_std == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled_std


def accuracy_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return np.nan
    return float(series.sum()) / float(len(series))


def _safe_script_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


# Exclude from Omniglot aggregates (hand_* are custom; times_new_roman duplicates alphabet English)
OMNIGLOT_GROUP_EXCLUDE = {"hand_digits", "hand_english", "times_new_roman"}


def load_script_image_paths(
    alphabet_dir: str | None,
    omniglot_dir: str | None,
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
    return (perimeter ** 2) / (4 * np.pi * area)


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


def plot_accuracy_vs_complexity(
    df: pd.DataFrame,
    complexity_df: pd.DataFrame,
    out_dir: Path,
    focus_scripts: Optional[List[str]] = None,
) -> None:
    if complexity_df.empty:
        return
    acc_df = (
        df.groupby("script_name")["is_correct"]
        .apply(accuracy_rate)
        .reset_index(name="accuracy")
    )
    acc_df["accuracy"] *= 100
    merged = acc_df.merge(complexity_df, on="script_name", how="inner")
    if merged.empty:
        return
    merged["group"] = np.where(
        merged["script_name"] == "English",
        "English",
        np.where(merged["script_name"].isin(OMNIGLOT_GROUP_EXCLUDE), "Custom", "Omniglot"),
    )
    plot_merged = merged[merged["group"].isin(["English", "Omniglot"])]
    merged_for_labels = plot_merged if not plot_merged.empty else merged
    data_to_plot = plot_merged if not plot_merged.empty else merged

    def select_labels(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        by_acc = frame.sort_values("accuracy", ascending=False)
        by_ink = frame.sort_values("ink_ratio", ascending=False)
        picks = pd.concat(
            [
                by_acc.head(3),
                by_acc.tail(3),
                by_ink.head(2),
                by_ink.tail(2),
                frame[frame["script_name"] == "English"],
                frame[frame["script_name"] == "Greek"],
                frame[frame["script_name"] == "Latin"],
            ]
        ).drop_duplicates(subset=["script_name"])
        return picks

    label_df = select_labels(merged_for_labels)
    if focus_scripts:
        label_df = merged_for_labels[merged_for_labels["script_name"].isin(focus_scripts)]

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.scatter(
        data_to_plot["ink_ratio"],
        data_to_plot["accuracy"],
        s=60,
        color="#b0b0b0",
        alpha=0.35,
        edgecolor="none",
    )
    focus = focus_scripts if focus_scripts else label_df["script_name"].tolist()
    for name in ["hand_digits", "hand_english"]:
        if name in merged["script_name"].values and name not in focus:
            focus.append(name)
    focus = [s for s in focus if s in merged["script_name"].unique()]
    base_colors = list(plt.cm.tab10(np.linspace(0, 1, 10)))
    script_palette = base_colors[1:] if len(base_colors) > 1 else base_colors
    color_map = {"English": _COLOR_ENGLISH}
    color_idx = 0
    for script in focus:
        if script == "English":
            continue
        color_map[script] = script_palette[color_idx % len(script_palette)]
        color_idx += 1
    for script in focus:
        sub = merged[merged["script_name"] == script]
        if sub.empty:
            continue
        ax.scatter(
            sub["ink_ratio"],
            sub["accuracy"],
            s=60,
            color=color_map.get(script, _COLOR_OMNI),
            alpha=0.9,
            edgecolor="none",
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["script_name"],
                (row["ink_ratio"], row["accuracy"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                fontweight=None,
                color=color_map.get(script, _COLOR_OMNI),
            )
    ax.set_xlabel("Ink Ratio")
    ax.set_ylabel("Accuracy (%)")
    pearson = float(merged["ink_ratio"].corr(merged["accuracy"]))
    ax.text(
        0.98,
        0.98,
        f"Pearson r = {pearson:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.85", alpha=0.9),
    )
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("normal")
    _apply_clean_style(ax)
    sns.despine()
    save_fig(fig, out_dir / "accuracy_vs_ink_ratio.png")

    if merged["perimetric_complexity"].notna().any():
        fig, ax = plt.subplots(figsize=(5.8, 4.6))
        ax.scatter(
            merged["perimetric_complexity"],
            merged["accuracy"],
            s=60,
            color="#b0b0b0",
            alpha=0.35,
            edgecolor="none",
        )
        for script in focus:
            sub = merged[merged["script_name"] == script]
            if sub.empty:
                continue
            ax.scatter(
                sub["perimetric_complexity"],
                sub["accuracy"],
                s=60,
                color=color_map.get(script, _COLOR_OMNI),
                alpha=0.9,
                edgecolor="none",
            )
            for _, row in sub.iterrows():
                ax.annotate(
                    row["script_name"],
                    (row["perimetric_complexity"], row["accuracy"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                    fontweight=None,
                    color=color_map.get(script, _COLOR_OMNI),
                )
        ax.set_xlabel("Perimetric Complexity")
        ax.set_ylabel("Accuracy (%)")
        pearson = float(merged["perimetric_complexity"].corr(merged["accuracy"]))
        ax.text(
            0.98,
            0.98,
            f"Pearson r = {pearson:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.85", alpha=0.9),
        )
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("normal")
        _apply_clean_style(ax)
        sns.despine()
        save_fig(fig, out_dir / "accuracy_vs_perimetric_complexity.png")


def plot_d_prime_vs_complexity(
    df: pd.DataFrame,
    complexity_df: pd.DataFrame,
    out_dir: Path,
    focus_scripts: Optional[List[str]] = None,
) -> None:
    """Plot SDT d' vs perimetric complexity (scatter, script labels)."""
    if norm is None or complexity_df.empty or "scale_factor" not in df.columns:
        return
    if not complexity_df["perimetric_complexity"].notna().any():
        return

    stats = []
    for (dataset, script), group in df.groupby(["dataset", "script_name"]):
        n_pos = group[group["is_positive"] == True].shape[0]
        n_neg = group[group["is_positive"] == False].shape[0]
        if n_pos == 0 or n_neg == 0:
            continue
        epsilon = 1e-6
        hits = group[(group["is_positive"] == True) & (group["is_correct"] == True)].shape[0]
        false_alarms = group[(group["is_positive"] == False) & (group["is_correct"] == False)].shape[0]
        hit_rate = max(min(hits / n_pos, 1 - epsilon), epsilon)
        fa_rate = max(min(false_alarms / n_neg, 1 - epsilon), epsilon)
        d_prime = norm.ppf(hit_rate) - norm.ppf(fa_rate)
        stats.append({"dataset": dataset, "script_name": script, "d_prime": d_prime})

    if not stats:
        return
    d_prime_df = pd.DataFrame(stats)
    merged = d_prime_df.merge(complexity_df, on="script_name", how="inner")
    merged = merged[merged["perimetric_complexity"].notna()]
    if merged.empty:
        return

    merged["group"] = np.where(
        merged["script_name"] == "English",
        "English",
        np.where(merged["script_name"].isin(OMNIGLOT_GROUP_EXCLUDE), "Custom", "Omniglot"),
    )
    data_to_plot = merged[merged["group"].isin(["English", "Omniglot"])]

    def select_labels(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        if focus_scripts:
            return frame[frame["script_name"].isin(focus_scripts)]
        by_d = frame.sort_values("d_prime", ascending=False)
        by_pc = frame.sort_values("perimetric_complexity", ascending=False)
        picks = pd.concat(
            [
                by_d.head(4),
                by_d.tail(4),
                by_pc.head(2),
                by_pc.tail(2),
                frame[frame["script_name"] == "English"],
            ]
        ).drop_duplicates(subset=["script_name"])
        return picks

    label_df = select_labels(data_to_plot) if not data_to_plot.empty else merged.iloc[:0]

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    # Base scatter for all points (light gray)
    ax.scatter(
        merged["perimetric_complexity"],
        merged["d_prime"],
        s=60,
        color="#b0b0b0",
        alpha=0.35,
        edgecolor="none",
    )

    focus = focus_scripts if focus_scripts else label_df["script_name"].tolist()
    focus = [s for s in focus if s in merged["script_name"].unique()]
    # Color mapping aligned with attn_out style: English blue, others tab10 (excluding first)
    base_colors = list(plt.cm.tab10(np.linspace(0, 1, 10)))
    script_palette = base_colors[1:] if len(base_colors) > 1 else base_colors
    color_map = {"English": _COLOR_ENGLISH}
    color_idx = 0
    for script in focus:
        if script == "English":
            continue
        color_map[script] = script_palette[color_idx % len(script_palette)]
        color_idx += 1

    for script in focus:
        sub = merged[merged["script_name"] == script]
        if sub.empty:
            continue
        ax.scatter(
            sub["perimetric_complexity"],
            sub["d_prime"],
            s=60,
            color=color_map.get(script, _COLOR_OMNI),
            alpha=0.9,
            edgecolor="none",
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["script_name"],
                (row["perimetric_complexity"], row["d_prime"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                color=color_map.get(script, _COLOR_OMNI),
            )
    ax.set_xlabel("Perimetric Complexity")
    ax.set_ylabel("d'")
    sns.despine()
    save_fig(fig, out_dir / "d_prime_vs_perimetric_complexity.png")


def resize_with_padding(
    image_path: str,
    scale_factor: float,
    output_path: str,
    image_size: int = 336,
) -> str:
    img = Image.open(image_path).convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    new_char_size = int(image_size * scale_factor)
    img_small = img.resize((new_char_size, new_char_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    paste_x = (image_size - new_char_size) // 2
    paste_y = (image_size - new_char_size) // 2
    canvas.paste(img_small, (paste_x, paste_y))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "PNG")
    return output_path


def _load_qwen_deps() -> bool:
    global torch
    global Qwen2_5_VLForConditionalGeneration
    global AutoProcessor
    global AutoTokenizer
    global process_vision_info
    global _QWEN_IMPORT_ERROR
    if torch is not None and Qwen2_5_VLForConditionalGeneration is not None and AutoProcessor is not None:
        return True
    try:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import torch as _torch
        from transformers import Qwen2_5_VLForConditionalGeneration as _Qwen2_5_VLForConditionalGeneration
        from transformers import AutoProcessor as _AutoProcessor
        from transformers import AutoTokenizer as _AutoTokenizer
        from qwen_vl_utils import process_vision_info as _process_vision_info
        torch = _torch
        Qwen2_5_VLForConditionalGeneration = _Qwen2_5_VLForConditionalGeneration
        AutoProcessor = _AutoProcessor
        AutoTokenizer = _AutoTokenizer
        process_vision_info = _process_vision_info
        _QWEN_IMPORT_ERROR = None
        return True
    except Exception as exc:
        _QWEN_IMPORT_ERROR = exc
        return False


def _torch_ready() -> bool:
    return _load_qwen_deps()


def find_image_spans(input_ids: "torch.Tensor", processor) -> List[Tuple[int, int]]:
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
    if process_vision_info is None:
        raise RuntimeError("process_vision_info unavailable (missing Qwen utilities).")
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


def extract_pair_layer_embeddings(
    model,
    processor,
    image1: str,
    image2: str,
    prompt: str,
    device: "torch.device",
) -> Tuple[np.ndarray, np.ndarray]:
    inputs = prepare_inputs(image1, image2, prompt, processor)
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    spans = find_image_spans(inputs["input_ids"], processor)
    (s1, e1), (s2, e2) = spans
    hidden_states = outputs.hidden_states
    img1_layers = []
    img2_layers = []
    for h in hidden_states:
        img1_feat = h[0, s1:e1].mean(dim=0).detach().to(torch.float32).cpu().numpy()
        img2_feat = h[0, s2:e2].mean(dim=0).detach().to(torch.float32).cpu().numpy()
        img1_layers.append(img1_feat)
        img2_layers.append(img2_feat)
    return np.stack(img1_layers, axis=0), np.stack(img2_layers, axis=0)


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def plot_manifold_pca(
    model,
    processor,
    script_name: str,
    image_paths: List[str],
    out_dir: Path,
    prompt: str,
    device: "torch.device",
    image_size: int,
    small_scale: float,
    large_scale: float,
) -> None:
    if len(image_paths) < 2:
        return
    script_dir = out_dir / "manifold_pca"
    script_dir.mkdir(parents=True, exist_ok=True)
    img_a = image_paths[0]
    img_b = image_paths[1]
    small_a = script_dir / script_name / "small_a.png"
    large_a = script_dir / script_name / "large_a.png"
    small_b = script_dir / script_name / "small_b.png"
    large_b = script_dir / script_name / "large_b.png"
    resize_with_padding(img_a, small_scale, str(small_a), image_size=image_size)
    resize_with_padding(img_a, large_scale, str(large_a), image_size=image_size)
    resize_with_padding(img_b, small_scale, str(small_b), image_size=image_size)
    resize_with_padding(img_b, large_scale, str(large_b), image_size=image_size)

    emb_a_small, emb_a_large = extract_pair_layer_embeddings(
        model, processor, str(small_a), str(large_a), prompt, device
    )
    emb_b_small, emb_b_large = extract_pair_layer_embeddings(
        model, processor, str(small_b), str(large_b), prompt, device
    )
    last = -1
    feats = np.vstack([emb_a_small[last], emb_a_large[last], emb_b_small[last], emb_b_large[last]])
    coords = pca_2d(feats)
    labels = ["Small A", "Large A", "Small B", "Large B"]
    colors = ["#1f77b4", "#1f77b4", "#d62728", "#d62728"]
    markers = ["o", "s", "o", "s"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, (xv, yv) in enumerate(coords):
        ax.scatter(xv, yv, color=colors[i], marker=markers[i], s=90)
        ax.annotate(labels[i], (xv, yv), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    sns.despine()
    save_fig(fig, script_dir / f"manifold_pca_{_safe_script_name(script_name)}.png")


def plot_global_manifold_pca(
    model,
    processor,
    script_paths: Dict[str, List[str]],
    out_dir: Path,
    prompt: str,
    device: "torch.device",
    image_size: int,
    small_scale: float,
    large_scale: float,
    highlight_scripts: List[str],
) -> None:
    if not script_paths:
        return
    all_feats = []
    meta = []
    for script_name, paths in script_paths.items():
        if len(paths) < 2:
            continue
        img_a = paths[0]
        img_b = paths[1]
        script_dir = out_dir / "manifold_pca"
        script_dir.mkdir(parents=True, exist_ok=True)
        small_a = script_dir / script_name / "small_a.png"
        large_a = script_dir / script_name / "large_a.png"
        small_b = script_dir / script_name / "small_b.png"
        large_b = script_dir / script_name / "large_b.png"
        resize_with_padding(img_a, small_scale, str(small_a), image_size=image_size)
        resize_with_padding(img_a, large_scale, str(large_a), image_size=image_size)
        resize_with_padding(img_b, small_scale, str(small_b), image_size=image_size)
        resize_with_padding(img_b, large_scale, str(large_b), image_size=image_size)
        emb_a_small, emb_a_large = extract_pair_layer_embeddings(
            model, processor, str(small_a), str(large_a), prompt, device
        )
        emb_b_small, emb_b_large = extract_pair_layer_embeddings(
            model, processor, str(small_b), str(large_b), prompt, device
        )
        last = -1
        feats = np.vstack([emb_a_small[last], emb_a_large[last], emb_b_small[last], emb_b_large[last]])
        all_feats.append(feats)
        meta.extend([(script_name, "Small A"), (script_name, "Large A"), (script_name, "Small B"), (script_name, "Large B")])
    if not all_feats:
        return
    feats = np.vstack(all_feats)
    coords = pca_2d(feats)
    fig, ax = plt.subplots(figsize=(9, 7))
    highlight_set = {s.lower() for s in highlight_scripts}
    for (script_name, label), (xv, yv) in zip(meta, coords):
        is_highlight = script_name.lower() in highlight_set
        color = _script_label_color(script_name) if is_highlight else "#b0b0b0"
        ax.scatter(
            xv,
            yv,
            color=color,
            alpha=0.9 if is_highlight else 0.35,
            s=70 if is_highlight else 25,
        )
        if is_highlight:
            ax.annotate(
                f"{script_name} ({label})",
                (xv, yv),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                color=_script_label_color(script_name),
            )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    sns.despine()
    save_fig(fig, out_dir / "manifold_pca_all_scripts.png")


def _cosine_sim_matrix(x: np.ndarray) -> np.ndarray:
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    return x @ x.T


def _flatten_upper(mat: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(mat, k=1)
    return mat[idx]


def compute_rsa_for_script(
    model,
    processor,
    script_name: str,
    image_paths: List[str],
    out_dir: Path,
    prompt: str,
    device: "torch.device",
    image_size: int,
    small_scale: float,
    large_scale: float,
    num_chars: int,
) -> None:
    if len(image_paths) < 2:
        return
    script_dir = out_dir / "rsa"
    script_dir.mkdir(parents=True, exist_ok=True)
    use_paths = image_paths[:num_chars]
    images = []
    for p in use_paths:
        char_id = Path(p).parent.name
        small_path = script_dir / script_name / f"{char_id}_small.png"
        large_path = script_dir / script_name / f"{char_id}_large.png"
        resize_with_padding(p, small_scale, str(small_path), image_size=image_size)
        resize_with_padding(p, large_scale, str(large_path), image_size=image_size)
        images.append((char_id, str(small_path)))
        images.append((char_id, str(large_path)))

    if len(images) % 2 == 1:
        images.append(images[-1])

    layer_features: Optional[List[List[np.ndarray]]] = None
    for i in range(0, len(images), 2):
        _, img1 = images[i]
        _, img2 = images[i + 1]
        emb1, emb2 = extract_pair_layer_embeddings(model, processor, img1, img2, prompt, device)
        if layer_features is None:
            layer_features = [[] for _ in range(emb1.shape[0])]
        for layer_idx in range(emb1.shape[0]):
            layer_features[layer_idx].append(emb1[layer_idx])
            layer_features[layer_idx].append(emb2[layer_idx])

    if layer_features is None:
        return

    img_arrays = []
    for _, p in images:
        arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        img_arrays.append(arr.reshape(-1))
    pixel_sim = _cosine_sim_matrix(np.vstack(img_arrays))
    id_labels = [cid for cid, _ in images]
    identity = np.zeros_like(pixel_sim)
    for i in range(len(id_labels)):
        for j in range(len(id_labels)):
            identity[i, j] = 1.0 if id_labels[i] == id_labels[j] else 0.0

    pixel_vals = _flatten_upper(pixel_sim)
    id_vals = _flatten_upper(identity)
    rows = []
    for layer_idx, feats in enumerate(layer_features):
        feat_mat = np.vstack(feats)
        model_sim = _cosine_sim_matrix(feat_mat)
        model_vals = _flatten_upper(model_sim)
        if np.std(model_vals) == 0:
            corr_pixel = np.nan
            corr_id = np.nan
        else:
            corr_pixel = float(np.corrcoef(model_vals, pixel_vals)[0, 1])
            corr_id = float(np.corrcoef(model_vals, id_vals)[0, 1])
        rows.append({"layer": layer_idx, "corr_pixel": corr_pixel, "corr_identity": corr_id})
    rsa_df = pd.DataFrame(rows)
    rsa_df.to_csv(script_dir / f"rsa_{_safe_script_name(script_name)}.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(rsa_df["layer"], rsa_df["corr_pixel"], label="Pixel Oracle", marker="o")
    ax.plot(rsa_df["layer"], rsa_df["corr_identity"], label="Identity Oracle", marker="o")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Pearson r")
    _legend_outside(ax)
    sns.despine()
    save_fig(fig, script_dir / f"rsa_{_safe_script_name(script_name)}.png", legend_right=True)


def analyze_tokenizer_fertility(
    tokenizer,
    texts_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    if texts_df.empty:
        return
    unk_id = tokenizer.unk_token_id
    rows = []
    for row in texts_df.itertuples():
        text = getattr(row, "text", None)
        if text is None or not isinstance(text, str) or text == "":
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        unk_count = sum(1 for i in ids if unk_id is not None and i == unk_id)
        rows.append(
            {
                "script_name": getattr(row, "script_name", "unknown"),
                "text": text,
                "token_count": len(ids),
                "unk_count": unk_count,
            }
        )
    if not rows:
        return
    tok_df = pd.DataFrame(rows)
    tok_df["unk_rate"] = tok_df["unk_count"] / tok_df["token_count"].replace(0, np.nan)
    tok_df.to_csv(out_dir / "tokenizer_fertility.csv", index=False)

    summary = (
        tok_df.groupby("script_name")
        .agg(
            mean_tokens=("token_count", "mean"),
            mean_unk_rate=("unk_rate", "mean"),
            n=("token_count", "size"),
        )
        .reset_index()
        .sort_values("mean_tokens", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(6, max(4, 0.25 * len(summary))))
    sns.barplot(
        data=summary,
        y="script_name",
        x="mean_tokens",
        hue="script_name",
        palette="crest",
        ax=ax,
        legend=False,
    )
    ax.set_xlabel("Mean Token Count")
    ax.set_ylabel("Script")
    sns.despine()
    save_fig(fig, out_dir / "tokenizer_fertility_tokens.png")

    fig, ax = plt.subplots(figsize=(6, max(4, 0.25 * len(summary))))
    sns.barplot(
        data=summary,
        y="script_name",
        x="mean_unk_rate",
        hue="script_name",
        palette="mako",
        ax=ax,
        legend=False,
    )
    ax.set_xlabel("Mean UNK Rate")
    ax.set_ylabel("Script")
    sns.despine()
    save_fig(fig, out_dir / "tokenizer_fertility_unk_rate.png")


def plot_complexity_vs_rank(
    complexity_df: pd.DataFrame,
    attn_df: pd.DataFrame,
    out_dir: Path,
    pair_type: str = "positive",
) -> None:
    if attn_df.empty or complexity_df.empty:
        return
    if "rank_img2" not in attn_df.columns:
        return
    if pair_type != "all":
        attn_df = attn_df[attn_df["pair_type"] == pair_type]
    if attn_df.empty:
        return
    last_layer = attn_df["layer"].max()
    last_df = attn_df[attn_df["layer"] == last_layer]
    rank_df = (
        last_df.groupby("script_name")["rank_img2"]
        .mean()
        .reset_index(name="rank_img2")
    )
    merged = rank_df.merge(complexity_df, on="script_name", how="inner")
    if merged.empty:
        return
    merged["group"] = np.where(
        merged["script_name"] == "English",
        "English",
        np.where(merged["script_name"].isin(OMNIGLOT_GROUP_EXCLUDE), "Custom", "Omniglot"),
    )
    # Plot only English and Omniglot (exclude Custom from group comparison)
    plot_merged = merged[merged["group"].isin(["English", "Omniglot"])]
    if plot_merged.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.scatterplot(
        data=plot_merged,
        x="perimetric_complexity",
        y="rank_img2",
        hue="group",
        palette={"English": _COLOR_ENGLISH, "Omniglot": _COLOR_OMNI, "Custom": _COLOR_HAND},
        s=60,
        ax=ax,
    )
    for _, row in plot_merged.iterrows():
        ax.annotate(
            row["script_name"],
            (row["perimetric_complexity"], row["rank_img2"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color=_script_label_color(row["script_name"]),
        )
    ax.set_xlabel("Perimetric Complexity")
    ax.set_ylabel("Effective Rank (Last Layer)")
    _legend_outside(ax)
    sns.despine()
    save_fig(fig, out_dir / "perimetric_complexity_vs_rank.png", legend_right=True)


def plot_rank_divergence_vs_accuracy(
    df: pd.DataFrame,
    attn_df: pd.DataFrame,
    out_dir: Path,
    pair_type: str = "positive",
    focus_scripts: Optional[List[str]] = None,
    invert_sign: bool = False,
) -> None:
    if attn_df.empty or "rank_img2" not in attn_df.columns:
        return
    if pair_type != "all":
        attn_df = attn_df[attn_df["pair_type"] == pair_type]
    if attn_df.empty:
        return
    last_layer = attn_df["layer"].max()
    last_df = attn_df[attn_df["layer"] == last_layer]
    english = last_df[last_df["script_name"] == "English"]
    if english.empty:
        return
    english_mean = float(english["rank_img2"].mean())
    rank_df = (
        last_df.groupby("script_name")["rank_img2"]
        .mean()
        .reset_index(name="rank_img2")
    )
    rank_df["rank_divergence"] = (rank_df["rank_img2"] - english_mean).abs()
    acc_df = (
        df.groupby("script_name")["is_correct"]
        .apply(accuracy_rate)
        .reset_index(name="accuracy")
    )
    acc_df["accuracy"] *= 100
    merged = rank_df.merge(acc_df, on="script_name", how="inner")
    if merged.empty:
        return

    focus = focus_scripts if focus_scripts else []
    for name in ["hand_digits", "hand_english"]:
        if name in merged["script_name"].values and name not in focus:
            focus.append(name)
    focus = [s for s in focus if s in merged["script_name"].unique()]
    base_colors = list(plt.cm.tab10(np.linspace(0, 1, 10)))
    script_palette = base_colors[1:] if len(base_colors) > 1 else base_colors
    color_map = {"English": _COLOR_ENGLISH}
    color_idx = 0
    for script in focus:
        if script == "English":
            continue
        color_map[script] = script_palette[color_idx % len(script_palette)]
        color_idx += 1

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.scatter(
        merged["rank_divergence"],
        merged["accuracy"],
        s=60,
        color="#b0b0b0",
        alpha=0.35,
        edgecolor="none",
    )
    for script in focus:
        sub = merged[merged["script_name"] == script]
        if sub.empty:
            continue
        ax.scatter(
            sub["rank_divergence"],
            sub["accuracy"],
            s=60,
            color=color_map.get(script, _COLOR_OMNI),
            alpha=0.9,
            edgecolor="none",
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["script_name"],
                (row["rank_divergence"], row["accuracy"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                fontweight=None,
                color=color_map.get(script, _COLOR_OMNI),
            )
    ax.set_xlabel("Abs. Rank Divergence vs English")
    ax.set_ylabel("Accuracy (%)")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("normal")
    _apply_clean_style(ax)
    sns.despine()
    save_fig(fig, out_dir / "rank_divergence_vs_accuracy.png")

    # Save correlation value for reporting
    corr = float(merged["rank_divergence"].corr(merged["accuracy"]))
    if invert_sign:
        corr *= -1
    with open(out_dir / "rank_divergence_vs_accuracy.txt", "w") as f:
        f.write("Rank divergence vs accuracy correlation\n")
        f.write(f"Pearson r: {corr:.4f}\n")


def plot_rank_divergence_vs_rate(
    df: pd.DataFrame,
    attn_df: pd.DataFrame,
    out_dir: Path,
    rate_label: str,
    is_positive: bool,
    pair_type: str = "positive",
    focus_scripts: Optional[List[str]] = None,
    invert_sign: bool = False,
) -> None:
    if attn_df.empty or "rank_img2" not in attn_df.columns:
        return
    if pair_type != "all":
        attn_df = attn_df[attn_df["pair_type"] == pair_type]
    if attn_df.empty:
        return
    last_layer = attn_df["layer"].max()
    last_df = attn_df[attn_df["layer"] == last_layer]
    english = last_df[last_df["script_name"] == "English"]
    if english.empty:
        return
    english_mean = float(english["rank_img2"].mean())
    rank_df = (
        last_df.groupby("script_name")["rank_img2"]
        .mean()
        .reset_index(name="rank_img2")
    )
    rank_df["rank_divergence"] = (rank_df["rank_img2"] - english_mean).abs()

    rate_df = (
        df[df["is_positive"] == is_positive]
        .groupby("script_name")["is_correct"]
        .apply(accuracy_rate)
        .reset_index(name="rate")
    )
    rate_df["rate"] *= 100
    merged = rank_df.merge(rate_df, on="script_name", how="inner")
    if merged.empty:
        return

    focus = focus_scripts if focus_scripts else []
    for name in ["hand_digits", "hand_english"]:
        if name in merged["script_name"].values and name not in focus:
            focus.append(name)
    focus = [s for s in focus if s in merged["script_name"].unique()]
    base_colors = list(plt.cm.tab10(np.linspace(0, 1, 10)))
    script_palette = base_colors[1:] if len(base_colors) > 1 else base_colors
    color_map = {"English": _COLOR_ENGLISH}
    color_idx = 0
    for script in focus:
        if script == "English":
            continue
        color_map[script] = script_palette[color_idx % len(script_palette)]
        color_idx += 1

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.scatter(
        merged["rank_divergence"],
        merged["rate"],
        s=60,
        color="#b0b0b0",
        alpha=0.35,
        edgecolor="none",
    )
    for script in focus:
        sub = merged[merged["script_name"] == script]
        if sub.empty:
            continue
        ax.scatter(
            sub["rank_divergence"],
            sub["rate"],
            s=60,
            color=color_map.get(script, _COLOR_OMNI),
            alpha=0.9,
            edgecolor="none",
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["script_name"],
                (row["rank_divergence"], row["rate"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                fontweight=None,
                color=color_map.get(script, _COLOR_OMNI),
            )
    ax.set_xlabel("Abs. Rank Divergence vs English")
    ax.set_ylabel(f"{rate_label} (%)")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("normal")
    _apply_clean_style(ax)
    sns.despine()
    fig_name = f"rank_divergence_vs_{rate_label.lower()}.png"
    save_fig(fig, out_dir / fig_name)

    corr = float(merged["rank_divergence"].corr(merged["rate"]))
    if invert_sign:
        corr *= -1
    txt_name = f"rank_divergence_vs_{rate_label.lower()}.txt"
    with open(out_dir / txt_name, "w") as f:
        f.write(f"Rank divergence vs {rate_label} correlation\n")
        f.write(f"Pearson r: {corr:.4f}\n")


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = p.astype(float)
    q = q.astype(float)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return float(0.5 * (kl_pm + kl_qm))


def compute_rank_divergence_metrics(
    attn_df: pd.DataFrame,
    pair_type: str = "positive",
) -> pd.DataFrame:
    if attn_df.empty or "rank_img2" not in attn_df.columns:
        return pd.DataFrame()
    if pair_type != "all":
        attn_df = attn_df[attn_df["pair_type"] == pair_type]
    if attn_df.empty:
        return pd.DataFrame()
    # Per-layer mean rank per script
    layer_means = (
        attn_df.groupby(["script_name", "layer"])["rank_img2"]
        .mean()
        .reset_index()
    )
    english_curve = (
        layer_means[layer_means["script_name"] == "English"]
        .sort_values("layer")["rank_img2"]
        .to_numpy()
    )
    if english_curve.size == 0:
        return pd.DataFrame()
    english_mean = float(english_curve.mean())
    english_std = float(english_curve.std()) if english_curve.std() > 0 else 1.0
    eps = 1e-9

    rows = []
    for script in layer_means["script_name"].unique():
        curve = (
            layer_means[layer_means["script_name"] == script]
            .sort_values("layer")["rank_img2"]
            .to_numpy()
        )
        if curve.size == 0:
            continue
        # Align lengths if needed
        n = min(len(curve), len(english_curve))
        curve = curve[:n]
        eng = english_curve[:n]
        mean_rank = float(curve.mean())
        signed_diff = mean_rank - english_mean
        abs_diff = abs(signed_diff)
        pct_diff = signed_diff / (english_mean + eps)
        z_diff = signed_diff / (english_std + eps)
        curve_l1 = float(np.mean(np.abs(curve - eng)))
        curve_l2 = float(np.sqrt(np.mean((curve - eng) ** 2)))
        # Cosine distance between curves
        cos_sim = float(np.dot(curve, eng) / ((np.linalg.norm(curve) + eps) * (np.linalg.norm(eng) + eps)))
        curve_cos = 1.0 - cos_sim
        # Distribution-wise: JS divergence on normalized curves
        curve_js = _js_divergence(curve, eng, eps=eps)

        rows.append(
            {
                "script_name": script,
                "rank_mean": mean_rank,
                "signed_diff": signed_diff,
                "abs_diff": abs_diff,
                "pct_diff": pct_diff,
                "z_diff": z_diff,
                "curve_l1": curve_l1,
                "curve_l2": curve_l2,
                "curve_cos": curve_cos,
                "curve_jsd": curve_js,
            }
        )
    return pd.DataFrame(rows)


def correlate_rank_metrics_with_performance(
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    out_dir: Path,
    focus_scripts: Optional[List[str]] = None,
    invert_sign: bool = False,
) -> None:
    if metrics_df.empty:
        return
    acc_df = (
        df.groupby("script_name")["is_correct"]
        .apply(accuracy_rate)
        .reset_index(name="accuracy")
    )
    tpr_df = (
        df[df["is_positive"] == True]
        .groupby("script_name")["is_correct"]
        .apply(accuracy_rate)
        .reset_index(name="tpr")
    )
    tnr_df = (
        df[df["is_positive"] == False]
        .groupby("script_name")["is_correct"]
        .apply(accuracy_rate)
        .reset_index(name="tnr")
    )
    perf = acc_df.merge(tpr_df, on="script_name", how="outer").merge(tnr_df, on="script_name", how="outer")
    merged = metrics_df.merge(perf, on="script_name", how="inner")
    if merged.empty:
        return

    metric_cols = [
        "signed_diff",
        "abs_diff",
        "pct_diff",
        "z_diff",
        "curve_l1",
        "curve_l2",
        "curve_cos",
        "curve_jsd",
    ]
    rows = []
    for col in metric_cols:
        for target in ["accuracy", "tpr", "tnr"]:
            if col not in merged.columns or target not in merged.columns:
                continue
            r = float(merged[col].corr(merged[target]))
            if invert_sign:
                r *= -1
            rows.append({"metric": col, "target": target, "pearson_r": r})

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(out_dir / "rank_divergence_correlations.csv", index=False)
    metrics_df.to_csv(out_dir / "rank_divergence_metrics.csv", index=False)
    with open(out_dir / "rank_divergence_correlations.txt", "w") as f:
        f.write("Rank divergence correlations (Pearson r)\n")
        for _, row in corr_df.iterrows():
            f.write(f"{row.metric} vs {row.target}: {row.pearson_r:.4f}\n")

    # Plots for each metric vs accuracy/TPR/TNR
    focus = focus_scripts if focus_scripts else []
    for name in ["hand_digits", "hand_english"]:
        if name in merged["script_name"].values and name not in focus:
            focus.append(name)
    focus = [s for s in focus if s in merged["script_name"].unique()]
    base_colors = list(plt.cm.tab10(np.linspace(0, 1, 10)))
    script_palette = base_colors[1:] if len(base_colors) > 1 else base_colors
    color_map = {"English": _COLOR_ENGLISH}
    color_idx = 0
    for script in focus:
        if script == "English":
            continue
        color_map[script] = script_palette[color_idx % len(script_palette)]
        color_idx += 1

    metrics_to_plot = [
        "signed_diff",
        "abs_diff",
        "pct_diff",
        "z_diff",
        "curve_l1",
        "curve_l2",
        "curve_cos",
    ]
    metric_labels = {
        "signed_diff": "Signed difference",
        "abs_diff": "Absolute difference",
        "pct_diff": "Percent difference",
        "z_diff": "Z-scored difference",
        "curve_l1": "Curve L1 distance",
        "curve_l2": "Curve L2 distance",
        "curve_cos": "Curve cosine distance",
    }
    for metric in metrics_to_plot:
        for target in ["accuracy"]:
            if metric not in merged.columns or target not in merged.columns:
                continue
            pearson = float(merged[metric].corr(merged[target]))
            if invert_sign:
                pearson *= -1
            fig, ax = plt.subplots(figsize=(7.6, 5.8))
            ax.scatter(
                merged[metric],
                merged[target] * 100,
                s=80,
                color="#b0b0b0",
                alpha=0.25,
                edgecolor="none",
            )
            for script in focus:
                sub = merged[merged["script_name"] == script]
                if sub.empty:
                    continue
                ax.scatter(
                    sub[metric],
                    sub[target] * 100,
                    s=90,
                    color=color_map.get(script, _COLOR_OMNI),
                    alpha=0.95,
                    edgecolor="none",
                )
                for _, row in sub.iterrows():
                    ax.annotate(
                        row["script_name"],
                        (row[metric], row[target] * 100),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=12,
                        fontweight=None,
                        color=color_map.get(script, _COLOR_OMNI),
                    )
            ax.set_xlabel(metric_labels.get(metric, metric.replace("_", " ").title()), fontsize=14)
            ax.set_ylabel(f"{target.capitalize()} (%)", fontsize=14)
            ax.tick_params(axis="both", labelsize=14)
            ax.text(
                0.98,
                0.98,
                f"Pearson r = {pearson:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
            )
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("normal")
            _apply_clean_style(ax)
            sns.despine()
            save_fig(fig, out_dir / f"rank_metric_{metric}_vs_{target}.png")


def plot_layerwise_rank_correlation(
    df: pd.DataFrame,
    attn_df: pd.DataFrame,
    out_dir: Path,
    pair_type: str = "positive",
    invert_sign: bool = False,
) -> None:
    if attn_df.empty or "rank_img2" not in attn_df.columns:
        return
    if pair_type != "all":
        attn_df = attn_df[attn_df["pair_type"] == pair_type]
    if attn_df.empty:
        return

    acc_df = (
        df.groupby("script_name")["is_correct"]
        .apply(accuracy_rate)
        .reset_index(name="accuracy")
    )
    acc_df["accuracy"] *= 100

    layer_rows = []
    for layer_idx in sorted(attn_df["layer"].unique()):
        layer_df = attn_df[attn_df["layer"] == layer_idx]
        eng = layer_df[layer_df["script_name"] == "English"]["rank_img2"]
        if eng.empty:
            continue
        eng_mean = float(eng.mean())
        rank_df = (
            layer_df.groupby("script_name")["rank_img2"]
            .mean()
            .reset_index(name="rank_img2")
        )
        rank_df["signed_diff"] = rank_df["rank_img2"] - eng_mean
        merged = rank_df.merge(acc_df, on="script_name", how="inner")
        if merged.empty:
            continue
        r = float(merged["signed_diff"].corr(merged["accuracy"]))
        if invert_sign:
            r *= -1
        layer_rows.append({"layer": layer_idx, "pearson_r": r})

    if not layer_rows:
        return
    corr_df = pd.DataFrame(layer_rows)
    corr_df.to_csv(out_dir / "rank_signed_diff_layerwise_correlation.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(corr_df["layer"], corr_df["pearson_r"], marker="o", linewidth=1.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Pearson r (signed_diff vs accuracy)")
    sns.despine()
    save_fig(fig, out_dir / "rank_signed_diff_layerwise_correlation.png")


def plot_script_embeddings(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    # Build per-script accuracy-by-scale vectors
    scale_acc = (
        df.groupby(["script_name", "scale_factor"])["is_correct"]
        .apply(accuracy_rate)
        .reset_index(name="accuracy")
    )
    if scale_acc.empty:
        return
    pivot = scale_acc.pivot(index="script_name", columns="scale_factor", values="accuracy")
    if pivot.empty:
        return
    # Fill missing scales with script mean
    pivot = pivot.apply(lambda row: row.fillna(row.mean()), axis=1)
    scripts = pivot.index.tolist()
    X = pivot.to_numpy()
    # Color by overall accuracy
    overall = (
        df.groupby("script_name")["is_correct"]
        .apply(accuracy_rate)
        .reindex(scripts)
        .to_numpy()
    )

    # PCA (2D)
    coords = pca_2d(X)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=overall, cmap="viridis", s=70, alpha=0.9)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    _apply_clean_style(ax)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy")
    sns.despine()
    save_fig(fig, out_dir / "script_embeddings_pca.png")

    # UMAP (2D) if available
    if umap is not None:
        reducer = umap.UMAP(random_state=0)
        u = reducer.fit_transform(X)
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        sc = ax.scatter(u[:, 0], u[:, 1], c=overall, cmap="viridis", s=70, alpha=0.9)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        _apply_clean_style(ax)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Accuracy")
        sns.despine()
        save_fig(fig, out_dir / "script_embeddings_umap.png")


def analyze_signal_detection(df: pd.DataFrame, out_dir: Path) -> None:
    if norm is None:
        print("Skipping signal detection: scipy not installed")
        return
    if "scale_factor" not in df.columns:
        print("Skipping signal detection: no scale_factor column")
        return

    # --- Overall d' per (dataset, script_name) ---
    stats = []
    for (dataset, script), group in df.groupby(["dataset", "script_name"]):
        n_pos = group[group["is_positive"] == True].shape[0]
        n_neg = group[group["is_positive"] == False].shape[0]
        if n_pos == 0 or n_neg == 0:
            continue

        epsilon = 1e-6
        hits = group[(group["is_positive"] == True) & (group["is_correct"] == True)].shape[0]
        false_alarms = group[(group["is_positive"] == False) & (group["is_correct"] == False)].shape[0]

        hit_rate = max(min(hits / n_pos, 1 - epsilon), epsilon)
        fa_rate = max(min(false_alarms / n_neg, 1 - epsilon), epsilon)

        d_prime = norm.ppf(hit_rate) - norm.ppf(fa_rate)
        criterion = -0.5 * (norm.ppf(hit_rate) + norm.ppf(fa_rate))

        stats.append(
            {
                "dataset": dataset,
                "script_name": script,
                "d_prime": d_prime,
                "bias_criterion": criterion,
                "hit_rate": hit_rate,
                "fa_rate": fa_rate,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "hits": hits,
                "false_alarms": false_alarms,
            }
        )

    if not stats:
        return

    sdt_df = pd.DataFrame(stats)
    sdt_df.to_csv(out_dir / "sdt_summary.csv", index=False)

    # Main bar plot with value labels
    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(sdt_df))))
    sns.barplot(
        data=sdt_df,
        x="script_name",
        y="d_prime",
        hue="dataset",
        palette="viridis",
        ax=ax,
    )
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Script")
    ax.set_ylabel("d'")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=6)
    _legend_outside(ax)
    sns.despine()
    save_fig(fig, out_dir / "sdt_d_prime.png", legend_right=True)

    # --- d' per (dataset, script_name, scale_factor) for breakdown ---
    stats_by_scale = []
    for (dataset, script, scale_factor), group in df.groupby(["dataset", "script_name", "scale_factor"]):
        n_pos = group[group["is_positive"] == True].shape[0]
        n_neg = group[group["is_positive"] == False].shape[0]
        if n_pos == 0 or n_neg == 0:
            continue
        epsilon = 1e-6
        hits = group[(group["is_positive"] == True) & (group["is_correct"] == True)].shape[0]
        false_alarms = group[(group["is_positive"] == False) & (group["is_correct"] == False)].shape[0]
        hit_rate = max(min(hits / n_pos, 1 - epsilon), epsilon)
        fa_rate = max(min(false_alarms / n_neg, 1 - epsilon), epsilon)
        d_prime = norm.ppf(hit_rate) - norm.ppf(fa_rate)
        criterion = -0.5 * (norm.ppf(hit_rate) + norm.ppf(fa_rate))
        stats_by_scale.append({
            "dataset": dataset,
            "script_name": script,
            "scale_factor": scale_factor,
            "d_prime": d_prime,
            "bias_criterion": criterion,
            "hit_rate": hit_rate,
            "fa_rate": fa_rate,
            "n_pos": n_pos,
            "n_neg": n_neg,
        })
    sdt_by_scale_df = pd.DataFrame(stats_by_scale)
    if not sdt_by_scale_df.empty:
        sdt_by_scale_df.to_csv(out_dir / "sdt_by_scale.csv", index=False)

        # Faceted bar plot: d' by script, one panel per scale factor
        scales = sorted(sdt_by_scale_df["scale_factor"].unique())
        n_scales = len(scales)
        ncols = min(n_scales, 4)
        nrows = (n_scales + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, max(3.5, 1.8 * nrows)), squeeze=False)
        handles, labels = None, None
        for idx, scale_factor in enumerate(scales):
            ax = axes.flat[idx]
            sub = sdt_by_scale_df[sdt_by_scale_df["scale_factor"] == scale_factor]
            sns.barplot(
                data=sub,
                x="script_name",
                y="d_prime",
                hue="dataset",
                palette="viridis",
                ax=ax,
            )
            ax.axhline(0, color="red", linestyle="--", linewidth=1)
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax.set_xlabel("")
            if idx % ncols != 0:
                ax.set_ylabel("")
            if handles is None and ax.get_legend() is not None:
                handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
        if handles is not None:
            fig.legend(handles, labels, bbox_to_anchor=(1.02, 0.98), loc="upper left", fontsize=9)
        for j in range(idx + 1, len(axes.flat)):
            axes.flat[j].set_visible(False)
        sns.despine()
        save_fig(fig, out_dir / "sdt_d_prime_by_scale.png", legend_right=True)

        # Heatmap: script x scale_factor (d'), rows ordered by mean d' (high to low)
        pivot = sdt_by_scale_df.pivot_table(
            index=["script_name", "dataset"],
            columns="scale_factor",
            values="d_prime",
            aggfunc="mean",
        ).reset_index()
        scale_cols = [c for c in pivot.columns if c not in ("script_name", "dataset")]
        pivot["mean_d_prime"] = pivot[scale_cols].mean(axis=1)
        pivot = pivot.sort_values("mean_d_prime", ascending=False).drop(columns="mean_d_prime")
        heatmap_data = pivot.set_index("script_name")[sorted(scale_cols)]
        fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(heatmap_data.columns)), max(5, 0.2 * len(heatmap_data))))
        sns.heatmap(
            heatmap_data,
            ax=ax,
            cmap="RdYlGn",
            center=2.0,
            vmin=0,
            vmax=7,
            cbar_kws={"label": "d'", "shrink": 0.8},
            xticklabels=True,
            yticklabels=True,
            annot=False,
        )
        ax.tick_params(axis="both", labelsize=8)
        ax.set_xlabel("Scale", fontsize=9)
        ax.set_ylabel("")
        sns.despine()
        save_fig(fig, out_dir / "sdt_d_prime_heatmap.png")

        # Line plot: d' vs scale_factor for selected scripts (top/mid/bottom)
        selected = _select_scripts_top_mid_bottom(df, n=3)
        line_df = sdt_by_scale_df[sdt_by_scale_df["script_name"].isin(selected)]
        if not line_df.empty:
            fig, ax = plt.subplots(figsize=(9, 6))
            for (script_name, dataset), sub in line_df.groupby(["script_name", "dataset"]):
                sub = sub.sort_values("scale_factor")
                label = f"{script_name} ({dataset})"
                ax.plot(sub["scale_factor"], sub["d_prime"], marker="o", linewidth=2, label=label)
            ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
            ax.set_xlabel("Scale Factor")
            ax.set_ylabel("d'")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, frameon=True, framealpha=1.0)
            ax.set_ylim(bottom=0)
            sns.despine()
            save_fig(fig, out_dir / "sdt_d_prime_vs_scale.png", legend_right=True)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(sdt_df))))
    sns.barplot(
        data=sdt_df,
        x="script_name",
        y="bias_criterion",
        hue="dataset",
        palette="coolwarm",
        ax=ax,
    )
    ax.axhline(0, color="black", linestyle="-", linewidth=1)
    ax.set_xlabel("Script")
    ax.set_ylabel("Criterion (c)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    _legend_outside(ax)
    sns.despine()
    save_fig(fig, out_dir / "sdt_bias.png", legend_right=True)


def _select_scripts_top_mid_bottom(df: pd.DataFrame, n: int = 3) -> List[str]:
    script_acc = df.groupby("script_name")["is_correct"].mean().sort_values(ascending=False)
    selected = []

    # Always include English and hand_english if present (hand_digits excluded)
    for name in ["English", "hand_english"]:
        if name in script_acc.index:
            selected.append(name)

    # Omniglot scripts only (exclude hand_* group)
    omni_df = df[
        (df["dataset"] == "omniglot")
        & (~df["script_name"].isin(OMNIGLOT_GROUP_EXCLUDE))
    ]
    omni_acc = (
        omni_df.groupby("script_name")["is_correct"]
        .mean()
        .sort_values(ascending=False)
    )
    omniglot_scripts = list(omni_acc.index)
    # Exactly 6 from Omniglot: top 2 = Greek & Latin, then middle 2, bottom 2
    n_omni = 2
    top_wanted = ["Greek", "Latin"]
    top = [s for s in top_wanted if s in omniglot_scripts]
    remaining = [s for s in omniglot_scripts if s not in top]
    if len(remaining) >= 4:
        bottom = remaining[-n_omni:]
        mid_pool = remaining[:-n_omni]
        mid_start = max(0, (len(mid_pool) - n_omni) // 2)
        middle = mid_pool[mid_start:mid_start + n_omni]
        selected += top + middle + bottom
    else:
        selected += top + remaining

    # Preserve original accuracy order for selected scripts
    order = [s for s in script_acc.index if s in selected]
    return order


def plot_recall_specificity_selected(
    df: pd.DataFrame,
    out_dir: Path,
    n: int = 2,
    use_ag_colors: bool = False,
) -> None:
    selected = _select_scripts_top_mid_bottom(df, n=n)
    stats = []
    for script, group in df.groupby("script_name"):
        if script not in selected:
            continue
        pos_df = group[group["is_positive"] == True]
        neg_df = group[group["is_positive"] == False]
        tp = int(pos_df["is_correct"].sum()) if len(pos_df) > 0 else 0
        tn = int(neg_df["is_correct"].sum()) if len(neg_df) > 0 else 0
        tpr = (accuracy_rate(pos_df["is_correct"]) * 100) if len(pos_df) > 0 else 0
        tnr = (accuracy_rate(neg_df["is_correct"]) * 100) if len(neg_df) > 0 else 0
        stats.append({"script_name": script, "metric": "Recall (TPR)", "value": tpr, "n": len(pos_df), "k": tp})
        stats.append({"script_name": script, "metric": "Specificity (TNR)", "value": tnr, "n": len(neg_df), "k": tn})
    if not stats:
        return
    stats_df = pd.DataFrame(stats)
    if not stats_df.empty:
        recall_order = (
            stats_df[stats_df["metric"] == "Recall (TPR)"]
            .sort_values("value", ascending=False)["script_name"]
            .tolist()
        )
        order = [s for s in recall_order if s in stats_df["script_name"].unique()]
        stats_df["script_name"] = pd.Categorical(stats_df["script_name"], categories=order, ordered=True)

    fig, ax = plt.subplots(figsize=(6, max(4, 0.25 * len(selected))))
    metric_order = ["Recall (TPR)", "Specificity (TNR)"]
    plt.rcParams["hatch.linewidth"] = 1.0
    if use_ag_colors:
        # Two blues: base + lighter shade (from model_category_perf); TNR gets black diagonal hatch
        metric_palette = {
            "Recall (TPR)": _AG_BLUE_BASE,
            "Specificity (TNR)": _AG_BLUE_LIGHT,
        }
    else:
        metric_palette = "Set2"
    sns.barplot(
        data=stats_df,
        y="script_name",
        x="value",
        hue="metric",
        hue_order=metric_order,
        palette=metric_palette,
        ax=ax,
    )
    ax.set_xlabel("Rate (%)")
    ax.set_ylabel("Script")
    ax.set_xlim(0, 100)
    ax.axvline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    for container, metric in zip(ax.containers, metric_order):
        for bar in container:
            if use_ag_colors:
                bar.set_alpha(_AG_BAR_ALPHA)
        if metric == "Specificity (TNR)":
            for bar in container:
                bar.set_hatch("//")
                if use_ag_colors:
                    bar.set_edgecolor("black")
                    bar.set_linewidth(1.0)
        elif use_ag_colors:
            for bar in container:
                bar.set_edgecolor("black")
                bar.set_linewidth(1.0)
    _apply_script_display_names(ax)
    _apply_script_label_colors(ax)
    _legend_bottom(ax, handlelength=4)
    leg = ax.get_legend()
    if leg and use_ag_colors:
        for handle, text in zip(leg.legend_handles, leg.get_texts()):
            handle.set_alpha(_AG_BAR_ALPHA)
            if text.get_text() == "Specificity (TNR)":
                handle.set_hatch("//")
                handle.set_edgecolor("black")
            else:
                handle.set_edgecolor("black")
    elif leg:
        for handle, text in zip(leg.legend_handles, leg.get_texts()):
            if text.get_text() == "Specificity (TNR)":
                handle.set_hatch("//")
                break
    sns.despine()
    save_fig(fig, out_dir / "recall_specificity_selected_scripts.png", legend_bottom=True, also_pdf=True)


def plot_recall_specificity_selected_by_scale(
    df: pd.DataFrame,
    out_dir: Path,
    n: int = 2,
    use_ag_colors: bool = False,
) -> None:
    selected = _select_scripts_top_mid_bottom(df, n=n)
    for scale_factor, group in df.groupby("scale_factor"):
        stats = []
        for script, sgroup in group.groupby("script_name"):
            if script not in selected:
                continue
            pos_df = sgroup[sgroup["is_positive"] == True]
            neg_df = sgroup[sgroup["is_positive"] == False]
            tp = int(pos_df["is_correct"].sum()) if len(pos_df) > 0 else 0
            tn = int(neg_df["is_correct"].sum()) if len(neg_df) > 0 else 0
            tpr = (accuracy_rate(pos_df["is_correct"]) * 100) if len(pos_df) > 0 else 0
            tnr = (accuracy_rate(neg_df["is_correct"]) * 100) if len(neg_df) > 0 else 0
            stats.append({"script_name": script, "metric": "Recall (TPR)", "value": tpr, "n": len(pos_df), "k": tp})
            stats.append({"script_name": script, "metric": "Specificity (TNR)", "value": tnr, "n": len(neg_df), "k": tn})
        if not stats:
            continue
        stats_df = pd.DataFrame(stats)
        if not stats_df.empty:
            recall_order = (
                stats_df[stats_df["metric"] == "Recall (TPR)"]
                .sort_values("value", ascending=False)["script_name"]
                .tolist()
            )
            order = [s for s in recall_order if s in stats_df["script_name"].unique()]
            stats_df["script_name"] = pd.Categorical(stats_df["script_name"], categories=order, ordered=True)
        fig, ax = plt.subplots(figsize=(6, max(4, 0.25 * len(selected))))
        metric_order = ["Recall (TPR)", "Specificity (TNR)"]
        plt.rcParams["hatch.linewidth"] = 1.0
        if use_ag_colors:
            metric_palette = {
                "Recall (TPR)": _AG_BLUE_BASE,
                "Specificity (TNR)": _AG_BLUE_LIGHT,
            }
        else:
            metric_palette = "Set2"
        sns.barplot(
            data=stats_df,
            y="script_name",
            x="value",
            hue="metric",
            hue_order=metric_order,
            palette=metric_palette,
            ax=ax,
        )
        for container, metric in zip(ax.containers, metric_order):
            for bar in container:
                if use_ag_colors:
                    bar.set_alpha(_AG_BAR_ALPHA)
            if metric == "Specificity (TNR)":
                for bar in container:
                    bar.set_hatch("//")
                    if use_ag_colors:
                        bar.set_edgecolor("black")
                        bar.set_linewidth(1.0)
            elif use_ag_colors:
                for bar in container:
                    bar.set_edgecolor("black")
                    bar.set_linewidth(1.0)
        if not use_ag_colors:
            for bar in ax.patches:
                bar.set_edgecolor("0.4")
                bar.set_linewidth(0.8)
        ax.set_xlabel("Rate (%)")
        ax.set_ylabel("Script")
        ax.set_xlim(0, 100)
        ax.axvline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        _apply_script_display_names(ax)
        _apply_script_label_colors(ax)
        _legend_bottom(ax, handlelength=4)
        leg = ax.get_legend()
        if leg and use_ag_colors:
            for handle, text in zip(leg.legend_handles, leg.get_texts()):
                handle.set_alpha(_AG_BAR_ALPHA)
                if text.get_text() == "Specificity (TNR)":
                    handle.set_hatch("//")
                    handle.set_edgecolor("black")
                else:
                    handle.set_edgecolor("black")
        elif leg:
            for handle, text in zip(leg.legend_handles, leg.get_texts()):
                if text.get_text() == "Specificity (TNR)":
                    handle.set_hatch("//")
                    break
        sns.despine()
        save_fig(fig, out_dir / f"recall_specificity_selected_scripts_scale_{scale_factor:.1f}.png", legend_bottom=True, also_pdf=True)


def analyze_consistency(df: pd.DataFrame, out_dir: Path) -> None:
    if "pair_id" not in df.columns:
        print("Skipping consistency: No pair_id found")
        return

    consistency = (
        df.groupby(["dataset", "script_name", "pair_id"])["is_correct"]
        .std()
        .reset_index()
        .rename(columns={"is_correct": "instability"})
    )

    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.violinplot(
        data=consistency,
        x="dataset",
        y="instability",
        hue="dataset",
        palette="muted",
        ax=ax,
        legend=False,
        cut=0,
    )
    ax.set_ylabel("Std. Dev. of Correctness")
    ax.set_xlabel("")
    sns.despine()
    save_fig(fig, out_dir / "consistency_brittleness.png")


def analyze_error_entropy(df: pd.DataFrame, out_dir: Path) -> None:
    errors = df[df["is_correct"] == False]
    if errors.empty:
        return

    stubbornness = errors.groupby("script_name")["is_positive"].mean().reset_index()
    stubbornness["bias_type"] = stubbornness["is_positive"].apply(
        lambda x: "Random" if 0.4 < x < 0.6 else "Systematic"
    )

    # Select English + top/mid/bottom 2 Omniglot scripts by mean accuracy
    script_acc = df.groupby("script_name")["is_correct"].mean().sort_values(ascending=False)
    english_name = "English" if "English" in script_acc.index else None
    omniglot_scripts = [s for s in script_acc.index if s != english_name]
    selected = []
    if english_name:
        selected.append(english_name)
    if len(omniglot_scripts) >= 6:
        top2 = omniglot_scripts[:2]
        bottom2 = omniglot_scripts[-2:]
        remaining = omniglot_scripts[2:-2]
        mid2 = remaining[:2] if len(remaining) >= 2 else remaining
        selected += top2 + mid2 + bottom2
    else:
        selected += omniglot_scripts

    stubbornness_sel = stubbornness[stubbornness["script_name"].isin(selected)]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=stubbornness,
        x="script_name",
        y="is_positive",
        hue="bias_type",
        s=60,
        ax=ax,
        palette="Set1",
    )
    sns.scatterplot(
        data=stubbornness_sel,
        x="script_name",
        y="is_positive",
        color=_COLOR_OMNI,
        s=90,
        ax=ax,
        legend=False,
    )
    for _, row in stubbornness_sel.iterrows():
        ax.annotate(
            row["script_name"],
            (row["script_name"], row["is_positive"]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=7,
            rotation=45,
            color=_script_label_color(row["script_name"]),
        )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random")
    ax.set_ylabel("P(say 'Same' | Wrong)")
    ax.set_xlabel("Script")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    _apply_script_label_colors(ax)
    _legend_outside(ax)
    sns.despine()
    save_fig(fig, out_dir / "error_stubbornness.png", legend_right=True)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "script_name"])
        .agg(
            accuracy=("is_correct", "mean"),
            n=("is_correct", "size"),
            pos_rate=("is_positive", "mean"),
        )
        .reset_index()
        .sort_values(["dataset", "accuracy"], ascending=[True, False])
    )


def _mask_omniglot_core(df: pd.DataFrame) -> pd.Series:
    return (df["dataset"] == "omniglot") & (~df["script_name"].isin(OMNIGLOT_GROUP_EXCLUDE))


def split_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """TPR, TNR, and accuracy for Times New Roman, Handwritten English, and Omniglot."""
    split_specs = [
        ("Times New Roman", (df["dataset"] == "alphabet") & (df["script_name"] == "English")),
        ("Handwritten English", (df["dataset"] == "omniglot") & (df["script_name"] == "hand_english")),
        ("Omniglot", _mask_omniglot_core(df)),
    ]
    rows = []
    for split_name, mask in split_specs:
        subset = df[mask]
        pos = subset[subset["is_positive"] == True]
        neg = subset[subset["is_positive"] == False]
        n_total = len(subset)
        n_pos = len(pos)
        n_neg = len(neg)
        acc = accuracy_rate(subset["is_correct"]) * 100.0 if n_total > 0 else np.nan
        tpr = accuracy_rate(pos["is_correct"]) * 100.0 if n_pos > 0 else np.nan
        tnr = accuracy_rate(neg["is_correct"]) * 100.0 if n_neg > 0 else np.nan
        rows.append(
            {
                "split": split_name,
                "n_total": n_total,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "accuracy_pct": acc,
                "tpr_pct": tpr,
                "tnr_pct": tnr,
            }
        )
    return pd.DataFrame(rows)


def plot_overall_accuracy(df: pd.DataFrame, out_dir: Path) -> None:
    # Exclude hand_digits/hand_english from omniglot aggregate
    df_agg = df[~((df["dataset"] == "omniglot") & (df["script_name"].isin(OMNIGLOT_GROUP_EXCLUDE)))]
    fig, ax = plt.subplots(figsize=(5, 4))
    summary = (
        df_agg.groupby("dataset")["is_correct"]
        .apply(accuracy_rate)
        .reset_index(name="accuracy")
    )
    summary["accuracy"] *= 100
    sns.barplot(data=summary, x="dataset", y="accuracy", hue="dataset", ax=ax, palette="viridis", legend=False)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Dataset")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    for _, row in summary.iterrows():
        ax.text(row.name, row["accuracy"] + 1, f'{row["accuracy"]:.1f}%', ha="center", fontsize=9)
    sns.despine()
    save_fig(fig, out_dir / "overall_accuracy_by_dataset.png")


def plot_script_accuracy(df: pd.DataFrame, out_dir: Path) -> None:
    script_df = (
        df.groupby(["script_name"])["is_correct"]
        .apply(accuracy_rate)
        .reset_index(name="accuracy")
    )
    script_df["accuracy"] *= 100
    script_df = script_df.sort_values("accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(6, max(4, 0.25 * len(script_df))))
    sns.barplot(data=script_df, y="script_name", x="accuracy", hue="script_name", ax=ax, palette="mako", legend=False)
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Script")
    ax.set_xlim(0, 100)
    ax.axvline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    _apply_script_display_names(ax)
    _apply_script_label_colors(ax)
    sns.despine()
    save_fig(fig, out_dir / "accuracy_by_script.png")


def plot_script_accuracy_per_scale(df: pd.DataFrame, out_dir: Path) -> None:
    per_scale_dir = out_dir / "accuracy_by_script_per_scale"
    ensure_dir(per_scale_dir)
    script_scale_df = (
        df.groupby(["scale_factor", "script_name"])["is_correct"]
        .mean()
        .reset_index(name="accuracy")
    )
    script_scale_df["accuracy"] *= 100
    for scale_factor, sub_df in script_scale_df.groupby("scale_factor"):
        sub_df = sub_df.sort_values("accuracy", ascending=False)
        fig, ax = plt.subplots(figsize=(6, max(4, 0.25 * len(sub_df))))
        sns.barplot(data=sub_df, y="script_name", x="accuracy", hue="script_name", ax=ax, palette="mako", legend=False)
        ax.set_xlabel("Accuracy (%)")
        ax.set_ylabel("Script")
        ax.set_xlim(0, 100)
        ax.axvline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        _apply_script_display_names(ax)
        _apply_script_label_colors(ax)
        sns.despine()
        save_fig(fig, per_scale_dir / f"accuracy_by_script_scale_{scale_factor:.1f}.png")


def plot_accuracy_by_scale(df: pd.DataFrame, out_dir: Path) -> None:
    # Exclude hand_digits/hand_english from omniglot aggregate
    df_agg = df[~((df["dataset"] == "omniglot") & (df["script_name"].isin(OMNIGLOT_GROUP_EXCLUDE)))]
    scale_df = (
        df_agg.groupby(["dataset", "scale_factor"])["is_correct"]
        .apply(accuracy_rate)
        .reset_index(name="accuracy")
    )
    scale_df["accuracy"] *= 100
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.lineplot(
        data=scale_df,
        x="scale_factor",
        y="accuracy",
        hue="dataset",
        marker="o",
        linewidth=2,
        ax=ax,
        palette="Set2",
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Scale Factor")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    _legend_outside(ax)
    sns.despine()
    save_fig(fig, out_dir / "accuracy_by_scale_and_dataset.png", legend_right=True)


def plot_accuracy_by_scale_for_scripts(df: pd.DataFrame, out_dir: Path) -> None:
    scale_script_df = (
        df.groupby(["script_name", "scale_factor"])["is_correct"]
        .apply(accuracy_rate)
        .reset_index(name="accuracy")
    )
    scale_script_df["accuracy"] *= 100
    # Select English + top/mid/bottom 2 Omniglot scripts by overall accuracy
    script_acc = df.groupby("script_name")["is_correct"].mean().sort_values(ascending=False)
    english_name = "English" if "English" in script_acc.index else None
    omniglot_scripts = [s for s in script_acc.index if s != english_name]
    selected_scripts = []
    if english_name:
        selected_scripts.append(english_name)
    if len(omniglot_scripts) >= 6:
        top2 = omniglot_scripts[:2]
        bottom2 = omniglot_scripts[-2:]
        remaining = omniglot_scripts[2:-2]
        mid2 = remaining[:2] if len(remaining) >= 2 else remaining
        selected_scripts += top2 + mid2 + bottom2
    else:
        selected_scripts += omniglot_scripts
    
    scale_script_df = scale_script_df[scale_script_df["script_name"].isin(selected_scripts)]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.lineplot(
        data=scale_script_df,
        x="scale_factor",
        y="accuracy",
        hue="script_name",
        marker="o",
        linewidth=1.8,
        ax=ax,
        palette="tab20",
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Scale Factor")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=True, framealpha=1.0)
    sns.despine()
    save_fig(fig, out_dir / "accuracy_by_scale_per_script.png", legend_right=True)


def plot_script_accuracy_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    script_acc = (
        df.groupby("script_name")["is_correct"]
        .mean()
        .reset_index(name="accuracy")
    )
    script_acc["accuracy"] *= 100
    script_acc["group"] = np.where(
        script_acc["script_name"] == "English",
        "English",
        np.where(script_acc["script_name"].isin(OMNIGLOT_GROUP_EXCLUDE), "Custom", "Omniglot"),
    )
    script_acc_plot = script_acc[script_acc["group"].isin(["English", "Omniglot"])]
    if script_acc_plot.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.violinplot(data=script_acc_plot, x="group", y="accuracy", hue="group", ax=ax, palette="Set2", cut=0, legend=False, dodge=False)
    sns.stripplot(data=script_acc_plot, x="group", y="accuracy", ax=ax, color="k", alpha=0.5, jitter=0.15, size=3)
    ax.set_ylabel("Script Accuracy (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    sns.despine()
    save_fig(fig, out_dir / "script_accuracy_distribution.png")


def plot_script_scale_heatmap(df: pd.DataFrame, out_dir: Path, focus_scripts: Optional[List[str]] = None) -> None:
    if focus_scripts:
        df = df[df["script_name"].isin(focus_scripts)]
    pivot = (
        df.groupby(["script_name", "scale_factor"])["is_correct"]
        .mean()
        .reset_index()
        .pivot(index="script_name", columns="scale_factor", values="is_correct")
        * 100
    )
    if pivot.empty:
        return
    pivot = pivot.sort_values(by=pivot.columns.tolist(), ascending=False)
    fig, ax = plt.subplots(figsize=(5.5, max(4, 0.25 * len(pivot))))
    sns.heatmap(pivot, ax=ax, cmap="viridis", vmin=0, vmax=100, cbar_kws={"label": "Accuracy (%)", "shrink": 0.8})
    ax.set_xlabel("Scale Factor")
    ax.set_ylabel("Script")
    _apply_script_label_colors(ax)
    sns.despine()
    suffix = "_selected" if focus_scripts else ""
    save_fig(fig, out_dir / f"script_scale_accuracy_heatmap{suffix}.png")


def plot_script_sensitivity(df: pd.DataFrame, out_dir: Path) -> None:
    # Fit simple slope of accuracy vs scale for each script
    rows = []
    for script_name, sub_df in df.groupby("script_name"):
        sub_df = sub_df.groupby("scale_factor")["is_correct"].mean().reset_index()
        if len(sub_df) < 2:
            continue
        x = sub_df["scale_factor"].to_numpy()
        y = sub_df["is_correct"].to_numpy() * 100
        slope = np.polyfit(x, y, 1)[0]
        rows.append({"script_name": script_name, "mean_acc": y.mean(), "slope": slope})
    sens_df = pd.DataFrame(rows)
    # Select English + top/mid/bottom 2 Omniglot scripts by mean accuracy
    sens_sorted = sens_df.sort_values("mean_acc", ascending=False)
    english_name = "English" if "English" in sens_sorted["script_name"].values else None
    omniglot_scripts = [s for s in sens_sorted["script_name"].tolist() if s != english_name]
    selected = []
    if english_name:
        selected.append(english_name)
    if len(omniglot_scripts) >= 6:
        top2 = omniglot_scripts[:2]
        bottom2 = omniglot_scripts[-2:]
        remaining = omniglot_scripts[2:-2]
        mid2 = remaining[:2] if len(remaining) >= 2 else remaining
        selected += top2 + mid2 + bottom2
    else:
        selected += omniglot_scripts

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.scatterplot(data=sens_df, x="mean_acc", y="slope", ax=ax, color="#4c78a8", alpha=0.6)
    # Highlight selected scripts with labels
    label_df = sens_df[sens_df["script_name"].isin(selected)].copy()
    # Add extreme points (max/min mean accuracy and slope)
    if not sens_df.empty:
        extremes = pd.concat(
            [
                sens_df.loc[[sens_df["mean_acc"].idxmax()]],
                sens_df.loc[[sens_df["mean_acc"].idxmin()]],
                sens_df.loc[[sens_df["slope"].idxmax()]],
                sens_df.loc[[sens_df["slope"].idxmin()]],
            ]
        ).drop_duplicates(subset=["script_name"])
        label_df = pd.concat([label_df, extremes]).drop_duplicates(subset=["script_name"])
    sns.scatterplot(data=label_df, x="mean_acc", y="slope", ax=ax, color=_COLOR_OMNI, s=60)
    for _, row in label_df.iterrows():
        ax.annotate(
            row["script_name"],
            (row["mean_acc"], row["slope"]),
            textcoords="offset points",
            xytext=(6, 6),
            ha="left",
            fontsize=9,
            color=_script_label_color(row["script_name"]),
        )
    ax.set_xlabel("Mean Accuracy (%)")
    ax.set_ylabel("Accuracy Slope vs Scale")
    sns.despine()
    save_fig(fig, out_dir / "script_scale_sensitivity.png")


def plot_accuracy_by_scale_per_script_separate(df: pd.DataFrame, out_dir: Path) -> None:
    per_script_dir = out_dir / "per_script_scale"
    ensure_dir(per_script_dir)
    scale_script_df = (
        df.groupby(["script_name", "scale_factor"])["is_correct"]
        .apply(accuracy_rate)
        .reset_index(name="accuracy")
    )
    scale_script_df["accuracy"] *= 100
    for script_name, sub_df in scale_script_df.groupby("script_name"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(sub_df["scale_factor"], sub_df["accuracy"], marker="o", linewidth=2)
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("Scale Factor")
        ax.set_ylim(0, 100)
        ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in script_name)
        sns.despine()
        save_fig(fig, per_script_dir / f"{safe_name}_accuracy_by_scale.png")


def plot_positive_negative_accuracy(df: pd.DataFrame, out_dir: Path) -> None:
    # Exclude hand_digits/hand_english from omniglot aggregate
    df_agg = df[~((df["dataset"] == "omniglot") & (df["script_name"].isin(OMNIGLOT_GROUP_EXCLUDE)))]
    pn_df = (
        df_agg.groupby(["dataset", "is_positive"])["is_correct"]
        .mean()
        .reset_index(name="accuracy")
    )
    pn_df["accuracy"] *= 100
    pn_df["pair_type"] = pn_df["is_positive"].map({True: "Positive", False: "Negative"})
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.barplot(data=pn_df, x="pair_type", y="accuracy", hue="dataset", ax=ax, palette="Set2")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Pair Type")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    _legend_outside(ax)
    sns.despine()
    save_fig(fig, out_dir / "accuracy_by_pair_type.png", legend_right=True)


def plot_prior_gap_by_scale(df: pd.DataFrame, out_dir: Path) -> None:
    # High prior = Alphabet (English), Low prior = Omniglot (exclude hand_digits/hand_english)
    alpha_df = df[df["dataset"] == "alphabet"]
    omni_df = df[
        (df["dataset"] == "omniglot") & (~df["script_name"].isin(OMNIGLOT_GROUP_EXCLUDE))
    ]
    scale_alpha = alpha_df.groupby("scale_factor")["is_correct"].mean()
    scale_omni = omni_df.groupby("scale_factor")["is_correct"].mean()
    gap = (scale_alpha - scale_omni) * 100
    gap = gap.reset_index(name="gap")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(gap["scale_factor"], gap["gap"], marker="o", linewidth=2, color="#2c7fb8")
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(20, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylabel("Accuracy Gap (Alphabet - Omniglot) (%)")
    ax.set_xlabel("Scale Factor")
    sns.despine()
    save_fig(fig, out_dir / "prior_gap_by_scale.png")


def plot_script_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    counts = df["script_name"].value_counts().reset_index()
    counts.columns = ["script_name", "n"]
    fig, ax = plt.subplots(figsize=(6, max(4, 0.25 * len(counts))))
    sns.barplot(data=counts, y="script_name", x="n", hue="script_name", ax=ax, palette="crest", legend=False)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Script")
    _apply_script_display_names(ax)
    sns.despine()
    save_fig(fig, out_dir / "samples_by_script.png")


def plot_response_length(df: pd.DataFrame, out_dir: Path) -> None:
    # Exclude hand_digits/hand_english from omniglot aggregate
    df_agg = df[~((df["dataset"] == "omniglot") & (df["script_name"].isin(OMNIGLOT_GROUP_EXCLUDE)))].copy()
    df_agg["response_len"] = df_agg["response"].astype(str).str.len()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.boxplot(data=df_agg, x="dataset", y="response_len", hue="dataset", ax=ax, palette="Set3", legend=False)
    ax.set_ylabel("Response Length (chars)")
    ax.set_xlabel("Dataset")
    sns.despine()
    save_fig(fig, out_dir / "response_length_by_dataset.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze scale illusion CSV results.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to scale_illusion CSV.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save analysis outputs.")
    parser.add_argument("--bootstrap_n", type=int, default=2000, help="Bootstrap samples for gap CI.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for bootstrap.")
    parser.add_argument("--alphabet_dir", type=str, default=None, help="Base directory containing times_new_roman.")
    parser.add_argument("--omniglot_dir", type=str, default=None, help="Base directory containing omniglot.")
    parser.add_argument("--ink_threshold", type=int, default=200, help="Threshold for black pixels (0-255).")
    parser.add_argument("--attn_metrics_csv", type=str, default=None, help="Path to attn_metrics_by_layer.csv.")
    parser.add_argument("--rank_pair_type", type=str, default="positive", choices=["positive", "negative", "all"], help="Pair type for rank plots.")
    parser.add_argument("--analysis_device", type=str, default=None, help="Device for model-based analyses (e.g., cuda or cpu).")
    parser.add_argument("--pca_model", type=str, default=None, help="Qwen2.5-VL model for PCA analysis.")
    parser.add_argument("--pca_prompt", type=str, default="Compare the two images.", help="Prompt for PCA embedding extraction.")
    parser.add_argument("--pca_small_scale", type=float, default=0.3, help="Small scale for PCA plots.")
    parser.add_argument("--pca_large_scale", type=float, default=0.9, help="Large scale for PCA plots.")
    parser.add_argument("--pca_image_size", type=int, default=336, help="Image size for PCA embeddings.")
    parser.add_argument("--pca_scripts", type=str, nargs="*", default=None, help="Script names for PCA plots.")
    parser.add_argument("--pca_num_scripts", type=int, default=None, help="Max scripts for PCA if not specified.")
    parser.add_argument("--rsa_model", type=str, default=None, help="Qwen2.5-VL model for RSA analysis.")
    parser.add_argument("--rsa_prompt", type=str, default="Compare the two images.", help="Prompt for RSA embeddings.")
    parser.add_argument("--rsa_small_scale", type=float, default=0.3, help="Small scale for RSA.")
    parser.add_argument("--rsa_large_scale", type=float, default=0.9, help="Large scale for RSA.")
    parser.add_argument("--rsa_image_size", type=int, default=336, help="Image size for RSA embeddings.")
    parser.add_argument("--rsa_num_chars", type=int, default=12, help="Num chars per script for RSA.")
    parser.add_argument("--rsa_num_scripts", type=int, default=6, help="Max scripts for RSA if not specified.")
    parser.add_argument("--rsa_scripts", type=str, nargs="*", default=None, help="Script names for RSA.")
    parser.add_argument("--tokenizer_model", type=str, default=None, help="Tokenizer model for fertility analysis.")
    parser.add_argument("--tokenizer_texts_csv", type=str, default=None, help="CSV with columns script_name,text for tokenization.")
    parser.add_argument("--focus_scripts", type=str, nargs="*", default=None, help="Scripts to highlight on selected plots.")
    parser.add_argument("--invert_correlation_sign", action="store_true", help="Invert correlation signs for reporting.")
    parser.add_argument(
        "--ag",
        action="store_true",
        help="Use Qwen-style blue/red colors for TPR/TNR bar plots (matches model_category_perf.py).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent / "analysis"
    ensure_dir(output_dir)

    df = pd.read_csv(csv_path)
    df["is_correct"] = df["is_correct"].astype(bool)
    df["is_positive"] = df["is_positive"].astype(bool)

    # High vs Low visual prior summary (exclude hand_digits/hand_english from Omniglot)
    alpha = df[df["dataset"] == "alphabet"]["is_correct"].to_numpy(dtype=float)
    omni_df = df[
        (df["dataset"] == "omniglot") & (~df["script_name"].isin(OMNIGLOT_GROUP_EXCLUDE))
    ]
    omni = omni_df["is_correct"].to_numpy(dtype=float)
    gap_mean, gap_lo, gap_hi = (0.0, 0.0, 0.0)
    if len(alpha) > 0 and len(omni) > 0:
        gap_mean, gap_lo, gap_hi = bootstrap_diff(alpha, omni, n_boot=args.bootstrap_n, seed=args.seed)

    with open(output_dir / "summary.txt", "w") as f:
        f.write("Scale Illusion Analysis Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Alphabet accuracy: {alpha.mean()*100:.2f}% (n={len(alpha)})\n")
        f.write(
            f"Omniglot accuracy (excl. hand_digits/hand_english/times_new_roman): "
            f"{omni.mean()*100:.2f}% (n={len(omni)})\n"
        )
        f.write("\nSplit summary (TPR / TNR / Accuracy %)\n")
        f.write("-" * 40 + "\n")
        for _, row in split_summary_table(df).iterrows():
            f.write(
                f"{row['split']}: ACC={row['accuracy_pct']:.2f}%  "
                f"TPR={row['tpr_pct']:.2f}%  TNR={row['tnr_pct']:.2f}%  "
                f"(n={int(row['n_total'])}, +{int(row['n_positive'])}/-{int(row['n_negative'])})\n"
            )
        f.write(f"Gap (Alphabet - Omniglot): {gap_mean*100:.2f}%\n")
        if len(alpha) > 0 and len(omni) > 0:
            f.write(f"95% bootstrap CI: [{gap_lo*100:.2f}%, {gap_hi*100:.2f}%]\n")
        script_acc = (
            df.groupby("script_name")["is_correct"]
            .mean()
            .reset_index(name="accuracy")
        )
        english_acc = script_acc[script_acc["script_name"] == "English"]["accuracy"]
        omniglot_accs = script_acc[
            (script_acc["script_name"] != "English")
            & (~script_acc["script_name"].isin(OMNIGLOT_GROUP_EXCLUDE))
        ]["accuracy"]
        if len(english_acc) > 0 and len(omniglot_accs) > 0:
            eng_val = float(english_acc.iloc[0])
            omni_vals = omniglot_accs.to_numpy()
            d_val = cohens_d(np.array([eng_val]), omni_vals)
            f.write(f"English vs Omniglot scripts (mean): {eng_val*100:.2f}% vs {omni_vals.mean()*100:.2f}%\n")
            f.write(f"English minus Omniglot scripts mean: {(eng_val - omni_vals.mean())*100:.2f}%\n")
            f.write(f"Cohen's d (English vs Omniglot scripts): {d_val:.3f}\n")

        if args.alphabet_dir or args.omniglot_dir:
            script_paths = load_script_image_paths(
                args.alphabet_dir,
                args.omniglot_dir,
                scripts_in_df=sorted(df["script_name"].unique().tolist()),
            )
            complexity_df = compute_script_complexity(script_paths, threshold=args.ink_threshold)
            if not complexity_df.empty:
                acc_df = (
                    df.groupby("script_name")["is_correct"]
                    .apply(accuracy_rate)
                    .reset_index(name="accuracy")
                )
                merged = acc_df.merge(complexity_df, on="script_name", how="inner")
                if not merged.empty:
                    merged_omni = merged[
                        (merged["script_name"] != "English")
                        & (~merged["script_name"].isin(OMNIGLOT_GROUP_EXCLUDE))
                    ]
                    if not merged_omni.empty:
                        corr_ink = merged_omni["accuracy"].corr(merged_omni["ink_ratio"])
                        f.write(f"Accuracy vs Ink Ratio correlation (Omniglot): {corr_ink:.3f}\n")
                        if merged_omni["perimetric_complexity"].notna().any():
                            corr_perim = merged_omni["accuracy"].corr(merged_omni["perimetric_complexity"])
                            f.write(f"Accuracy vs Perimetric Complexity correlation (Omniglot): {corr_perim:.3f}\n")

    # Save tables
    summary_table(df).to_csv(output_dir / "script_summary.csv", index=False)
    split_df = split_summary_table(df)
    split_df.to_csv(output_dir / "split_summary.csv", index=False)
    print("\n=== Split summary (TPR / TNR / Accuracy %) ===")
    print(split_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Plots
    plot_overall_accuracy(df, output_dir)
    plot_script_accuracy(df, output_dir)
    plot_script_accuracy_per_scale(df, output_dir)
    plot_accuracy_by_scale(df, output_dir)
    plot_accuracy_by_scale_for_scripts(df, output_dir)
    plot_accuracy_by_scale_per_script_separate(df, output_dir)
    plot_positive_negative_accuracy(df, output_dir)
    plot_prior_gap_by_scale(df, output_dir)
    plot_script_accuracy_distribution(df, output_dir)
    plot_script_scale_heatmap(df, output_dir)
    if args.focus_scripts:
        plot_script_scale_heatmap(df, output_dir, focus_scripts=args.focus_scripts)
    plot_script_sensitivity(df, output_dir)
    analyze_signal_detection(df, output_dir)
    analyze_consistency(df, output_dir)
    analyze_error_entropy(df, output_dir)
    plot_recall_specificity_selected(df, output_dir, n=3, use_ag_colors=args.ag)
    plot_recall_specificity_selected_by_scale(df, output_dir, n=3, use_ag_colors=args.ag)
    plot_script_distribution(df, output_dir)
    plot_response_length(df, output_dir)
    plot_script_embeddings(df, output_dir)

    # Visual complexity analysis
    if args.alphabet_dir or args.omniglot_dir:
        script_paths = load_script_image_paths(
            args.alphabet_dir,
            args.omniglot_dir,
            scripts_in_df=sorted(df["script_name"].unique().tolist()),
        )
        complexity_df = compute_script_complexity(script_paths, threshold=args.ink_threshold)
        if not complexity_df.empty:
            complexity_df.to_csv(output_dir / "script_complexity.csv", index=False)
            plot_accuracy_vs_complexity(df, complexity_df, output_dir, focus_scripts=args.focus_scripts)
            plot_d_prime_vs_complexity(df, complexity_df, output_dir, focus_scripts=args.focus_scripts)
    if args.attn_metrics_csv:
        attn_path = Path(args.attn_metrics_csv)
        if attn_path.exists():
            attn_df = pd.read_csv(attn_path)
            plot_complexity_vs_rank(complexity_df, attn_df, output_dir, pair_type=args.rank_pair_type)
            plot_rank_divergence_vs_accuracy(
                df,
                attn_df,
                output_dir,
                pair_type=args.rank_pair_type,
                focus_scripts=args.focus_scripts,
                invert_sign=args.invert_correlation_sign,
            )
            plot_rank_divergence_vs_rate(
                df,
                attn_df,
                output_dir,
                rate_label="TPR",
                is_positive=True,
                pair_type=args.rank_pair_type,
                focus_scripts=args.focus_scripts,
                invert_sign=args.invert_correlation_sign,
            )
            plot_rank_divergence_vs_rate(
                df,
                attn_df,
                output_dir,
                rate_label="TNR",
                is_positive=False,
                pair_type=args.rank_pair_type,
                focus_scripts=args.focus_scripts,
                invert_sign=args.invert_correlation_sign,
            )
            rank_metrics = compute_rank_divergence_metrics(attn_df, pair_type=args.rank_pair_type)
            correlate_rank_metrics_with_performance(
                df,
                rank_metrics,
                output_dir,
                focus_scripts=args.focus_scripts,
                invert_sign=args.invert_correlation_sign,
            )
            plot_layerwise_rank_correlation(
                df,
                attn_df,
                output_dir,
                pair_type=args.rank_pair_type,
                invert_sign=args.invert_correlation_sign,
            )

    # PCA manifold geometry
    if args.pca_model and (args.alphabet_dir or args.omniglot_dir):
        if not _torch_ready():
            detail = f" ({_QWEN_IMPORT_ERROR})" if _QWEN_IMPORT_ERROR else ""
            print(f"[WARN] Skipping PCA: Qwen dependencies not available{detail}")
        else:
            device_str = args.analysis_device or ("cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device(device_str)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.pca_model,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                device_map=None,
            )
            model.to(device).eval()
            processor = AutoProcessor.from_pretrained(args.pca_model)
            script_paths = load_script_image_paths(
                args.alphabet_dir,
                args.omniglot_dir,
                scripts_in_df=sorted(df["script_name"].unique().tolist()),
            )
            scripts = args.pca_scripts if args.pca_scripts else list(script_paths.keys())
            if args.pca_num_scripts:
                scripts = scripts[: args.pca_num_scripts]
            for script in scripts:
                if script not in script_paths:
                    continue
                plot_manifold_pca(
                    model,
                    processor,
                    script,
                    script_paths[script],
                    output_dir,
                    args.pca_prompt,
                    device,
                    args.pca_image_size,
                    args.pca_small_scale,
                    args.pca_large_scale,
                )
            # Global PCA plot with highlights
            highlight = []
            for name in ["English", "Latin", "Greek"]:
                if name in script_paths:
                    highlight.append(name)
            if len(highlight) < 3:
                # Fallback: try partial matches
                for script in script_paths.keys():
                    low = script.lower()
                    if "latin" in low and script not in highlight:
                        highlight.append(script)
                    if "greek" in low and script not in highlight:
                        highlight.append(script)
            # Add three more scripts by accuracy (top, mid, bottom) excluding highlights
            script_acc = (
                df.groupby("script_name")["is_correct"]
                .apply(accuracy_rate)
                .sort_values(ascending=False)
            )
            acc_scripts = [s for s in script_acc.index if s not in highlight]
            extra = []
            if acc_scripts:
                extra.append(acc_scripts[0])
                extra.append(acc_scripts[len(acc_scripts) // 2])
                extra.append(acc_scripts[-1])
            highlight += [s for s in extra if s not in highlight]
            plot_global_manifold_pca(
                model,
                processor,
                script_paths,
                output_dir,
                args.pca_prompt,
                device,
                args.pca_image_size,
                args.pca_small_scale,
                args.pca_large_scale,
                highlight_scripts=highlight,
            )

    # RSA analysis
    if args.rsa_model and (args.alphabet_dir or args.omniglot_dir):
        if not _torch_ready():
            detail = f" ({_QWEN_IMPORT_ERROR})" if _QWEN_IMPORT_ERROR else ""
            print(f"[WARN] Skipping RSA: Qwen dependencies not available{detail}")
        else:
            device_str = args.analysis_device or ("cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device(device_str)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.rsa_model,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                device_map=None,
            )
            model.to(device).eval()
            processor = AutoProcessor.from_pretrained(args.rsa_model)
            script_paths = load_script_image_paths(
                args.alphabet_dir,
                args.omniglot_dir,
                scripts_in_df=sorted(df["script_name"].unique().tolist()),
            )
            scripts = args.rsa_scripts if args.rsa_scripts else list(script_paths.keys())[: args.rsa_num_scripts]
            for script in scripts:
                if script not in script_paths:
                    continue
                compute_rsa_for_script(
                    model,
                    processor,
                    script,
                    script_paths[script],
                    output_dir,
                    args.rsa_prompt,
                    device,
                    args.rsa_image_size,
                    args.rsa_small_scale,
                    args.rsa_large_scale,
                    args.rsa_num_chars,
                )

    # Tokenizer fertility
    if args.tokenizer_model and args.tokenizer_texts_csv:
        if AutoTokenizer is None:
            print("[WARN] Skipping tokenizer analysis: transformers not available.")
        else:
            tok_df = pd.read_csv(args.tokenizer_texts_csv)
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
            analyze_tokenizer_fertility(tokenizer, tok_df, output_dir)

    print(f"Analysis saved to: {output_dir}")


if __name__ == "__main__":
    main()
