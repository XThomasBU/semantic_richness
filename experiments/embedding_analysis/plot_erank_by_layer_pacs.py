#!/usr/bin/env python3
"""
PACS-only eRank vs layer plot.

Plots mean eRank (+/- 1 std band) for the four PACS domains:
Photo, Cartoon, Art, Sketch.

Input CSV can be either:
1) per-domain summary: columns [domain, layer, erank_mean, erank_std]
2) per-image values:   columns [domain, layer, erank]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AXIS_LABEL_FONTSIZE = 20
AXIS_TICK_FONTSIZE = 16
LINE_WIDTH = 2.4
SCATTER_POINT_SIZE = 170

DOMAIN_STYLE = {
    "photo": ("Photo", "#1F77B4"),
    "cartoon": ("Cartoon", "#E6914B"),
    "art_painting": ("Art", "#2CA02C"),
    "sketch": ("Sketch", "#9467BD"),
}
PACS_DOMAIN_ORDER = ["photo", "cartoon", "art_painting", "sketch"]


def _savefig_png_and_pdf(out_path: Path | str) -> None:
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".png":
        out_path = out_path.with_suffix(".png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")


def _style_axes(ax) -> None:
    ax.xaxis.label.set_fontsize(AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_fontsize(AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=AXIS_TICK_FONTSIZE, width=1.1, length=6)
    ax.grid(True, alpha=0.28, linestyle="-", linewidth=0.6, color="0.45", zorder=0)
    ax.set_axisbelow(True)


def _set_layer_xticks(ax, n_layers: int) -> None:
    if n_layers <= 32:
        ax.set_xticks(np.arange(0, n_layers, 1))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
    else:
        step = max(1, n_layers // 20)
        ax.set_xticks(np.arange(0, n_layers, step))


def _normalize_domain_name(v: str) -> str:
    s = str(v).strip().lower()
    if s in {"art", "artpainting", "art_painting"}:
        return "art_painting"
    return s


def _load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    req_summary = {"domain", "layer", "erank_mean", "erank_std"}
    req_per_image = {"domain", "layer", "erank"}

    if req_summary.issubset(df.columns):
        out = df[["domain", "layer", "erank_mean", "erank_std"]].copy()
    elif req_per_image.issubset(df.columns):
        out = (
            df.groupby(["domain", "layer"], as_index=False)
            .agg(erank_mean=("erank", "mean"), erank_std=("erank", "std"))
            .copy()
        )
    else:
        raise ValueError(
            f"{csv_path} must have either columns {sorted(req_summary)} "
            f"or {sorted(req_per_image)}"
        )

    out["domain"] = out["domain"].map(_normalize_domain_name)
    out["layer"] = out["layer"].astype(int)
    out["erank_mean"] = out["erank_mean"].astype(float)
    out["erank_std"] = out["erank_std"].fillna(0.0).astype(float)
    return out


def _load_domain_accuracy(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    req = {"domain", "accuracy"}
    if not req.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain columns: {sorted(req)}")
    out = df[["domain", "accuracy"]].copy()
    out["domain"] = out["domain"].map(_normalize_domain_name)
    out["accuracy"] = out["accuracy"].astype(float)
    # Handle both [0,1] and [0,100] conventions.
    if out["accuracy"].max() <= 1.0:
        out["accuracy"] = out["accuracy"] * 100.0
    return out


def _pearson_r_xy(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def plot_pacs_erank_by_layer(summary: pd.DataFrame, out_path: Path) -> None:
    df = summary[summary["domain"].isin(PACS_DOMAIN_ORDER)].copy()
    if df.empty:
        raise ValueError("No PACS domains found in input CSV (expected photo/cartoon/art_painting/sketch).")

    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    y_min, y_max = np.inf, -np.inf
    for dom in PACS_DOMAIN_ORDER:
        g = df[df["domain"] == dom].sort_values("layer")
        if g.empty:
            continue
        label, color = DOMAIN_STYLE[dom]
        x = g["layer"].to_numpy(dtype=float)
        m = g["erank_mean"].to_numpy(dtype=float)
        s = g["erank_std"].to_numpy(dtype=float)
        lo = np.maximum(m - s, 0.0)
        hi = m + s
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0.0, zorder=2)
        ax.plot(x, m, linewidth=LINE_WIDTH, color=color, label=label, zorder=3)
        y_min = min(y_min, float(lo.min()))
        y_max = max(y_max, float(hi.max()))

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        raise ValueError("Unable to compute y-range for PACS plot.")

    n_layers = int(df["layer"].max()) + 1
    ax.set_xlim(-0.5, n_layers - 0.5)
    y_pad = max(0.5, 0.08 * (y_max - y_min))
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel("Layer")
    ax.set_ylabel("eRank")
    _set_layer_xticks(ax, n_layers)
    _style_axes(ax)
    ax.legend(loc="best", frameon=True, edgecolor="0.8", fontsize=AXIS_TICK_FONTSIZE - 1)

    plt.tight_layout()
    _savefig_png_and_pdf(out_path)
    plt.close(fig)


def plot_pacs_accuracy_vs_erank(
    summary: pd.DataFrame,
    acc_df: pd.DataFrame,
    layer: int,
    out_path: Path,
) -> None:
    s = summary[summary["domain"].isin(PACS_DOMAIN_ORDER)].copy()
    a = acc_df[acc_df["domain"].isin(PACS_DOMAIN_ORDER)].copy()
    s = s[s["layer"] == int(layer)][["domain", "erank_mean"]].copy()
    if s.empty:
        raise ValueError(f"No eRank rows found for layer={layer}.")
    df = s.merge(a, on="domain", how="inner")
    if df.empty:
        raise ValueError("No overlapping PACS domains between eRank and accuracy CSV.")

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    for dom in PACS_DOMAIN_ORDER:
        g = df[df["domain"] == dom]
        if g.empty:
            continue
        label, color = DOMAIN_STYLE[dom]
        x = float(g["erank_mean"].iloc[0])
        y = float(g["accuracy"].iloc[0])
        ax.scatter(
            [x],
            [y],
            s=SCATTER_POINT_SIZE,
            alpha=0.95,
            c=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
            label=label,
        )
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(8, 8),
            ha="left",
            va="bottom",
            fontsize=AXIS_TICK_FONTSIZE - 1,
            color=color,
            zorder=5,
        )

    r = _pearson_r_xy(df["erank_mean"].to_numpy(), df["accuracy"].to_numpy())
    ax.text(
        0.98,
        0.94,
        f"Pearson r = {r:.2f}" if np.isfinite(r) else "Pearson r = nan",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=AXIS_TICK_FONTSIZE - 1,
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.78", linewidth=1.0, alpha=0.95),
    )

    ax.set_xlabel(f"eRank (layer {layer})")
    ax.set_ylabel("Accuracy (%)")
    _style_axes(ax)
    # Keep legend compact even with labels annotated.
    ax.legend(loc="lower right", frameon=True, edgecolor="0.8", fontsize=AXIS_TICK_FONTSIZE - 2)
    plt.tight_layout()
    _savefig_png_and_pdf(out_path)
    plt.close(fig)


def _best_accuracy_layer(summary: pd.DataFrame, acc_df: pd.DataFrame) -> tuple[int, float]:
    """Select layer with strongest absolute Pearson correlation |r|."""
    s_all = summary[summary["domain"].isin(PACS_DOMAIN_ORDER)].copy()
    a = acc_df[acc_df["domain"].isin(PACS_DOMAIN_ORDER)][["domain", "accuracy"]].copy()
    best_layer = -1
    best_abs_r = -np.inf
    best_r = float("nan")
    for layer, s_layer in s_all.groupby("layer", sort=True):
        merged = s_layer[["domain", "erank_mean"]].merge(a, on="domain", how="inner")
        if len(merged) < 3:
            continue
        r = _pearson_r_xy(merged["erank_mean"].to_numpy(), merged["accuracy"].to_numpy())
        if np.isfinite(r) and abs(r) > best_abs_r:
            best_abs_r = abs(r)
            best_layer = int(layer)
            best_r = float(r)
    if best_layer < 0:
        raise ValueError("Could not determine best layer for PACS accuracy-vs-eRank.")
    return best_layer, best_r


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PACS-only eRank by layer plot (Photo/Cartoon/Art/Sketch).")
    p.add_argument(
        "--input_csv",
        type=str,
        default="pacs_erank_sanity/erank_mean_by_domain_layer.csv",
        help="Input CSV with domain/layer/eRank stats.",
    )
    p.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Output PNG path (PDF also saved). Default: <input_dir>/pacs_erank_by_layer.png",
    )
    p.add_argument(
        "--accuracy_csv",
        type=str,
        default="results_pacs/scale_illusion_pacs_gpt-5_2/scale_illusion_pacs_prompt_v2/gpt_5_2_scale_illusion_pacs_domain_summary.csv",
        help="PACS domain accuracy CSV (columns: domain, accuracy).",
    )
    p.add_argument(
        "--accuracy_layer",
        type=int,
        default=-1,
        help="Transformer layer for PACS accuracy-vs-eRank; set -1 to auto-select best |r| layer.",
    )
    p.add_argument(
        "--accuracy_out_path",
        type=str,
        default=None,
        help="Output PNG for accuracy-vs-eRank (PDF also saved). "
        "Default: <input_dir>/pacs_accuracy_vs_erank_layer<L>.png",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    out_path = (
        Path(args.out_path).resolve()
        if args.out_path
        else input_csv.parent / "pacs_erank_by_layer.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(input_csv)
    plot_pacs_erank_by_layer(summary, out_path)
    print(f"[OK] Saved: {out_path} and {out_path.with_suffix('.pdf')}")

    accuracy_csv = Path(args.accuracy_csv).resolve()
    if accuracy_csv.is_file():
        acc_df = _load_domain_accuracy(accuracy_csv)
        selected_layer = int(args.accuracy_layer)
        if selected_layer < 0:
            selected_layer, selected_r = _best_accuracy_layer(summary, acc_df)
            print(f"[info] Selected best PACS layer={selected_layer} (|r|={abs(selected_r):.3f}, r={selected_r:.3f})")
        acc_out = (
            Path(args.accuracy_out_path).resolve()
            if args.accuracy_out_path
            else input_csv.parent / f"pacs_accuracy_vs_erank_layer{selected_layer}.png"
        )
        plot_pacs_accuracy_vs_erank(summary, acc_df, selected_layer, acc_out)
        print(f"[OK] Saved: {acc_out} and {acc_out.with_suffix('.pdf')}")
    else:
        print(f"[warn] accuracy CSV not found; skip accuracy-vs-eRank: {accuracy_csv}")


if __name__ == "__main__":
    main()
