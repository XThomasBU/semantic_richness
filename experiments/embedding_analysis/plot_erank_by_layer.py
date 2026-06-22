#!/usr/bin/env python3
"""
Plot eRank / accuracy / perimetric relationships for a single transformer layer.

Points are colored by script family; **script names** are drawn only for highlighted scripts (no legend).

Reads cached outputs from effective_rank_analysis.py (e.g. merged_script_metrics.csv).

Also writes **−Pearson r vs layer** (inverted for readability; CSV still stores raw `pearson_r`)
in `erank_accuracy_pearson_vs_layer_<suffix>.png`/`.pdf` + `erank_accuracy_pearson_by_layer_<suffix>.csv`.
Each figure is saved as **PNG and PDF** (same basename).

Example:
  python experiments/embedding_analysis/plot_erank_by_layer.py \\
    --data_dir ./erank_out_FINAL --layer 12 --out_dir ./erank_out_FINAL/plots_layer12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Never used in any figure or correlation (drop rows from merged metrics).
EXCLUDE_FROM_ANALYSIS = frozenset({"hand_digits"})

# Excluded from --omniglot_only (core Omniglot scripts for paper-style correlations)
OMNIGLOT_ONLY_EXCLUDE = {"English", "hand_english"}

SCATTER_POINT_SIZE = 120
# One body size for ticks, script labels, and Pearson r box on scatter plots
SCATTER_TEXT_FONTSIZE = 18

# Axis titles slightly larger; correlation-vs-layer plot uses same tick/body sizes
AXIS_LABEL_FONTSIZE = 20
AXIS_TICK_FONTSIZE = SCATTER_TEXT_FONTSIZE

# All layer scatter panels share the same canvas size
SCATTER_FIGSIZE = (9.0, 6.5)

# Plot labels for CSV script_name (Omniglot scripts keep their script_name as-is).
ANNOTATE_DISPLAY_NAME: dict[str, str] = {
    "English": "Times New Roman",
    "hand_english": "Handwritten English",
}

LABEL_TR = "Times New Roman"
LABEL_HE = "Handwritten English"
LABEL_RED = "Latin, Greek, Malayalam, Braille, Keble, Oriya"
LABEL_OMNI = "Omniglot"

# Scripts drawn in rose/red (same hue as former Handwritten English)
HIGHLIGHT_RED_SCRIPTS = frozenset({"Latin", "Greek", "Malayalam", "Braille", "Keble", "Oriya"})

# Label on figure only for these script_name values (English / hand_english / rose scripts).
ANNOTATE_SCRIPT_NAMES = frozenset({"English", "hand_english", *HIGHLIGHT_RED_SCRIPTS})


# Plot order: (label, color, alpha, zorder); same point size for all groups
GROUP_STYLES: list[tuple[str, str, float, int]] = [
    (LABEL_TR, "#1F77B4", 0.95, 4),
    (LABEL_HE, "#9467BD", 0.9, 3),
    (LABEL_RED, "#E6914B", 0.88, 3),
    (LABEL_OMNI, "#E6914B", 0.56, 2),
]
GROUP_COLORS: dict[str, str] = {lbl: c for lbl, c, _, _ in GROUP_STYLES}


def _style_axes_typography(ax) -> None:
    """Same font size for x/y axis titles and tick labels."""
    ax.xaxis.label.set_fontsize(AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_fontsize(AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=AXIS_TICK_FONTSIZE, width=1.1, length=6)


def _style_scatter_axes(ax) -> None:
    """Typography + light grid for readability (same for all scatter figures)."""
    _style_axes_typography(ax)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.55, color="0.45", zorder=0)
    ax.set_axisbelow(True)


def _savefig_png_and_pdf(out_path: Path | str) -> None:
    """Write current figure as PNG (hi-dpi) and vector PDF next to it."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    if out_path.suffix.lower() != ".png":
        out_path = out_path.with_suffix(".png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")


def _add_pearson_stats_box(ax, r: float) -> None:
    """Top-right Pearson r; font matches tick labels and point annotations."""
    ax.text(
        0.98,
        0.94,
        f"Pearson r = {r:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=SCATTER_TEXT_FONTSIZE,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.78", linewidth=1.0, alpha=0.96),
    )


def drop_excluded_scripts(df: pd.DataFrame) -> pd.DataFrame:
    """Remove scripts excluded from all analyses (e.g. hand_digits)."""
    if df.empty or "script_name" not in df.columns:
        return df
    return df[~df["script_name"].isin(EXCLUDE_FROM_ANALYSIS)].copy()


def _annotate_script_names(
    ax,
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    *,
    xytext_overrides: dict[str, tuple[int, int]] | None = None,
    include_accuracy: bool = True,
) -> None:
    """Label highlighted points with script_name (optionally incl. accuracy)."""
    overrides = xytext_overrides or {}
    for _, row in df.iterrows():
        if row["script_name"] not in ANNOTATE_SCRIPT_NAMES:
            continue
        grp = row["group"]
        c = GROUP_COLORS.get(grp, "#333333")
        name = str(row["script_name"])
        label = ANNOTATE_DISPLAY_NAME.get(name, name)
        if include_accuracy and "accuracy" in df.columns:
            try:
                acc = float(row["accuracy"])
                if np.isfinite(acc):
                    label = f"{label} ({acc * 100.0:.1f}%)"
            except Exception:
                pass
        # Times New Roman (English): below the point, offset to the right (not centered)
        if name == "English":
            ax.annotate(
                label,
                (float(row[xcol]), float(row[ycol])),
                fontsize=SCATTER_TEXT_FONTSIZE,
                ha="left",
                va="top",
                color=c,
                alpha=0.95,
                xytext=(4, -12),
                textcoords="offset points",
                zorder=10,
            )
        else:
            xytext = overrides.get(name, (8, 8))
            ax.annotate(
                label,
                (float(row[xcol]), float(row[ycol])),
                fontsize=SCATTER_TEXT_FONTSIZE,
                ha="left",
                va="bottom",
                color=c,
                alpha=0.95,
                xytext=xytext,
                textcoords="offset points",
                zorder=10,
            )


def pearson_r_xy(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def script_plot_group(name: str) -> str:
    """Maps script_name to legend/color bucket."""
    if name == "English":
        return LABEL_TR
    if name == "hand_english":
        return LABEL_HE
    if name in HIGHLIGHT_RED_SCRIPTS:
        return LABEL_RED
    return LABEL_OMNI


def load_merged_script_metrics(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "merged_script_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def compute_erank_accuracy_pearson_by_layer(merged: pd.DataFrame, omniglot_only: bool) -> pd.DataFrame:
    """One Pearson r per layer between script-level eRank (image2) and accuracy."""
    req = {"layer", "script_name", "erank_token_img2_mean", "accuracy"}
    if not req.issubset(merged.columns):
        raise ValueError(f"merged_script_metrics.csv needs columns: {req}")
    df = drop_excluded_scripts(merged.copy())
    if omniglot_only:
        df = df[~df["script_name"].isin(OMNIGLOT_ONLY_EXCLUDE)]
    rows = []
    for layer, g in df.groupby("layer"):
        r = pearson_r_xy(g["erank_token_img2_mean"].to_numpy(), g["accuracy"].to_numpy())
        rows.append({"layer": int(layer), "pearson_r": r, "n_scripts": g["script_name"].nunique()})
    return pd.DataFrame(rows).sort_values("layer")


def plot_correlation_vs_layer(corr_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if corr_df.empty:
        return
    # Plot −r so upward trends match “higher agreement” with expected eRank–accuracy story
    pearson = corr_df["pearson_r"].to_numpy(dtype=float)
    layers = corr_df["layer"].to_numpy(dtype=float)
    neg_r = -pearson

    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    ax.plot(
        layers,
        neg_r,
        marker="o",
        markersize=10,
        linewidth=2.4,
        color="#2E86AB",
        zorder=2,
    )

    # Strongest |r| layer: star + Pearson r label only
    p = np.asarray(pearson, dtype=float)
    mask = np.isfinite(p)
    if mask.any():
        abs_masked = np.where(mask, np.abs(p), -np.inf)
        idx = int(np.argmax(abs_masked))
        x_star = float(layers[idx])
        y_star = float(neg_r[idx])
        r_star = float(pearson[idx])
        ax.scatter(
            [x_star],
            [y_star],
            s=520,
            marker="*",
            color="#C1666B",
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
        )
        ax.annotate(
            rf"$r={r_star:.2f}$",
            (x_star, y_star),
            textcoords="offset points",
            xytext=(14, 12),
            fontsize=AXIS_TICK_FONTSIZE,
            ha="left",
            va="bottom",
            color="#2a2a2a",
            zorder=6,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$-$Pearson $r$ (eRank vs accuracy)")
    _set_layer_xticks_corr(ax, int(corr_df["layer"].max()) + 1)
    _style_axes_typography(ax)
    plt.tight_layout()
    _savefig_png_and_pdf(out_path)
    plt.close()


def _set_layer_xticks_corr(ax, n_layers: int) -> None:
    """Integer layer ticks for correlation plot (reuse logic from pacs script)."""
    if n_layers <= 32:
        ax.set_xticks(np.arange(0, n_layers, 1))
        ax.tick_params(axis="x", labelsize=AXIS_TICK_FONTSIZE, width=1.1, length=6)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
    else:
        step = max(1, n_layers // 20)
        ax.set_xticks(np.arange(0, n_layers, step))


def load_layer_slice(data_dir: Path, layer: int) -> pd.DataFrame:
    path = data_dir / "merged_script_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    if "layer" not in df.columns:
        raise ValueError(f"{path} has no 'layer' column.")
    sub = df[df["layer"] == layer].copy()
    if sub.empty:
        avail = sorted(df["layer"].unique().tolist())
        raise ValueError(f"No rows for layer={layer}. Available layers: {avail[0]}..{avail[-1]} ({len(avail)} total).")
    return drop_excluded_scripts(sub)


def plot_accuracy_vs_erank(
    sub: pd.DataFrame,
    out_path: Path,
    omniglot_only: bool,
) -> None:
    import matplotlib.pyplot as plt

    df = sub.copy()
    if omniglot_only:
        df = df[~df["script_name"].isin(OMNIGLOT_ONLY_EXCLUDE)]
    df["group"] = df["script_name"].apply(script_plot_group)
    if df.empty or len(df) < 3:
        print(f"[skip] accuracy_vs_erank: not enough points (n={len(df)})")
        return

    r = pearson_r_xy(df["erank_token_img2_mean"].to_numpy(), df["accuracy"].to_numpy())

    fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)
    for grp, color, alpha, z in GROUP_STYLES:
        g = df[df["group"] == grp]
        if g.empty:
            continue
        ax.scatter(
            g["erank_token_img2_mean"],
            g["accuracy"] * 100.0,
            s=SCATTER_POINT_SIZE,
            alpha=alpha,
            c=color,
            edgecolors="white",
            linewidths=0.65,
            zorder=z,
        )

    df["_acc_pct"] = df["accuracy"] * 100.0
    _annotate_script_names(
        ax,
        df,
        "erank_token_img2_mean",
        "_acc_pct",
        xytext_overrides={
            "Keble": (-8, 8),
            "Braille": (8, -2),
            "Oriya": (-2, 8),
        },
    )

    ax.set_xlabel("eRank")
    ax.set_ylabel("Accuracy (%)")
    _style_scatter_axes(ax)
    _add_pearson_stats_box(ax, r)
    plt.tight_layout(pad=1.0)
    _savefig_png_and_pdf(out_path)
    plt.close()


def plot_perimetric_vs_erank(
    sub: pd.DataFrame,
    out_path: Path,
    omniglot_only: bool,
) -> None:
    import matplotlib.pyplot as plt

    df = sub[sub["perimetric_complexity"].notna()].copy()
    if omniglot_only:
        df = df[~df["script_name"].isin(OMNIGLOT_ONLY_EXCLUDE)]
    df["group"] = df["script_name"].apply(script_plot_group)
    if df.empty or len(df) < 3:
        print(f"[skip] perimetric_vs_erank: not enough points (n={len(df)})")
        return

    r = pearson_r_xy(df["perimetric_complexity"].to_numpy(), df["erank_token_img2_mean"].to_numpy())

    fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)
    for grp, color, alpha, z in GROUP_STYLES:
        g = df[df["group"] == grp]
        if g.empty:
            continue
        ax.scatter(
            g["perimetric_complexity"],
            g["erank_token_img2_mean"],
            s=SCATTER_POINT_SIZE,
            alpha=alpha,
            c=color,
            edgecolors="white",
            linewidths=0.65,
            zorder=z,
        )

    _annotate_script_names(ax, df, "perimetric_complexity", "erank_token_img2_mean")

    ax.set_xlabel("Perimetric complexity")
    ax.set_ylabel("eRank")
    _style_scatter_axes(ax)
    _add_pearson_stats_box(ax, r)
    plt.tight_layout(pad=1.0)
    _savefig_png_and_pdf(out_path)
    plt.close()


def plot_accuracy_vs_perimetric_layer(
    sub: pd.DataFrame,
    out_path: Path,
    omniglot_only: bool,
) -> None:
    """Same relationship as the global accuracy–perimetric plot."""
    import matplotlib.pyplot as plt

    df = sub[sub["perimetric_complexity"].notna()].copy()
    if omniglot_only:
        df = df[~df["script_name"].isin(OMNIGLOT_ONLY_EXCLUDE)]
    df["group"] = df["script_name"].apply(script_plot_group)
    if df.empty or len(df) < 3:
        print(f"[skip] accuracy_vs_perimetric: not enough points (n={len(df)})")
        return

    r = pearson_r_xy(df["perimetric_complexity"].to_numpy(), df["accuracy"].to_numpy())

    fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)
    for grp, color, alpha, z in GROUP_STYLES:
        g = df[df["group"] == grp]
        if g.empty:
            continue
        ax.scatter(
            g["perimetric_complexity"],
            g["accuracy"] * 100.0,
            s=SCATTER_POINT_SIZE,
            alpha=alpha,
            c=color,
            edgecolors="white",
            linewidths=0.65,
            zorder=z,
        )
    df["_acc_pct"] = df["accuracy"] * 100.0
    _annotate_script_names(ax, df, "perimetric_complexity", "_acc_pct")

    ax.set_xlabel("Perimetric complexity")
    ax.set_ylabel("Accuracy (%)")
    _style_scatter_axes(ax)
    _add_pearson_stats_box(ax, r)
    plt.tight_layout(pad=1.0)
    _savefig_png_and_pdf(out_path)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Layer-specific eRank / accuracy / complexity plots from merged_script_metrics.csv.")
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing merged_script_metrics.csv (e.g. erank_out_FINAL).",
    )
    p.add_argument("--layer", type=int, required=True, help="Transformer layer index (must exist in CSV).")
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Where to save PNG/PDF figures. Default: <data_dir>/plots_layer<L>.",
    )
    p.add_argument(
        "--omniglot_only",
        action="store_true",
        help="Only core Omniglot scripts (exclude Times New Roman and Handwritten English; hand_digits is never used).",
    )
    p.add_argument(
        "--skip_perimetric",
        action="store_true",
        help="Do not write perimetric-related figures.",
    )
    p.add_argument(
        "--skip_correlation_vs_layer",
        action="store_true",
        help="Do not write Pearson r vs layer (uses full merged_script_metrics.csv).",
    )
    args = p.parse_args()

    data_dir = Path(args.data_dir).resolve()
    layer = int(args.layer)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else data_dir / f"plots_layer{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_omniglot_only" if args.omniglot_only else "_all_scripts"

    if not args.skip_correlation_vs_layer:
        merged_path = data_dir / "merged_script_metrics.csv"
        if merged_path.is_file():
            try:
                merged = load_merged_script_metrics(data_dir)
                corr = compute_erank_accuracy_pearson_by_layer(merged, args.omniglot_only)
                corr.to_csv(data_dir / f"erank_accuracy_pearson_by_layer{suffix}.csv", index=False)
                plot_correlation_vs_layer(
                    corr,
                    data_dir / f"erank_accuracy_pearson_vs_layer{suffix}.png",
                )
            except Exception as exc:
                print(f"[warn] correlation vs layer skipped: {exc}")
        else:
            print(f"[warn] {merged_path} not found; skip correlation vs layer")

    sub = load_layer_slice(data_dir, layer)

    plot_accuracy_vs_erank(sub, out_dir / f"accuracy_vs_erank_img2_layer{layer}{suffix}.png", args.omniglot_only)

    if not args.skip_perimetric:
        if sub["perimetric_complexity"].notna().any():
            plot_perimetric_vs_erank(sub, out_dir / f"perimetric_vs_erank_img2_layer{layer}{suffix}.png", args.omniglot_only)
            plot_accuracy_vs_perimetric_layer(
                sub, out_dir / f"accuracy_vs_perimetric_layer{layer}_tag{suffix}.png", args.omniglot_only
            )
        else:
            print("[info] No perimetric_complexity column values; skip perimetric plots.")

    print(f"[OK] Saved PNG + PDF figures under {out_dir}")
    if not args.skip_correlation_vs_layer and (data_dir / "merged_script_metrics.csv").is_file():
        print(f"     Pearson r vs layer: {data_dir}/erank_accuracy_pearson_vs_layer{suffix}.png / .pdf")


if __name__ == "__main__":
    main()
