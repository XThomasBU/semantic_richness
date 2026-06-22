#!/usr/bin/env python3
"""
Plot eRank across layers: English vs selected scripts.

Each subplot overlays:
- English (always present)
- one target script

Colors intentionally match `experiments/embedding_analysis/plot_erank_by_layer.py`:
- English (Times New Roman): blue (#1F77B4)
- Handwritten English: purple (#9467BD)
- Other highlighted scripts: orange (#E6914B)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_TR = "Times New Roman"
LABEL_HE = "Handwritten English"

COLOR_TR = "#1F77B4"
TARGET_SCRIPT_COLORS = {
    "hand_english": "#9467BD",
    "Latin": "#E6914B",
    "Greek": "#E45756",
    "Mongolian": "#54A24B",
    "Keble": "#F1CE63",
    "Braille": "#B279A2",
}

LINESTYLES = {
    "English": "-",
    "hand_english": (0, (5, 2)),  # dashed
    "Latin": "-.",
    "Greek": ":",
    "Mongolian": (0, (3, 1, 1, 1)),  # dash-dot-dash
    "Keble": (0, (7, 2)),  # longer dash
    "Braille": (0, (2, 2)),  # short dash
}

SCATTER_TEXT_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 22
AXIS_TICK_FONTSIZE = SCATTER_TEXT_FONTSIZE

LINE_WIDTH = 2.4
COL_FIGWIDTH = 6.2
ROW_FIGHEIGHT = 5.2

TARGET_SCRIPTS_DEFAULT = [
    "hand_english",
    "Latin",
    "Greek",
    "Mongolian",
    "Keble",
    "Braille",
]

DISPLAY_NAME = {
    "English": LABEL_TR,
    "hand_english": LABEL_HE,
}


def _style_axes(ax) -> None:
    ax.xaxis.label.set_fontsize(AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_fontsize(AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=AXIS_TICK_FONTSIZE, width=1.1, length=6)
    ax.grid(False)
    ax.set_axisbelow(True)


def _savefig_png_and_pdf(out_path: Path | str) -> None:
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".png":
        out_path = out_path.with_suffix(".png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")


def _display(name: str) -> str:
    return DISPLAY_NAME.get(name, name)


def _script_accuracy_pct(merged: pd.DataFrame) -> dict[str, float]:
    """Mean accuracy per script (percent). Accuracy is constant per script in merged CSV."""
    if merged.empty or "script_name" not in merged.columns or "accuracy" not in merged.columns:
        return {}
    g = merged.groupby("script_name", as_index=True)["accuracy"].mean()
    out: dict[str, float] = {}
    for k, v in g.items():
        try:
            fv = float(v) * 100.0
            if np.isfinite(fv):
                out[str(k)] = fv
        except Exception:
            continue
    return out


def _display_with_acc(name: str, acc_pct: dict[str, float]) -> str:
    base = _display(name)
    if name in acc_pct:
        return f"{base} ({acc_pct[name]:.1f}%)"
    return base


def _line_color(script_name: str) -> str:
    if script_name == "English":
        return COLOR_TR
    return TARGET_SCRIPT_COLORS.get(script_name, "#E6914B")


def _line_style(script_name: str) -> str | tuple:
    return LINESTYLES.get(script_name, "-")


def _load_merged(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "merged_script_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    req = {"layer", "script_name", "erank_token_img2_mean"}
    if not req.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: {sorted(req)}")
    return df


def _pearson_r_xy(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _script_layer_series(df: pd.DataFrame, script_name: str) -> pd.DataFrame:
    sub = df[df["script_name"] == script_name].copy()
    if sub.empty:
        return sub
    # Aggregate per-layer mean/std for variance bands.
    sub = (
        sub.groupby("layer", as_index=False)["erank_token_img2_mean"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "erank_mean", "std": "erank_std"})
    )
    sub["erank_std"] = sub["erank_std"].fillna(0.0)
    return sub.sort_values("layer")


def plot_correlation_vs_layer(
    merged: pd.DataFrame,
    target_scripts: list[str] | None,
    out_path: Path,
) -> None:
    req = {"layer", "script_name", "erank_token_img2_mean", "accuracy"}
    if not req.issubset(merged.columns):
        print("[warn] Missing accuracy column(s); skip correlation-vs-layer plot.")
        return

    if target_scripts is None:
        # All scripts (except hand_digits) and keep hand_english excluded to reflect
        # "all Omniglot scripts + English baseline" style analyses.
        df = merged[~merged["script_name"].isin({"hand_digits", "hand_english"})].copy()
    else:
        keep = {"English", *target_scripts}
        df = merged[merged["script_name"].isin(keep)].copy()
    rows = []
    for layer, g in df.groupby("layer", sort=True):
        r = _pearson_r_xy(g["erank_token_img2_mean"].to_numpy(), g["accuracy"].to_numpy())
        rows.append({"layer": int(layer), "pearson_r": r, "n_scripts": int(g["script_name"].nunique())})
    corr_df = pd.DataFrame(rows).sort_values("layer")
    if corr_df.empty:
        print("[warn] No data for correlation-vs-layer plot.")
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(
        corr_df["layer"].to_numpy(dtype=float),
        corr_df["pearson_r"].to_numpy(dtype=float),
        marker="o",
        markersize=7.5,
        linewidth=LINE_WIDTH,
        color="#2E86AB",
        linestyle="-",
        zorder=3,
    )
    ax.axhline(0, color="0.55", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Pearson r (eRank vs accuracy)")
    _style_axes(ax)
    ax.grid(False)

    layer_int = sorted({int(x) for x in corr_df["layer"].unique().tolist()})
    if layer_int:
        start, stop = layer_int[0], layer_int[-1]
        xticks = list(range(start, stop + 1, 4))
        if len(xticks) == 1 and stop > start:
            xticks = [start, stop]
        ax.set_xticks(xticks)

    # Highlight layer 11 explicitly (requested callout).
    p = corr_df["pearson_r"].to_numpy(dtype=float)
    m = np.isfinite(p)
    if m.any():
        layer_vals = corr_df["layer"].to_numpy(dtype=float)
        if np.any(layer_vals == 11):
            idx = int(np.where(layer_vals == 11)[0][0])
        else:
            # Fallback to minimum r if layer 11 is unavailable.
            idx = int(np.nanargmin(p))
        x_star = float(corr_df.iloc[idx]["layer"])
        y_star = float(corr_df.iloc[idx]["pearson_r"])
        ax.scatter(
            [x_star],
            [y_star],
            s=360,
            marker="*",
            color="#C1666B",
            edgecolors="white",
            linewidths=1.0,
            zorder=6,
        )
        ax.annotate(
            rf"$r={y_star:.2f}$",
            (x_star, y_star),
            textcoords="offset points",
            xytext=(12, 10),
            fontsize=AXIS_TICK_FONTSIZE - 1,
            ha="left",
            va="bottom",
            color="#2a2a2a",
            zorder=7,
        )

    # Requested: invert y-axis for this figure.
    ax.invert_yaxis()

    fig.tight_layout(pad=1.2)
    _savefig_png_and_pdf(out_path)
    plt.close(fig)


def plot_english_vs_scripts(
    merged: pd.DataFrame,
    target_scripts: list[str],
    out_path: Path,
) -> None:
    eng = _script_layer_series(merged, "English")
    if eng.empty:
        raise ValueError("No 'English' rows found in merged_script_metrics.csv")

    acc_pct = _script_accuracy_pct(merged)

    n = len(target_scripts)
    nrows = 2 if n > 3 else 1
    ncols = int(np.ceil(n / nrows))
    fig_w = COL_FIGWIDTH * ncols
    fig_h = ROW_FIGHEIGHT * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    y_min, y_max = np.inf, -np.inf
    plotted_axes = 0
    legend_handles: dict[str, object] = {}

    for i, target in enumerate(target_scripts):
        ax = axes[i]
        tgt = _script_layer_series(merged, target)
        if tgt.empty:
            ax.text(0.5, 0.5, f"Missing script:\n{_display(target)}", ha="center", va="center", fontsize=14)
            ax.set_axis_off()
            continue

        eng_x = eng["layer"].to_numpy(dtype=float)
        eng_y = eng["erank_mean"].to_numpy(dtype=float)
        eng_s = eng["erank_std"].to_numpy(dtype=float)
        ax.fill_between(
            eng_x,
            eng_y - eng_s,
            eng_y + eng_s,
            color=_line_color("English"),
            alpha=0.18,
            linewidth=0.0,
            zorder=2,
        )
        (line_eng_plot,) = ax.plot(
            eng["layer"].to_numpy(dtype=float),
            eng["erank_mean"].to_numpy(dtype=float),
            linewidth=LINE_WIDTH,
            color=_line_color("English"),
            linestyle=_line_style("English"),
            label=_display("English"),
            zorder=3,
        )
        tgt_x = tgt["layer"].to_numpy(dtype=float)
        tgt_y = tgt["erank_mean"].to_numpy(dtype=float)
        tgt_s = tgt["erank_std"].to_numpy(dtype=float)
        ax.fill_between(
            tgt_x,
            tgt_y - tgt_s,
            tgt_y + tgt_s,
            color=_line_color(target),
            alpha=0.18,
            linewidth=0.0,
            zorder=3,
        )
        (line_tgt,) = ax.plot(
            tgt["layer"].to_numpy(dtype=float),
            tgt["erank_mean"].to_numpy(dtype=float),
            linewidth=LINE_WIDTH,
            color=_line_color(target),
            linestyle=_line_style(target),
            label=_display(target),
            zorder=4,
        )
        if "English" not in legend_handles:
            # Build a consistent, single bottom legend like similarity-vs-angle.
            legend_handles["English"] = line_eng_plot
        legend_handles[target] = line_tgt

        y_min = min(y_min, (eng["erank_mean"] - eng["erank_std"]).min(), (tgt["erank_mean"] - tgt["erank_std"]).min())
        y_max = max(y_max, (eng["erank_mean"] + eng["erank_std"]).max(), (tgt["erank_mean"] + tgt["erank_std"]).max())
        ax.set_title(f"{_display_with_acc(target, acc_pct)}", fontsize=AXIS_LABEL_FONTSIZE, pad=8)
        _style_axes(ax)
        plotted_axes += 1

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_axis_off()

    if plotted_axes == 0:
        raise ValueError("No target scripts available to plot.")

    y_pad = max(0.5, (y_max - y_min) * 0.08) if np.isfinite(y_max - y_min) else 1.0
    all_layers = sorted(eng["layer"].unique())
    layer_int = sorted({int(x) for x in all_layers})
    if layer_int:
        # More spacing on x-axis ticks for readability.
        start = layer_int[0]
        stop = layer_int[-1]
        xticks = list(range(start, stop + 1, 4))
        if len(xticks) == 1 and stop > start:
            xticks = [start, stop]
    else:
        xticks = all_layers
    for ax in axes[:n]:
        if ax.axison:
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_xlabel("Layer")
            ax.set_ylabel("eRank")
            # Reduce tick density to avoid crowding.
            ax.set_xticks(xticks)
            ax.tick_params(axis="x", rotation=0)

    if legend_handles:
        desired_scripts = ["English"] + list(target_scripts)
        present = [s for s in desired_scripts if s in legend_handles]
        present += [s for s in legend_handles.keys() if s not in set(present)]
        labels = [_display(s) for s in present]
        handles = [legend_handles[s] for s in present]
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=max(1, len(labels)),
            fontsize=AXIS_TICK_FONTSIZE,
            frameon=True,
            fancybox=False,
            edgecolor="0.78",
            bbox_to_anchor=(0.5, -0.03),
        )

    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.98), pad=1.3, w_pad=1.8, h_pad=1.4)
    _savefig_png_and_pdf(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot English vs selected scripts eRank across layers.")
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing merged_script_metrics.csv (e.g. erank_out_FINAL).",
    )
    p.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Output .png path (PDF also saved). Default: <data_dir>/english_vs_selected_scripts_erank_by_layer.png",
    )
    p.add_argument(
        "--target_scripts",
        type=str,
        nargs="+",
        default=TARGET_SCRIPTS_DEFAULT,
        help="Target scripts to compare against English.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    out_path = (
        Path(args.out_path).resolve()
        if args.out_path
        else data_dir / "english_vs_selected_scripts_erank_by_layer.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged = _load_merged(data_dir)
    plot_english_vs_scripts(merged, args.target_scripts, out_path)
    corr_out = out_path.with_name("english_vs_selected_erank_accuracy_pearson_vs_layer.png")
    plot_correlation_vs_layer(merged, args.target_scripts, corr_out)
    print(f"[OK] Saved: {out_path} and {out_path.with_suffix('.pdf')}")
    print(f"[OK] Saved: {corr_out} and {corr_out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

