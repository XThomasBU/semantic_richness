from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from analyze_scale_illusion import OMNIGLOT_GROUP_EXCLUDE, accuracy_rate


def _char_id_to_int(char_id: str) -> int | None:
    """Extract numeric part from char_id (e.g. 'character01' -> 1, 'character25' -> 25)."""
    if not isinstance(char_id, str) or "character" not in char_id.lower():
        return None
    try:
        return int(char_id.lower().replace("character", "").strip())
    except ValueError:
        return None


def _drop_gibberish_response_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where ``response`` looks corrupted (spam, mixed scripts, replacement chars)."""
    if "response" not in df.columns:
        return df

    def is_gibberish(val: object) -> bool:
        if pd.isna(val):
            return False
        t = str(val)
        # Valid YES/NO-style answers are short; corrupted runs are typically much longer.
        if len(t) > 80:
            return True
        if "\ufffd" in t:
            return True
        low = t.lower()
        if "ationtoken" in low or "alicalic" in low:
            return True
        if any("\u4e00" <= c <= "\u9fff" for c in t):
            return True
        return False

    return df.loc[~df["response"].map(is_gibberish)].copy()


def _filter_character_ranges(df: pd.DataFrame, lowercase_only: bool) -> pd.DataFrame:
    """
    Character selection for metrics:
    - English: always keep only character01–character26 (lowercase a–z set).
    - If lowercase_only: handwritten English (hand_english) character0–character25 only.
    - If not lowercase_only: other scripts unchanged; hand_english is not restricted here.
    """
    if "char_id" not in df.columns:
        return df

    def keep_row(row: pd.Series) -> bool:
        script = str(row.get("script_name", ""))
        char_id = row.get("char_id")
        n = _char_id_to_int(char_id)

        if script == "English":
            if n is None:
                return True
            return 1 <= n <= 26

        if not lowercase_only:
            return True
        if n is None:
            return True
        if script == "hand_english":
            return 0 <= n <= 25
        return True

    mask = df.apply(keep_row, axis=1)
    return df.loc[mask].copy()


def _load_scale_illusion_csv(path: Path, lowercase_only: bool = True) -> pd.DataFrame:
    """Load a scale-illusion CSV and normalize key columns."""
    df = pd.read_csv(path)
    required = {"dataset", "script_name", "is_positive", "scale_factor", "is_correct"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")

    # Coerce booleans robustly in case they are stored as strings.
    for col in ["is_positive", "is_correct"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )

    # Ensure scale_factor is numeric.
    df["scale_factor"] = pd.to_numeric(df["scale_factor"], errors="coerce")
    df = df.dropna(subset=["scale_factor"])

    df = _drop_gibberish_response_rows(df)
    df = _filter_character_ranges(df, lowercase_only=lowercase_only)
    return df


def _compute_metrics(group: pd.DataFrame) -> dict:
    """Compute TPR, TNR, and overall accuracy for a group of trials."""
    pos = group[group["is_positive"] == True]
    neg = group[group["is_positive"] == False]

    n_pos = int(len(pos))
    n_neg = int(len(neg))
    n_total = int(len(group))

    tpr = float(accuracy_rate(pos["is_correct"])) * 100.0 if n_pos > 0 else np.nan
    tnr = float(accuracy_rate(neg["is_correct"])) * 100.0 if n_neg > 0 else np.nan
    acc = float(accuracy_rate(group["is_correct"])) * 100.0 if n_total > 0 else np.nan

    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_total": n_total,
        "TPR": tpr,
        "TNR": tnr,
        "Accuracy": acc,
    }


def compute_overall_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Metrics aggregated over all scales: one row per (dataset, script)."""
    rows = []
    for (dataset, script), group in df.groupby(["dataset", "script_name"]):
        metrics = _compute_metrics(group)
        metrics.update({"dataset": dataset, "script_name": script})
        rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    cols = ["dataset", "script_name", "n_pos", "n_neg", "n_total", "TPR", "TNR", "Accuracy"]
    return pd.DataFrame(rows)[cols]


def compute_per_scale_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Metrics computed separately for each scale: one row per (dataset, script, scale_factor)."""
    rows = []
    for (dataset, script, scale), group in df.groupby(["dataset", "script_name", "scale_factor"]):
        metrics = _compute_metrics(group)
        metrics.update({"dataset": dataset, "script_name": script, "scale_factor": float(scale)})
        rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    cols = ["dataset", "script_name", "scale_factor", "n_pos", "n_neg", "n_total", "TPR", "TNR", "Accuracy"]
    return pd.DataFrame(rows)[cols].sort_values(["dataset", "script_name", "scale_factor"])


def _add_dataset_group_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add dataset_group: English, hand_english, Omniglot (excluding hand_digits etc.)."""

    def _map_row(row: pd.Series) -> str | None:
        script = str(row["script_name"])
        dataset = str(row["dataset"])
        if script == "English":
            return "Times New Roman"
        if script == "hand_english":
            return "Handwritten English"
        if dataset == "omniglot" and script not in OMNIGLOT_GROUP_EXCLUDE:
            return "Omniglot"
        return None

    df = df.copy()
    df["dataset_group"] = df.apply(_map_row, axis=1)
    return df.dropna(subset=["dataset_group"])


def compute_dataset_group_overall_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Metrics aggregated over all scales per dataset group."""
    df_g = _add_dataset_group_column(df)
    rows = []
    for group, gdf in df_g.groupby("dataset_group"):
        metrics = _compute_metrics(gdf)
        metrics.update({"dataset_group": group})
        rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    cols = ["dataset_group", "n_pos", "n_neg", "n_total", "TPR", "TNR", "Accuracy"]
    return pd.DataFrame(rows)[cols]


def compute_dataset_group_per_scale_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Metrics per scale per dataset group."""
    df_g = _add_dataset_group_column(df)
    rows = []
    for (group, scale), gdf in df_g.groupby(["dataset_group", "scale_factor"]):
        metrics = _compute_metrics(gdf)
        metrics.update({"dataset_group": group, "scale_factor": float(scale)})
        rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    cols = ["dataset_group", "scale_factor", "n_pos", "n_neg", "n_total", "TPR", "TNR", "Accuracy"]
    return pd.DataFrame(rows)[cols].sort_values(["dataset_group", "scale_factor"])


def _write_metrics_for_file(
    csv_path: Path, out_dir: Path | None = None, lowercase_only: bool = True
) -> None:
    df = _load_scale_illusion_csv(csv_path, lowercase_only=lowercase_only)
    overall = compute_overall_metrics(df)
    per_scale = compute_per_scale_metrics(df)
    overall_ds = compute_dataset_group_overall_metrics(df)
    per_scale_ds = compute_dataset_group_per_scale_metrics(df)

    base = csv_path.stem
    if out_dir is None:
        out_dir = csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_path = out_dir / f"{base}_metrics_overall.csv"
    per_scale_path = out_dir / f"{base}_metrics_by_scale.csv"
    overall_ds_path = out_dir / f"{base}_metrics_dataset_overall.csv"
    per_scale_ds_path = out_dir / f"{base}_metrics_dataset_by_scale.csv"

    overall.to_csv(overall_path, index=False)
    per_scale.to_csv(per_scale_path, index=False)
    overall_ds.to_csv(overall_ds_path, index=False)
    per_scale_ds.to_csv(per_scale_ds_path, index=False)

    print(f"Wrote overall metrics to: {overall_path}")
    print(f"Wrote per-scale metrics to: {per_scale_path}")
    print(f"Wrote dataset-group overall metrics to: {overall_ds_path}")
    print(f"Wrote dataset-group per-scale metrics to: {per_scale_ds_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute TPR, TNR, and Accuracy (overall and per scale) "
            "from scale-illusion CSV result files."
        )
    )
    parser.add_argument(
        "csvs",
        type=Path,
        nargs="+",
        help="One or more scale-illusion CSV files (with columns like dataset, script_name, is_positive, scale_factor, is_correct).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Optional directory to write summaries into (defaults to each CSV's own directory).",
    )
    parser.add_argument(
        "--all_characters",
        action="store_true",
        help=(
            "Include all characters for non-English scripts; English is always restricted to "
            "character01–26. Default also restricts hand_english to character0–25."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    lowercase_only = not getattr(args, "all_characters", False)
    for csv_path in args.csvs:
        _write_metrics_for_file(csv_path, out_dir=args.out_dir, lowercase_only=lowercase_only)


if __name__ == "__main__":
    main()

