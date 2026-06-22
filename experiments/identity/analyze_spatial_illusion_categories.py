"""
Summarize Spatial Illusion results for three data categories:

1. alphabet + script_name=English
2. omniglot + script_name=hand_english
3. omniglot + all other script_name values

For each category:
- Accuracy = mean(is_correct)
- TPR = mean(prediction == 'yes' over is_positive == True)
- TNR = mean(prediction == 'no' over is_positive == False)

Rows with angle 180° or 270° are excluded (matches 10–90° step schedule only).

Omniglot aggregates exclude hand_english, hand_digits, and times_new_roman.
Also prints top / mid / bottom 3 Omniglot scripts by accuracy.
"""

import argparse
import os
from typing import List, Optional, Tuple

import pandas as pd

EXCLUDED_ANGLES = {180, 270}
OMNIGLOT_EXCLUDE = {"hand_english", "hand_digits", "times_new_roman"}
TOP_MID_BOTTOM_N = 3

PRINT_COLS = [
    "category",
    "n_total",
    "n_scripts",
    "accuracy_frac",
    "accuracy_pct",
    "tpr_frac",
    "tpr_pct",
    "tnr_frac",
    "tnr_pct",
]

SCRIPT_PRINT_COLS = [
    "tier",
    "script_name",
    "n_total",
    "accuracy_frac",
    "accuracy_pct",
    "tpr_frac",
    "tpr_pct",
    "tnr_frac",
    "tnr_pct",
]

CATEGORIES = [
    {
        "category": "English (alphabet)",
        "dataset": "alphabet",
        "script_name": "English",
        "omniglot_other": False,
    },
    {
        "category": "hand_english (omniglot)",
        "dataset": "omniglot",
        "script_name": "hand_english",
        "omniglot_other": False,
    },
    {
        "category": "Omniglot (other scripts)",
        "dataset": "omniglot",
        "script_name": None,
        "omniglot_other": True,
    },
]


def _safe_pct(numer: float, denom: float) -> float:
    if denom <= 0:
        return float("nan")
    return float(numer / denom * 100.0)


def _mask_omniglot_core(df: pd.DataFrame) -> pd.Series:
    return (df["dataset"] == "omniglot") & (~df["script_name"].isin(OMNIGLOT_EXCLUDE))


def _mask_for_category(df: pd.DataFrame, spec: dict) -> pd.Series:
    if spec["omniglot_other"]:
        return _mask_omniglot_core(df)
    return (df["dataset"] == spec["dataset"]) & (df["script_name"] == spec["script_name"])


def _metrics_for_subset(sdf: pd.DataFrame) -> dict:
    sdf_pos = sdf[sdf["is_positive"] == True]
    sdf_neg = sdf[sdf["is_positive"] == False]

    n_total = len(sdf)
    n_pos = len(sdf_pos)
    n_neg = len(sdf_neg)

    accuracy_pct = float(sdf["is_correct"].mean() * 100.0) if n_total > 0 else float("nan")
    tpr_numer = int((sdf_pos["prediction_norm"] == "yes").sum())
    tnr_numer = int((sdf_neg["prediction_norm"] == "no").sum())
    tpr_pct = _safe_pct(tpr_numer, n_pos)
    tnr_pct = _safe_pct(tnr_numer, n_neg)

    correct_n = int(sdf["is_correct"].sum())
    return {
        "n_total": n_total,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "accuracy_frac": f"{correct_n}/{n_total}" if n_total > 0 else "nan",
        "accuracy_pct": accuracy_pct,
        "tpr_frac": f"{tpr_numer}/{n_pos}" if n_pos > 0 else "nan",
        "tpr_pct": tpr_pct,
        "tnr_frac": f"{tnr_numer}/{n_neg}" if n_neg > 0 else "nan",
        "tnr_pct": tnr_pct,
        "n_correct": correct_n,
        "n_pred_yes_on_positive": tpr_numer,
        "n_pred_no_on_negative": tnr_numer,
    }


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    usecols = ["dataset", "script_name", "angle", "is_positive", "prediction", "is_correct"]
    df = pd.read_csv(csv_path, usecols=usecols)

    required_cols = ["dataset", "script_name", "angle", "is_positive", "prediction", "is_correct"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {missing}. Have columns: {list(df.columns)}"
        )

    df = df.copy()
    df["angle"] = pd.to_numeric(df["angle"], errors="coerce")
    n_before = len(df)
    df = df[~df["angle"].isin(EXCLUDED_ANGLES)].copy()
    n_excluded = n_before - len(df)
    if n_excluded:
        print(f"Excluded {n_excluded} rows with angle in {sorted(EXCLUDED_ANGLES)}")

    df["prediction_norm"] = df["prediction"].astype(str).str.strip().str.lower()
    return df


def _select_top_mid_bottom(script_names: List[str], n: int = TOP_MID_BOTTOM_N) -> List[Tuple[str, str]]:
    """Return (tier, script_name) for top n, middle n, bottom n by accuracy rank."""
    if not script_names:
        return []
    if len(script_names) < 2 * n + 1:
        tiers = [("top", script_names[:n]), ("mid", []), ("bottom", script_names[-n:])]
    else:
        top = script_names[:n]
        bottom = script_names[-n:]
        remaining = script_names[n:-n]
        mid_start = max(0, (len(remaining) - n) // 2)
        middle = remaining[mid_start : mid_start + n]
        tiers = [("top", top), ("mid", middle), ("bottom", bottom)]
    out: List[Tuple[str, str]] = []
    for tier, names in tiers:
        for name in names:
            out.append((tier, name))
    return out


def omniglot_per_script_table(df: pd.DataFrame) -> pd.DataFrame:
    omni = df[_mask_omniglot_core(df)]
    rows = []
    for script_name, sdf in omni.groupby("script_name", dropna=False):
        rows.append({"script_name": script_name, **_metrics_for_subset(sdf)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("accuracy_pct", ascending=False, kind="mergesort")


def omniglot_top_mid_bottom_table(df: pd.DataFrame) -> pd.DataFrame:
    per_script = omniglot_per_script_table(df)
    if per_script.empty:
        return per_script

    ranked = per_script["script_name"].tolist()
    selected = _select_top_mid_bottom(ranked, n=TOP_MID_BOTTOM_N)
    tier_map = {name: tier for tier, name in selected}
    out = per_script[per_script["script_name"].isin(tier_map)].copy()
    out["tier"] = out["script_name"].map(tier_map)
    tier_order = {"top": 0, "mid": 1, "bottom": 2}
    out["_tier_ord"] = out["tier"].map(tier_order)
    out = out.sort_values(["_tier_ord", "accuracy_pct"], ascending=[True, False]).drop(
        columns="_tier_ord"
    )
    cols = ["tier", "script_name"] + [c for c in out.columns if c not in ("tier", "script_name")]
    return out[cols]


def analyze(
    csv_path: str, out_csv: Optional[str] = None, scripts_out_csv: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_and_prepare(csv_path)

    out_rows = []
    for spec in CATEGORIES:
        mask = _mask_for_category(df, spec)
        sdf = df[mask]
        row = {"category": spec["category"], **_metrics_for_subset(sdf)}
        if spec["omniglot_other"]:
            row["n_scripts"] = int(sdf["script_name"].nunique()) if len(sdf) else 0
        else:
            row["n_scripts"] = 1
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    scripts_df = omniglot_top_mid_bottom_table(df)

    base, _ = os.path.splitext(csv_path)
    if out_csv is None:
        out_csv = base + "_category_summary.csv"
    if scripts_out_csv is None:
        scripts_out_csv = base + "_omniglot_top_mid_bottom.csv"

    out_df.to_csv(out_csv, index=False)
    if not scripts_df.empty:
        scripts_df.to_csv(scripts_out_csv, index=False)

    return out_df, scripts_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spatial Illusion: Accuracy/TPR/TNR for alphabet English, hand_english, and other Omniglot."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to *_spatial_illusion.csv",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Output CSV path (default: <input>_category_summary.csv)",
    )
    parser.add_argument(
        "--scripts_out_csv",
        type=str,
        default=None,
        help="Output CSV for top/mid/bottom Omniglot scripts (default: <input>_omniglot_top_mid_bottom.csv)",
    )
    args = parser.parse_args()

    out_df, scripts_df = analyze(
        args.csv_path, out_csv=args.out_csv, scripts_out_csv=args.scripts_out_csv
    )

    print("=== Category summary ===")
    print(out_df[PRINT_COLS].to_string(index=False))

    print(
        "\n=== Omniglot scripts (top / mid / bottom by accuracy; "
        "excl. hand_english, hand_digits, times_new_roman) ==="
    )
    if scripts_df.empty:
        print("(no omniglot scripts)")
    else:
        print(scripts_df[SCRIPT_PRINT_COLS].to_string(index=False))

    base, _ = os.path.splitext(args.csv_path)
    cat_path = args.out_csv or f"{base}_category_summary.csv"
    scripts_path = args.scripts_out_csv or f"{base}_omniglot_top_mid_bottom.csv"
    print(f"\nWrote: {cat_path}")
    if not scripts_df.empty:
        print(f"Wrote: {scripts_path}")


if __name__ == "__main__":
    main()
