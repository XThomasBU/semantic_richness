"""
Summarize Spatial Illusion results by script.

Reads a CSV produced by `spatial_illusion_idefics.py` (or equivalent),
then computes for each `script_name`:
- Accuracy
- TPR (True Positive Rate) = mean(prediction == 'yes' over is_positive == True)
- TNR (True Negative Rate) = mean(prediction == 'no' over is_positive == False)
"""

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd


def _safe_pct(numer: float, denom: float) -> float:
    if denom <= 0:
        return float("nan")
    return float(numer / denom * 100.0)


def analyze(csv_path: str, out_csv: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "script_name" not in df.columns:
        # Fallback: at least split alphabet vs omniglot.
        df = df.copy()
        df["script_name"] = df.get("dataset", "unknown")

    required_cols = ["is_positive", "prediction", "is_correct"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}. Have columns: {list(df.columns)}")

    # Normalize prediction strings for robustness.
    pred = df["prediction"].astype(str).str.strip().str.lower()
    df = df.copy()
    df["prediction_norm"] = pred

    out_rows = []
    for script_name, sdf in df.groupby("script_name", dropna=False):
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
        accuracy_frac = f"{correct_n}/{n_total}" if n_total > 0 else "nan"
        tpr_frac = f"{tpr_numer}/{n_pos}" if n_pos > 0 else "nan"
        tnr_frac = f"{tnr_numer}/{n_neg}" if n_neg > 0 else "nan"

        # Angle coverage (include all angles present in the CSV for this script).
        if "angle" in sdf.columns:
            angle_series = pd.to_numeric(sdf["angle"], errors="coerce").dropna()
            angle_min = float(angle_series.min()) if len(angle_series) else float("nan")
            angle_max = float(angle_series.max()) if len(angle_series) else float("nan")
            n_unique_angles = int(angle_series.nunique()) if len(angle_series) else 0
        else:
            angle_min = float("nan")
            angle_max = float("nan")
            n_unique_angles = 0

        out_rows.append(
            {
                "script_name": script_name,
                "n_total": n_total,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "accuracy_frac": accuracy_frac,
                "accuracy_pct": accuracy_pct,
                "tpr_pct": tpr_pct,
                "tpr_frac": tpr_frac,
                "tnr_pct": tnr_pct,
                "tnr_frac": tnr_frac,
                "angle_min": angle_min,
                "angle_max": angle_max,
                "n_unique_angles": n_unique_angles,
                "n_correct": int(sdf["is_correct"].sum()) if "is_correct" in sdf.columns else np.nan,
                "n_pred_yes_on_positive": int((sdf_pos["prediction_norm"] == "yes").sum()),
                "n_pred_no_on_negative": int((sdf_neg["prediction_norm"] == "no").sum()),
            }
        )

    out_df = pd.DataFrame(out_rows)

    # A nice ordering: alphabet first, then the rest by accuracy desc.
    out_df = out_df.sort_values(
        by=["script_name"],
        key=lambda col: col.map(lambda x: (0 if str(x).lower() == "english" else 1, str(x))),
    )

    if out_csv is None:
        base, _ = os.path.splitext(csv_path)
        out_csv = base + "_by_script_summary.csv"

    out_df.to_csv(out_csv, index=False)
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial Illusion: Accuracy/TPR/TNR per script.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to *_spatial_illusion.csv")
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Output CSV path (default: <input>_by_script_summary.csv)",
    )
    args = parser.parse_args()

    out_df = analyze(args.csv_path, out_csv=args.out_csv)
    # Print a compact view to help quick inspection.
    cols = [
        "script_name",
        "n_total",
        "accuracy_frac",
        "accuracy_pct",
        "tpr_frac",
        "tpr_pct",
        "tnr_frac",
        "tnr_pct",
        "n_unique_angles",
        "angle_min",
        "angle_max",
    ]
    print(out_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()

