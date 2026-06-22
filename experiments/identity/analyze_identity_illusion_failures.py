#!/usr/bin/env python3
"""
Sample failing rows from an identity_illusion results CSV and write resolved image paths.

For each script_name, take up to N failing rows (is_correct == False), resolve paths the same way
as identity_illusion.py. For negative trials, only the primary character image path is known;
the random distractor path was not logged in the CSV (image2_path left empty).

Example:
  python -m experiments.identity.analyze_identity_illusion_failures \\
    --results_csv results/identity_illusion/qwen_qwen2_5_vl_7b_instruct_identity_illusion.csv \\
    --alphabet_dir <repo_root> \\
    --omniglot_dir <repo_root> \\
    --per_script 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def _first_png(char_dir: Path) -> Optional[Path]:
    if not char_dir.is_dir():
        return None
    pngs = sorted(char_dir.glob("*.png"))
    return pngs[0] if pngs else None


def resolve_image1_path(
    dataset: str,
    script_name: str,
    char_id: str,
    alphabet_dir: Path,
    omniglot_dir: Path,
) -> str:
    if str(dataset).strip().lower() == "alphabet":
        p = alphabet_dir / "times_new_roman" / char_id / "image.png"
        return str(p.resolve()) if p.is_file() else str(p.resolve())
    root = (
        omniglot_dir
        / "omniglot"
        / "omniglot-master"
        / "python"
        / "images_all"
        / script_name
        / char_id
    )
    first = _first_png(root)
    return str(first.resolve()) if first and first.is_file() else str(root / "<first.png>")


def main() -> None:
    p = argparse.ArgumentParser(description="Export sample failing identity_illusion rows with image paths.")
    p.add_argument(
        "--results_csv",
        type=str,
        required=True,
        help="Path to *_identity_illusion.csv",
    )
    p.add_argument(
        "--alphabet_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Base dir containing times_new_roman/",
    )
    p.add_argument(
        "--omniglot_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Base dir containing omniglot/omniglot-master/.../images_all/",
    )
    p.add_argument("--per_script", type=int, default=5, help="Max failing examples per script_name.")
    p.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Output CSV path. Default: <results_csv_stem>_failure_paths_sample.csv next to input.",
    )
    args = p.parse_args()

    results_path = Path(args.results_csv).resolve()
    if not results_path.is_file():
        raise FileNotFoundError(results_path)

    alphabet_dir = Path(args.alphabet_dir).resolve()
    omniglot_dir = Path(args.omniglot_dir).resolve()

    df = pd.read_csv(results_path)
    if "is_correct" not in df.columns:
        raise ValueError("CSV must contain is_correct column")

    # Normalize boolean
    ic = df["is_correct"]
    if ic.dtype == object:
        fail = ic.astype(str).str.lower().isin(["false", "0"])
    else:
        fail = ~ic.astype(bool) if ic.dtype == bool else (ic == False)

    bad = df[fail].copy()
    if bad.empty:
        print("No failing rows (is_correct=False) in CSV.")
        out = results_path.parent / f"{results_path.stem}_failure_paths_sample.csv"
        pd.DataFrame().to_csv(out, index=False)
        print(f"Wrote empty: {out}")
        return

    out_rows = []
    for script_name, g in bad.groupby("script_name", sort=True):
        g2 = g.head(int(args.per_script))
        for _, row in g2.iterrows():
            dataset = row["dataset"]
            char_id = row["char_id"]
            is_pos = bool(row["is_positive"]) if not pd.isna(row["is_positive"]) else False
            img1 = resolve_image1_path(str(dataset), str(script_name), str(char_id), alphabet_dir, omniglot_dir)
            if is_pos:
                img2 = img1
                image2_note = "same_as_image1"
            else:
                img2 = ""
                image2_note = "not_logged_random_distractor"

            out_rows.append(
                {
                    "script_name": script_name,
                    "dataset": dataset,
                    "char_id": char_id,
                    "is_positive": is_pos,
                    "prediction": row.get("prediction", ""),
                    "response": row.get("response", ""),
                    "image1_path": img1,
                    "image2_path": img2,
                    "image2_note": image2_note,
                }
            )

    out_df = pd.DataFrame(out_rows)
    out_path = (
        Path(args.out_csv).resolve()
        if args.out_csv
        else results_path.parent / f"{results_path.stem}_failure_paths_sample.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(out_df)} rows to {out_path}")
    print(f"     Scripts with failures: {bad['script_name'].nunique()}")


if __name__ == "__main__":
    main()
