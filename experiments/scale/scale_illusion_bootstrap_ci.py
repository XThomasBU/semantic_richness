#!/usr/bin/env python3
"""
Bootstrap confidence intervals for scale-illusion results.

Categories (matching paper plots):
  - Times New Roman
  - Handwritten English
  - Omniglot

Reads CSVs from results/scale_illusion_* (read-only); writes under
results/scale_illusion_bootstrap_ci/.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "scale_illusion_bootstrap_ci"

CATEGORIES = ("Times New Roman", "Handwritten English", "Omniglot")
OMNIGLOT_EXCLUDE = frozenset({"hand_digits", "hand_english", "times_new_roman"})
PER_SCRIPT_SUFFIX_RE = re.compile(
    r"^(scale_illusion_.+?-(?:instruct|thinking|a3b-instruct|a22b-thinking|pro))_(.+)$"
)

# Matches tab:scale_results aggregation (prompt version + filters per model).
PAPER_TABLE_MODELS: Tuple[dict, ...] = (
    {
        "model": "qwen_qwen2_5-vl-7b-instruct",
        "model_dir": "scale_illusion_qwen_qwen2_5-vl-7b-instruct",
        "prompt_name": "prompt_v3",
        "english_char_filter": False,
        "hand_char_filter": False,
        "handwritten_scripts": ("hand_english", "hand_digits"),
    },
    {
        "model": "qwen_qwen2_5-vl-32b-instruct",
        "model_dir": "scale_illusion_qwen_qwen2_5-vl-32b-instruct",
        "prompt_name": "prompt_v3",
        "english_char_filter": False,
        "hand_char_filter": False,
        "handwritten_scripts": ("hand_english", "hand_digits"),
    },
    {
        "model": "qwen_qwen3-vl-8b-instruct",
        "model_dir": "scale_illusion_qwen_qwen3-vl-8b-instruct",
        "prompt_name": "prompt_v2",
        "english_char_filter": True,
        "hand_char_filter": True,
        "handwritten_scripts": ("hand_english",),
    },
    {
        "model": "qwen_qwen3-vl-30b-a3b-instruct",
        "model_dir": "scale_illusion_qwen_qwen3-vl-30b-a3b-instruct",
        "prompt_name": "prompt_v2",
        "english_char_filter": False,
        "hand_char_filter": False,
        "handwritten_scripts": ("hand_english",),
    },
    {
        "model": "gpt_5_2",
        "model_dir": "scale_illusion_prompt_v2",
        "prompt_name": "prompt_v2",
        "english_char_filter": False,
        "hand_char_filter": False,
        "handwritten_scripts": ("hand_english",),
    },
    {
        "model": "gemini_2_5_pro",
        "model_dir": "scale_illusion_gemini_2_5_pro_prompt_v2",
        "prompt_name": "prompt_v2",
        "english_char_filter": True,
        "hand_char_filter": True,
        "handwritten_scripts": ("hand_english",),
    },
)


def _char_id_to_int(char_id: str) -> int | None:
    if not isinstance(char_id, str) or "character" not in char_id.lower():
        return None
    try:
        return int(char_id.lower().replace("character", "").strip())
    except ValueError:
        return None


def _drop_gibberish_response_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "response" not in df.columns:
        return df

    def is_gibberish(val: object) -> bool:
        if pd.isna(val):
            return False
        t = str(val)
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


def _filter_character_ranges(
    df: pd.DataFrame,
    *,
    english_char_filter: bool,
    hand_char_filter: bool,
) -> pd.DataFrame:
    if "char_id" not in df.columns or not (english_char_filter or hand_char_filter):
        return df

    def keep_row(row: pd.Series) -> bool:
        script = str(row.get("script_name", ""))
        n = _char_id_to_int(row.get("char_id"))

        if english_char_filter and script == "English":
            return n is None or 1 <= n <= 26
        if hand_char_filter and script == "hand_english":
            return n is None or 0 <= n <= 25
        return True

    return df.loc[df.apply(keep_row, axis=1)].copy()


def load_scale_illusion_csv(
    path: Path,
    *,
    drop_gibberish: bool = True,
    english_char_filter: bool = False,
    hand_char_filter: bool = False,
    lowercase_only: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"dataset", "script_name", "is_positive", "scale_factor", "is_correct"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")

    for col in ["is_positive", "is_correct"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )

    df["scale_factor"] = pd.to_numeric(df["scale_factor"], errors="coerce")
    df = df.dropna(subset=["scale_factor"])
    if drop_gibberish:
        df = _drop_gibberish_response_rows(df)
    if lowercase_only and not (english_char_filter or hand_char_filter):
        english_char_filter = True
        hand_char_filter = True
    df = _filter_character_ranges(
        df,
        english_char_filter=english_char_filter,
        hand_char_filter=hand_char_filter,
    )
    return df


def assign_category(
    df: pd.DataFrame,
    handwritten_scripts: Sequence[str] = ("hand_english",),
) -> pd.DataFrame:
    hand_scripts = frozenset(handwritten_scripts)
    out = df.copy()
    out["category"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out.loc[(out["dataset"] == "alphabet") & (out["script_name"] == "English"), "category"] = (
        "Times New Roman"
    )
    out.loc[(out["dataset"] == "omniglot") & (out["script_name"].isin(hand_scripts)), "category"] = (
        "Handwritten English"
    )
    out.loc[
        (out["dataset"] == "omniglot") & (~out["script_name"].isin(OMNIGLOT_EXCLUDE)),
        "category",
    ] = "Omniglot"
    return out.dropna(subset=["category"]).copy()


def accuracy(df: pd.DataFrame) -> float:
    return float(df["is_correct"].astype(float).mean()) if len(df) else np.nan


def recall(df: pd.DataFrame) -> float:
    pos = df[df["is_positive"] == True]
    return float(pos["is_correct"].astype(float).mean()) if len(pos) else np.nan


def specificity(df: pd.DataFrame) -> float:
    neg = df[df["is_positive"] == False]
    return float(neg["is_correct"].astype(float).mean()) if len(neg) else np.nan


METRICS: dict[str, Callable[[pd.DataFrame], float]] = {
    "accuracy": accuracy,
    "recall": recall,
    "specificity": specificity,
}


def bootstrap_ci(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    n_boot: int,
    rng: np.random.Generator,
    alpha: float,
) -> Tuple[float, float, float, int]:
    n = len(df)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    point = metric_fn(df)
    if n == 1:
        return point, point, point, n

    boot_stats = np.empty(n_boot, dtype=float)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_stats[_] = metric_fn(df.iloc[idx])

    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return float(point), lo, hi, n


def summarize_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    n_boot: int,
    alpha: float,
    seed: int,
) -> List[dict]:
    rng = np.random.default_rng(seed)
    rows: List[dict] = []

    if not group_cols:
        groups = [("all", df)]
    else:
        groups = [(name, gdf) for name, gdf in df.groupby(list(group_cols), dropna=False)]

    for group_name, gdf in groups:
        if isinstance(group_name, tuple):
            group_label = "|".join(str(x) for x in group_name)
            group_dict = dict(zip(group_cols, group_name))
        else:
            group_label = str(group_name)
            group_dict = {}

        for metric_name, metric_fn in METRICS.items():
            point, lo, hi, n = bootstrap_ci(gdf, metric_fn, n_boot, rng, alpha)
            rows.append(
                {
                    **group_dict,
                    "group": group_label,
                    "metric": metric_name,
                    "n_trials": n,
                    "point_estimate": point,
                    "ci_low": lo,
                    "ci_high": hi,
                    "point_pct": point * 100 if np.isfinite(point) else np.nan,
                    "ci_low_pct": lo * 100 if np.isfinite(lo) else np.nan,
                    "ci_high_pct": hi * 100 if np.isfinite(hi) else np.nan,
                }
            )
    return rows


def build_summaries(
    df: pd.DataFrame,
    n_boot: int,
    alpha: float,
    seed: int,
    by_scale: bool,
    by_script: bool,
) -> pd.DataFrame:
    specs: List[Tuple[str, Sequence[str]]] = [
        ("by_category", ("category",)),
    ]
    if by_script and "script_name" in df.columns:
        specs.append(("by_script", ("script_name",)))
    if by_scale and "scale_factor" in df.columns:
        specs.append(("by_category_scale", ("category", "scale_factor")))
        specs.append(("by_scale", ("scale_factor",)))

    all_rows: List[dict] = []
    for breakdown, cols in specs:
        for row in summarize_group(df, cols, n_boot=n_boot, alpha=alpha, seed=seed):
            row["breakdown"] = breakdown
            all_rows.append(row)
    if not all_rows:
        return pd.DataFrame(
            columns=["category", "group", "metric", "n_trials", "point_estimate",
                     "ci_low", "ci_high", "point_pct", "ci_low_pct", "ci_high_pct", "breakdown"]
        )
    return pd.DataFrame(all_rows)


def _valid_csv(path: Path) -> bool:
    name = path.name.lower()
    return (
        path.suffix == ".csv"
        and "scale_illusion" in name
        and "pacs" not in name
        and "metrics" not in name
    )


def find_csv(model_dir: Path, prompt_name: Optional[str] = None) -> Optional[Path]:
    if prompt_name:
        if model_dir.name.endswith(f"_{prompt_name}") or model_dir.name.endswith(prompt_name):
            for path in sorted(model_dir.glob("*scale_illusion.csv")):
                if _valid_csv(path):
                    return path

        for sub in (f"scale_illusion_{prompt_name}", prompt_name):
            prompt_dir = model_dir / sub
            if prompt_dir.is_dir():
                for path in sorted(prompt_dir.glob("*scale_illusion.csv")):
                    if _valid_csv(path):
                        return path

        candidates = sorted(
            (
                p
                for p in model_dir.rglob("*scale_illusion.csv")
                if _valid_csv(p) and prompt_name in str(p)
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    candidates = sorted(
        model_dir.rglob("*scale_illusion.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if _valid_csv(path):
            return path
    return None


def find_latest_csv(model_dir: Path) -> Optional[Path]:
    return find_csv(model_dir, prompt_name=None)


def is_main_model_dir(path: Path) -> bool:
    name = path.name
    if "copy" in name.lower() or "pacs" in name.lower():
        return False
    if not name.startswith("scale_illusion_"):
        return False
    if PER_SCRIPT_SUFFIX_RE.match(name):
        return False
    return True


def model_label_from_dir(path: Path) -> str:
    return path.name.removeprefix("scale_illusion_")


def discover_models(
    results_root: Path, prompt_name: Optional[str] = None
) -> List[Tuple[str, Path]]:
    models: List[Tuple[str, Path]] = []
    for path in sorted(results_root.iterdir()):
        if not path.is_dir() or not is_main_model_dir(path):
            continue
        if prompt_name and not (
            path.name.endswith(f"_{prompt_name}")
            or path.name.endswith(prompt_name)
            or (path / f"scale_illusion_{prompt_name}").is_dir()
            or (path / prompt_name).is_dir()
        ):
            continue
        csv_path = find_csv(path, prompt_name=prompt_name)
        if csv_path is None:
            continue
        models.append((model_label_from_dir(path), csv_path))
    return models


def paper_table_model_specs(results_root: Path) -> List[Tuple[str, Path, dict]]:
    specs: List[Tuple[str, Path, dict]] = []
    for entry in PAPER_TABLE_MODELS:
        model_dir = results_root / entry["model_dir"]
        csv_path = find_csv(model_dir, prompt_name=entry["prompt_name"])
        if csv_path is None:
            raise FileNotFoundError(
                f"No {entry['prompt_name']} CSV for paper-table model {entry['model']} under {model_dir}"
            )
        specs.append((entry["model"], csv_path, entry))
    return specs


def run_bootstrap_for_csv(
    model: str,
    csv_path: Path,
    output_dir: Path,
    tag: Optional[str],
    n_boot: int,
    alpha: float,
    seed: int,
    by_scale: bool,
    by_script: bool,
    prompt_name: Optional[str] = None,
    english_char_filter: bool = False,
    hand_char_filter: bool = False,
    handwritten_scripts: Sequence[str] = ("hand_english",),
    lowercase_only: bool = False,
    drop_gibberish: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    raw = load_scale_illusion_csv(
        csv_path,
        drop_gibberish=drop_gibberish,
        english_char_filter=english_char_filter,
        hand_char_filter=hand_char_filter,
        lowercase_only=lowercase_only,
    )
    df = assign_category(raw, handwritten_scripts=handwritten_scripts)

    if verbose:
        print(f"  CSV: {csv_path}")
        print(f"  Loaded {len(df)} trials across {df['category'].nunique() if len(df) else 0} categories")

    if df.empty:
        if verbose:
            print("  Warning: no trials after filtering; skipping metrics.")
        return pd.DataFrame()

    summary = build_summaries(
        df,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
        by_scale=by_scale,
        by_script=by_script,
    )
    summary.insert(0, "model", model)
    summary["csv_path"] = str(csv_path)
    summary["prompt_name"] = prompt_name

    tag = tag or model.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{tag}_bootstrap_ci.csv"
    summary.to_csv(out_path, index=False)

    if verbose:
        print(f"  Wrote {out_path}")
        print_category_table(summary, model=model)
    return summary


def print_category_table(summary: pd.DataFrame, model: Optional[str] = None) -> None:
    rows = summary[
        (summary["breakdown"] == "by_category")
        & (summary["category"].isin(CATEGORIES))
    ].sort_values(["category", "metric"])
    header = f"\n{model} — category bootstrap CIs (95%):" if model else "\nCategory bootstrap CIs (95%):"
    print(header)
    for category in CATEGORIES:
        cat_rows = rows[rows["category"] == category]
        if cat_rows.empty:
            print(f"  {category}: (no data)")
            continue
        print(f"  {category}:")
        for _, r in cat_rows.iterrows():
            print(
                f"    {r['metric']:12s}: {r['point_pct']:6.2f}%  "
                f"[{r['ci_low_pct']:6.2f}%, {r['ci_high_pct']:6.2f}%]  (n={int(r['n_trials'])})"
            )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap CIs for scale-illusion results (Times New Roman, Handwritten English, Omniglot)."
    )
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--csv_path", type=Path, default=None, help="Single scale_illusion CSV.")
    parser.add_argument("--model", type=str, default=None, help="Model label when using --csv_path.")
    parser.add_argument(
        "--all_models",
        action="store_true",
        help="Run bootstrap for every main scale_illusion_* directory under results_root.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Explicit model directory names (e.g. scale_illusion_qwen_qwen2_5-vl-7b-instruct).",
    )
    parser.add_argument(
        "--prompt_name",
        type=str,
        default=None,
        help="Prompt folder/tag to use (e.g. prompt_v2 -> scale_illusion_prompt_v2/).",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--by_scale", action="store_true")
    parser.add_argument("--by_script", action="store_true")
    parser.add_argument(
        "--paper_table",
        action="store_true",
        help="Use the six tab:scale_results models with per-model prompt/filter settings.",
    )
    parser.add_argument(
        "--all_characters",
        action="store_true",
        help="Legacy alias: restrict English/hand_english character ranges only.",
    )
    args = parser.parse_args(argv)

    lowercase_only = not args.all_characters
    all_summaries: List[pd.DataFrame] = []

    def _run_one(
        model: str,
        csv_path: Path,
        *,
        prompt_name: Optional[str],
        english_char_filter: bool,
        hand_char_filter: bool,
        handwritten_scripts: Sequence[str],
        use_lowercase_filter: bool = False,
    ) -> None:
        print(f"\n=== {model} ===")
        try:
            summary = run_bootstrap_for_csv(
                model=model,
                csv_path=csv_path,
                output_dir=args.output_dir,
                tag=args.tag,
                n_boot=args.n_boot,
                alpha=args.alpha,
                seed=args.seed,
                by_scale=args.by_scale,
                by_script=args.by_script,
                prompt_name=prompt_name,
                english_char_filter=english_char_filter,
                hand_char_filter=hand_char_filter,
                handwritten_scripts=handwritten_scripts,
                lowercase_only=use_lowercase_filter,
                verbose=True,
            )
            if not summary.empty:
                all_summaries.append(summary)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  Skipping {model}: {exc}")

    if args.csv_path is not None:
        model = args.model or args.csv_path.parent.parent.name.removeprefix("scale_illusion_")
        _run_one(
            model,
            args.csv_path,
            prompt_name=args.prompt_name,
            english_char_filter=lowercase_only,
            hand_char_filter=lowercase_only,
            handwritten_scripts=("hand_english",),
            use_lowercase_filter=lowercase_only,
        )
    elif args.paper_table:
        specs = paper_table_model_specs(args.results_root)
        print(f"Paper table models: {', '.join(m for m, _, _ in specs)}")
        for model, csv_path, entry in specs:
            _run_one(
                model,
                csv_path,
                prompt_name=entry["prompt_name"],
                english_char_filter=entry["english_char_filter"],
                hand_char_filter=entry["hand_char_filter"],
                handwritten_scripts=entry["handwritten_scripts"],
            )
    else:
        if args.models is not None:
            model_specs: List[Tuple[str, Path]] = []
            for name in args.models:
                model_dir = args.results_root / name
                if not model_dir.is_dir():
                    model_dir = args.results_root / f"scale_illusion_{name}"
                csv_path = find_csv(model_dir, prompt_name=args.prompt_name)
                if csv_path is None:
                    print(f"Skipping {name}: no scale_illusion.csv found")
                    continue
                model_specs.append((model_label_from_dir(model_dir), csv_path))
        elif args.all_models:
            model_specs = discover_models(args.results_root, prompt_name=args.prompt_name)
            prompt_label = f" ({args.prompt_name})" if args.prompt_name else ""
            print(
                f"Discovered {len(model_specs)} models{prompt_label}: "
                f"{', '.join(m for m, _ in model_specs)}"
            )
        else:
            raise SystemExit("Provide --csv_path, --paper_table, --all_models, or --models.")

        for model, csv_path in model_specs:
            _run_one(
                model,
                csv_path,
                prompt_name=args.prompt_name,
                english_char_filter=lowercase_only,
                hand_char_filter=lowercase_only,
                handwritten_scripts=("hand_english",),
                use_lowercase_filter=lowercase_only,
            )

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined_path = args.output_dir / "all_models_bootstrap_ci.csv"
        combined.to_csv(combined_path, index=False)

        category_summary = combined[
            (combined["breakdown"] == "by_category")
            & (combined["category"].isin(CATEGORIES))
        ].sort_values(["model", "category", "metric"])
        category_path = args.output_dir / "all_models_category_bootstrap_ci.csv"
        category_summary.to_csv(category_path, index=False)
        print("\nWrote combined tables:")
        print(f"  {combined_path}")
        print(f"  {category_path}")


if __name__ == "__main__":
    main()
