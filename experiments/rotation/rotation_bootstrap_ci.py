#!/usr/bin/env python3
"""
Bootstrap confidence intervals for rotation-recognition results.

Categories (matching paper plots):
  - PACS
  - Omniglot
  - Handwritten English
  - Times New Roman

Reads CSVs (read-only); writes under experiments/rotation/rotation_bootstrap_ci/results/.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_PAC_ROOT = Path(__file__).resolve().parents[2] / "DATA"
DEFAULT_OMNIGLOT_ROOT = Path(__file__).resolve().parents[2] / "DATA" / "omniglot-master"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "rotation_bootstrap_ci" / "results"

PAC_PROMPT_CANDIDATES = ("prompt_rotate_modified", "prompt_rotate")
OMNIGLOT_PROMPT_CANDIDATES = ("prompt_rotate", "prompt_rotate_modified")

PACS_DOMAINS = ("art_painting", "cartoon", "photo", "sketch")
YES_NO = frozenset({"yes", "no"})
CATEGORIES = ("PACS", "Omniglot", "Handwritten English", "Times New Roman")
PAPER_TABLE_CATEGORIES = ("Times New Roman", "Handwritten English", "Omniglot")
OMNIGLOT_EXCLUDE = frozenset({"hand_digits", "hand_english", "times_new_roman", "English"})
ANGLES_10_TO_90 = list(range(10, 91, 10))

# Matches tab:rotation_results (omniglot only; angle filter per model).
PAPER_TABLE_MODELS: Tuple[dict, ...] = (
    {
        "model": "qwen_2.5_7B",
        "datasets": frozenset({"omniglot"}),
        "omniglot_prompt": "prompt_rotate",
        "angles": ANGLES_10_TO_90,
    },
    {
        "model": "qwen_2.5_32B",
        "datasets": frozenset({"omniglot"}),
        "omniglot_prompt": "prompt_rotate",
        "angles": None,
    },
    {
        "model": "qwen_3_8B",
        "datasets": frozenset({"omniglot"}),
        "omniglot_prompt": "prompt_rotate",
        "angles": ANGLES_10_TO_90,
    },
    {
        "model": "qwen_3_30B-A3B",
        "datasets": frozenset({"omniglot"}),
        "omniglot_prompt": "prompt_rotate",
        "angles": ANGLES_10_TO_90,
    },
    {
        "model": "gpt_5.2_",
        "datasets": frozenset({"omniglot"}),
        "omniglot_prompt": "prompt_rotate",
        "angles": None,
    },
    {
        "model": "gemini_2.5_",
        "datasets": frozenset({"omniglot"}),
        "omniglot_prompt": "prompt_rotate",
        "angles": None,
    },
)


def parse_yes_no(raw) -> Optional[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    text = str(raw).strip()
    if not text:
        return None
    m = re.search(r"\{([^}]*)\}", text, flags=re.IGNORECASE)
    if m:
        text = m.group(1)
    text = text.strip("[]'\" ").lower()
    if text in YES_NO:
        return text
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return None


def add_parsed_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ground_truth"] = out["ground_truth"].astype(str).str.strip().str.lower()
    out["parsed_response"] = out["response"].map(parse_yes_no)
    out["correct"] = (out["parsed_response"] == out["ground_truth"]).astype(float)
    return out


def omniglot_category(script: str) -> Optional[str]:
    if script in {"times_new_roman", "English"}:
        return "Times New Roman"
    if script == "hand_english":
        return "Handwritten English"
    if script in OMNIGLOT_EXCLUDE:
        return None
    return "Omniglot"


def load_csv_paths(data_dir: Path) -> List[Path]:
    if data_dir.is_file() and data_dir.suffix == ".csv":
        return [data_dir]
    csv_dir = data_dir / "parallel_csvs" if (data_dir / "parallel_csvs").is_dir() else data_dir
    paths = sorted(csv_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files in {csv_dir}")
    return paths


def load_pacs(data_dir: Path) -> pd.DataFrame:
    frames = []
    for path in load_csv_paths(data_dir):
        df = pd.read_csv(path)
        if "domain" not in df.columns:
            for domain in PACS_DOMAINS:
                if path.stem.endswith(f"_{domain}"):
                    df = df.copy()
                    df["domain"] = domain
                    break
        df["source_file"] = path.name
        df["category"] = "PACS"
        frames.append(df)
    return add_parsed_columns(pd.concat(frames, ignore_index=True))


def load_omniglot(data_dir: Path) -> pd.DataFrame:
    all_csv = sorted(data_dir.glob("*_all.csv"))
    paths = [all_csv[0]] if all_csv else load_csv_paths(data_dir)

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if "script" not in df.columns:
            raise ValueError(f"Expected 'script' column in omniglot CSV: {path}")
        df = df.copy()
        df["source_file"] = path.name
        df["category"] = df["script"].map(omniglot_category)
        df = df[df["category"].notna()].copy()
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    if "correct" not in out.columns:
        out = add_parsed_columns(out)
    else:
        out["parsed_response"] = out["response"].map(parse_yes_no)
        out["correct"] = (out["parsed_response"] == out["ground_truth"]).astype(float)
    return out


def accuracy(df: pd.DataFrame) -> float:
    return float(df["correct"].mean()) if len(df) else np.nan


def recall(df: pd.DataFrame) -> float:
    pos = df[df["ground_truth"] == "yes"]
    return float((pos["parsed_response"] == "yes").mean()) if len(pos) else np.nan


def specificity(df: pd.DataFrame) -> float:
    neg = df[df["ground_truth"] == "no"]
    return float((neg["parsed_response"] == "no").mean()) if len(neg) else np.nan


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
    by_angle: bool,
    by_domain: bool,
    by_script: bool,
) -> pd.DataFrame:
    specs: List[Tuple[str, Sequence[str]]] = [
        ("by_category", ("category",)),
    ]
    if by_domain and "domain" in df.columns:
        specs.append(("by_category_domain", ("category", "domain")))
    if by_script and "script" in df.columns:
        specs.append(("by_script", ("script",)))
    if by_angle and "angle" in df.columns:
        specs.append(("by_category_angle", ("category", "angle")))
        specs.append(("by_angle", ("angle",)))

    all_rows: List[dict] = []
    for breakdown, cols in specs:
        for row in summarize_group(df, cols, n_boot=n_boot, alpha=alpha, seed=seed):
            row["breakdown"] = breakdown
            all_rows.append(row)
    return pd.DataFrame(all_rows)


def resolve_data_dir(
    root: Path, model_dir: str, prompt_name: str, data_dir: Optional[Path]
) -> Path:
    if data_dir is not None:
        return data_dir
    return root / f"{model_dir}_responses" / prompt_name


def prompt_dir_has_data(prompt_dir: Path) -> bool:
    if not prompt_dir.is_dir():
        return False
    if list(prompt_dir.glob("*_all.csv")):
        return True
    parallel = prompt_dir / "parallel_csvs"
    return parallel.is_dir() and bool(list(parallel.glob("*.csv")))


def resolve_prompt(root: Path, model_dir: str, candidates: Sequence[str]) -> Optional[str]:
    for prompt in candidates:
        if prompt_dir_has_data(root / f"{model_dir}_responses" / prompt):
            return prompt
    return None


def discover_models(
    pac_root: Path,
    omniglot_root: Path,
    datasets: set[str],
) -> List[str]:
    models: set[str] = set()
    if "pacs" in datasets:
        for path in pac_root.glob("*_responses"):
            model = path.name.replace("_responses", "")
            if resolve_prompt(pac_root, model, PAC_PROMPT_CANDIDATES):
                models.add(model)
    if "omniglot" in datasets:
        for path in omniglot_root.glob("*_responses"):
            model = path.name.replace("_responses", "")
            if resolve_prompt(omniglot_root, model, OMNIGLOT_PROMPT_CANDIDATES):
                models.add(model)
    return sorted(models)


def load_model_data(
    model_dir: str,
    datasets: set[str],
    pac_root: Path,
    omniglot_root: Path,
    pac_prompt: Optional[str],
    omni_prompt: Optional[str],
    pac_data_dir: Optional[Path],
    omniglot_data_dir: Optional[Path],
    angles: Optional[Sequence[int]],
) -> Tuple[pd.DataFrame, Optional[str], Optional[str], List[str]]:
    resolved_pac_prompt = pac_prompt
    resolved_omni_prompt = omni_prompt
    frames = []
    loaded = []

    if "pacs" in datasets:
        if resolved_pac_prompt is None:
            resolved_pac_prompt = resolve_prompt(pac_root, model_dir, PAC_PROMPT_CANDIDATES)
        if resolved_pac_prompt is not None:
            pac_dir = resolve_data_dir(
                pac_root, model_dir, resolved_pac_prompt, pac_data_dir
            )
            if pac_dir.exists():
                frames.append(load_pacs(pac_dir))
                loaded.append("pacs")

    if "omniglot" in datasets:
        if resolved_omni_prompt is None:
            resolved_omni_prompt = resolve_prompt(
                omniglot_root, model_dir, OMNIGLOT_PROMPT_CANDIDATES
            )
        if resolved_omni_prompt is not None:
            omni_dir = resolve_data_dir(
                omniglot_root, model_dir, resolved_omni_prompt, omniglot_data_dir
            )
            if omni_dir.exists():
                frames.append(load_omniglot(omni_dir))
                loaded.append("omniglot")

    if not frames:
        raise FileNotFoundError(f"No rotation results found for model {model_dir}")

    df = pd.concat(frames, ignore_index=True)
    if angles is not None and "angle" in df.columns:
        df = df[df["angle"].isin(angles)].copy()
    return df, resolved_pac_prompt, resolved_omni_prompt, loaded


def run_bootstrap_for_model(
    model_dir: str,
    datasets: set[str],
    pac_root: Path,
    omniglot_root: Path,
    output_dir: Path,
    pac_prompt: Optional[str],
    omni_prompt: Optional[str],
    pac_data_dir: Optional[Path],
    omniglot_data_dir: Optional[Path],
    tag: Optional[str],
    n_boot: int,
    alpha: float,
    seed: int,
    by_angle: bool,
    by_domain: bool,
    by_script: bool,
    angles: Optional[Sequence[int]],
    verbose: bool = True,
) -> pd.DataFrame:
    df, pac_prompt_used, omni_prompt_used, loaded = load_model_data(
        model_dir=model_dir,
        datasets=datasets,
        pac_root=pac_root,
        omniglot_root=omniglot_root,
        pac_prompt=pac_prompt,
        omni_prompt=omni_prompt,
        pac_data_dir=pac_data_dir,
        omniglot_data_dir=omniglot_data_dir,
        angles=angles,
    )

    if verbose:
        print(f"  Loaded datasets: {', '.join(loaded)}")
        if pac_prompt_used:
            print(f"  PACS prompt: {pac_prompt_used}")
        if omni_prompt_used:
            print(f"  Omniglot prompt: {omni_prompt_used}")
        print(f"  Loaded {len(df)} trials")

    n_unparsed = int(df["parsed_response"].isna().sum())
    if n_unparsed and verbose:
        print(f"  Warning: {n_unparsed} unparseable responses (counted incorrect).")

    summary = build_summaries(
        df,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
        by_angle=by_angle,
        by_domain=by_domain,
        by_script=by_script,
    )
    summary.insert(0, "model", model_dir)
    summary["pac_prompt"] = pac_prompt_used
    summary["omniglot_prompt"] = omni_prompt_used

    tag = tag or f"{model_dir}_{pac_prompt_used or 'no_pacs'}_{omni_prompt_used or 'no_omni'}"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{tag}_bootstrap_ci.csv"
    summary.to_csv(out_path, index=False)

    if verbose:
        print(f"  Wrote {out_path}")
        print_category_table(summary, model=model_dir)
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
        description="Bootstrap CIs for rotation results (PACS + Omniglot categories)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["pacs", "omniglot", "all"],
        default=["all"],
        help="Which datasets to load (default: all).",
    )
    parser.add_argument("--pac_data_dir", type=Path, default=None)
    parser.add_argument("--omniglot_data_dir", type=Path, default=None)
    parser.add_argument("--pac_root", type=Path, default=DEFAULT_PAC_ROOT)
    parser.add_argument("--omniglot_root", type=Path, default=DEFAULT_OMNIGLOT_ROOT)
    parser.add_argument("--model_dir", type=str, default="qwen_3_8B")
    parser.add_argument(
        "--all_models",
        action="store_true",
        help="Run bootstrap for every model with available rotation CSVs.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Explicit model list (overrides --model_dir / --all_models discovery).",
    )
    parser.add_argument(
        "--pac_prompt_name",
        type=str,
        default=None,
        help="PACS prompt folder (auto-detected per model if omitted).",
    )
    parser.add_argument(
        "--omniglot_prompt_name",
        type=str,
        default=None,
        help="Omniglot prompt folder (auto-detected per model if omitted).",
    )
    # Back-compat alias
    parser.add_argument(
        "--prompt_name",
        type=str,
        default=None,
        help="Sets both pac and omniglot prompt names if the specific flags are omitted.",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--by_angle", action="store_true")
    parser.add_argument("--by_domain", action="store_true")
    parser.add_argument("--by_script", action="store_true")
    parser.add_argument("--angles", type=int, nargs="*", default=None)
    parser.add_argument(
        "--paper_table",
        action="store_true",
        help="Use tab:rotation_results models (omniglot only, per-model angle filters).",
    )
    args = parser.parse_args(argv)

    if args.prompt_name is not None:
        pac_prompt = args.prompt_name
        omni_prompt = args.prompt_name
    else:
        pac_prompt = args.pac_prompt_name
        omni_prompt = args.omniglot_prompt_name

    datasets = set(args.datasets)
    if "all" in datasets:
        datasets = {"pacs", "omniglot"}

    all_summaries = []

    if args.paper_table:
        print(
            "Paper table models: "
            + ", ".join(entry["model"] for entry in PAPER_TABLE_MODELS)
        )
        for entry in PAPER_TABLE_MODELS:
            model_dir = entry["model"]
            print(f"\n=== {model_dir} ===")
            try:
                summary = run_bootstrap_for_model(
                    model_dir=model_dir,
                    datasets=set(entry["datasets"]),
                    pac_root=args.pac_root,
                    omniglot_root=args.omniglot_root,
                    output_dir=args.output_dir,
                    pac_prompt=None,
                    omni_prompt=entry.get("omniglot_prompt"),
                    pac_data_dir=args.pac_data_dir,
                    omniglot_data_dir=args.omniglot_data_dir,
                    tag=args.tag,
                    n_boot=args.n_boot,
                    alpha=args.alpha,
                    seed=args.seed,
                    by_angle=args.by_angle,
                    by_domain=args.by_domain,
                    by_script=args.by_script,
                    angles=entry.get("angles"),
                    verbose=True,
                )
                all_summaries.append(summary)
            except FileNotFoundError as exc:
                print(f"  Skipping {model_dir}: {exc}")
    else:
        if args.models is not None:
            models = args.models
        elif args.all_models:
            models = discover_models(args.pac_root, args.omniglot_root, datasets)
            print(f"Discovered {len(models)} models: {', '.join(models)}")
        else:
            models = [args.model_dir]

        for model_dir in models:
            print(f"\n=== {model_dir} ===")
            try:
                summary = run_bootstrap_for_model(
                    model_dir=model_dir,
                    datasets=datasets,
                    pac_root=args.pac_root,
                    omniglot_root=args.omniglot_root,
                    output_dir=args.output_dir,
                    pac_prompt=pac_prompt,
                    omni_prompt=omni_prompt,
                    pac_data_dir=args.pac_data_dir,
                    omniglot_data_dir=args.omniglot_data_dir,
                    tag=args.tag,
                    n_boot=args.n_boot,
                    alpha=args.alpha,
                    seed=args.seed,
                    by_angle=args.by_angle,
                    by_domain=args.by_domain,
                    by_script=args.by_script,
                    angles=args.angles,
                    verbose=True,
                )
                all_summaries.append(summary)
            except FileNotFoundError as exc:
                print(f"  Skipping {model_dir}: {exc}")

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined_path = args.output_dir / "all_models_bootstrap_ci.csv"
        combined.to_csv(combined_path, index=False)

        paper_categories = PAPER_TABLE_CATEGORIES if args.paper_table else CATEGORIES
        category_summary = combined[
            (combined["breakdown"] == "by_category")
            & (combined["category"].isin(paper_categories))
        ].sort_values(["model", "category", "metric"])
        category_path = args.output_dir / "all_models_category_bootstrap_ci.csv"
        category_summary.to_csv(category_path, index=False)
        print(f"\nWrote combined tables:")
        print(f"  {combined_path}")
        print(f"  {category_path}")


if __name__ == "__main__":
    main()
