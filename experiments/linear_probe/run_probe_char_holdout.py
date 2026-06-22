#!/usr/bin/env python3
"""
Angle linear probe with character-level holdout.

Reuses precomputed all-strokes features from results/linear_probe/features.
Train: all strokes of train characters. Test: all strokes of held-out characters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

_RESEARCH_ROOT = Path(__file__).resolve().parents[2]
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

from experiments.linear_probe.feature_store import (  # noqa: E402
    PROBE_GROUPS,
    FeatureStore,
    angle_class_map,
    build_angle_classification_tensors,
    class_counts,
    filter_keys_by_group,
    split_train_val_keys,
)
from experiments.linear_probe.probe import (  # noqa: E402
    FeatureNormalizer,
    evaluate_angle_probe,
    train_angle_probe,
)
from experiments.linear_probe.splits import (  # noqa: E402
    assign_character_holdout_keys,
)

EVAL_SPLIT = "character_holdout"
RESULT_TAG = "char_holdout"


def parse_args():
    p = argparse.ArgumentParser(
        description="Angle linear probe — character holdout (reuses existing features)"
    )
    p.add_argument(
        "--features_path",
        type=str,
        default="./results/linear_probe/features/siglip_images_all_all_strokes_features.pt",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="./results/linear_probe_char_holdout",
    )
    p.add_argument(
        "--feature_mode",
        type=str,
        default="both_modes",
        choices=["diff", "concat", "both", "both_modes"],
        help="both_modes: run diff and concat; both: [f0;f_rot;diff]",
    )
    p.add_argument(
        "--test_ratio",
        type=float,
        default=None,
        help="Fraction of characters held out for test (default: meta test_ratio or 0.2).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for character split (default: meta seed or 42).",
    )
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=128)
    return p.parse_args()


def _plot_confusion(cm: List[List[int]], class_to_angle: dict, path: str, title: str):
    labels = [str(int(class_to_angle[i])) for i in range(len(class_to_angle))]
    cm_arr = np.array(cm, dtype=float)
    row_sum = cm_arr.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_arr, row_sum, where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted angle (°)")
    ax.set_ylabel("True angle (°)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _metrics_dict(m):
    return {
        "accuracy": m.accuracy,
        "balanced_accuracy": m.balanced_accuracy,
        "macro_recall": m.macro_recall,
        "n_samples": m.n_samples,
        "per_class_accuracy": m.per_class_accuracy,
        "per_angle_recall": m.per_class_accuracy,
        "per_class_count": m.per_class_count,
        "per_angle_count": m.per_class_count,
        "confusion_matrix": m.confusion_matrix,
    }


def _sorted_angles(per_class_accuracy: dict) -> List[int]:
    return sorted(int(a) for a in per_class_accuracy)


def _print_per_angle_recall(label: str, per_class_accuracy: dict, indent: str = "  "):
    print(f"{indent}{label} per-angle recall:")
    for angle in _sorted_angles(per_class_accuracy):
        print(f"{indent}  {angle}°: {per_class_accuracy[str(angle)]:.3f}")


def _plot_by_group_per_angle(
    test_by_group: dict,
    path: str,
    title: str,
    num_classes: int,
):
    groups = [g for g in PROBE_GROUPS if g in test_by_group]
    if not groups:
        return
    angles = _sorted_angles(test_by_group[groups[0]]["per_class_accuracy"])
    colors = {
        "times_new_roman": "#4c78a8",
        "hand_english": "#f58518",
        "omniglot": "#54a24b",
    }

    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 5), sharey=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, group in zip(axes, groups):
        m = test_by_group[group]
        accs = [m["per_class_accuracy"][str(a)] for a in angles]
        ax.bar(
            angles,
            accs,
            color=colors.get(group, "steelblue"),
            alpha=0.85,
            edgecolor="black",
        )
        ax.axhline(
            m["balanced_accuracy"],
            color="green",
            linestyle="-",
            linewidth=2,
            label=f"Balanced ({m['balanced_accuracy']:.2f})",
        )
        ax.axhline(1.0 / num_classes, color="gray", linestyle="--", label="Chance")
        ax.set_xlabel("True angle (°)")
        ax.set_ylabel("Per-class recall")
        ax.set_title(group.replace("_", " "))
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_angle_by_group_combined(test_by_group: dict, path: str, title: str):
    groups = [g for g in PROBE_GROUPS if g in test_by_group]
    if not groups:
        return
    angles = _sorted_angles(test_by_group[groups[0]]["per_class_accuracy"])
    colors = {
        "times_new_roman": "#4c78a8",
        "hand_english": "#f58518",
        "omniglot": "#54a24b",
    }

    x = np.arange(len(angles))
    width = 0.8 / len(groups)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, group in enumerate(groups):
        accs = [test_by_group[group]["per_class_accuracy"][str(a)] for a in angles]
        offset = (i - (len(groups) - 1) / 2) * width
        ax.bar(
            x + offset,
            accs,
            width,
            label=group.replace("_", " "),
            color=colors.get(group, "steelblue"),
            alpha=0.85,
        )
    ax.set_xticks(x, [str(a) for a in angles])
    ax.set_xlabel("True rotation angle (degrees)")
    ax.set_ylabel("Per-class recall")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def run_single_mode(
    store: FeatureStore,
    feature_mode: str,
    fit_keys,
    val_keys,
    test_keys,
    class_to_angle: dict,
    num_classes: int,
    args,
    device: str,
    *,
    test_ratio: float,
    split_seed: int,
) -> dict:
    encoder = store.meta.get("encoder", "probe")

    print(f"\n{'=' * 60}")
    print(f"Feature mode: {feature_mode}")
    print(f"{'=' * 60}")

    cache_all_strokes = not store.meta.get("first_stroke_only", True)
    if not cache_all_strokes:
        raise ValueError(
            "Character holdout requires an all-strokes feature cache "
            "(meta first_stroke_only=false). Use *_all_strokes_features.pt."
        )

    print(
        f"  eval_split={EVAL_SPLIT}: fit/val use all strokes of train chars; "
        f"test uses all strokes of {len(test_keys)} held-out chars "
        f"(test_ratio={test_ratio}, seed={split_seed})"
    )

    stroke_kw = dict(first_stroke_only=False, exclude_first_stroke=False)

    X_fit, y_fit, _, _ = build_angle_classification_tensors(
        store, fit_keys, feature_mode=feature_mode, **stroke_kw
    )
    X_val, y_val, _, _ = build_angle_classification_tensors(
        store, val_keys, feature_mode=feature_mode, **stroke_kw
    )
    X_test, y_test, _, _ = build_angle_classification_tensors(
        store, test_keys, feature_mode=feature_mode, **stroke_kw
    )

    print(f"  samples: fit={len(y_fit)}, val={len(y_val)}, test={len(y_test)}")
    print(
        f"  fit chars={len(fit_keys)}, val chars={len(val_keys)}, test chars={len(test_keys)}"
    )
    print(f"  fit class counts: {class_counts(y_fit, class_to_angle)}")
    print(f"  input dim: {X_fit.shape[1]}")

    normalizer = FeatureNormalizer().fit(X_fit)
    X_fit = normalizer.transform(X_fit)
    X_val = normalizer.transform(X_val)
    X_test = normalizer.transform(X_test)

    print(f"Training up to {args.epochs} epochs (patience={args.patience})...")
    result = train_angle_probe(
        X_fit,
        y_fit,
        X_val,
        y_val,
        num_classes=num_classes,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        patience=args.patience,
        device=device,
    )
    probe = result.model
    print(f"  best epoch={result.best_epoch}, best val_loss={result.best_val_loss:.4f}")

    fit_m = evaluate_angle_probe(probe, X_fit, y_fit, class_to_angle, device=device)
    val_m = evaluate_angle_probe(probe, X_val, y_val, class_to_angle, device=device)
    test_m = evaluate_angle_probe(probe, X_test, y_test, class_to_angle, device=device)

    test_by_group = {}
    print("Test by dataset group:")
    for group in PROBE_GROUPS:
        gkeys = filter_keys_by_group(test_keys, store.char_to_group, group)
        if not gkeys:
            print(f"  {group}: (no test characters)")
            continue
        X_g, y_g, _, _ = build_angle_classification_tensors(
            store, gkeys, feature_mode=feature_mode, **stroke_kw
        )
        X_g = normalizer.transform(X_g)
        g_m = evaluate_angle_probe(probe, X_g, y_g, class_to_angle, device=device)
        test_by_group[group] = _metrics_dict(g_m)
        print(
            f"  {group}: n_chars={len(gkeys)} n_samples={g_m.n_samples} "
            f"acc={g_m.accuracy:.3f} balanced={g_m.balanced_accuracy:.3f}"
        )
        _print_per_angle_recall(group, g_m.per_class_accuracy, indent="    ")

    results = {
        "task": "angle_classification",
        "feature_mode": feature_mode,
        "num_classes": num_classes,
        "class_to_angle": {str(k): v for k, v in class_to_angle.items()},
        "features_path": str(args.features_path),
        "eval_split": EVAL_SPLIT,
        "cache_meta_eval_split": store.meta.get("eval_split"),
        "train_all_strokes": True,
        "test_all_strokes": True,
        "test_ratio": test_ratio,
        "split_seed": split_seed,
        "encoder": encoder,
        "epochs_max": args.epochs,
        "patience": args.patience,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "training_history": result.history,
        "n_fit_chars": len(fit_keys),
        "n_val_chars": len(val_keys),
        "n_test_chars": len(test_keys),
        "fit": _metrics_dict(fit_m),
        "val": _metrics_dict(val_m),
        "test": _metrics_dict(test_m),
        "test_by_group": test_by_group,
    }

    tag = f"{encoder}_angle_probe_{RESULT_TAG}_{feature_mode}"
    out_json = os.path.join(args.output_dir, f"{tag}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Fit  acc={fit_m.accuracy:.3f}  balanced={fit_m.balanced_accuracy:.3f}")
    print(f"Val  acc={val_m.accuracy:.3f}  balanced={val_m.balanced_accuracy:.3f}")
    print(f"Test acc={test_m.accuracy:.3f}  balanced={test_m.balanced_accuracy:.3f}")
    print("Test per-angle recall:")
    for angle_str, acc in sorted(
        test_m.per_class_accuracy.items(), key=lambda x: int(x[0])
    ):
        print(f"  {angle_str}°: {acc:.3f}")

    angle_vals = sorted(int(a) for a in test_m.per_class_accuracy)
    accs = [test_m.per_class_accuracy[str(a)] for a in angle_vals]
    plt.figure(figsize=(10, 5))
    plt.bar(angle_vals, accs, color="steelblue", alpha=0.8, edgecolor="black")
    plt.axhline(
        test_m.balanced_accuracy,
        color="green",
        linestyle="-",
        linewidth=2,
        label=f"Balanced ({test_m.balanced_accuracy:.2f})",
    )
    plt.axhline(1.0 / num_classes, color="gray", linestyle="--", label="Chance")
    plt.xlabel("True rotation angle (degrees)")
    plt.ylabel("Per-class recall")
    plt.title(f"Angle probe ({encoder}, {feature_mode}) — test ({RESULT_TAG})")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plot_path = os.path.join(args.output_dir, f"{tag}_per_class.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    cm_path = os.path.join(args.output_dir, f"{tag}_confusion.png")
    _plot_confusion(
        test_m.confusion_matrix,
        class_to_angle,
        cm_path,
        f"Test confusion — {encoder} ({feature_mode}, {RESULT_TAG})",
    )
    if test_by_group:
        groups = [g for g in PROBE_GROUPS if g in test_by_group]
        bal_accs = [test_by_group[g]["balanced_accuracy"] for g in groups]
        plt.figure(figsize=(8, 5))
        plt.bar(
            groups,
            bal_accs,
            color=["#4c78a8", "#f58518", "#54a24b"][: len(groups)],
            alpha=0.85,
        )
        plt.ylabel("Balanced accuracy")
        plt.title(f"Test by group ({encoder}, {feature_mode}, {RESULT_TAG})")
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3, axis="y")
        grp_path = os.path.join(args.output_dir, f"{tag}_by_group.png")
        plt.savefig(grp_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Group plot saved to {grp_path}")

        per_angle_path = os.path.join(args.output_dir, f"{tag}_by_group_per_angle.png")
        _plot_by_group_per_angle(
            test_by_group,
            per_angle_path,
            f"Test per-angle recall by group — {encoder} ({feature_mode}, {RESULT_TAG})",
            num_classes,
        )
        print(f"Per-group per-angle plot saved to {per_angle_path}")

        combined_path = os.path.join(args.output_dir, f"{tag}_angle_by_group.png")
        _plot_angle_by_group_combined(
            test_by_group,
            combined_path,
            f"Test per-angle recall — groups compared ({encoder}, {RESULT_TAG})",
        )
        print(f"Angle×group comparison plot saved to {combined_path}")

    print(f"Saved {out_json}")
    return results


def _plot_comparison(all_results: List[dict], output_dir: str, encoder: str):
    angles = sorted(int(a) for a in all_results[0]["test"]["per_class_accuracy"])

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(angles))
    width = 0.35
    for i, res in enumerate(all_results):
        accs = [res["test"]["per_class_accuracy"][str(a)] for a in angles]
        offset = (i - 0.5) * width
        ax.bar(x + offset, accs, width, label=res["feature_mode"], alpha=0.85)

    ax.set_xticks(x, [str(a) for a in angles])
    ax.set_xlabel("True angle (°)")
    ax.set_ylabel("Per-class recall")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title(f"diff vs concat — test per-angle recall ({encoder}, {RESULT_TAG})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{encoder}_angle_probe_{RESULT_TAG}_diff_vs_concat.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot: {path}")

    summary = {
        "encoder": encoder,
        "eval_split": EVAL_SPLIT,
        "comparison": [
            {
                "feature_mode": r["feature_mode"],
                "best_epoch": r["best_epoch"],
                "best_val_loss": r["best_val_loss"],
                "test_accuracy": r["test"]["accuracy"],
                "test_balanced_accuracy": r["test"]["balanced_accuracy"],
                "val_balanced_accuracy": r["val"]["balanced_accuracy"],
            }
            for r in all_results
        ],
    }
    summary_path = os.path.join(
        output_dir, f"{encoder}_angle_probe_{RESULT_TAG}_comparison.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Comparison summary: {summary_path}")


def _save_split_manifest(
    output_dir: str,
    encoder: str,
    *,
    train_keys,
    test_keys,
    fit_keys,
    val_keys,
    test_ratio: float,
    split_seed: int,
    features_path: str,
) -> None:
    manifest = {
        "eval_split": EVAL_SPLIT,
        "encoder": encoder,
        "features_path": features_path,
        "test_ratio": test_ratio,
        "seed": split_seed,
        "n_train_chars": len(train_keys),
        "n_test_chars": len(test_keys),
        "n_fit_chars": len(fit_keys),
        "n_val_chars": len(val_keys),
        "train_char_keys": [list(k) for k in sorted(train_keys)],
        "test_char_keys": [list(k) for k in sorted(test_keys)],
        "fit_char_keys": [list(k) for k in sorted(fit_keys)],
        "val_char_keys": [list(k) for k in sorted(val_keys)],
    }
    path = os.path.join(output_dir, f"{encoder}_{RESULT_TAG}_split.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  split manifest: {path}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    if args.feature_mode == "both_modes":
        modes = ["diff", "concat"]
    else:
        modes = [args.feature_mode]

    print(f"Loading feature cache from {args.features_path}...")
    store = FeatureStore.from_cache(args.features_path)
    if store.meta.get("first_stroke_only", True):
        raise ValueError(
            "Expected all-strokes cache (first_stroke_only=false). "
            "Point --features_path at *_all_strokes_features.pt."
        )

    stroke_mode = "all strokes per character"
    print(f"  stroke mode: {stroke_mode} (tag={store.meta.get('dataset_tag', '?')})")
    print(f"  cache meta eval_split={store.meta.get('eval_split')} (ignored for splits)")

    angles = store.probe_angles()
    num_classes = len(angles)
    _, class_to_angle = angle_class_map(angles)

    test_ratio = (
        args.test_ratio
        if args.test_ratio is not None
        else float(store.meta.get("test_ratio", 0.2))
    )
    split_seed = args.seed if args.seed is not None else int(store.meta.get("seed", 42))

    train_keys, test_keys = assign_character_holdout_keys(
        store, test_ratio=test_ratio, seed=split_seed
    )
    overlap = train_keys & test_keys
    if overlap:
        raise RuntimeError(f"Train/test character overlap: {len(overlap)} keys")

    fit_keys, val_keys = split_train_val_keys(
        train_keys, val_ratio=args.val_ratio, seed=split_seed
    )
    if fit_keys & test_keys or val_keys & test_keys:
        raise RuntimeError("Val/fit characters must be disjoint from test")

    encoder = store.meta.get("encoder", "probe")
    print(f"  encoder={encoder}")
    print(f"  eval_split={EVAL_SPLIT} (runtime)")
    print(f"  modes={modes}, epochs={args.epochs}, patience={args.patience}")
    print(
        f"  chars: train={len(train_keys)}, test={len(test_keys)} "
        f"(fit={len(fit_keys)}, val={len(val_keys)})"
    )

    _save_split_manifest(
        args.output_dir,
        encoder,
        train_keys=train_keys,
        test_keys=test_keys,
        fit_keys=fit_keys,
        val_keys=val_keys,
        test_ratio=test_ratio,
        split_seed=split_seed,
        features_path=str(args.features_path),
    )

    all_results = []
    for mode in modes:
        all_results.append(
            run_single_mode(
                store,
                mode,
                fit_keys,
                val_keys,
                test_keys,
                class_to_angle,
                num_classes,
                args,
                device,
                test_ratio=test_ratio,
                split_seed=split_seed,
            )
        )

    if len(all_results) > 1:
        _plot_comparison(all_results, args.output_dir, encoder)

    best = max(all_results, key=lambda r: r["test"]["balanced_accuracy"])
    print(
        f"\nBest test balanced accuracy: {best['test']['balanced_accuracy']:.3f} "
        f"({best['feature_mode']})"
    )


if __name__ == "__main__":
    main()
