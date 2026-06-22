#!/usr/bin/env python3
"""Precompute and save frozen vision features for Omniglot rotation views."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_RESEARCH_ROOT = Path(__file__).resolve().parents[2]
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

from experiments.linear_probe.data import (  # noqa: E402
    PROBE_ANGLES,
    default_images_all_root,
    load_characters,
    load_images_all,
    load_probe_group_characters,
    render_rotated_views,
    rotated_cache_path,
    assign_train_test_keys,
)
from experiments.linear_probe.encoder_config import (  # noqa: E402
    resolve_encoder_args,
    resolve_model_name,
)
from experiments.linear_probe.features import (  # noqa: E402
    dataset_cache_tag,
    feature_cache_path,
    features_for_rotated_views,
    load_frozen_encoder,
    save_feature_cache,
)


def parse_args():
    p = argparse.ArgumentParser(description="Precompute frozen Omniglot features")
    p.add_argument(
        "--omniglot_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Repo root containing omniglot/omniglot-master/...",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="images_all",
        choices=["images_all", "evaluation", "probe_groups"],
        help="probe_groups: TNR (standalone) + hand_english + omniglot scripts; "
        "images_all: all scripts under images_all; evaluation: Angelic only",
    )
    p.add_argument(
        "--images_root",
        type=str,
        default=None,
        help="Override path to images_all (default: .../python/images_all)",
    )
    p.add_argument(
        "--script",
        type=str,
        default="Angelic",
        help="Script name when --dataset evaluation",
    )
    p.add_argument(
        "--scripts",
        type=str,
        nargs="*",
        default=None,
        help="Optional filter for images_all script folder names",
    )
    p.add_argument(
        "--encoder",
        type=str,
        default="clip",
        choices=["clip", "dino", "diffusion", "dit", "qwen", "siglip"],
    )
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument(
        "--features_dir",
        type=str,
        default="./results/linear_probe/features",
    )
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument(
        "--eval_split",
        type=str,
        default=None,
        choices=["character_holdout", "first_stroke_test"],
        help="character_holdout: 20%% of characters in test only. "
        "first_stroke_test: all characters in train and test; test uses first PNG per "
        "character (paper-style, use with --all_strokes). Default: first_stroke_test "
        "if --all_strokes else character_holdout.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_size", type=int, default=336)
    p.add_argument(
        "--angles",
        type=float,
        nargs="+",
        default=None,
        help=f"Rotation angles in degrees (default: {PROBE_ANGLES}; 0° always included).",
    )
    p.add_argument("--feature_batch_size", type=int, default=32)
    p.add_argument("--image_cache_dir", type=str, default=None)
    p.add_argument("--skip_render", action="store_true")
    p.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Re-render even if cached PNG exists",
    )
    p.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Limit number of source PNGs (debug)",
    )
    p.add_argument(
        "--all_strokes",
        action="store_true",
        help="Use every PNG per character folder (~46k strokes). Default: first PNG "
        "only (~1.7k). Writes to {encoder}_{dataset}_all_strokes_features.pt and "
        "rotated_images/{dataset}_all_strokes/.",
    )
    p.add_argument(
        "--canonical_only",
        action="store_true",
        help="Only 0° (no rotations). ~1.7k features with default loader",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Replace existing feature cache"
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.eval_split is None:
        args.eval_split = (
            "first_stroke_test" if args.all_strokes else "character_holdout"
        )
    if args.eval_split == "first_stroke_test" and not args.all_strokes:
        raise ValueError(
            "eval_split=first_stroke_test requires --all_strokes "
            "(otherwise each character has only one PNG)."
        )
    if args.canonical_only:
        angles = [0.0]
    else:
        angles = args.angles if args.angles is not None else list(PROBE_ANGLES)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = resolve_model_name(args.encoder, args.model_name)
    encoder_args = resolve_encoder_args(args.encoder, args.model_name)

    base_tag = (
        "probe_groups"
        if args.dataset == "probe_groups"
        else "images_all" if args.dataset == "images_all" else args.script
    )
    dataset_tag = dataset_cache_tag(base_tag, args.all_strokes)
    out_path = (
        Path(args.output)
        if args.output
        else feature_cache_path(
            args.features_dir, args.encoder, base_tag, all_strokes=args.all_strokes
        )
    )
    if out_path.exists() and not args.overwrite:
        print(f"Feature cache already exists: {out_path}")
        print("Pass --overwrite to recompute.")
        return

    images_root = (
        Path(args.images_root)
        if args.images_root
        else default_images_all_root(args.omniglot_dir)
    )

    print(f"Loading dataset={args.dataset} ...")
    if args.dataset == "probe_groups":
        images = load_probe_group_characters(
            args.omniglot_dir,
            images_all_root=images_root,
            first_stroke_only=not args.all_strokes,
        )
        from collections import Counter

        stroke_mode = (
            "all strokes" if args.all_strokes else "first stroke per character"
        )
        group_counts = Counter(c.dataset_group for c in images)
        n_chars = len({i.key for i in images})
        print(f"  {len(images)} source images ({stroke_mode}), {n_chars} characters")
        print(f"  by group: {dict(group_counts)}")
        print(f"  TNR source: {Path(args.omniglot_dir) / 'times_new_roman'}")
        print(f"  Omniglot scripts: {images_root}")
    elif args.dataset == "images_all":
        if not images_root.exists():
            raise FileNotFoundError(f"images_all not found: {images_root}")
        images = load_images_all(
            images_root,
            scripts=args.scripts,
            max_images=args.max_images,
            first_stroke_only=not args.all_strokes,
        )
        n_scripts = len({i.script_name for i in images})
        n_chars = len({i.key for i in images})
        stroke_mode = (
            "all strokes" if args.all_strokes else "first stroke per character"
        )
        print(f"  {len(images)} source images ({stroke_mode})")
        print(f"  {n_scripts} scripts, {n_chars} characters")
        print(f"  source: {images_root}")
    else:
        images = load_characters(args.omniglot_dir, script_name=args.script)
        print(f"  {len(images)} characters ({args.script})")

    train_keys, test_keys = assign_train_test_keys(
        images,
        eval_split=args.eval_split,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    if args.eval_split == "first_stroke_test":
        n_chars = len(train_keys)
        print(
            f"  eval_split=first_stroke_test: all {n_chars} characters; "
            "test = first PNG per character, train = other strokes"
        )
    else:
        print(f"  split: {len(train_keys)} train chars, {len(test_keys)} test chars")

    image_cache = args.image_cache_dir
    if image_cache is None:
        if args.dataset == "evaluation":
            cache_name = args.script
        else:
            cache_name = dataset_tag
        image_cache = str(
            Path(args.features_dir).parent / "rotated_images" / cache_name
        )
    if image_cache:
        os.makedirs(image_cache, exist_ok=True)

    if args.skip_render:
        if not image_cache:
            raise ValueError("--skip_render requires --image_cache_dir")
        all_angles = sorted(set([0.0, *angles]))
        cache_root = Path(image_cache)
        path_by_key = {}
        for img in images:
            for angle in all_angles:
                p = rotated_cache_path(cache_root, img, angle)
                if not p.exists():
                    raise FileNotFoundError(f"Missing rendered image: {p}")
                path_by_key[(*img.image_key, float(angle))] = str(p)
        print(f"  loaded {len(path_by_key)} cached views from {image_cache}")
    else:
        n_angles = len(set([0.0, *angles]))
        n_views = len(images) * n_angles
        print(
            f"Rendering {len(images)} images x {n_angles} angles = {n_views} views -> {image_cache}"
        )
        path_by_key = render_rotated_views(
            images,
            angles=angles,
            image_size=args.image_size,
            cache_dir=image_cache,
            skip_existing=not args.no_skip_existing,
        )

    print(f"Loading encoder {args.encoder} ({model_name}) on {device}...")
    encoder, processor = load_frozen_encoder(
        args.encoder, model_name, encoder_args, device=device
    )
    feat_batch = 1 if "Qwen" in encoder.__class__.__name__ else args.feature_batch_size

    print("Extracting frozen features...")
    features = features_for_rotated_views(
        path_by_key,
        encoder,
        processor,
        batch_size=feat_batch,
        device=device,
    )
    del encoder
    if device == "cuda":
        torch.cuda.empty_cache()

    meta = {
        "encoder": args.encoder,
        "model_name": model_name,
        "dataset": args.dataset,
        "dataset_tag": dataset_tag,
        "eval_split": args.eval_split,
        "script": args.script if args.dataset == "evaluation" else "images_all",
        "images_root": str(images_root),
        "omniglot_dir": args.omniglot_dir,
        "image_size": args.image_size,
        "angles": angles,
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "n_images": len(images),
        "first_stroke_only": not args.all_strokes,
        "n_scripts": len({i.script_name for i in images}),
        "n_characters": len({i.key for i in images}),
        "n_views": len(features),
        "train_char_keys": [list(k) for k in sorted(train_keys)],
        "test_char_keys": [list(k) for k in sorted(test_keys)],
        "char_to_group": [
            [c.script_name, c.char_id, c.dataset_group]
            for c in sorted(
                images, key=lambda x: (x.dataset_group, x.script_name, x.char_id)
            )
        ],
        "image_cache_dir": image_cache,
    }

    save_feature_cache(out_path, features, meta)
    print("Done.")


if __name__ == "__main__":
    main()
