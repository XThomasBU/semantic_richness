#!/usr/bin/env python3
"""
Utility script to export example original + scaled (with padding) image pairs
for the scale task, for:
  - Alphabet (Times New Roman)
  - Omniglot
  - PACS

Each pair is saved as two separate image files:
  * one for the original full-size version (same canvas size)
  * one for the scaled version (character/object smaller with white padding)

Default output layout:
  <output_dir>/
    alphabet/
      alphabet_<char_id>_orig_full_<k>.png
      alphabet_<char_id>_scaled_<scale>_<k>.png
    omniglot/
      omniglot_<script>_<char_id>_orig_full_<k>.png
      omniglot_<script>_<char_id>_scaled_<scale>_<k>.png
    pacs/
      pacs_<domain>_<object>_orig_full_<k>.png
      pacs_<domain>_<object>_scaled_<scale>_<k>.png
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, Sequence

from experiments.scale.scale_illusion import (
    load_alphabet_images,
    load_omniglot_images,
    resize_with_padding,
)
from experiments.scale.scale_illusion_pacs import load_pacs_images


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _pick_scales(defaults: Sequence[float]) -> Sequence[float]:
    return defaults if defaults else (0.3, 0.5, 0.9)


def save_alphabet_examples(
    alphabet_dir: Path,
    output_dir: Path,
    num_examples: int = 8,
    image_size: int = 336,
    scale_factors: Iterable[float] = (0.3, 0.5, 0.9),
    rng: random.Random | None = None,
) -> None:
    rng = rng or random
    images = load_alphabet_images(str(alphabet_dir))
    if not images:
        raise ValueError(f"No alphabet images found under {alphabet_dir}")

    out_dir = output_dir / "alphabet"
    _ensure_dir(out_dir)
    scale_factors = tuple(_pick_scales(scale_factors))

    for k in range(num_examples):
        char_id, img_path_first, img_path_second = rng.choice(images)
        scale = float(rng.choice(scale_factors))

        orig_path = out_dir / f"alphabet_{char_id}_orig_full_{k}.png"
        scaled_path = out_dir / f"alphabet_{char_id}_scaled_{scale:.2f}_{k}.png"

        # Original: full canvas (resize to image_size if needed, no scaling down)
        resize_with_padding(
            img_path_first,
            scale_factor=1.0,
            output_path=str(orig_path),
            image_size=image_size,
        )
        # Scaled: same character smaller with white padding
        resize_with_padding(
            img_path_second,
            scale_factor=scale,
            output_path=str(scaled_path),
            image_size=image_size,
        )


def save_omniglot_examples(
    omniglot_dir: Path,
    output_dir: Path,
    num_examples: int = 8,
    image_size: int = 336,
    scale_factors: Iterable[float] = (0.3, 0.5, 0.9),
    max_scripts: int | None = None,
    rng: random.Random | None = None,
) -> None:
    rng = rng or random
    omniglot_images = load_omniglot_images(str(omniglot_dir))
    if not omniglot_images:
        raise ValueError(f"No Omniglot images found under {omniglot_dir}")

    if max_scripts is not None and max_scripts > 0:
        by_script: dict[str, list] = {}
        for script_name, char_id, img_first, img_second in omniglot_images:
            by_script.setdefault(script_name, []).append(
                (script_name, char_id, img_first, img_second)
            )
        scripts = sorted(by_script.keys())
        chosen = set(rng.sample(scripts, min(len(scripts), max_scripts)))
        omniglot_images = [
            (s, c, p1, p2) for (s, c, p1, p2) in omniglot_images if s in chosen
        ]

    out_dir = output_dir / "omniglot"
    _ensure_dir(out_dir)
    scale_factors = tuple(_pick_scales(scale_factors))

    for k in range(num_examples):
        script_name, char_id, img_path_first, img_path_second = rng.choice(omniglot_images)
        scale = float(rng.choice(scale_factors))

        safe_script = script_name.replace(" ", "_").replace("/", "_")
        orig_path = out_dir / f"omniglot_{safe_script}_{char_id}_orig_full_{k}.png"
        scaled_path = out_dir / f"omniglot_{safe_script}_{char_id}_scaled_{scale:.2f}_{k}.png"

        # Use the same image for orig and scaled so they show the same character instance.
        resize_with_padding(
            img_path_first,
            scale_factor=1.0,
            output_path=str(orig_path),
            image_size=image_size,
        )
        resize_with_padding(
            img_path_first,
            scale_factor=scale,
            output_path=str(scaled_path),
            image_size=image_size,
        )


def save_pacs_examples(
    pacs_dir: Path,
    output_dir: Path,
    num_examples: int = 8,
    image_size: int = 336,
    scale_factors: Iterable[float] = (0.3, 0.5, 0.9),
    max_images_per_subcategory: int | None = None,
    rng: random.Random | None = None,
) -> None:
    rng = rng or random
    pacs_images = load_pacs_images(
        str(pacs_dir), max_images_per_subcategory=max_images_per_subcategory
    )
    if not pacs_images:
        raise ValueError(f"No PACS images found under {pacs_dir}")

    out_dir = output_dir / "pacs"
    _ensure_dir(out_dir)
    scale_factors = tuple(_pick_scales(scale_factors))

    for k in range(num_examples):
        domain, obj, img_path = rng.choice(pacs_images)
        scale = float(rng.choice(scale_factors))

        safe_domain = str(domain).replace(" ", "_").replace("/", "_")
        safe_obj = str(obj).replace(" ", "_").replace("/", "_")
        orig_path = out_dir / f"pacs_{safe_domain}_{safe_obj}_orig_full_{k}.png"
        scaled_path = out_dir / f"pacs_{safe_domain}_{safe_obj}_scaled_{scale:.2f}_{k}.png"

        resize_with_padding(
            img_path,
            scale_factor=1.0,
            output_path=str(orig_path),
            image_size=image_size,
        )
        resize_with_padding(
            img_path,
            scale_factor=scale,
            output_path=str(scaled_path),
            image_size=image_size,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save example original + scaled (with padding) image pairs for "
            "Alphabet, Omniglot, and PACS (scale task)."
        )
    )
    parser.add_argument(
        "--alphabet_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Base directory containing 'times_new_roman'.",
    )
    parser.add_argument(
        "--omniglot_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Base directory containing 'omniglot/omniglot-master/python/images_all'.",
    )
    parser.add_argument(
        "--pacs_dir",
        type=Path,
        default=None,
        help="Directory containing PACS subset folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "example_scale_images",
        help="Root directory to write example images into.",
    )
    parser.add_argument(
        "--num_examples_alphabet",
        type=int,
        default=8,
        help="Number of Alphabet pairs to export.",
    )
    parser.add_argument(
        "--num_examples_omniglot",
        type=int,
        default=8,
        help="Number of Omniglot pairs to export.",
    )
    parser.add_argument(
        "--num_examples_pacs",
        type=int,
        default=8,
        help="Number of PACS pairs to export.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=336,
        help="Target image size (canvas size for scale task).",
    )
    parser.add_argument(
        "--scale_factors",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.9],
        help="Scale factors to sample from (0–1; 1 = full size).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max_omniglot_scripts",
        type=int,
        default=None,
        help="Optional limit on number of Omniglot scripts to sample from.",
    )
    parser.add_argument(
        "--max_pacs_per_subcategory",
        type=int,
        default=None,
        help="Optional max PACS images per subcategory when sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    output_dir: Path = args.output_dir
    _ensure_dir(output_dir)

    save_alphabet_examples(
        alphabet_dir=args.alphabet_dir,
        output_dir=output_dir,
        num_examples=args.num_examples_alphabet,
        image_size=args.image_size,
        scale_factors=args.scale_factors,
        rng=rng,
    )

    save_omniglot_examples(
        omniglot_dir=args.omniglot_dir,
        output_dir=output_dir,
        num_examples=args.num_examples_omniglot,
        image_size=args.image_size,
        scale_factors=args.scale_factors,
        max_scripts=args.max_omniglot_scripts,
        rng=rng,
    )

    save_pacs_examples(
        pacs_dir=args.pacs_dir,
        output_dir=output_dir,
        num_examples=args.num_examples_pacs,
        image_size=args.image_size,
        scale_factors=args.scale_factors,
        max_images_per_subcategory=args.max_pacs_per_subcategory,
        rng=rng,
    )


if __name__ == "__main__":
    main()
