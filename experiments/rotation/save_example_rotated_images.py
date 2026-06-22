#!/usr/bin/env python3
"""
Utility script to export example original + rotated image pairs
for:
  - Alphabet (Times New Roman)
  - Omniglot
  - PACS

Each pair is saved as two separate image files:
  * one for the original (0°) version
  * one for the rotated version (e.g. 30°, 60°, 90°)

Default output layout:
  <output_dir>/
    alphabet/
      alphabet_<char_id>_orig_0deg_<k>.png
      alphabet_<char_id>_rot_<angle>deg_<k>.png
    omniglot/
      omniglot_<script>_<char_id>_orig_0deg_<k>.png
      omniglot_<script>_<char_id>_rot_<angle>deg_<k>.png
    pacs/
      pacs_<domain>_<object>_orig_0deg_<k>.png
      pacs_<domain>_<object>_rot_<angle>deg_<k>.png
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, Sequence

from experiments.scale.scale_illusion import load_alphabet_images, load_omniglot_images
from experiments.identity.spatial_illusion import rotate_image
from experiments.scale.scale_illusion_pacs import load_pacs_images


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _pick_angles(defaults: Sequence[float]) -> Sequence[float]:
    # Ensure we always have at least one angle
    return defaults if defaults else (30.0,)


def save_alphabet_examples(
    alphabet_dir: Path,
    output_dir: Path,
    num_examples: int = 8,
    image_size: int = 336,
    angles: Iterable[float] = (30.0, 60.0, 90.0, 180.0),
    rng: random.Random | None = None,
) -> None:
    rng = rng or random
    images = load_alphabet_images(str(alphabet_dir))
    if not images:
        raise ValueError(f"No alphabet images found under {alphabet_dir}")

    out_dir = output_dir / "alphabet"
    _ensure_dir(out_dir)
    angles = tuple(_pick_angles(angles))

    for k in range(num_examples):
        char_id, img_path_first, _ = rng.choice(images)
        angle = float(rng.choice(angles))
        src = Path(img_path_first)

        orig_path = out_dir / f"alphabet_{char_id}_orig_0deg_{k}.png"
        rot_path = out_dir / f"alphabet_{char_id}_rot_{int(angle)}deg_{k}.png"

        # Use the same pipeline for orig + rotated to guarantee same size/resolution.
        rotate_image(str(src), angle=0.0, output_path=str(orig_path), image_size=image_size)
        rotate_image(str(src), angle=angle, output_path=str(rot_path), image_size=image_size)


def save_omniglot_examples(
    omniglot_dir: Path,
    output_dir: Path,
    num_examples: int = 8,
    image_size: int = 336,
    angles: Iterable[float] = (30.0, 60.0, 90.0, 180.0),
    max_scripts: int | None = None,
    rng: random.Random | None = None,
) -> None:
    rng = rng or random
    # Reuse the helper from scale_illusion; allowed_scripts/max_scripts is handled there.
    omniglot_images = load_omniglot_images(str(omniglot_dir))
    if not omniglot_images:
        raise ValueError(f"No Omniglot images found under {omniglot_dir}")

    if max_scripts is not None and max_scripts > 0:
        # Restrict to a subset of scripts for diversity, not required but sometimes convenient.
        by_script: dict[str, list[tuple[str, str, str, str]]] = {}
        for script_name, char_id, img_first, img_second in omniglot_images:
            by_script.setdefault(script_name, []).append((script_name, char_id, img_first, img_second))
        scripts = sorted(by_script.keys())
        chosen = set(rng.sample(scripts, min(len(scripts), max_scripts)))
        omniglot_images = [
            (s, c, p1, p2) for (s, c, p1, p2) in omniglot_images if s in chosen
        ]

    out_dir = output_dir / "omniglot"
    _ensure_dir(out_dir)
    angles = tuple(_pick_angles(angles))

    for k in range(num_examples):
        script_name, char_id, img_path_first, _ = rng.choice(omniglot_images)
        angle = float(rng.choice(angles))
        src = Path(img_path_first)

        safe_script = script_name.replace(" ", "_").replace("/", "_")
        orig_path = out_dir / f"omniglot_{safe_script}_{char_id}_orig_0deg_{k}.png"
        rot_path = out_dir / f"omniglot_{safe_script}_{char_id}_rot_{int(angle)}deg_{k}.png"

        # Use the same pipeline for orig + rotated to guarantee same size/resolution.
        rotate_image(str(src), angle=0.0, output_path=str(orig_path), image_size=image_size)
        rotate_image(str(src), angle=angle, output_path=str(rot_path), image_size=image_size)


def save_pacs_examples(
    pacs_dir: Path,
    output_dir: Path,
    num_examples: int = 8,
    image_size: int = 336,
    angles: Iterable[float] = (30.0, 60.0, 90.0, 180.0),
    max_images_per_subcategory: int | None = None,
    rng: random.Random | None = None,
) -> None:
    rng = rng or random
    pacs_images = load_pacs_images(str(pacs_dir), max_images_per_subcategory=max_images_per_subcategory)
    if not pacs_images:
        raise ValueError(f"No PACS images found under {pacs_dir}")

    out_dir = output_dir / "pacs"
    _ensure_dir(out_dir)
    angles = tuple(_pick_angles(angles))

    for k in range(num_examples):
        domain, obj, img_path = rng.choice(pacs_images)
        angle = float(rng.choice(angles))
        src = Path(img_path)

        safe_domain = str(domain).replace(" ", "_").replace("/", "_")
        safe_obj = str(obj).replace(" ", "_").replace("/", "_")
        orig_path = out_dir / f"pacs_{safe_domain}_{safe_obj}_orig_0deg_{k}.png"
        rot_path = out_dir / f"pacs_{safe_domain}_{safe_obj}_rot_{int(angle)}deg_{k}.png"

        # Use the same pipeline for orig + rotated to guarantee same size/resolution.
        rotate_image(str(src), angle=0.0, output_path=str(orig_path), image_size=image_size)
        rotate_image(str(src), angle=angle, output_path=str(rot_path), image_size=image_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save example original + rotated image pairs for Alphabet, Omniglot, and PACS.\n"
            "Run as: python -m experiments.rotation.save_example_rotated_images"
        )
    )
    parser.add_argument(
        "--alphabet_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Base directory containing 'times_new_roman' (as in scale_illusion).",
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
        help="Directory containing PACS subset folders (as in scale_illusion_pacs).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "example_rotated_images",
        help="Root directory to write example images into.",
    )
    parser.add_argument(
        "--num_examples_alphabet",
        type=int,
        default=8,
        help="Number of Alphabet (Times New Roman) pairs to export.",
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
        help="Target image size used in rotation (must match model experiments).",
    )
    parser.add_argument(
        "--angles",
        type=float,
        nargs="+",
        default=[30.0, 60.0, 90.0, 180.0],
        help="List of rotation angles (degrees) to sample from.",
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
        help="Optional limit on the number of Omniglot scripts to sample from.",
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

    # Alphabet (Times New Roman)
    save_alphabet_examples(
        alphabet_dir=args.alphabet_dir,
        output_dir=output_dir,
        num_examples=args.num_examples_alphabet,
        image_size=args.image_size,
        angles=args.angles,
        rng=rng,
    )

    # Omniglot
    save_omniglot_examples(
        omniglot_dir=args.omniglot_dir,
        output_dir=output_dir,
        num_examples=args.num_examples_omniglot,
        image_size=args.image_size,
        angles=args.angles,
        max_scripts=args.max_omniglot_scripts,
        rng=rng,
    )

    # PACS
    save_pacs_examples(
        pacs_dir=args.pacs_dir,
        output_dir=output_dir,
        num_examples=args.num_examples_pacs,
        image_size=args.image_size,
        angles=args.angles,
        max_images_per_subcategory=args.max_pacs_per_subcategory,
        rng=rng,
    )


if __name__ == "__main__":
    main()

