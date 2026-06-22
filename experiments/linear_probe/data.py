"""Omniglot loading, rotation, pair generation, and train/test splits."""

from __future__ import annotations

import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

# Repo root on path for cross-package imports
_RESEARCH_ROOT = Path(__file__).resolve().parents[2]
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

from experiments.linear_probe.probe_types import (  # noqa: E402
    PROBE_GROUPS,
    CharacterKey,
    FeatureKey,
    ImageKey,
    MatchPair,
    OmniglotCharacter,
)
from experiments.linear_probe.image_rotate import rotate_image  # noqa: E402

# Linear-probe rotations: 10°–90° in 10° steps (0° added at render time)
PROBE_ANGLES = [float(a) for a in range(10, 100, 10)]
ROTATION_ANGLES = PROBE_ANGLES  # alias for compatibility


def default_images_all_root(omniglot_dir: str) -> Path:
    return Path(omniglot_dir) / "omniglot" / "omniglot-master" / "python" / "images_all"


def load_characters(
    omniglot_dir: str,
    script_name: str = "Angelic",
) -> List[OmniglotCharacter]:
    """Load Omniglot characters (Angelic evaluation set, one image per character)."""
    eval_path = (
        Path(omniglot_dir)
        / "omniglot"
        / "omniglot-master"
        / "python"
        / "images_evaluation"
        / script_name
    )
    if not eval_path.exists():
        raise FileNotFoundError(f"Omniglot evaluation path not found: {eval_path}")

    characters: List[OmniglotCharacter] = []
    for char_dir in sorted(eval_path.glob("character*")):
        pngs = sorted(char_dir.glob("*.png"))
        if pngs:
            characters.append(
                OmniglotCharacter(
                    script_name=script_name,
                    char_id=char_dir.name,
                    image_path=str(pngs[0]),
                    image_stem=pngs[0].stem,
                )
            )
    if not characters:
        raise FileNotFoundError(
            f"No Omniglot characters found for script '{script_name}' under {omniglot_dir}"
        )
    return characters


def script_to_probe_group(script_name: str) -> str:
    if script_name == "times_new_roman":
        return "times_new_roman"
    if script_name == "hand_english":
        return "hand_english"
    return "omniglot"


def load_times_new_roman_standalone(repo_root: str | Path) -> List[OmniglotCharacter]:
    """Latin alphabet from VLM_BLIND/times_new_roman/character*/image.png."""
    tnr = Path(repo_root) / "times_new_roman"
    if not tnr.exists():
        raise FileNotFoundError(f"times_new_roman not found: {tnr}")
    images: List[OmniglotCharacter] = []
    for char_dir in sorted(tnr.glob("character*")):
        img = char_dir / "image.png"
        if not img.exists():
            continue
        images.append(
            OmniglotCharacter(
                script_name="times_new_roman",
                char_id=char_dir.name,
                image_path=str(img),
                image_stem="image",
                dataset_group="times_new_roman",
            )
        )
    if not images:
        raise FileNotFoundError(f"No characters under {tnr}")
    return images


def load_probe_group_characters(
    repo_root: str | Path,
    images_all_root: str | Path | None = None,
    first_stroke_only: bool = True,
) -> List[OmniglotCharacter]:
    """
    Characters for three-way probe eval:
      - times_new_roman: standalone Latin alphabet (repo/times_new_roman)
      - hand_english: Omniglot script hand_english
      - omniglot: all other images_all scripts (excl. TNR, hand_english, hand_digits)
    """
    repo_root = Path(repo_root)
    if images_all_root is None:
        images_all_root = default_images_all_root(repo_root)
    images_all_root = Path(images_all_root)

    images: List[OmniglotCharacter] = []
    images.extend(load_times_new_roman_standalone(repo_root))
    images.extend(
        load_images_all(
            images_all_root,
            scripts=["hand_english"],
            dataset_group="hand_english",
            first_stroke_only=first_stroke_only,
        )
    )
    images.extend(
        load_images_all(
            images_all_root,
            exclude_scripts={"times_new_roman", "hand_english", "hand_digits"},
            dataset_group="omniglot",
            first_stroke_only=first_stroke_only,
        )
    )
    return images


def load_images_all(
    images_root: str | Path,
    scripts: Optional[Sequence[str]] = None,
    exclude_scripts: Optional[Set[str]] = None,
    max_images: Optional[int] = None,
    first_stroke_only: bool = True,
    dataset_group: Optional[str] = None,
) -> List[OmniglotCharacter]:
    """
    Load PNGs from omniglot .../python/images_all.

    By default (first_stroke_only=True): one row per character folder, using the
    first PNG (sorted by filename). That is ~1,685 characters across 52 scripts,
    not all ~46k stroke images.

    Set first_stroke_only=False to load every PNG (multiple strokes per character).
    """
    root = Path(images_root)
    if not root.exists():
        raise FileNotFoundError(f"images_all not found: {root}")

    allowed = {s.lower() for s in scripts} if scripts else None
    excluded = exclude_scripts or set()
    images: List[OmniglotCharacter] = []

    script_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    for script_dir in script_dirs:
        script_name = script_dir.name
        if script_name in excluded:
            continue
        if allowed is not None and script_name.lower() not in allowed:
            continue
        group = dataset_group or script_to_probe_group(script_name)
        for char_dir in sorted(script_dir.glob("character*")):
            pngs = sorted(char_dir.glob("*.png"))
            if not pngs:
                continue
            stroke_paths = pngs[:1] if first_stroke_only else pngs
            for png in stroke_paths:
                images.append(
                    OmniglotCharacter(
                        script_name=script_name,
                        char_id=char_dir.name,
                        image_path=str(png),
                        image_stem=png.stem,
                        dataset_group=group,
                    )
                )
                if max_images is not None and len(images) >= max_images:
                    return images

    if not images:
        raise FileNotFoundError(f"No PNG images found under {root}")
    return images


def unique_characters(images: Sequence[OmniglotCharacter]) -> List[OmniglotCharacter]:
    """One entry per (script, char_id), using the first stroke image."""
    seen: Dict[CharacterKey, OmniglotCharacter] = {}
    for img in images:
        if img.key not in seen:
            seen[img.key] = img
    return list(seen.values())


def split_characters(
    characters: Sequence[OmniglotCharacter],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[OmniglotCharacter], List[OmniglotCharacter]]:
    """Character-level train/test split (no character appears in both splits)."""
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")
    rng = random.Random(seed)
    # Deduplicate to character keys, then assign all strokes to the same split
    char_pool = unique_characters(characters)
    rng.shuffle(char_pool)
    n_test = max(1, int(len(char_pool) * test_ratio))
    if len(char_pool) - n_test < 2:
        raise ValueError(
            f"Need at least 2 train characters; got {len(char_pool)} total "
            f"with test_ratio={test_ratio}"
        )
    test_keys = {c.key for c in char_pool[:n_test]}
    train_keys = {c.key for c in char_pool[n_test:]}
    train_chars = [c for c in characters if c.key in train_keys]
    test_chars = [c for c in characters if c.key in test_keys]
    return train_chars, test_chars


def split_character_keys(
    characters: Sequence[OmniglotCharacter],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[set[CharacterKey], set[CharacterKey]]:
    train, test = split_characters(characters, test_ratio=test_ratio, seed=seed)
    return {c.key for c in unique_characters(train)}, {c.key for c in unique_characters(test)}


def all_character_keys(characters: Sequence[OmniglotCharacter]) -> set[CharacterKey]:
    return {c.key for c in unique_characters(characters)}


def assign_train_test_keys(
    characters: Sequence[OmniglotCharacter],
    *,
    eval_split: str = "character_holdout",
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[set[CharacterKey], set[CharacterKey]]:
    """
    character_holdout: disjoint train/test character sets (default).
    first_stroke_test: every character in both splits; probe uses non-first strokes
    for train and first stroke for test (no character holdout).
    """
    if eval_split == "first_stroke_test":
        keys = all_character_keys(characters)
        return keys, keys
    if eval_split == "character_holdout":
        return split_character_keys(characters, test_ratio=test_ratio, seed=seed)
    raise ValueError(
        f"Unknown eval_split={eval_split!r}; use character_holdout or first_stroke_test"
    )


def _pick_negative(
    anchor: OmniglotCharacter,
    pool: Sequence[OmniglotCharacter],
    rng: random.Random,
) -> OmniglotCharacter:
    """Hard negative: different character from the same script."""
    candidates = [c for c in pool if c.key != anchor.key and c.script_name == anchor.script_name]
    if not candidates:
        candidates = [c for c in pool if c.key != anchor.key]
    return rng.choice(candidates)


def build_match_pairs(
    train_chars: Sequence[OmniglotCharacter],
    test_chars: Sequence[OmniglotCharacter],
    angles: Sequence[float],
    pairs_per_char_per_angle: int = 1,
    positive_ratio: float = 0.5,
    seed: int = 42,
) -> List[MatchPair]:
    """
    Build same/different rotation-matching pairs.

    Positive: char at 0° vs same char rotated by angle.
    Negative: char at 0° vs different char (same script) rotated by angle.
    """
    rng = random.Random(seed)
    pairs: List[MatchPair] = []

    def _add_split(split_name: str, char_pool: Sequence[OmniglotCharacter]) -> None:
        if len(char_pool) < 2:
            return
        for _ in range(pairs_per_char_per_angle):
            for anchor in char_pool:
                for angle in angles:
                    if angle == 0:
                        continue
                    is_positive = rng.random() < positive_ratio
                    if is_positive:
                        other = anchor
                        label = 1
                    else:
                        other = _pick_negative(anchor, char_pool, rng)
                        label = 0
                    pairs.append(
                        MatchPair(
                            char_a=anchor.key,
                            char_b=other.key,
                            angle=float(angle),
                            label=label,
                            split=split_name,
                        )
                    )

    _add_split("train", train_chars)
    _add_split("test", test_chars)
    rng.shuffle(pairs)
    return pairs


def _angle_slug(angle: float) -> str:
    return str(int(angle)) if float(angle).is_integer() else str(angle).replace(".", "p")


def rotated_cache_path(
    cache_dir: Path,
    img: OmniglotCharacter,
    angle: float,
) -> Path:
    """Nested cache path: {script}/{char_id}/{image_stem}_a{angle}.png"""
    return (
        cache_dir
        / img.script_name
        / img.char_id
        / f"{img.image_stem}_a{_angle_slug(angle)}.png"
    )


def render_rotated_views(
    characters: Sequence[OmniglotCharacter],
    angles: Sequence[float],
    image_size: int = 336,
    cache_dir: str | None = None,
    skip_existing: bool = True,
) -> Dict[FeatureKey, str]:
    """
    Render rotated PNGs for each (image, angle).

    Returns mapping (script, char_id, image_stem, angle) -> path on disk.
    """
    all_angles = sorted(set([0.0, *angles]))
    out_dir = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="linear_probe_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[FeatureKey, str] = {}
    n_skip = 0

    for img in tqdm(characters, desc="Rendering rotations"):
        for angle in all_angles:
            out_path = rotated_cache_path(out_dir, img, angle)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            key: FeatureKey = (*img.image_key, float(angle))
            if skip_existing and out_path.exists():
                paths[key] = str(out_path)
                n_skip += 1
                continue
            rotate_image(img.image_path, angle, str(out_path), image_size=image_size)
            paths[key] = str(out_path)

    if n_skip:
        print(f"  skipped {n_skip} existing renders")
    return paths


def characters_for_keys(
    all_characters: Sequence[OmniglotCharacter],
) -> Dict[CharacterKey, OmniglotCharacter]:
    return {c.key: c for c in all_characters}
