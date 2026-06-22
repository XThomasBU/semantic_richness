"""Load precomputed features + metadata; build probe pairs and label tensors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import torch

from experiments.linear_probe.feature_index import build_index_map, load_index_map
from experiments.linear_probe.probe_types import (  # noqa: F401
    PROBE_GROUPS,
    CharacterKey,
    FeatureKey,
    ImageKey,
    OmniglotCharacter,
)

def _infer_probe_group(script_name: str) -> str:
    if script_name == "times_new_roman":
        return "times_new_roman"
    if script_name == "hand_english":
        return "hand_english"
    return "omniglot"



def probe_input_features(
    f0: torch.Tensor,
    f_rot: torch.Tensor,
    mode: str = "diff",
) -> torch.Tensor:
    """
    Build probe input from canonical and rotated embeddings.

    diff (default): f_rot - f0 — rotation signal, more balanced across angles
    concat: [f0; f_rot]
    both: [f0; f_rot; f_rot - f0]
    """
    if mode == "diff":
        return f_rot - f0
    if mode == "concat":
        return torch.cat([f0, f_rot], dim=-1)
    if mode == "both":
        return torch.cat([f0, f_rot, f_rot - f0], dim=-1)
    raise ValueError(f"Unknown feature_mode: {mode}")


def angle_class_map(angles: Sequence[float]) -> Tuple[Dict[float, int], Dict[int, float]]:
    """Map angle (10..90) <-> class index 0..9."""
    sorted_angles = sorted(float(a) for a in angles)
    angle_to_class = {a: i for i, a in enumerate(sorted_angles)}
    class_to_angle = {i: a for a, i in angle_to_class.items()}
    return angle_to_class, class_to_angle


@dataclass
class FeatureStore:
    """Stacked feature matrix aligned with index_map entries."""

    matrix: torch.Tensor  # [N, D]
    rows: List[dict]
    meta: dict
    key_to_index: Dict[FeatureKey, int]
    char_to_stem: Dict[CharacterKey, str]
    char_to_group: Dict[CharacterKey, str]

    @classmethod
    def from_cache(cls, features_path: str | Path) -> "FeatureStore":
        path = Path(features_path)
        pt_path = path if path.suffix == ".pt" else path.with_suffix(".pt")

        payload = torch.load(pt_path, map_location="cpu", weights_only=True)
        matrix = payload["features"].float()
        key_strings = payload["keys"]

        meta_path = pt_path.with_name(f"{pt_path.stem}_meta.json")
        index_path = pt_path.with_name(f"{pt_path.stem}_index_map.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}
        if index_path.exists():
            index_data = load_index_map(index_path)
            rows = index_data["entries"]
        else:
            rows = build_index_map(key_strings)

        key_to_index: Dict[FeatureKey, int] = {}
        char_to_stem: Dict[CharacterKey, str] = {}
        char_to_group: Dict[CharacterKey, str] = {}
        for row in rows:
            k: FeatureKey = (
                row["script"],
                row["char_id"],
                row["image_stem"],
                float(row["angle"]),
            )
            key_to_index[k] = int(row["index"])
            ck: CharacterKey = (row["script"], row["char_id"])
            char_to_stem.setdefault(ck, row["image_stem"])

        if "char_to_group" in meta:
            for script, char_id, group in meta["char_to_group"]:
                char_to_group[(script, char_id)] = group
        else:
            for ck in char_to_stem:
                char_to_group[ck] = _infer_probe_group(ck[0])

        return cls(
            matrix=matrix,
            rows=rows,
            meta=meta,
            key_to_index=key_to_index,
            char_to_stem=char_to_stem,
            char_to_group=char_to_group,
        )

    @property
    def feature_dim(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def n_views(self) -> int:
        return int(self.matrix.shape[0])

    def train_character_keys(self) -> Set[CharacterKey]:
        keys = self.meta.get("train_char_keys")
        if not keys:
            raise ValueError("meta.json missing train_char_keys; re-run precompute_features.py")
        return {tuple(k) for k in keys}

    def test_character_keys(self) -> Set[CharacterKey]:
        keys = self.meta.get("test_char_keys")
        if not keys:
            raise ValueError("meta.json missing test_char_keys; re-run precompute_features.py")
        return {tuple(k) for k in keys}

    def probe_angles(self) -> List[float]:
        """Non-zero rotation angles stored in meta (0° used as canonical reference)."""
        return [float(a) for a in self.meta.get("angles", [])]

    def vector(self, char_key: CharacterKey, angle: float, image_stem: str = "") -> torch.Tensor:
        stem = image_stem or self.char_to_stem[char_key]
        fk: FeatureKey = (char_key[0], char_key[1], stem, float(angle))
        if fk not in self.key_to_index:
            raise KeyError(f"No feature for {fk}")
        return self.matrix[self.key_to_index[fk]]

    def characters_in_split(self, split: str) -> List[OmniglotCharacter]:
        keys = self.train_character_keys() if split == "train" else self.test_character_keys()
        return [
            OmniglotCharacter(
                script_name=s,
                char_id=c,
                image_path="",
                image_stem=self.char_to_stem[(s, c)],
            )
            for s, c in sorted(keys)
        ]


def _first_stroke_by_character(
    strokes: List[ImageKey],
) -> Dict[CharacterKey, ImageKey]:
    first_by_char: Dict[CharacterKey, ImageKey] = {}
    for ik in strokes:
        ck = (ik[0], ik[1])
        if ck not in first_by_char or ik[2] < first_by_char[ck][2]:
            first_by_char[ck] = ik
    return first_by_char


def list_strokes_for_characters(
    store: FeatureStore,
    char_keys: Set[CharacterKey],
    *,
    first_stroke_only: bool = False,
    exclude_first_stroke: bool = False,
) -> List[ImageKey]:
    """
    Unique (script, char_id, image_stem) rows for the given characters.

    first_stroke_only: keep the lexicographically smallest image_stem per character
    (matches sorted(glob('*.png'))[0] in load_images_all).
    exclude_first_stroke: drop that first stroke (for train when test uses first only).
    """
    seen: Set[ImageKey] = set()
    strokes: List[ImageKey] = []
    for row in store.rows:
        ck: CharacterKey = (row["script"], row["char_id"])
        if ck not in char_keys:
            continue
        ik: ImageKey = (row["script"], row["char_id"], row["image_stem"])
        if ik in seen:
            continue
        seen.add(ik)
        strokes.append(ik)
    strokes = sorted(strokes)
    first_by_char = _first_stroke_by_character(strokes)
    if first_stroke_only:
        return sorted(first_by_char.values())
    if exclude_first_stroke:
        return [
            ik
            for ik in strokes
            if ik[2] != first_by_char[(ik[0], ik[1])][2]
        ]
    return strokes


def build_angle_classification_tensors(
    store: FeatureStore,
    char_keys: Set[CharacterKey],
    feature_mode: str = "diff",
    *,
    first_stroke_only: bool | None = None,
    exclude_first_stroke: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[float, int], Dict[int, float]]:
    """
    Build angle classification tensors for a character set.

    first_stroke_only caches: one sample per (character, angle).
    all_strokes caches: one sample per (stroke, angle) unless first_stroke_only=True
    is passed (e.g. paper-matched test eval: train on all strokes, test on first only).
    exclude_first_stroke: train on strokes 2..N per character (eval_split=first_stroke_test).
    """
    angles = store.probe_angles()
    angle_to_class, class_to_angle = angle_class_map(angles)
    cache_all_strokes = not store.meta.get("first_stroke_only", True)
    use_first_stroke = (
        first_stroke_only if first_stroke_only is not None else not cache_all_strokes
    )

    xs, ys = [], []
    if cache_all_strokes:
        for script, char_id, stem in list_strokes_for_characters(
            store,
            char_keys,
            first_stroke_only=use_first_stroke,
            exclude_first_stroke=exclude_first_stroke,
        ):
            char_key = (script, char_id)
            f0 = store.vector(char_key, 0.0, stem)
            for angle in angles:
                f_rot = store.vector(char_key, float(angle), stem)
                xs.append(probe_input_features(f0, f_rot, mode=feature_mode))
                ys.append(angle_to_class[float(angle)])
    else:
        for char_key in sorted(char_keys):
            f0 = store.vector(char_key, 0.0)
            for angle in angles:
                f_rot = store.vector(char_key, float(angle))
                xs.append(probe_input_features(f0, f_rot, mode=feature_mode))
                ys.append(angle_to_class[float(angle)])

    X = torch.stack(xs)
    y = torch.tensor(ys, dtype=torch.long)
    return X, y, angle_to_class, class_to_angle


def class_counts(y: torch.Tensor, class_to_angle: Dict[int, float]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for cls_idx, angle in class_to_angle.items():
        counts[str(int(angle))] = int((y == cls_idx).sum().item())
    return counts


def filter_keys_by_group(
    keys: Set[CharacterKey],
    char_to_group: Dict[CharacterKey, str],
    group: str,
) -> Set[CharacterKey]:
    return {k for k in keys if char_to_group.get(k) == group}


def split_train_val_keys(
    train_keys: Set[CharacterKey],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Set[CharacterKey], Set[CharacterKey]]:
    """Hold out characters from train for validation (same # of angles per char)."""
    import random

    keys = sorted(train_keys)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * val_ratio))
    if len(keys) - n_val < 2:
        raise ValueError("Not enough train characters for val split")
    val_keys = set(keys[:n_val])
    fit_keys = set(keys[n_val:])
    return fit_keys, val_keys
