"""Character-level holdout splits from an existing all-strokes feature cache."""

from __future__ import annotations

from typing import Set, Tuple

from experiments.linear_probe.data import split_character_keys
from experiments.linear_probe.feature_store import FeatureStore
from experiments.linear_probe.probe_types import CharacterKey, OmniglotCharacter


def all_character_keys_in_store(store: FeatureStore) -> Set[CharacterKey]:
    """Every (script, char_id) present in the feature index."""
    keys: Set[CharacterKey] = set()
    for row in store.rows:
        keys.add((row["script"], row["char_id"]))
    return keys


def assign_character_holdout_keys(
    store: FeatureStore,
    *,
    test_ratio: float | None = None,
    seed: int | None = None,
) -> Tuple[Set[CharacterKey], Set[CharacterKey]]:
    """
    Disjoint train / test character sets (no character in both).

    Uses the same shuffle + test_ratio logic as linear_probe.data.split_characters.
    Ignores train_char_keys / test_char_keys in meta (e.g. from first_stroke_test caches).
    """
    ratio = test_ratio if test_ratio is not None else float(store.meta.get("test_ratio", 0.2))
    split_seed = seed if seed is not None else int(store.meta.get("seed", 42))
    keys = all_character_keys_in_store(store)
    characters = [
        OmniglotCharacter(
            script_name=script,
            char_id=char_id,
            image_path="",
            image_stem=store.char_to_stem[(script, char_id)],
        )
        for script, char_id in sorted(keys)
    ]
    return split_character_keys(characters, test_ratio=ratio, seed=split_seed)
