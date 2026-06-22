"""Shared types for linear probe (no heavy dependencies)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

CharacterKey = Tuple[str, str]  # (script_name, char_id)
ImageKey = Tuple[str, str, str]  # (script_name, char_id, image_stem)
FeatureKey = Tuple[str, str, str, float]


PROBE_GROUPS = ("times_new_roman", "hand_english", "omniglot")


@dataclass(frozen=True)
class OmniglotCharacter:
    script_name: str
    char_id: str
    image_path: str
    image_stem: str = ""
    dataset_group: str = "omniglot"

    def __post_init__(self):
        if not self.image_stem:
            object.__setattr__(self, "image_stem", Path(self.image_path).stem)

    @property
    def key(self) -> CharacterKey:
        return (self.script_name, self.char_id)

    @property
    def image_key(self) -> ImageKey:
        return (self.script_name, self.char_id, self.image_stem)


@dataclass(frozen=True)
class MatchPair:
    char_a: CharacterKey
    char_b: CharacterKey
    angle: float
    label: int  # 1 = same character, 0 = different
    split: str  # "train" or "test"
