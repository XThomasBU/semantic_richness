"""Index mapping for stacked feature matrices (no torch model imports)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_SEP = "\x1f"


def _str_to_key(s: str) -> Tuple:
    if _SEP in s:
        script, char_id, image_stem, angle = s.split(_SEP, 3)
        return (script, char_id, image_stem, float(angle))
    script, char_id, angle = s.rsplit("|", 2)
    return (script, char_id, "", float(angle))


def parse_key_string(key_str: str) -> Dict[str, Any]:
    key = _str_to_key(key_str)
    if len(key) == 4:
        script, char_id, image_stem, angle = key
    else:
        script, char_id, angle = key
        image_stem = ""
    return {
        "script": script,
        "char_id": char_id,
        "image_stem": image_stem,
        "angle": float(angle),
        "is_canonical": float(angle) == 0.0,
        "key": key_str,
    }


def build_index_map(key_strings: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Row i in features.pt matches entries[i].

    Sort order when saving: lexicographic on (script, char_id, image_stem, angle).
    Per character (10 views): index i..i+9 are 0°, 10°, 20°, ..., 90°.
    """
    return [
        {"index": i, **parse_key_string(key_strings[i])}
        for i in range(len(key_strings))
    ]


def save_index_map(
    pt_path: str | Path,
    key_strings: Sequence[str],
    meta: Dict[str, Any] | None = None,
) -> Path:
    pt_path = Path(pt_path)
    out_path = pt_path.with_name(f"{pt_path.stem}_index_map.json")
    entries = build_index_map(key_strings)
    payload = {
        "description": (
            "features[i] in the .pt file is entries[i]. "
            "Sorted by (script, char_id, image_stem, angle). "
            "Each character has 10 consecutive rows: 0°, 10°, …, 90°."
        ),
        "n_features": len(entries),
        "feature_dim": meta.get("feature_dim") if meta else None,
        "angles_in_cache": sorted({e["angle"] for e in entries}),
        "views_per_character": (
            len(
                [
                    e
                    for e in entries
                    if e["char_id"] == entries[0]["char_id"]
                    and e["script"] == entries[0]["script"]
                ]
            )
            if entries
            else 0
        ),
        "entries": entries,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Index map saved to {out_path}")
    return out_path


def load_index_map(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if path.suffix != ".json":
        path = path.with_name(f"{path.stem}_index_map.json")
    with open(path) as f:
        return json.load(f)
