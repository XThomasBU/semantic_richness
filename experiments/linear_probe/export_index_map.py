#!/usr/bin/env python3
"""Build index_map.json for an existing feature cache .pt file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import json

import torch

_RESEARCH_ROOT = Path(__file__).resolve().parents[2]
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

from experiments.linear_probe.feature_index import save_index_map  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Export index map for a feature .pt cache")
    p.add_argument(
        "--features_path",
        type=str,
        required=True,
        help="Path to clip_images_all_features.pt",
    )
    args = p.parse_args()

    pt_path = Path(args.features_path)
    payload = torch.load(pt_path, map_location="cpu", weights_only=True)
    meta_path = pt_path.with_name(f"{pt_path.stem}_meta.json")
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    save_index_map(pt_path, payload["keys"], meta)
    print(f"features.shape = {tuple(payload['features'].shape)}")
    print("Example index 0:", payload["keys"][0])


if __name__ == "__main__":
    main()
