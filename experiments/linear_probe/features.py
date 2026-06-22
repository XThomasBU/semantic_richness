"""Extract frozen features from images; save/load precomputed caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from experiments.cosine_similarity.cos_similarity import load_encoder
from experiments.linear_probe.feature_index import (  # noqa: E402
    build_index_map,
    load_index_map,
    parse_key_string,
    save_index_map,
)


class PathImageDataset(Dataset):
    def __init__(self, image_paths: Sequence[str]):
        self.image_paths = list(image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return path, img
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            return None, None


def extract_features_from_paths(
    image_paths: Sequence[str],
    encoder,
    processor,
    batch_size: int = 32,
    device: str | None = None,
) -> Dict[str, torch.Tensor]:
    """Return path -> feature vector (CPU float tensor)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder.eval()

    dataset = PathImageDataset(image_paths)
    unique_paths = dataset.image_paths

    def collate_fn(batch):
        batch = [item for item in batch if item is not None and item[1] is not None]
        if not batch:
            return None, None, None
        paths = [item[0] for item in batch]
        images = [item[1] for item in batch]
        try:
            inputs = processor(images=images, return_tensors="pt")
        except Exception as e:
            print(f"Processor error: {e}")
            return None, None, None
        batch_tensors = inputs["pixel_values"]
        batch_grid_thw = inputs.get("image_grid_thw", None)
        return paths, batch_tensors, batch_grid_thw

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=device != "cpu",
        collate_fn=collate_fn,
    )

    features: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for batch_paths, batch_tensors, batch_grid_thw in tqdm(
            dataloader, desc="Extracting features"
        ):
            if batch_paths is None:
                continue
            batch_tensors = batch_tensors.to(device=device, dtype=encoder.dtype)
            if batch_grid_thw is not None:
                batch_grid_thw = batch_grid_thw.to(device=device)
                out = encoder(batch_tensors, grid_thw=batch_grid_thw)
            else:
                out = encoder(batch_tensors)
            if out.dim() == 3:
                out = out.mean(dim=1)
            for path, feat in zip(batch_paths, out.cpu()):
                features[path] = feat.float().flatten()

    missing = set(unique_paths) - set(features.keys())
    if missing:
        raise RuntimeError(f"Failed to extract features for {len(missing)} images")
    return features


def load_frozen_encoder(
    encoder_name: str, model_name: str, args: dict, device: str | None = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return load_encoder(encoder_name, model_name, args=args, device=device)


def features_for_rotated_views(
    path_by_key_angle: Dict[Tuple, str],
    encoder,
    processor,
    batch_size: int = 32,
    device: str | None = None,
) -> Dict[Tuple, torch.Tensor]:
    """Map feature key -> tensor using rendered PNG paths."""
    path_to_key = {p: k for k, p in path_by_key_angle.items()}
    path_features = extract_features_from_paths(
        list(path_by_key_angle.values()),
        encoder,
        processor,
        batch_size=batch_size,
        device=device,
    )
    return {path_to_key[path]: feat for path, feat in path_features.items()}


_SEP = "\x1f"


def _key_to_str(key: Tuple) -> str:
    if len(key) == 3:
        script, char_id, angle = key
        image_stem = ""
    elif len(key) == 4:
        script, char_id, image_stem, angle = key
    else:
        raise ValueError(f"Invalid feature key: {key}")
    return _SEP.join([script, char_id, image_stem, str(float(angle))])


def _str_to_key(s: str) -> Tuple:
    if _SEP in s:
        script, char_id, image_stem, angle = s.split(_SEP, 3)
        return (script, char_id, image_stem, float(angle))
    # Legacy cache: script|char_id|angle
    script, char_id, angle = s.rsplit("|", 2)
    return (script, char_id, "", float(angle))


def load_feature_matrix(
    path: str | Path,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load stacked features [N, D] plus per-row metadata and meta.json.

    Example:
        features, rows, meta = load_feature_matrix("clip_images_all_features.pt")
        row = rows[0]  # index 0
        vec = features[0]
    """
    path = Path(path)
    if path.suffix == ".pt":
        pt_path = path
    else:
        pt_path = path.with_suffix(".pt")

    payload = torch.load(pt_path, map_location="cpu", weights_only=True)
    key_strings = payload["keys"]
    matrix = payload["features"]
    _, meta = load_feature_cache(pt_path)
    rows = build_index_map(key_strings)
    return matrix, rows, meta


def save_feature_cache(
    path: str | Path,
    features: Dict[Tuple, torch.Tensor],
    meta: Dict[str, Any],
) -> None:
    """
    Save precomputed features to disk.

    Writes:
      - {stem}.pt   : stacked feature matrix + string keys
      - {stem}_meta.json : run configuration (encoder, angles, split, etc.)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.stem if path.suffix else path.name
    parent = path.parent if path.suffix else path
    if path.suffix == ".pt":
        pt_path, meta_path = path, path.with_name(f"{path.stem}_meta.json")
    else:
        pt_path = parent / f"{stem}.pt"
        meta_path = parent / f"{stem}_meta.json"

    keys = sorted(features.keys(), key=lambda k: _key_to_str(k))
    stacked = torch.stack([features[k] for k in keys])
    torch.save({"keys": [_key_to_str(k) for k in keys], "features": stacked}, pt_path)

    meta = dict(meta)
    meta["feature_dim"] = int(stacked.shape[1])
    meta["n_views"] = int(stacked.shape[0])
    meta["index_order"] = (
        "features[i] aligns with keys sorted lexicographically by "
        "(script, char_id, image_stem, angle); see {stem}_index_map.json"
    )
    key_strings = [_key_to_str(k) for k in keys]
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    save_index_map(pt_path, key_strings, meta)

    print(
        f"Saved {stacked.shape[0]} feature vectors (dim={stacked.shape[1]}) to {pt_path}"
    )
    print(f"Metadata saved to {meta_path}")


def load_feature_cache(
    path: str | Path,
) -> Tuple[Dict[Tuple, torch.Tensor], Dict[str, Any]]:
    """Load precomputed features written by save_feature_cache."""
    path = Path(path)
    if path.suffix == ".pt":
        pt_path = path
        meta_path = path.with_name(f"{path.stem}_meta.json")
    elif path.is_dir():
        pt_files = sorted(path.glob("*.pt"))
        if len(pt_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one .pt file in {path}, found {len(pt_files)}"
            )
        pt_path = pt_files[0]
        meta_path = pt_path.with_name(f"{pt_path.stem}_meta.json")
    else:
        pt_path = path.with_suffix(".pt")
        meta_path = path.parent / f"{path.name}_meta.json"

    if not pt_path.exists():
        raise FileNotFoundError(f"Feature cache not found: {pt_path}")

    payload = torch.load(pt_path, map_location="cpu", weights_only=True)
    keys = [_str_to_key(s) for s in payload["keys"]]
    features = {k: v for k, v in zip(keys, payload["features"])}

    meta: Dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return features, meta


def dataset_cache_tag(dataset: str, all_strokes: bool = False) -> str:
    """Filesystem tag for rotated images / feature caches."""
    if all_strokes and dataset in ("images_all", "probe_groups"):
        return f"{dataset}_all_strokes"
    return dataset


def feature_cache_path(
    features_dir: str | Path,
    encoder: str,
    tag: str,
    *,
    all_strokes: bool = False,
) -> Path:
    """Default cache filename for an encoder/dataset tag."""
    tag = dataset_cache_tag(tag, all_strokes)
    return Path(features_dir) / f"{encoder}_{tag}_features.pt"
