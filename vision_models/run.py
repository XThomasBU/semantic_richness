#!/usr/bin/env python3
"""
Extract and save features from all vision models for omniglot images.

Usage:
    python -m vision_models.run [--output_dir OUTPUT_DIR] [--device DEVICE] [--models MODEL1,MODEL2,...]
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from vision_models.clip_encoder import ClipVisionTower
from vision_models.dino_encoder import DinoVisionTower
from vision_models.diffusion_encoder import DiffusionVisionTower, DiTTower
from vision_models.qwen_encoder import QwenVisionTower

from experiments.cosine_similarity.image_loader import ImageLoader


ENCODER_REGISTRY = {
    "clip": (ClipVisionTower, "openai/clip-vit-base-patch32", "args"),
    "dino": (DinoVisionTower, "facebook/dinov2-base", "args"),
    "diffusion": (DiffusionVisionTower, "stabilityai/stable-diffusion-2-1", "config"),
    "dit": (DiTTower, "facebook/DiT-XL-2-512", "config"),
    "qwen": (QwenVisionTower, "Qwen/Qwen2.5-VL-7B-Instruct", "config"),
}

ENCODER_CONFIGS = {
    "clip": {"mm_vision_select_layer": -1, "unfreeze_mm_vision_tower": False, "mm_vision_select_feature": "patch_mean"},
    "dino": {"mm_vision_select_layer": -1, "unfreeze_mm_vision_tower": False, "mm_vision_select_feature": "patch_mean"},
    "diffusion": {"mm_vision_select_layer": 1, "unfreeze_mm_vision_tower": False, "model_name": "stabilityai/stable-diffusion-2-1", "diffusion_step_type": "onestep", "time_step": 25},
    "dit": {"mm_vision_select_layer": 13, "model_name": "facebook/DiT-XL-2-512", "time_step": 25, "unfreeze_mm_vision_tower": False},
    "qwen": {"mm_vision_select_layer": -1, "model_name": "Qwen/Qwen2.5-VL-7B-Instruct", "unfreeze_mm_vision_tower": False, "mm_vision_select_feature": "cls"},
}

BATCH_SIZES = {"qwen": 1, "clip": 32, "dino": 32, "diffusion": 8, "dit": 8}

DEFAULT_OMNIGLOT_PATH = _REPO_ROOT / "DATA" / "omniglot-master" / "python" / "images_all"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "vision_models" / "omniglot_features"


def load_encoder(encoder_name, device="cuda"):
    if encoder_name not in ENCODER_REGISTRY:
        raise ValueError("Unknown encoder: " + encoder_name)
    encoder_cls, model_name, config_key = ENCODER_REGISTRY[encoder_name]
    config = ENCODER_CONFIGS[encoder_name].copy()
    if "model_name" in config:
        model_name = config["model_name"]
    if config_key == "config":
        encoder = encoder_cls(model_name, config=config)
    else:
        encoder = encoder_cls(model_name, args=config)
    if device != "cpu":
        encoder = encoder.to(device)
    encoder.eval()
    return encoder, encoder.image_processor


def extract_features(encoder, processor, folder_path, device, batch_size):
    dataset = ImageLoader(folder_path=folder_path)
    def collate_fn(batch):
        batch = [item for item in batch if item is not None and item[1] is not None]
        if not batch:
            return None, None, None
        paths = [item[0] for item in batch]
        images = [item[1] for item in batch]
        try:
            inputs = processor(images=images, return_tensors="pt")
        except Exception as e:
            print("Processor error:", e)
            return None, None, None
        pv = inputs["pixel_values"]
        if isinstance(pv, list):
            pv = torch.stack(pv)
        return paths, pv, inputs.get("image_grid_thw", None)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    all_features = []
    with torch.no_grad():
        for batch_paths, batch_tensors, batch_grid_thw in tqdm(dataloader, desc=Path(folder_path).name):
            if batch_paths is None:
                continue
            batch_tensors = batch_tensors.to(device=device, dtype=encoder.dtype)
            if batch_grid_thw is not None:
                features = encoder(batch_tensors, grid_thw=batch_grid_thw.to(device=device))
            else:
                features = encoder(batch_tensors)
            if features.dim() == 3:
                features = features.mean(dim=1)
            for path, feat in zip(batch_paths, features.cpu()):
                all_features.append((path, feat))
    return all_features


def run_model(encoder_name, omniglot_path, output_dir, device):
    print("\n" + "=" * 60 + "\nLoading " + encoder_name + "...")
    encoder, processor = load_encoder(encoder_name, device=device)
    batch_size = BATCH_SIZES.get(encoder_name, 16)
    (output_dir / encoder_name).mkdir(parents=True, exist_ok=True)
    script_dirs = sorted([p for p in omniglot_path.iterdir() if p.is_dir()])
    if not script_dirs:
        print("No script dirs in", omniglot_path)
        return
    for script_dir in script_dirs:
        script_name = script_dir.name
        print("Processing", script_name, "...")
        features_list = extract_features(encoder, processor, str(script_dir), device, batch_size)
        if not features_list:
            continue
        paths = [p for p, _ in features_list]
        features = torch.stack([f for _, f in features_list])
        torch.save({"paths": paths, "features": features, "script": script_name, "encoder": encoder_name},
                   output_dir / encoder_name / (script_name + ".pt"))
        print("  Saved", len(features_list), "features")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omniglot_path", default=str(DEFAULT_OMNIGLOT_PATH))
    ap.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models", default="clip,dino,diffusion,dit,qwen", help="Comma-separated model names")
    args = ap.parse_args()
    omniglot_path = Path(args.omniglot_path)
    if not omniglot_path.exists():
        raise FileNotFoundError("Omniglot path not found: " + str(omniglot_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        if name not in ENCODER_REGISTRY:
            print("Unknown model", name)
            continue
        try:
            run_model(name, omniglot_path, output_dir, args.device)
        except Exception as e:
            print("Error running", name, ":", e)
            import traceback
            traceback.print_exc()
    print("\nDone. Features saved to", output_dir)


if __name__ == "__main__":
    main()
