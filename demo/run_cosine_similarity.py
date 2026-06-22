"""
Demo: cosine similarity across rotation angles.

Computes how similar rotated Omniglot character images are to their
unrotated originals, averaged over all samples, for one or more encoders.

Usage (from repo root):
    python demo/run_cosine_similarity.py
    python demo/run_cosine_similarity.py --encoder dino
    python demo/run_cosine_similarity.py --encoder clip dino siglip
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from experiments.cosine_similarity.cos_similarity import CosineSimilarity, load_encoder


ENCODER_DEFAULTS = {
    "clip": ("openai/clip-vit-base-patch32", {"mm_vision_select_layer": -1, "unfreeze_mm_vision_tower": False, "mm_vision_select_feature": "patch"}),
    "dino": ("facebook/dinov2-base", {"mm_vision_select_layer": -1, "unfreeze_mm_vision_tower": False, "mm_vision_select_feature": "patch"}),
    "siglip": ("google/siglip-so400m-patch14-384", {"mm_vision_select_layer": -1, "unfreeze_mm_vision_tower": False, "mm_vision_select_feature": "patch"}),
    "qwen": ("Qwen/Qwen2.5-VL-7B-Instruct", {"mm_vision_select_layer": -1, "model_name": "Qwen/Qwen2.5-VL-7B-Instruct", "unfreeze_mm_vision_tower": False, "mm_vision_select_feature": "cls"}),
}


def run_encoder(encoder_name, data_dir, angles):
    """Run cosine similarity for one encoder, return {angle: mean_score}."""
    model_name, encoder_args = ENCODER_DEFAULTS[encoder_name]
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, processor = load_encoder(encoder_name, model_name, args=encoder_args, device=device)

    _, dirs, _ = next(os.walk(data_dir))
    all_scores = defaultdict(list)

    for dir_name in sorted(dirs):
        dir_path = os.path.join(data_dir, dir_name)
        sim = CosineSimilarity(dir_path, encoder, processor, device)
        features_by_angle = sim.feature_score(angles)
        scores = sim.similarity_score(angles, features_by_angle)
        for angle, score in scores.items():
            all_scores[angle].append(score)

    avg_scores = {angle: np.mean(vals) for angle, vals in sorted(all_scores.items())}
    std_scores = {angle: np.std(vals) for angle, vals in sorted(all_scores.items())}
    return avg_scores, std_scores


def plot_results(results, output_path):
    """Plot averaged similarity vs angle for each encoder."""
    plt.figure(figsize=(8, 5))
    colors = {"clip": "#1f77b4", "dino": "#ff7f0e", "siglip": "#2ca02c", "qwen": "#d62728"}

    for name, (avg, std) in results.items():
        angles = sorted(avg.keys())
        means = [avg[a] for a in angles]
        stds = [std[a] for a in angles]
        color = colors.get(name, None)
        plt.plot(angles, means, marker="o", label=name.upper(), color=color)
        plt.fill_between(angles,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         alpha=0.15, color=color)

    plt.title("Cosine Similarity vs Rotation Angle (averaged over samples)")
    plt.xlabel("Rotation Angle (degrees)")
    plt.ylabel("Cosine Similarity")
    plt.ylim(0.85, 1.005)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Demo: cosine similarity on rotated omniglot images")
    parser.add_argument("--encoder", nargs="+", default=["clip"], choices=list(ENCODER_DEFAULTS.keys()))
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "demo" / "sample_data"))
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "demo" / "output"))
    args = parser.parse_args()

    angles = list(range(0, 100, 10))
    results = {}

    for enc_name in args.encoder:
        print(f"\n{'='*50}")
        print(f"Running: {enc_name.upper()}")
        print(f"{'='*50}")
        avg, std = run_encoder(enc_name, args.data_dir, angles)
        results[enc_name] = (avg, std)
        print(f"\n{enc_name.upper()} average similarity by angle:")
        for angle in sorted(avg.keys()):
            print(f"  {angle:3d}°: {avg[angle]:.4f} ± {std[angle]:.4f}")

    output_path = os.path.join(args.output_dir, "cosine_similarity_avg.png")
    plot_results(results, output_path)


if __name__ == "__main__":
    main()
