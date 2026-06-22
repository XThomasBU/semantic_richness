import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict

from experiments.cosine_similarity.feature_extractor import FeatureExtractor

from vision_models.clip_encoder import ClipVisionTower
from vision_models.dino_encoder import DinoVisionTower

# from vision_models.diffusion_encoder import DiffusionVisionTower, DiTTower
from vision_models.qwen_encoder import QwenVisionTower
from vision_models.siglip_encoder import SiglipVisionTower

ENCODER = {
    "clip": ClipVisionTower,
    "dino": DinoVisionTower,
    # "diffusion": DiffusionVisionTower,
    # "dit": DiTTower,
    "qwen": QwenVisionTower,
    "siglip": SiglipVisionTower,
}


def load_encoder(encoder_name, model_name, args=None, device="cpu"):
    if encoder_name not in ENCODER:
        raise ValueError(f"Invalid encoder: {encoder_name}")

    encoder_class = ENCODER[encoder_name]
    if encoder_name in ["diffusion", "dit", "qwen"]:
        # Diffusion/DiT/Qwen towers expect the 'config' keyword argument
        encoder = encoder_class(model_name, config=args)
    else:
        # CLIP and DINO towers expect the 'args' keyword argument
        encoder = encoder_class(model_name, args=args)

    if device != "cpu":
        encoder = encoder.to(device)

    encoder._device = torch.device(device)
    processor = encoder.image_processor

    print(f"Encoder loaded on {encoder._device}")
    return encoder, processor


class CosineSimilarity:
    def __init__(self, image_folder_path, encoder, processor, device):
        self.image_folder_path = image_folder_path
        self.encoder = encoder
        self.processor = processor
        self.device = device
        self.feature_extractor = FeatureExtractor()

    # Use a smaller batch size for DiT/SD, larger size for Clip/Dino
    def feature_score(self, angles):
        current_batch_size = 1 if "Qwen" in self.encoder.__class__.__name__ else 32
        print(f"Using batch size: {current_batch_size}")

        # qwen tiles images
        all_features = self.feature_extractor.extract_features(
            folder_path=self.image_folder_path,
            processor=self.processor,
            encoder=self.encoder,
            batch_size=current_batch_size,
            device=self.device,
        )

        print("\nReorganizing features by angle...")
        features_by_angles = defaultdict(list)
        for path, feature in all_features:
            try:
                # Assumes path is like .../Angle_10/image.png
                angle = int(path.split(os.sep)[-2].split("_")[1])
                features_by_angles[angle].append((path, feature))
            except (IndexError, ValueError):
                print(f"Could not parse angle from path: {path}")

        return features_by_angles

    def similarity_score(self, angles, features_by_angles):
        avg_score = defaultdict(list)

        features_by_character = {
            path.split(os.sep)[-4]: feature for path, feature in features_by_angles[0]
        }

        for angle in angles:
            if angle == 0:
                continue
            for path, feature in features_by_angles[angle]:
                char_key = path.split(os.sep)[-4]
                if char_key in features_by_character:
                    original_feature = features_by_character[char_key]
                    similarity = F.cosine_similarity(
                        original_feature.unsqueeze(0), feature.unsqueeze(0)
                    ).item()

                    avg_score[angle].append(similarity)

        final_scores = {
            angle: sum(scores) / len(scores) for angle, scores in avg_score.items()
        }

        return final_scores

    def plot_score(self, omniglot_name, encoder_name, similarity_scores, output_folder):
        angles = list(similarity_scores.keys())
        scores = list(similarity_scores.values())

        plt.figure(figsize=(8, 5))
        plt.plot(angles, scores, marker="o", linestyle="-", color="b")
        plt.title(f"{omniglot_name}: {encoder_name.upper()} vision encoder")
        plt.xlabel("Angle")
        plt.ylabel("Similarity Score")
        plt.ylim(0, 1.0)
        plt.grid(True)

        output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

        filename = f"{omniglot_name}_{encoder_name.upper()}_similarity_scores.png"
        full_path = os.path.join(output_folder, filename)
        plt.savefig(full_path)
        plt.close()

    @staticmethod
    def process(
        image_folder_path, output_folder, angles, encoder_name, model_name, args
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # loading encoder
        encoder, processor = load_encoder(
            encoder_name, model_name, args=args, device=device
        )

        _, dirs, _ = next(os.walk(image_folder_path))
        for dir_name in dirs:
            dir_folder_path = os.path.join(image_folder_path, dir_name)
            # image -> tensor -> feature -> similarity score
            similarity = CosineSimilarity(dir_folder_path, encoder, processor, device)
            feature_scores = similarity.feature_score(angles)
            similarity_scores = similarity.similarity_score(angles, feature_scores)

            print(f"Found all similarity scores for {dir_name}")
            print(similarity_scores)
            # saves the similarity score plot as a png
            similarity.plot_score(
                dir_name, encoder_name, similarity_scores, output_folder
            )


# Usage
if __name__ == "__main__":
    args = {
        "mm_vision_select_layer": -1,
        "unfreeze_mm_vision_tower": False,
        "mm_vision_select_feature": "patch",
    }

    diffusion_args = {
        "mm_vision_select_layer": 1,
        "unfreeze_mm_vision_tower": False,
        "model_name": "stabilityai/stable-diffusion-2-1",
        "diffusion_step_type": "onestep",
        "time_step": 25,
    }

    dit_args = {
        "mm_vision_select_layer": 13,
        "model_name": "facebook/DiT-XL-2-512",
        "time_step": 25,
        "unfreeze_mm_vision_tower": False,
    }

    qwen_args = {
        "mm_vision_select_layer": -1,
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "unfreeze_mm_vision_tower": False,
        "mm_vision_select_feature": "cls",
    }

    image_folder_path = "path/to/images"
    output_folder = "output/folder/path"
    angles = [x for x in range(0, 100, 10)]

    # example: clip or dino
    encoder_name = "diffusion"
    # example: openai/clip-vit-base-patch32 or facebook/dinov2-large
    model_name = "stabilityai/stable-diffusion-2-1"

    CosineSimilarity.process(
        image_folder_path,
        output_folder,
        angles,
        encoder_name,
        model_name,
        diffusion_args,
    )
