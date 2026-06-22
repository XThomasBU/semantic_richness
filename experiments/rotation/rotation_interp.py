import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vision_models.clip_encoder import ClipVisionTower
from PIL import Image
from tqdm import tqdm
import torch

config = {
    "model_name": "openai/clip-vit-large-patch14-336",
    "mm_vision_select_layer": -1,
    "select_feature": "cls_patch",
}

model = ClipVisionTower(config["model_name"], args=config, delay_load=False)
model = model.to('cuda')
model.eval()

_REPO_ROOT = Path(__file__).resolve().parents[2]
ROTATED_DIR = str(_REPO_ROOT / "DATA" / "omniglot-master" / "python" / "images_rotated" / "images_background")
ROTATED_FEATURES_DIR = str(_REPO_ROOT / "DATA" / "omniglot-master" / "python" / "images_rotated" / "features")
os.makedirs(ROTATED_FEATURES_DIR, exist_ok=True)
ANGLES = sorted(os.listdir(ROTATED_DIR))

print(ANGLES)

def get_feat(image, model):
    features = model.forward(image)
    return features

for angle in tqdm(ANGLES, total=len(ANGLES)):
    angle_folder = os.path.join(ROTATED_DIR, angle)
    scripts = sorted(os.listdir(angle_folder))
    for script in tqdm(scripts, total=len(scripts)):
        script_folder = os.path.join(angle_folder, script)
        characters = sorted(os.listdir(script_folder))
        for character in tqdm(characters, total=len(characters)):
            char_folder = os.path.join(script_folder, character)
            images = sorted(os.listdir(char_folder))
            char_paths = [os.path.join(char_folder, img) for img in images]
            for char_path in char_paths:
                char_name = os.path.basename(char_path)
                image = Image.open(f"{char_path}/{char_name}.png").convert("RGB")
                image = model.image_processor.preprocess(images=[image], return_tensors="pt")['pixel_values']
                image = image.to('cuda')
                features = get_feat(image, model)
                features_path = os.path.join(ROTATED_FEATURES_DIR, angle, script, character)
                cls_feature = features[:, 0, :]
                os.makedirs(features_path, exist_ok=True)
                torch.save(features, f"{features_path}/{char_name}.pt")