import os
from torch.utils.data import Dataset
from glob import glob
from PIL import Image

class ImageLoader(Dataset):
    def __init__(self, folder_path):
        self.image_paths = []
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        for ext in extensions:
            self.image_paths.extend(glob(os.path.join(folder_path, "**", ext), recursive=True))
        
        if not self.image_paths:
            print(f"No images in {folder_path}")


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            return path, img
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            return None, None