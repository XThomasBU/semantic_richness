import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from experiments.cosine_similarity.image_loader import ImageLoader

class FeatureExtractor:
    def extract_features(self, folder_path, processor, encoder, batch_size, device = None):
        device = device or next(encoder.parameters()).device
        encoder.eval()

        dataset = ImageLoader(folder_path=folder_path)
        
        def collate_fn(batch):
            batch = [item for item in batch if item is not None and item[1] is not None]
            if not batch:
                return None, None, None
            
            paths = [item[0] for item in batch]
            images = [item[1] for item in batch]
            
            try:
                inputs = processor(
                    images=images, 
                    return_tensors="pt"
                )
            except Exception as e:
                print(f"Error during processor call: {e}")
                return None, None, None

            batch_tensors = inputs['pixel_values']
            
            #It will be None for CLIP/DiT/SD.
            batch_grid_thw = inputs.get('image_grid_thw', None) 
            
            return paths, batch_tensors, batch_grid_thw


        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn
        )

        all_features = []

        with torch.no_grad():
            for batch_paths, batch_tensors, batch_grid_thw in tqdm(dataloader, desc = "Extracting Features"):
                
                if batch_paths is None:
                    continue 

                batch_tensors = batch_tensors.to(device=device, dtype=encoder.dtype)

                if batch_grid_thw is not None:
                    # for qwen
                    batch_grid_thw = batch_grid_thw.to(device=device)
                    features = encoder(batch_tensors, grid_thw=batch_grid_thw)
                else:
                    # for every other encoder
                    features = encoder(batch_tensors)

                if features.dim() == 3:
                    features = features.mean(dim = 1)

                for path, feature in zip(batch_paths, features.cpu()):
                    all_features.append((path, feature))

        return all_features