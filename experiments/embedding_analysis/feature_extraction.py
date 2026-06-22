import os
import torch
import numpy as np

from experiments.cosine_similarity.feature_extractor import FeatureExtractor
from experiments.cosine_similarity.cos_similarity import load_encoder


def load_model(encoder_name, args, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    args = args.copy()
    args["mm_vision_select_feature"] = 'cls'
    
    encoder, processor = load_encoder(
        encoder_name,
        args["model_name"],
        args=args,
        device=device
    )
    feature_extractor = FeatureExtractor()
    
    return encoder, processor, feature_extractor, device


def extract_features_from_dirs(dir_list, processor, encoder, feature_extractor, 
                                device, batch_size=1):
    features, labels, paths = [], [], []
    
    for class_dir in dir_list:
        class_name = os.path.basename(class_dir)
        print(f"Extracting class: {class_name} from {class_dir}")

        features_list = feature_extractor.extract_features(
            folder_path=class_dir,
            processor=processor,
            encoder=encoder,
            batch_size=batch_size,
            device=device
        )

        for p, feature in features_list:
            features.append(feature.float().numpy())
            labels.append(class_name)
            paths.append(p)

    return np.array(features), np.array(labels), paths


def extract_features_from_paths(dataset_paths, processor, encoder, 
                                 feature_extractor, device, batch_size=1):
    all_features, all_labels, all_paths = [], [], []

    for i, path in enumerate(dataset_paths):
        features_list = feature_extractor.extract_features(
            folder_path=path,
            processor=processor,
            encoder=encoder,
            batch_size=batch_size,
            device=device
        )
        
        for path_str, feature in features_list:
            all_features.append(feature.float().numpy())
            all_labels.append(i)
            all_paths.append(path_str)

    return all_features, all_labels, all_paths