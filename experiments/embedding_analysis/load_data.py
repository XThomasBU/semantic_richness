import os
import numpy as np


def load_performance_sets(filepath):
    performance_set = set()
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    script_name = parts[0].strip()
                    character_name = parts[1].strip()
                    performance_set.add((script_name, character_name))
    except FileNotFoundError:
        print(f"Didn't find file at {filepath}")
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
    
    return performance_set


def collect_domain_subfolders(root, domain_prefix):
    dirs = []
    for name in os.listdir(root):
        if name.startswith(domain_prefix):
            full_path = os.path.join(root, name)
            for sub in os.listdir(full_path):
                sub_path = os.path.join(full_path, sub)
                if os.path.isdir(sub_path):
                    dirs.append(sub_path)
    return dirs


def filter_omniglot_by_performance(features, labels, paths, 
    top_chars, bottom_chars, omniglot_label=0):
    filtered_features = []
    performance_labels = []

    for feature, label, path in zip(features, labels, paths):
        if label != omniglot_label:
            continue
            
        try:
            parts = path.split(os.sep)
            script_name = parts[-3]
            character_name = parts[-2]
            char_tuple = (script_name, character_name)

            if char_tuple in bottom_chars:
                filtered_features.append(feature)
                performance_labels.append(0)
            elif char_tuple in top_chars:
                filtered_features.append(feature)
                performance_labels.append(1)
        except IndexError:
            print(f"Error parsing character from path: {path}")

    return np.array(filtered_features), np.array(performance_labels)