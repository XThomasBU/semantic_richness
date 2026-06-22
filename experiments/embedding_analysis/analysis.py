import os
import numpy as np

from feature_extraction import load_model, extract_features_from_dirs, extract_features_from_paths
from dimensionality import run_pca, run_tsne, run_kmeans, compute_nmi, labels_to_numeric
from visualization import plot_pca, plot_tsne, plot_cross_dataset_pca, plot_performance_pca
from load_data import collect_domain_subfolders, load_performance_sets, filter_omniglot_by_performance


def run_pacs_identity_analysis(qwen_args, pacs_root, output_folder):
    """
    Currently for Photo vs Art domain
    """
    print("Loading Qwen model")
    encoder, processor, feature_extractor, device = load_model("qwen", qwen_args)

    # Change this for other domains
    photo_dirs = collect_domain_subfolders(pacs_root, "photo")
    art_dirs = collect_domain_subfolders(pacs_root, "art_painting")
    print(f"Found {len(photo_dirs)} PHOTO class folders")
    print(f"Found {len(art_dirs)} ART class folders")

    print("Extracting PACS Photo features")
    photo_feats, photo_labels, _ = extract_features_from_dirs(
        photo_dirs, processor, encoder, feature_extractor, device
    )

    print("Extracting PACS Art features")
    art_feats, art_labels, _ = extract_features_from_dirs(
        art_dirs, processor, encoder, feature_extractor, device
    )

    os.makedirs(output_folder, exist_ok=True)
    
    for feats, labels, domain in [(photo_feats, photo_labels, "Photo"),
                                   (art_feats, art_labels, "Art")]:
        projected, pca = run_pca(feats)
        numeric_labels, _ = labels_to_numeric(labels)
        
        n_classes = len(set(labels))
        cluster_labels = run_kmeans(feats, n_clusters=n_classes)
        nmi = compute_nmi(numeric_labels, cluster_labels)
        print(f"[{domain}] NMI = {nmi:.4f}")
        
        save_path = os.path.join(output_folder, f"PACS_{domain}_PCA_identity.png")
        plot_pca(projected, labels, pca, 
                 f"PACS {domain} (PCA Identity)", save_path)

    return photo_feats, photo_labels, art_feats, art_labels


def run_cross_dataset_pca(encoder_name, args, dataset_paths, output_folder,
                          dataset_names=None):
    """
    Run PCA analysis across multiple datasets.
    """
    
    if dataset_names is None:
        dataset_names = ['Omniglot', 'MNIST', 'English Alphabets']
    
    encoder, processor, feature_extractor, device = load_model(encoder_name, args)
    
    all_features, all_labels, all_paths = extract_features_from_paths(
        dataset_paths, processor, encoder, feature_extractor, device
    )

    features_np = np.array(all_features)
    labels_np = np.array(all_labels)
    print(f"Total features extracted: {features_np.shape[0]}")

    projected, pca = run_pca(features_np)
    
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, "pca_cross_dataset.png")
    plot_cross_dataset_pca(projected, labels_np, pca, dataset_names, save_path)

    return all_features, all_labels, all_paths


def run_omniglot_tsne_analysis(all_features, all_labels, all_paths, output_folder,
                                top_k_file, bottom_k_file, plot_title, plot_filename):
    """
    t-SNE for top and bottom performing omniglot scripts
    """
    top_chars = load_performance_sets(top_k_file)
    bottom_chars = load_performance_sets(bottom_k_file)
    
    if not top_chars and not bottom_chars:
        print("Error: No performance data loaded")
        return

    features_np, labels_np = filter_omniglot_by_performance(
        all_features, all_labels, all_paths, top_chars, bottom_chars
    )

    if features_np.shape[0] < 2:
        print("Not enough data to run analysis.")
        return

    # Compute NMI with K-Means
    print(f"Running K-Means (k=2) for {plot_filename}")
    cluster_labels = run_kmeans(features_np, n_clusters=2)
    nmi = compute_nmi(labels_np, cluster_labels)
    print(f"NMI Score for {plot_filename}: {nmi:.4f}")

    # Run t-SNE and plot
    print(f"Running t-SNE for {plot_filename}")
    projected = run_tsne(features_np)
    
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, f"{plot_filename}_tsne.png")
    plot_tsne(projected, labels_np, 
              ['Bottom Performing', 'Top Performing'],
              plot_title, save_path)


def run_omniglot_pca_analysis(all_features, all_labels, all_paths, output_folder,
                               top_k_file, bottom_k_file, plot_title, plot_filename):
    """
    PCA for top and bottom performing omniglot scripts
    """
    top_chars = load_performance_sets(top_k_file)
    bottom_chars = load_performance_sets(bottom_k_file)
    
    if not top_chars and not bottom_chars:
        print("Error: No performance data loaded")
        return

    features_np, labels_np = filter_omniglot_by_performance(
        all_features, all_labels, all_paths, top_chars, bottom_chars
    )

    if features_np.shape[0] < 2:
        print("Not enough data to run analysis.")
        return

    projected, pca = run_pca(features_np)
    
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, plot_filename)
    plot_performance_pca(projected, labels_np, pca, plot_title, save_path)