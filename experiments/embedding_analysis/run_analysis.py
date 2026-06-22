import argparse
from pathlib import Path
from analysis import run_pacs_identity_analysis, run_cross_dataset_pca, run_omniglot_tsne_analysis

_REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
    "paths": {
        "omniglot": str(_REPO_ROOT / "DATA" / "omniglot"),
        "mnist": None,
        "alphabets": None,
        "pacs_root": None,
        "output": str(_REPO_ROOT / "results" / "embedding_analysis"),
    },
    "performance_files": {
        "identity_top": None,
        "identity_bottom": None,
        "rotation_top": None,
        "rotation_bottom": None,
    },
    "qwen_args": {
        "mm_vision_select_layer": -1,
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "unfreeze_mm_vision_tower": False,
        "mm_vision_select_feature": "cls",
    },
}


def run_all_experiments():
    paths = CONFIG["paths"]
    perf_files = CONFIG["performance_files"]
    qwen_args = CONFIG["qwen_args"]
    
    dataset_paths = [paths["omniglot"], paths["mnist"], paths["alphabets"]]

    # PACS identity analysis
    print("Running PACS identity analysis")
    run_pacs_identity_analysis(
        qwen_args=qwen_args,
        pacs_root=paths["pacs_root"],
        output_folder=paths["output"]
    )

    # Cross-dataset PCA
    print("Running Cross-Dataset PCA")
    all_features, all_labels, all_paths = run_cross_dataset_pca(
        "qwen",
        qwen_args,
        dataset_paths,
        paths["output"]
    )

    # Identity task t-SNE
    print("Running t-SNE for IDENTITY Task")
    run_omniglot_tsne_analysis(
        all_features, all_labels, all_paths,
        paths["output"],
        top_k_file=perf_files["identity_top"],
        bottom_k_file=perf_files["identity_bottom"],
        plot_title="t-SNE of Omniglot (Identity Task Performance)",
        plot_filename="TSNE_omniglot_performance_IDENTITY"
    )

    # Rotation task t-SNE
    print("Running t-SNE for ROTATION Task")
    run_omniglot_tsne_analysis(
        all_features, all_labels, all_paths,
        paths["output"],
        top_k_file=perf_files["rotation_top"],
        bottom_k_file=perf_files["rotation_bottom"],
        plot_title="t-SNE of Omniglot (Rotation Task Performance)",
        plot_filename="TSNE_omniglot_performance_ROTATION"
    )

    print("All analyses completed")


def main():
    parser = argparse.ArgumentParser(
        description="Run vision embedding analysis experiments"
    )
    parser.add_argument("--pacs", action="store_true", help="Run PACS identity analysis")
    parser.add_argument("--cross-dataset", action="store_true", help="Run cross-dataset PCA")
    parser.add_argument("--identity", action="store_true", help="Run identity task t-SNE")
    parser.add_argument("--rotation", action="store_true", help="Run rotation task t-SNE")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    
    args = parser.parse_args()
    
    # All is default
    if not any([args.pacs, args.cross_dataset, args.identity, 
                args.rotation, args.all]):
        args.all = True
    
    if args.all:
        run_all_experiments()
        return
    
    paths = CONFIG["paths"]
    perf_files = CONFIG["performance_files"]
    qwen_args = CONFIG["qwen_args"]
    dataset_paths = [paths["omniglot"], paths["mnist"], paths["alphabets"]]
    
    all_features, all_labels, all_paths = None, None, None
    
    if args.pacs:
        run_pacs_identity_analysis(qwen_args, paths["pacs_root"], paths["output"])
    
    if args.cross_dataset or args.identity or args.rotation:
        all_features, all_labels, all_paths = run_cross_dataset_pca(
            "qwen", qwen_args, dataset_paths, paths["output"]
        )
    
    if args.identity and all_features:
        run_omniglot_tsne_analysis(
            all_features, all_labels, all_paths, paths["output"],
            perf_files["identity_top"], perf_files["identity_bottom"],
            "t-SNE of Omniglot (Identity Task Performance)",
            "TSNE_omniglot_performance_IDENTITY"
        )
    
    if args.rotation and all_features:
        run_omniglot_tsne_analysis(
            all_features, all_labels, all_paths, paths["output"],
            perf_files["rotation_top"], perf_files["rotation_bottom"],
            "t-SNE of Omniglot (Rotation Task Performance)",
            "TSNE_omniglot_performance_ROTATION"
        )


if __name__ == "__main__":
    # Options: --pacs, --cross-dataset, --identity, --rotation, --all
    main()