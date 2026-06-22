import os
import matplotlib.pyplot as plt


def plot_pca(projected, labels, pca, title, save_path, figsize=(12, 10)):
    plt.figure(figsize=figsize)
    
    classes = sorted(list(set(labels)))
    for cls in classes:
        idx = (labels == cls)
        plt.scatter(
            projected[idx, 0],
            projected[idx, 1],
            s=18,
            alpha=0.55,
            label=cls
        )

    plt.title(title, fontsize=24)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA plot: {save_path}")


def plot_tsne(projected, labels, label_names, title, save_path, 
              colors=None, figsize=(12, 10)):
    if colors is None:
        colors = ['r', 'g', 'b']
    
    plt.figure(figsize=figsize)
    
    for i, name in enumerate(label_names):
        idx = (labels == i)
        plt.scatter(
            projected[idx, 0],
            projected[idx, 1],
            c=colors[i % len(colors)],
            label=name,
            alpha=0.5,
            s=15
        )
    
    plt.title(title, fontsize=26)
    plt.xlabel("t-SNE Component 1", fontsize=22)
    plt.ylabel("t-SNE Component 2", fontsize=22)
    plt.legend(fontsize=25, markerscale=6)
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot: {save_path}")


def plot_cross_dataset_pca(projected, labels, pca, dataset_names, save_path,
                            colors=None, figsize=(12, 10)):
    if colors is None:
        colors = ['r', 'g', 'b']
    
    plt.figure(figsize=figsize)
    
    for i, name in enumerate(dataset_names):
        idx = (labels == i)
        plt.scatter(
            projected[idx, 0],
            projected[idx, 1],
            c=colors[i % len(colors)],
            label=name,
            alpha=0.5,
            s=15
        )
    
    plt.title("PCA of Vision Embeddings (Shape vs. Semantics)", fontsize=25)
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)", fontsize=22)
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)", fontsize=22)
    plt.legend(fontsize=30, markerscale=5, loc='lower right')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved cross-dataset PCA: {save_path}")


def plot_performance_pca(projected, labels, pca, title, save_path, figsize=(12, 10)):
    plt.figure(figsize=figsize)
    
    label_names = ['Bottom Performing', 'Top Performing']
    colors = ['r', 'g']
    
    for i, name in enumerate(label_names):
        idx = (labels == i)
        plt.scatter(
            projected[idx, 0],
            projected[idx, 1],
            c=colors[i],
            label=name,
            alpha=0.5,
            s=15
        )
    
    plt.title(title, fontsize=30)
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)", fontsize=30)
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)", fontsize=30)
    plt.legend(fontsize=40, markerscale=6)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved performance PCA: {save_path}")