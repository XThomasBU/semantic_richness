import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


def run_pca(features, n_components=2):
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(features)
    return projected, pca


def run_tsne(features, n_components=2, perplexity=30, random_state=42):
    tsne = TSNE(
        n_components=n_components,
        random_state=random_state,
        perplexity=perplexity
    )
    return tsne.fit_transform(features)


def run_kmeans(features, n_clusters, n_init=10, random_state=42):
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state
    )
    return kmeans.fit_predict(features)


def compute_nmi(labels_true, labels_pred):
    return normalized_mutual_info_score(labels_true, labels_pred)


def labels_to_numeric(labels):
    """
    For mapping labels into numbers(PACS dataset)
    """
    classes = sorted(list(set(labels)))
    class_to_int = {c: i for i, c in enumerate(classes)}
    numeric = np.array([class_to_int[c] for c in labels])
    return numeric, class_to_int