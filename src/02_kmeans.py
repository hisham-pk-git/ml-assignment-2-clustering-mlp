import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

SEED = 42
STD_CSV = "outputs/StdGasProperties.csv"
OUT_LABELS = "outputs/kmeans_labels.csv"

def within_cluster_sse(X, labels, centers):
    sse_per_cluster = []
    for k in range(centers.shape[0]):
        pts = X[labels == k]
        sse = ((pts - centers[k]) ** 2).sum()
        sse_per_cluster.append(sse)
    return sse_per_cluster

def main():
    X = pd.read_csv(STD_CSV).values

    # To report "initial centroids" clearly, we force a reproducible random init.
    # (k-means++ initial centers are less direct to report.)
    init_method = "random"
    kmeans = KMeans(
        n_clusters=3,
        init=init_method,
        n_init=1,
        random_state=SEED
    )

    # We need initial centroids: with init='random' + n_init=1, sklearn chooses them deterministically from seed.
    kmeans.fit(X)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Total within-cluster SSE:
    inertia = kmeans.inertia_
    # Per-cluster SSE:
    sse_per_cluster = within_cluster_sse(X, labels, centers)
    sizes = np.bincount(labels, minlength=3)

    print("=== KMeans (K=3) ===")
    print(f"Initialization method: {init_method}")
    
    rng = np.random.RandomState(SEED)
    init_idx = rng.choice(X.shape[0], 3, replace=False)
    init_centroids = X[init_idx]
    print("Initial centroids (reported from seeded random selection):")
    print(init_centroids)

    print(f"\nIterations until convergence: {kmeans.n_iter_}")
    print("\nFinal centroids:")
    print(centers)

    print("\nCluster variances / within-cluster SSE (per cluster):")
    for k, sse in enumerate(sse_per_cluster):
        print(f"  Cluster {k}: {sse:.6f}")

    print(f"\nTotal within-cluster SSE (inertia): {inertia:.6f}")
    print("\nCluster sizes:")
    for k, c in enumerate(sizes):
        print(f"  Cluster {k}: {c}")

    pd.DataFrame({"kmeans_label": labels}).to_csv(OUT_LABELS, index=False)
    print(f"\nSaved: {OUT_LABELS}")

if __name__ == "__main__":
    main()