import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.cluster import KMeans

SEED = 42
STD_CSV = "outputs/StdGasProperties.csv"
OUT_LABELS = "outputs/som_labels.csv"

def main():
    X = pd.read_csv(STD_CSV).values
    n, d = X.shape

    # ---- SOM hyperparameters  ----
    grid_x, grid_y = 15, 15
    sigma = 2.0                    # neighborhood radius
    learning_rate = 0.5            # initial LR
    iterations = 5000              # training steps
    neighborhood = "gaussian"      # minisom default neighborhood is gaussian
    metric = "euclidean"

    print("=== SOM (2D) ===")
    print(f"Grid size: {grid_x}x{grid_y}")
    print(f"Neighborhood function: {neighborhood}")
    print(f"Learning rate (initial): {learning_rate}")
    print(f"Sigma (neighborhood radius): {sigma}")
    print(f"Iterations: {iterations}")
    print(f"Similarity metric: {metric}")

    som = MiniSom(
        x=grid_x, y=grid_y, input_len=d,
        sigma=sigma, learning_rate=learning_rate,
        neighborhood_function=neighborhood,
        random_seed=SEED
    )
    som.random_weights_init(X)

    som.train_random(X, iterations)

    # ---- Extract prototypes (neuron weights) ----
    prototypes = som.get_weights().reshape(grid_x * grid_y, d)

    # Cluster prototypes using KMeans K=3
    kmeans = KMeans(n_clusters=3, random_state=SEED, n_init=10)
    proto_labels = kmeans.fit_predict(prototypes)

    # Map each BMU to a prototype cluster
    def bmu_index(x):
        i, j = som.winner(x)
        return i * grid_y + j

    labels = np.array([proto_labels[bmu_index(x)] for x in X])

    # Compute final cluster centroids in data space + sizes
    centroids = np.vstack([X[labels == k].mean(axis=0) for k in range(3)])
    sizes = np.array([(labels == k).sum() for k in range(3)])

    print("\nSOM-derived cluster centroids (mean of assigned data points):")
    print(centroids)
    print("\nSOM-derived cluster sizes:")
    for k in range(3):
        print(f"  Cluster {k}: {sizes[k]}")

    pd.DataFrame({"som_label": labels}).to_csv(OUT_LABELS, index=False)
    print(f"\nSaved: {OUT_LABELS}")

if __name__ == "__main__":
    main()