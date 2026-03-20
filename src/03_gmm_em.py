import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

SEED = 42
STD_CSV = "outputs/StdGasProperties.csv"
OUT_LABELS = "outputs/gmm_labels.csv"

def main():
    X = pd.read_csv(STD_CSV).values

    # Use "full" covariance so each cluster can model correlations between features.
    cov_type = "full"
    gmm = GaussianMixture(
        n_components=3,
        covariance_type=cov_type,
        init_params="kmeans",
        tol=1e-3,
        max_iter=200,
        random_state=SEED
    )
    gmm.fit(X)

    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)

    print("=== GMM / EM (K=3) ===")
    print("Initialization method: init_params='kmeans' (means initialized from KMeans)")
    print("Convergence criterion:")
    print(f"  tol={gmm.tol}, max_iter={gmm.max_iter}, converged={gmm.converged_}, n_iter={gmm.n_iter_}")
    print(f"Covariance type: {cov_type} (allows feature correlations)")

    print("\nMixture weights (pi_k):")
    print(gmm.weights_)
    print("\nMeans (mu_k):")
    print(gmm.means_)
    print("\nCovariances (Sigma_k):")
    print(gmm.covariances_)

    print("\nPosterior probabilities p(z=k|x) for 3 samples:")
    for i in range(3):
        print(f"Sample {i}: {probs[i]}")

    pd.DataFrame({"gmm_label": labels}).to_csv(OUT_LABELS, index=False)
    print(f"\nSaved: {OUT_LABELS}")

if __name__ == "__main__":
    main()