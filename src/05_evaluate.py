import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score

SEED = 42
RAW_CSV = "data/GasProperties.csv"
STD_CSV = "outputs/StdGasProperties.csv"

KM_LABELS = "outputs/kmeans_labels.csv"
GMM_LABELS = "outputs/gmm_labels.csv"
SOM_LABELS = "outputs/som_labels.csv"

FEATURES = ["T", "P", "TC", "SV"]

def make_quality_classes(idx_series):
    # 33/33/33 quantile split 
    q1 = idx_series.quantile(1/3)
    q2 = idx_series.quantile(2/3)

    def label(v):
        if v <= q1:
            return "Regular"
        elif v <= q2:
            return "Medium"
        else:
            return "Premium"

    return idx_series.apply(label), (q1, q2)

def main():
    df_raw = pd.read_csv(RAW_CSV)
    X = pd.read_csv(STD_CSV).values  # standardized features for silhouette

    km = pd.read_csv(KM_LABELS)["kmeans_label"].values
    gmm = pd.read_csv(GMM_LABELS)["gmm_label"].values
    som = pd.read_csv(SOM_LABELS)["som_label"].values

    print("=== Silhouette Scores (higher is better separation) ===")
    print("KMeans:", silhouette_score(X, km))
    print("GMM   :", silhouette_score(X, gmm))
    print("SOM   :", silhouette_score(X, som))

    # ---- Quality classes from Idx ----
    quality, (q1, q2) = make_quality_classes(df_raw["Idx"])
    print("\n=== Quality class thresholds (Idx) ===")
    print(f"Regular <= {q1:.4f}, Medium <= {q2:.4f}, Premium > {q2:.4f}")

    # Stats per class on ORIGINAL feature scale (better for interpretation)
    df_raw["Quality"] = quality
    print("\n=== Feature mean/variance by quality class (original scale) ===")
    stats = df_raw.groupby("Quality")[FEATURES].agg(["mean", "var"])
    print(stats)

    # ---- ARI: clustering vs quality labels ----
    # convert text labels to integers
    quality_int = quality.map({"Regular": 0, "Medium": 1, "Premium": 2}).values

    print("\n=== Adjusted Rand Index (ARI) vs quality classes ===")
    print("ARI(KMeans, Quality):", adjusted_rand_score(quality_int, km))
    print("ARI(GMM, Quality)   :", adjusted_rand_score(quality_int, gmm))
    print("ARI(SOM, Quality)   :", adjusted_rand_score(quality_int, som))

if __name__ == "__main__":
    main()