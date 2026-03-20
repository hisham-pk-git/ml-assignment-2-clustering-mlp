import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# Configurations
SEED = 42
STD_DATA = "outputs/StdGasProperties.csv"
RAW_DATA = "data/GasProperties.csv"
OUTPUT_LABELS = "outputs/rbf_labels.csv"
OUTPUT_CM_PLOT = "outputs/rbf_confusion_matrix.png"
OUTPUT_REPORT = "outputs/rbf_report.txt"
N_CLUSTERS = 200

def make_quality_classes(idx_series):
    q1 = idx_series.quantile(1/3)
    q2 = idx_series.quantile(2/3)
    def label(v):
        if v <= q1: return 0
        elif v <= q2: return 1
        else: return 2
    return idx_series.apply(label)

def rbf_kernel(X, centers, sigma):
    """
    Computes the RBF transformation efficiently.
    """
    n_samples = X.shape[0]
    n_centers = centers.shape[0]
    G = np.zeros((n_samples, n_centers))
    gamma = 1.0 / (2 * sigma**2)
    
    for i in range(n_centers):
        dist_sq = np.sum((X - centers[i])**2, axis=1)
        G[:, i] = np.exp(-dist_sq * gamma)
    return G

def main():
    print(f"=== High-Performance RBF Neural Network (N_CLUSTERS={N_CLUSTERS}) ===")
    
    # 1. Load Data
    X = pd.read_csv(STD_DATA).values
    df_raw = pd.read_csv(RAW_DATA)
    y = make_quality_classes(df_raw["Idx"]).values

    # 2. Extract RBF Centers using K-Means
    print(f"Extracting {N_CLUSTERS} centers using K-Means")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=3)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_

    # 3. Calculate Kernel Width (sigma)
    avg_dist = np.mean([np.linalg.norm(c1 - c2) for c1 in centers[:50] for c2 in centers[:50] if not np.array_equal(c1, c2)])
    sigma = avg_dist / np.sqrt(2 * N_CLUSTERS)
    
    print(f"Calculated Kernel width (sigma): {sigma:.4f}")

    # 4. Transform Data to RBF Space
    print("Transforming input data into high-dimensional RBF space")
    X_rbf = rbf_kernel(X, centers, sigma)

    # 5. Train-Validation-Test Split (70-15-15)
    print("Splitting data into 70% train, 15% val, 15% test sets...")
    # First split: 70% train, 30% remainder
    X_train, X_rem, y_train, y_rem = train_test_split(
        X_rbf, y, train_size=0.7, random_state=SEED, stratify=y
    )
    # Second split: split remainder into 50/50 (15% / 15% of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rem, y_rem, test_size=0.5, random_state=SEED, stratify=y_rem
    )

    # 6. Train Linear Model (Output Layer)
    print("Training the output layer (Logistic Regression)...")
    clf = LogisticRegression(max_iter=2000, solver='lbfgs')
    clf.fit(X_train, y_train)

    # 7. Evaluate
    print("\n=== Model Evaluation (High Capacity RBF) ===")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Test Accuracy: {acc:.4%}")
    print(f"F1-Score:      {f1:.4f}")
    
    print("\nClassification Report:")
    target_names = ["Regular", "Medium", "Premium"]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 8. Save Artifacts to /outputs
    print(f"\nSaving predictions to {OUTPUT_LABELS}...")
    all_preds = clf.predict(X_rbf)
    pd.DataFrame({"rbf_label": all_preds}).to_csv(OUTPUT_LABELS, index=False)

    print(f"Saving confusion matrix plot to {OUTPUT_CM_PLOT}...")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap='Blues', ax=ax)
    plt.title(f"RBF Classifier Confusion Matrix (N={N_CLUSTERS})")
    plt.savefig(OUTPUT_CM_PLOT)
    plt.close()

    # --- Final Report for Assignment ---
    report_text = f"""
        ========================================
        FINAL RBF ASSIGNMENT REPORT DATA:
        - Number of Hidden Units: {N_CLUSTERS}
        - Kernel Width (sigma):   {sigma:.4f}
        - Final Test Accuracy:    {acc:.4%}
        - F1-Score (Macro):      {f1:.4f}
        ========================================
        """
    print(report_text)
    
    with open(OUTPUT_REPORT, "w") as f:
        f.write(report_text)
    print(f"Summary report saved to {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()
