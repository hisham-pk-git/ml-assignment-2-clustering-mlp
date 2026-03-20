import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# Configurations
SEED = 42
RAW_DATA = "data/GasProperties.csv"
STD_DATA = "outputs/StdGasProperties.csv"
OUTPUT_LABELS = "outputs/mlp_labels.csv"
OUTPUT_PLOTS = "outputs/mlp_training_curve.png"

def make_quality_classes(idx_series):
    """
    Same 33/33/33 quantile split as src/05_evaluate.py.
    Returns integer labels: 0=Regular, 1=Medium, 2=Premium.
    """
    q1 = idx_series.quantile(1/3)
    q2 = idx_series.quantile(2/3)

    def label(v):
        if v <= q1: return 0
        elif v <= q2: return 1
        else: return 2
    return idx_series.apply(label), (q1, q2)

def main():
    print("=== MLP Classification Starting ===")
    
    # 1. Load Data
    print(f"Loading data from {STD_DATA} and {RAW_DATA}...")
    X = pd.read_csv(STD_DATA)
    df_raw = pd.read_csv(RAW_DATA)
    
    # 2. Prepare Target Labels
    print("Generating quality labels using tertiles of 'Idx'...")
    y, (q1, q2) = make_quality_classes(df_raw["Idx"])
    print(f"Thresholds: Regular <= {q1:.4f}, Medium <= {q2:.4f}, Premium > {q2:.4f}")

    # 3. Train-Validation-Test Split (70-15-15)
    print("Splitting data into 70% train, 15% val, 15% test sets...")
    # First split: 70% train, 30% remainder
    X_train, X_rem, y_train, y_rem = train_test_split(
        X, y, train_size=0.7, random_state=SEED, stratify=y
    )
    # Second split: split remainder into 50/50 (15% / 15% of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rem, y_rem, test_size=0.5, random_state=SEED, stratify=y_rem
    )

    # 4. Initialize and Train MLP
    print("Training high-capacity MLP model with (256, 128, 64) layers...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-5,                        # Less regularization
        batch_size=128,                    # Smaller batches
        learning_rate='adaptive',          # Adaptive schedule
        learning_rate_init=0.001,
        max_iter=500,                      # More time to converge
        early_stopping=True,
        validation_fraction=0.15,          # Matches your 15% val set
        random_state=SEED,
        verbose=True
    )

    mlp.fit(X_train, y_train)

    # 5. Evaluate Performance
    print("\n=== Model Evaluation ===")
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4%}")

    print("\nClassification Report:")
    target_names = ["Regular", "Medium", "Premium"]
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print(f"Saving confusion matrix plot to outputs/mlp_confusion_matrix.png...")
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap='Blues', ax=ax)
    plt.title("MLP Classifier Confusion Matrix")
    plt.savefig("outputs/mlp_confusion_matrix.png")
    plt.close()

    # 6. Save Results
    print(f"\nSaving predictions to {OUTPUT_LABELS}...")
    all_preds = mlp.predict(X)
    results_df = pd.DataFrame({"mlp_label": all_preds})
    results_df.to_csv(OUTPUT_LABELS, index=False)

    # 8. Final Report for Assignment
    print("\n" + "="*40)
    print("ASSIGNMENT REPORT DATA (MLP):")
    print(f"- Topology:              {mlp.hidden_layer_sizes} neurons")
    print(f"- Activation Functions:  {mlp.activation}")
    print(f"- Optimizer:             {mlp.solver} (Adam)")
    print(f"- Initial Learning Rate: {mlp.learning_rate_init}")
    print(f"- Number of Epochs:      {mlp.n_iter_} (Stopped early)")
    print(f"- Batch Size:            {mlp.batch_size}")
    print(f"- Regularization:        alpha={mlp.alpha} (L2)")
    print(f"- Test Accuracy:         {acc:.4%}")
    # Extract F1-score (macro) from classification report
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"- F1-Score (Macro):      {f1:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
