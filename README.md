# Machine Learning Assignment: Clustering & MLP Classification

This repository contains the implementation of the clustering and classification portion of our second ML assignment.

---

## (a) MLP Classifier Report

**Objective**: Predict quality levels (Regular, Medium, Premium) from standardized gas properties.

*   **Topology**: 3 Hidden Layers with `(256, 128, 64)` neurons.
*   **Activation Function**: ReLU (Rectified Linear Unit).
*   **Optimizer**: Adam (Adaptive Moment Estimation).
*   **Initial Learning Rate**: 0.001.
*   **Learning Rate Schedule**: Adaptive (reduces upon plateau).
*   **Number of Epochs**: 53 (Early stopping triggered after 10 epochs of no improvement).
*   **Training Method**: Stochastic Gradient Descent with Batch size = `128`.
*   **Regularization**: L2 Regularization with `alpha = 1e-05`.

### Performance Evaluation (Test Set - 15%)
*   **Accuracy**: 82.4127%
*   **F1-Score (Macro)**: 0.8264
*   **Confusion Matrix**:
    ```
    [[18167  2833     0]   (True Regular)
     [ 1466 16722  2812]   (True Medium)
     [    0  3969 17031]]   (True Premium)
    ```

---

## (b) RBF Classifier Report

**Objective**: Train an RBF network using K-means centroids as the basis functions.

*   **Hidden Units**: 200 (Selected centers using K-Means with 200 clusters).
*   **Kernel Width ($\sigma$)**: 0.1224 (Calculated via spread heuristic).
*   **Output Layer**: Logistic Regression.

### Performance Evaluation (Test Set - 15%)
*   **Accuracy**: 75.1667%
*   **F1-Score (Macro)**: 0.7521
*   **Confusion Matrix**:
    ```
    [[18368  2632     0]
     [ 3624 13650  3726]
     [    0  4376 16624]]
    ```

---

## (c) Model Comparison

1.  **Which model was easier to tune?**  
    The **MLP** was significantly "easier" to tune in terms of achieving high accuracy quickly. It uses hierarchical feature extraction, and hyperparameters like layer size and batch size have well-understood effects. The **RBF** is highly sensitive to the initial selection of centers and the choice of the kernel width ($\sigma$).

2.  **Which model performed better?**  
    The **MLP** performed better overall (82.4% vs 75.2% accuracy). 

3.  **Why might their behaviors differ?**  
    *   **MLP (Global Learning)**: The MLP learns global decision boundaries by layering non-linear transformations. It can model complex relations across all features simultaneously.
    *   **RBF (Local Learning)**: The RBF classifier works based on proximity to "prototypes" (centers). If a point is far from all prototypes, or if the prototypes don't cover a specific region of the feature space densely enough, the model fails to categorize it accurately. For a dataset with 420,000 samples, even 200 prototypes cover a much smaller percentage of the data's diversity compared to the continuous decision surface learned by an MLP.

---

## Running the Code

1.  **Setup Environment**:
    ```bash
    python3 -m venv .
    source bin/activate
    pip install numpy pandas scikit-learn matplotlib
    ```
2.  **Run Models**:
    *   MLP: `python3 src/06_mlp_classification.py`
    *   RBF: `python3 src/07_rbf_classification.py`

Final outputs and plots are saved to the `outputs/` directory.