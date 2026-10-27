# Deep Learning Lab 1: Regression & Multi-Class Classification with PyTorch

## 📌 Objective
The objective of this lab is to design, train, tune, and evaluate Deep Neural Network (DNN / MLP) architectures using **PyTorch** for both **Regression** and **Multi-Class Classification** tasks, integrating `GridSearchCV` hyperparameter optimization, SMOTE data augmentation, and regularization techniques (L1, L2, Dropout).

---

## Tools & Libraries
* **Deep Learning:** PyTorch (`torch.nn`, `Dataset`, `DataLoader`, Kaiming & Xavier initialization)
* **Hyperparameter Tuning & Evaluation:** Scikit-Learn (`GridSearchCV`, `StandardScaler`, `classification_report`) & `Skorch`
* **Data Augmentation:** `imbalanced-learn` (`SMOTE`)
* **EDA & Visualization:** `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Environment:** Google Colab

---

## Part 1: Regression (NYSE Stock Prices Dataset)
* **Dataset:** [NYSE Dataset (Kaggle)](https://www.kaggle.com/datasets/dgawlik/nyse) (`851,264` rows, 7 features). Predicts `close` price from `open`, `low`, `high`, and `volume`.
* **Architecture:** 3-layer MLP (`4 -> 32 -> 16 -> 1`) with `ReLU` activations, Kaiming Uniform initialization on hidden layers, and Xavier Uniform initialization on the linear output layer.
* **Best Hyperparameters (`GridSearchCV`):**
  * Optimizer: `Adam` | Learning Rate: `0.001` | Epochs: `40` | Hidden Size: `32`
* **Results Comparison:**

| Model Variant | Regularization Applied | Final Test MSE | Final Test $R^2$ Score |
| :--- | :--- | :---: | :---: |
| **Baseline MLP** | None | **0.000509** | **0.9994 (99.94%)** |
| **Regularized MLP** | Dropout (`0.1`) + L2 (`1e-4`) + L1 (`1e-5`) | 0.008245 | 0.9906 (99.06%) |

* **Key Takeaway:** Because `open`, `low`, and `high` have a near-perfect linear correlation ($1.00$) with `close`[cite: 1], the baseline MLP converged within 2 epochs without overfitting[cite: 2]. Adding Dropout and L1/L2 penalties slightly constrained capacity on noise-free linear features.

---

## Part 2: Multi-Class Classification (Predictive Maintenance Dataset)
* **Dataset:** [Machine Predictive Maintenance Classification (Kaggle)](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) (`10,000` rows, 6 target failure classes).
* **Preprocessing & SMOTE Balancing:**
  * Dropped identifiers (`UDI`, `Product ID`), one-hot encoded product `Type`, label-encoded `Failure Type`, and standardized sensor features.
  * Used **SMOTE** to balance the severe class imbalance (`9,652` No Failure vs. `18` Random Failures), expanding the dataset to `57,912` balanced samples (`9,652` per class).
* **Architecture:** 3-layer MLP (`7 -> 128 -> 64 -> 6`) optimized with `CrossEntropyLoss`.
* **Best Hyperparameters (`GridSearchCV`):**
  * Optimizer: `Adam` | Learning Rate: `0.01` | Epochs: `40` | Hidden Size: `128` (CV Accuracy: `98.83%`)
* **Performance Metrics:**

| Dataset Split / Variant | Accuracy | Macro Sensitivity (Recall) | Macro Precision | Macro F1-Score | Cross-Entropy Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Train Set)** | 99.37% | 99.37% | 99.37% | 99.37% | 0.0261 |
| **Baseline (Test Set)** | **99.37%** | **99.38%** | **99.37%** | **99.37%** | **0.0346** |
| **Regularized (Test Set)** | 98.58% | 98.58% | 98.59% | 98.58% | 0.0564 |

---

## Synthesis: Key Learnings
1. **End-to-End PyTorch Workflow:** Building custom `Dataset` and `DataLoader` pipelines alongside modular `nn.Module` classes with proper weight initialization (`kaiming_uniform_` and `xavier_uniform_`) is essential for fast convergence.
2. **Importance of Data Balancing:** Without SMOTE, a classifier on the Predictive Maintenance dataset could achieve 96.5% accuracy by simply predicting `"No Failure"` every time while failing to detect actual breakdowns. Balancing the classes yielded 99%+ F1-score across all 6 failure modes.
3. **Hyperparameter Tuning & Regularization:** Combining PyTorch with `GridSearchCV` demonstrated the superiority of adaptive learning rate optimizers (`Adam`) over vanilla `SGD` for tabular MLPs. Furthermore, testing Dropout and L1/L2 regularization highlighted that regularization is most beneficial when a network suffers from high variance (overfitting).
