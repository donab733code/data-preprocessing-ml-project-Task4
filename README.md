# 🏡 Boston Housing Classification (Logistic Regression)

This project applies **Logistic Regression** to classify Boston housing prices into:
- Low Price
- High Price

The original regression target (`MEDV`) is converted into a binary classification problem using the median value.

---

## 📊 Dataset

- Source: Boston Housing dataset (preprocessed version)
- Target: `MEDV` (Median home value)
- Converted to:
  - `0` → Low Price
  - `1` → High Price

---

## ⚙️ Workflow

1. Load dataset
2. Create binary target
3. Train-test split (stratified)
4. Feature scaling (StandardScaler)
5. Logistic Regression model
6. Hyperparameter tuning (GridSearchCV)
7. Cross-validation (Stratified K-Fold)
8. Threshold optimization (F1-based)
9. Evaluation (Accuracy, Precision, Recall, ROC-AUC)
10. Feature importance analysis

---

## 📈 Model Performance

- ROC-AUC: ~0.90
- Accuracy: ~0.85
- Balanced classification between classes

---

## 🔥 Key Features

- Pipeline-based ML workflow (no data leakage)
- Hyperparameter tuning using GridSearchCV
- Automatic threshold optimization
- ROC curve analysis
- Feature importance visualization

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt