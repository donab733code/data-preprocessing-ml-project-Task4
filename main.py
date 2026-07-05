import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("preprocessed_data.csv")

# Convert regression target to classification
median_price = df["MEDV"].median()
df["target"] = (df["MEDV"] >= median_price).astype(int)

X = df.drop(["MEDV", "target"], axis=1)
y = df["target"]

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# PIPELINE
# ============================================================

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        solver="lbfgs"
    ))
])

# ============================================================
# GRID SEARCH (FIXED - NO PENALTY WARNING)
# ============================================================

param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Params:", grid.best_params_)
print("Best CV ROC-AUC:", grid.best_score_)

# ============================================================
# PREDICTIONS
# ============================================================

y_prob = best_model.predict_proba(X_test)[:, 1]

# ============================================================
# OPTIMAL THRESHOLD (F1-BASED)
# ============================================================

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

f1 = (2 * precision * recall) / (precision + recall + 1e-9)
best_idx = np.argmax(f1)

best_threshold = thresholds[max(best_idx - 1, 0)]
y_pred = (y_prob >= best_threshold).astype(int)

print("\nOptimal Threshold:", best_threshold)

# ============================================================
# EVALUATION
# ============================================================

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Low Price", "High Price"],
    zero_division=0
))

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(
    cm,
    display_labels=["Low Price", "High Price"]
).plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.show()

# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],"--",color="red")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

model = best_model.named_steps["model"]

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\nFeature Importance:\n")
print(importance)

plt.figure(figsize=(8,6))
plt.barh(importance["Feature"], importance["Coefficient"])
plt.title("Logistic Regression Feature Importance")
plt.gca().invert_yaxis()
plt.show()

print("\n MODEL TRAINING COMPLETE")