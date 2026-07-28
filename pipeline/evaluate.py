"""
evaluate.py — XGBoost tray-rating model: cross-validation and inference utilities.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (cohen_kappa_score, confusion_matrix,
                              mean_absolute_error)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import xgboost as xgb
import matplotlib.pyplot as plt

# Must match the columns produced by features.py
FEATURE_COLS  = ["mean_prob_bad", "n_beans"]

# Decision threshold for per-bean Bad/Good classification
BAD_THRESHOLD = 0.5


def evaluate_tray_model(df: pd.DataFrame) -> dict:
    """
    Evaluate the tray-rating XGBoost model via 5-fold cross-validation.

    Args:
        df: DataFrame produced by features.load_tray_table().
            Must contain FEATURE_COLS + 'human_rating' + 'year'.

    Returns:
        dict with keys: qwk, mae, confusion_matrix
    """
    X  = df[FEATURE_COLS].values
    y  = df["human_rating"].round().astype(int).clip(1, 5).values
    y0 = y - 1    # zero-indexed for XGBoost

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=42,
    )

    kf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred_cv = cross_val_predict(model, X, y0, cv=kf) + 1

    qwk = cohen_kappa_score(y, pred_cv, weights="quadratic")
    mae = mean_absolute_error(y, pred_cv)
    cm  = confusion_matrix(y, pred_cv, labels=[1, 2, 3, 4, 5])

    print(f"5-fold CV QWK = {qwk:.4f}")
    print(f"MAE           = {mae:.4f}")
    print(f"Confusion matrix:\n{cm}")

    return {"qwk": qwk, "mae": mae, "confusion_matrix": cm}


def flag_bad_beans(prob_bad: np.ndarray,
                   threshold: float = BAD_THRESHOLD) -> np.ndarray:
    """
    Classify individual beans as Bad (1) or Good (0) using a fixed threshold.

    Args:
        prob_bad:  Array of CNN bad-bean probabilities, shape (N,)
        threshold: Decision boundary applied to the softmax Bad-class output

    Returns:
        Binary int8 array, 1 = Bad, 0 = Good
    """
    return (prob_bad >= threshold).astype(np.int8)


def plot_confusion(cm: np.ndarray, title: str = "Confusion Matrix"):
    """Visualise a 5x5 confusion matrix for ratings 1–5."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([1, 2, 3, 4, 5])
    ax.set_yticklabels([1, 2, 3, 4, 5])
    ax.set_xlabel("Predicted rating")
    ax.set_ylabel("True rating")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Run evaluation on the synthetic sample data
    import sys
    sys.path.insert(0, ".")
    from pipeline.features import load_tray_table
    from pathlib import Path

    df = load_tray_table(Path("data/sample_beans"), Path("data/ratings.csv"))
    print(f"Evaluating on {len(df)} trays...\n")
    results = evaluate_tray_model(df)
