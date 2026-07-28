"""
features.py — Tray-level feature engineering from per-bean CNN scores.

Each tray produces one row of features that feeds the XGBoost rating model.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def compute_tray_features(prob_bad: np.ndarray) -> Optional[dict]:
    """
    Summarise per-bean prob_bad scores (values in [0, 1]) into a
    fixed-length feature vector for the tray-rating model.

    Args:
        prob_bad: 1-D array of CNN defect probabilities, one per bean.
                  A tray typically contains 150-350 beans.

    Returns:
        Feature dictionary, or None if the array is empty.
    """
    if len(prob_bad) == 0:
        return None

    return {
        "mean_prob_bad": float(np.mean(prob_bad)),
        "n_beans":       int(len(prob_bad)),
    }


def load_tray_table(csv_dir: Path, rating_csv: Path) -> pd.DataFrame:
    """
    Build the full tray-level feature table by joining per-tray CNN scores
    with human appearance ratings.

    Args:
        csv_dir:    Directory containing <tray_id>_results.csv files.
                    Each file has columns: bean_id, prob_bad
        rating_csv: CSV with columns: tray_id, year, human_rating

    Returns:
        DataFrame with one row per tray: features + tray_id + year + human_rating
    """
    ratings = pd.read_csv(rating_csv)
    rows    = []

    for _, row in ratings.iterrows():
        csv_path = csv_dir / f"{row['tray_id']}_results.csv"
        if not csv_path.exists():
            continue
        bean_df  = pd.read_csv(csv_path)
        prob_bad = bean_df["prob_bad"].dropna().values
        feats    = compute_tray_features(prob_bad)
        if feats is None:
            continue
        feats.update({
            "tray_id":      row["tray_id"],
            "year":         int(row["year"]),
            "human_rating": float(row["human_rating"]),
        })
        rows.append(feats)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick smoke-test using the synthetic sample data
    csv_dir    = Path("data/sample_beans")
    rating_csv = Path("data/ratings.csv")
    df = load_tray_table(csv_dir, rating_csv)
    print(f"Loaded {len(df)} trays")
    print(df.head())
    print(df.describe())
