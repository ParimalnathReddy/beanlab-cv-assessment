"""
Generate synthetic sample data for CV_Assessment.

Creates:
  data/ratings.csv              — 60 trays with tray_id, year, human_rating
  data/sample_beans/<tray>_results.csv — per-bean prob_bad scores per tray

Rating distribution mirrors real BeanLab data:
  1 (very bad) ~8%,  2 (bad) ~15%,  3 (average) ~27%,
  4 (good) ~32%,  5 (excellent) ~18%

No real bean images or proprietary data are used.
Run once before using features.py or evaluate.py.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

OUT_DIR  = Path("data/sample_beans")
RATE_CSV = Path("data/ratings.csv")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Target rating distribution (matches approximate real-world proportions)
# Each tray gets a latent quality tier that drives both prob_bad scores and rating
TIER_CONFIG = [
    # (tier, weight, beta_a, beta_b, rating_range)
    ("very_bad",   0.08, 9, 2,  [1, 1, 2]),
    ("bad",        0.15, 6, 3,  [2, 2, 3]),
    ("average",    0.27, 3, 4,  [2, 3, 3, 4]),
    ("good",       0.32, 2, 7,  [3, 4, 4, 5]),
    ("excellent",  0.18, 1, 12, [4, 5, 5, 5]),
]

tray_records = []

years      = [2023, 2024, 2025]
trays_each = 20   # 20 trays per year = 60 total

for year in years:
    for i in range(1, trays_each + 1):
        tray_id = f"{year}_TRAY_{i:02d}"
        n_beans = int(RNG.integers(160, 320))

        # Pick a quality tier for this tray
        tier_weights = [t[1] for t in TIER_CONFIG]
        tier_idx = RNG.choice(len(TIER_CONFIG), p=tier_weights / np.sum(tier_weights))
        _, _, beta_a, beta_b, rating_pool = TIER_CONFIG[tier_idx]

        # Generate bimodal prob_bad: most beans near tier baseline,
        # a random fraction clearly defective
        defect_frac  = RNG.beta(beta_a, beta_b)
        n_defective  = int(n_beans * defect_frac)
        n_good       = n_beans - n_defective

        prob_bad_good = RNG.beta(1.2, 10,   size=max(n_good, 1))
        prob_bad_bad  = RNG.beta(8,   1.5,  size=max(n_defective, 0))
        prob_bad      = np.concatenate([prob_bad_good, prob_bad_bad[:n_defective]])
        RNG.shuffle(prob_bad)
        prob_bad      = np.clip(prob_bad, 0.0, 1.0)

        # Human rating: drawn from tier's rating pool with small noise
        rating = int(RNG.choice(rating_pool))

        tray_records.append({
            "tray_id":      tray_id,
            "year":         year,
            "human_rating": rating,
        })

        bean_df = pd.DataFrame({
            "bean_id":  np.arange(len(prob_bad)),
            "prob_bad": np.round(prob_bad, 6),
        })
        bean_df.to_csv(OUT_DIR / f"{tray_id}_results.csv", index=False)

ratings_df = pd.DataFrame(tray_records)
ratings_df.to_csv(RATE_CSV, index=False)

print(f"Generated {len(tray_records)} trays ({trays_each} per year × {len(years)} years)")
print(f"  Per-bean CSVs → {OUT_DIR}/")
print(f"  Ratings table → {RATE_CSV}")
print()
print("Rating distribution:")
print(ratings_df["human_rating"].value_counts().sort_index()
      .rename("count").to_frame()
      .assign(pct=lambda d: (d["count"] / len(ratings_df) * 100).round(1)))
print()
print("Per-year breakdown:")
print(ratings_df.groupby("year")["human_rating"]
      .describe()[["count","mean","std","min","max"]].round(2))
