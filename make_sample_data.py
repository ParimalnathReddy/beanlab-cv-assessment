"""
Generate synthetic sample data for CV_Assessment.

Creates:
  data/ratings.csv              — 30 trays with tray_id, year, human_rating
  data/sample_beans/<tray>_results.csv — per-bean prob_bad scores per tray

No real bean images or proprietary data are used.
Run once before using features.py or evaluate.py.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

OUT_DIR   = Path("data/sample_beans")
RATE_CSV  = Path("data/ratings.csv")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Tray catalogue ────────────────────────────────────────────────────────────
# 10 trays per year; 2025 has a slightly elevated prob_bad baseline
# (simulating the real covariate shift we observe year-to-year)

tray_records = []
year_base_bad = {2023: 0.10, 2024: 0.12, 2025: 0.19}

for year, base in year_base_bad.items():
    for i in range(1, 11):
        tray_id      = f"{year}_TRAY_{i:02d}"
        n_beans      = int(RNG.integers(150, 320))
        # Each tray has a latent "badness" level that drives both the
        # prob_bad distribution and the human rating
        latent       = RNG.beta(2, 8) + base           # 0..~0.6
        latent       = float(np.clip(latent, 0.0, 1.0))

        # Simulate a bimodal distribution: most beans are good,
        # a fraction are defective
        n_bad        = int(n_beans * latent * RNG.uniform(0.5, 1.5))
        n_bad        = min(n_bad, n_beans)
        n_good       = n_beans - n_bad

        prob_bad_good = RNG.beta(1.5, 12, size=n_good)          # low scores
        prob_bad_bad  = RNG.beta(5,   2,  size=n_bad)           # high scores
        prob_bad      = np.concatenate([prob_bad_good, prob_bad_bad])
        RNG.shuffle(prob_bad)
        prob_bad      = np.clip(prob_bad, 0.0, 1.0)

        # Human rating: strongly driven by the 95th-percentile bean score,
        # not the mean — this is intentional (see features.py)
        p95      = float(np.percentile(prob_bad, 95))
        raw_r    = 1.0 + 4.0 * p95 + RNG.normal(0, 0.4)
        rating   = int(np.clip(round(raw_r), 1, 5))

        tray_records.append({
            "tray_id":      tray_id,
            "year":         year,
            "human_rating": rating,
        })

        # Save per-bean CSV
        bean_df = pd.DataFrame({
            "bean_id":  np.arange(n_beans),
            "prob_bad": np.round(prob_bad, 6),
        })
        bean_df.to_csv(OUT_DIR / f"{tray_id}_results.csv", index=False)

ratings_df = pd.DataFrame(tray_records)
ratings_df.to_csv(RATE_CSV, index=False)

print(f"Generated {len(tray_records)} trays:")
print(f"  Per-bean CSVs  → {OUT_DIR}/")
print(f"  Ratings table  → {RATE_CSV}")
print()
print(ratings_df.groupby("year")["human_rating"].describe().round(2))
