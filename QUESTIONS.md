# Assessment Questions — A2025-03

Write your answers in `SOLUTION.md`. There is no length minimum or maximum —
calibrate to your findings.

---

## Q1 — Bug Report

List every issue you find across all five files in `pipeline/`.

For each issue provide:

- **(a)** File name and approximate line number
- **(b)** What observable symptom or metric outcome it produces — not just
  "it's wrong" but *what number is wrong and in what direction*
- **(c)** A concrete fix

There are between seven and nine issues total. Issues exist at the level of
design decisions, not syntax — the code runs without errors.

---

## Q2 — Diagnosing a Cross-Year Performance Drop

You have just joined the lab. The pipeline (assume all Q1 bugs are fixed)
is trained on data from 2023 and 2024, then evaluated blind on 2025 —
a new growing year with a different field team and a different camera.
Performance drops significantly below the human inter-rater agreement baseline.

You have access to:
- Tray photos from all three seasons
- Per-bean CNN scores (`prob_bad`) for each tray
- Human ratings for each tray
- The trained model weights

You do **not** yet know *why* performance dropped. It could be the images,
the labels, the beans themselves, or some combination.

**Describe your investigation and recovery plan:**

1. **Diagnosis** — Before touching any code, what would you measure or
   visualise to understand *where* the failure is coming from? Walk through
   your reasoning step by step.

2. **Experiment** — Based on what your diagnosis might reveal, propose
   **two different recovery strategies** — one for each of the two most
   plausible root causes you identified. For each: what you would implement,
   and what result would tell you it worked.

3. **Risk** — For each strategy, what is the main way it could produce
   misleading results even if the numbers improve?

---

## Q3 — Honest Evaluation Protocol

The pipeline reports **5-fold CV QWK = 0.62** as its headline result.

Looking specifically at `evaluate.py`, explain why this number may be
misleading for a multi-year field dataset. Then design an alternative
evaluation protocol that gives a more realistic estimate of deployment
performance. Specify exactly what you would compute and what you would report.

How does your proposed metric compare against the human-human inter-rater
agreement as a baseline, and why does that comparison matter?

---

## Q4 — Segmentation Validation Without Ground Truth

`segment_beans()` in `segment.py` produces instance counts used downstream.
You need to validate that it produces reliable output before trusting it,
but you have **no pixel-level annotation budget**.

Propose a practical validation strategy:
- What data would you collect?
- What would you measure?
- How would you decide whether the segmentation is good enough to trust?

---

## Q5 — Feature Reasoning

`features.py` computes two numbers per tray. Propose exactly **three
additional features** you would add.

For each feature:
- Give the formula (in Python or plain math)
- Explain what human-rater behaviour it is designed to capture
- Under what conditions might this feature become unreliable or misleading?

---

## Q6 — Pipeline Reproducibility

This pipeline will re-run every growing season as new data arrives.
A new lab member six months from now needs to reproduce last year's
results exactly — same split, same features, same model checkpoint.

Reading the five files in `pipeline/` and `requirements.txt`, identify
**three specific reproducibility or maintainability problems**.

For each problem:
- **(a)** File and line where the problem lives
- **(b)** What concretely breaks when the pipeline runs again on new-year data
- **(c)** A concrete fix — code or pseudocode is fine
