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
is trained on data from seasons A and B, then evaluated blind on season C —
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

Explain why this number may be misleading for a multi-year field dataset.
Then design an alternative evaluation protocol that gives a more realistic
estimate of deployment performance. Specify exactly what you would compute
and what you would report.

---

## Q4 — Segmentation Validation Without Ground Truth

`segment_beans()` in `segment.py` produces instance counts used downstream.
You suspect it under-counts touching bean pairs but have **no pixel-level
annotation budget**.

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
- Identify whether it could become noisier or less stable under specific
  conditions (class imbalance, CNN miscalibration, very small or large trays)
