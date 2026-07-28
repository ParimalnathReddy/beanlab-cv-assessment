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

## Q2 — Cross-Year Generalisation

Suppose the pipeline is retrained with all the bugs from Q1 fixed.
It now achieves **QWK = 0.65** when evaluated within a single growing season,
but drops to **QWK = 0.45** when the model trained on 2023–2024 data is tested
on the following year (2025). The human-human inter-rater QWK baseline is 0.49,
so the model is currently *below* human agreement on new-year data.

Additional context:
- A year classifier trained on raw image statistics (mean hue, saturation,
  brightness) achieves **AUC = 0.9993** — the three years are almost perfectly
  separable from pixel statistics alone.
- A Reinhard LAB colour transfer was applied to 2025 tray photos to match the
  2023–2024 colour distribution. It moved the year-classifier AUC from 0.9993
  to 0.9887 — essentially no improvement in downstream QWK.
- The rater pool also changed between years (different field teams), introducing
  possible label drift on top of image shift.

**Propose two concrete approaches** you would investigate to close the
cross-year QWK gap. For each approach:

- Describe what you would implement (be specific enough that a labmate could
  reproduce it)
- State what metric or diagnostic you would use to know whether it worked
- Identify the main risk or failure mode — what could make it not work or
  make the results misleading

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
