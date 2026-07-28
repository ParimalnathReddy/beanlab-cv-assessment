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

## Q2 — Prioritisation Under Time Pressure

You have **three working days** before a lab meeting where results will be
shown to the PI. Which **two** issues do you fix first, and why?

Defend your answer with an expected effect on the headline metric. Do not
simply list the most obviously wrong issues — argue for the highest expected
payoff per hour of engineering time.

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
