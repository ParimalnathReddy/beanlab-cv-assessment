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

**After listing all issues**, answer this one additional question:

> Issue 7 (missing `hue=` in `ColorJitter`) and Issue 8 (`BAD_THRESHOLD = 0.5`
> hardcoded) interact with each other. Explain how fixing Issue 7 alone —
> without fixing Issue 8 — can make the downstream tray-rating pipeline
> *worse*, not better. Be specific about which direction the error shifts and why.

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

First, run the evaluation script from the repo root and paste the exact
terminal output into your answer:

```bash
python3 pipeline/evaluate.py
```

Then, looking specifically at `evaluate.py`, explain why the reported QWK
may be misleading for a multi-year field dataset. Design an alternative
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

For **Problem 1** only: implement the fix directly in the repo and paste
the `git diff` output into your answer.

---

## Q7 — LLM-Assisted Lab Query System

The lab accumulates data every growing season: tray photos, per-bean CNN
scores, human ratings, segmentation outputs, and field notes written by the
field team (free text, e.g. *"camera was misaligned on rows 3–5"*,
*"new variety introduced mid-season"*).

A researcher wants to query this data in plain English without writing SQL.
Using the existing files in this repo (`data/ratings.csv` and
`data/sample_beans/`) as your starting point:

1. **Schema** — Design the database schema that would store all of the above
   (tray metadata, per-bean scores, human ratings, field notes). Specify table
   names, columns, and data types. Justify any indexing decisions.

2. **Query pipeline** — Describe how you would build a system that takes a
   plain-English question from a researcher and returns the correct rows.
   For example:
   - *"Which trays from 2024 had a human rating ≥ 4 but the model predicted ≤ 2?"*
   - *"Which field team had the most disagreements with the model this season?"*
   - *"Show me all trays where the field notes mention camera or lighting issues."*
   What components does it need? Where does an LLM fit in, and where does it not?

3. **Risk** — What is the most likely way this system gives a researcher a
   wrong answer without them realising it? How would you detect and prevent that?

---

## Q7 — LLM-Assisted Lab Query System

The lab accumulates data every growing season: tray photos, per-bean CNN
scores, human ratings, segmentation outputs, and field notes written by the
field team (free text, e.g. *"camera was misaligned on rows 3–5"*,
*"new variety introduced mid-season"*).

A researcher wants to query this data in plain English without writing SQL.
Examples:
- *"Which trays from 2024 had a human rating ≥ 4 but the model predicted ≤ 2?"*
- *"Which field team had the most disagreements with the model this season?"*
- *"Show me all trays where the field notes mention camera or lighting issues."*

Using the existing data files in this repo (`data/ratings.csv` and
`data/sample_beans/`) as your starting point:

1. **Schema** — Design the database schema that would store all of the above
   (tray metadata, per-bean scores, human ratings, field notes). Specify table
   names, columns, and data types. Justify any indexing decisions.

2. **Query pipeline** — Describe how you would build a system that takes a
   plain-English question from a researcher and returns the correct rows.
   What components does it need? Where does an LLM fit in, and where does it not?

3. **Risk** — What is the most likely way this system gives a researcher a
   wrong answer without them realising it? How would you detect and prevent that?
