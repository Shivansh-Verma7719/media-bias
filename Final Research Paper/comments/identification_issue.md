# RQ1/RQ2 identification problem — must be resolved before submission

**Author:** Sashwat (with Claude-assisted verification, 2026-07-08)
**Status:** OPEN — blocks arXiv/COMPASS submission
**Affects:** RQ1 and RQ2 specifications and results, in both v1 and the new v2
(`v2/final-paper.tex` Eq. 2 and Eq. 3). RQ3/VAR is unaffected.

---

## The problem in one paragraph

`Post_t = 1[t >= 2020-01-01]` is a deterministic function of the calendar date. Both
regressions include time fixed effects — daily for RQ1, monthly for RQ2 — and Post
switches exactly at a day/month boundary, so **Post is perfectly collinear with the
time effects**. In a correctly implemented two-way fixed-effects regression the Post
coefficient is not estimable at all; any β we report for it is an artifact of the
implementation. A referee who knows panel econometrics will see "Post + time fixed
effects" in the equations and reject the design on sight.

## Why we got numbers anyway (the artifact)

`scripts/run_rq_todos.py :: _within_transform` does **single-pass** double-demeaning
(entity means, then time means) on an **unbalanced** panel. For a date-only regressor
this leaves behind a firm-level constant — each firm's share of post-period
observations — instead of exactly zero.

Numerical proof (run on `results/rq1/rq1_daily_panel.csv`):

- within-firm std of the transformed Post regressor: **~4e-17 (machine zero)**
- cross-firm spread of the leftover constants: **−0.30 to +0.40**
- overall std the regression used: 0.076 — i.e. **100% of the variation the
  estimator used is cross-firm**, none of it is the within-firm pre/post change.

So the reported RQ1 β (+0.01096, p=0.712) and RQ2 β (−0.00013, p=0.921) are
coefficients on *firms' post-period coverage share* (e.g. ABNB and BAC start late so
they have high post shares) — not on the within-firm shift our hypotheses state.

Two further problems found while tracing numbers:

1. **Misdescribed estimator (v1):** the paper said RQ2 uses monthly time effects,
   but the two-way run in `run_rq_todos.py` uses `time_col="date"` (daily FE). The
   new v2 numbers come from `relevant_only_rerun.json: rq2_month_relevant`
   (S&P beta = 1.10 is now sensible, so the month-FE run is real) — but note its
   **n = 20,537**, while STORY.md and v1 claim n = 62,667. Reconcile.
2. **Untraceable robustness claims (v1, don't port to v2):** the joint day-FE F-test
   "F = 1.05, p = 0.019" (the only recorded test is F = 1.135, p ≈ 1e-8, old panel),
   the τ ∈ {0, 0.6, 0.9} sensitivity, and the log-volume check on the relevant panel
   have **no artifacts in the repo**. Regenerate on the VM or drop them.

## The contradiction we must confront

The one **identified** specification that already exists — RQ2 with firm FE plus the
S&P 500 return and VIX, *no time FE* (`results/rq2/rq2_summary.json`) — gives:

> Post β = −0.000344, HC1 SE = 0.000172, **p = 0.045**, n = 62,667

i.e. a *marginal rejection* of the null (−3.4bp/day post-2020). The "clean null"
headline currently depends on the degenerate specification. Whatever we decide, we
cannot ship the null while an identified spec in our own results folder rejects it.

## Fix options (pick one, team + prof decision)

**Option A — drop the time FE from the Post regressions (smallest change).**
- RQ1: firm FE + Post + volume, HC1 + day-clustered SE. (Needs a rerun on the
  relevant panel; the old full-panel run `rq1_summary.json` has this shape.)
- RQ2: report the existing identified run (β = −0.00034, p = 0.045) honestly.
- Narrative shifts to: "no stance shift; weak evidence of a small negative return
  shift." Day-clustered SEs may well push RQ2 back over 0.05 — run and see.

**Option B — keep time FE, drop Post, reframe as an event study.**
- Replace the single Post coefficient with monthly event-study coefficients around
  January 2020 (reference period = 2019), plot with CIs. Identified, more
  informative, and closer to what the professor's "did it shift?" question wants.
- More work: new estimation + a figure + rewritten hypothesis framing.

**Option C — change the estimand to heterogeneity.**
- Interact Post with firm characteristics (e.g. pre-period coverage intensity), so
  the coefficient is identified *within* time periods. This changes the research
  question and needs prof sign-off.

**Recommendation:** A for speed, B if we have a week. Either way, rerun on the VM
against the relevance-filtered panels, save the JSONs into `results/`, and update
Eq. 2/Eq. 3 and the results tables in `v2/final-paper.tex` to match.

## What is NOT affected

- The VAR/Granger pipeline (RQ3): break-dated, stationarity-checked, HAC — verified
  end to end against `VAR/diagnostics/`, all numbers trace.
- The relevance classifier and stance construction sections.
- Data/corpus numbers (per STORY.md DB ground truth).
