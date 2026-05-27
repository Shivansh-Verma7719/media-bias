# Step 3: Gap Analysis & Imputation
**Analysis window:** 2015-01-01 → 2025-12-31
**Frequency:** Weekly
**Bias metric:** `simple_mean_stance` (simple mean of daily stance scores)
**Companies:** 16 (ABNB, AMZN, T, BA, BAC, GM, GS, INTC, MCD, MSFT, MS, SBUX, UBER, V, WMT, WFC)

---
## 3.1 — Gap Classification

| Ticker | Total Weeks | Weeks w/ Data | Missing | % Covered | Gaps A (1-2w) | Gaps B (3-8w) | Gaps C (>8w) | Longest Gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ABNB | 574 | 572 | 2 | 99.7% | 1 | 0 | 0 | 2 |
| AMZN | 574 | 574 | 0 | 100.0% | 0 | 0 | 0 | 0 |
| T | 574 | 567 | 7 | 98.8% | 0 | 1 | 0 | 7 |
| BA | 574 | 574 | 0 | 100.0% | 0 | 0 | 0 | 0 |
| BAC | 574 | 572 | 2 | 99.7% | 2 | 0 | 0 | 1 |
| GM | 574 | 574 | 0 | 100.0% | 0 | 0 | 0 | 0 |
| GS | 574 | 573 | 1 | 99.8% | 1 | 0 | 0 | 1 |
| INTC | 574 | 574 | 0 | 100.0% | 0 | 0 | 0 | 0 |
| MCD | 574 | 574 | 0 | 100.0% | 0 | 0 | 0 | 0 |
| MSFT | 574 | 574 | 0 | 100.0% | 0 | 0 | 0 | 0 |
| MS | 574 | 574 | 0 | 100.0% | 0 | 0 | 0 | 0 |
| SBUX | 574 | 567 | 7 | 98.8% | 0 | 1 | 0 | 7 |
| UBER | 574 | 567 | 7 | 98.8% | 0 | 1 | 0 | 7 |
| V | 574 | 565 | 9 | 98.4% | 1 | 1 | 0 | 7 |
| WMT | 574 | 567 | 7 | 98.8% | 0 | 1 | 0 | 7 |
| WFC | 574 | 558 | 16 | 97.2% | 9 | 1 | 0 | 7 |

---
## 3.2 — Minimum Coverage Threshold

**Thresholds:** ≥60% weeks with data, no gap >8 weeks.

| Ticker | % Covered | Longest Gap | Passes | Reason |
|---|---:|---:|---|---|
| ABNB | 99.7% | 2 | ✅ | OK |
| AMZN | 100.0% | 0 | ✅ | OK |
| T | 98.8% | 7 | ✅ | OK |
| BA | 100.0% | 0 | ✅ | OK |
| BAC | 99.7% | 1 | ✅ | OK |
| GM | 100.0% | 0 | ✅ | OK |
| GS | 99.8% | 1 | ✅ | OK |
| INTC | 100.0% | 0 | ✅ | OK |
| MCD | 100.0% | 0 | ✅ | OK |
| MSFT | 100.0% | 0 | ✅ | OK |
| MS | 100.0% | 0 | ✅ | OK |
| SBUX | 98.8% | 7 | ✅ | OK |
| UBER | 98.8% | 7 | ✅ | OK |
| V | 98.4% | 7 | ✅ | OK |
| WMT | 98.8% | 7 | ✅ | OK |
| WFC | 97.2% | 7 | ✅ | OK |

> **16 companies pass**, **0 fail** the coverage threshold.

---
## 3.3 — Imputation Summary

| Method | Gap Type | Observations Imputed |
|---|---|---:|
| forward_fill | Type A | 16 |
| linear_interpolation | Type B | 42 |
| **Total** | — | **58** |

### Per-Company Imputation Counts

| Ticker | Imputed Weeks |
|---|---:|
| ABNB | 2 |
| T | 7 |
| BAC | 2 |
| GS | 1 |
| SBUX | 7 |
| UBER | 7 |
| V | 9 |
| WMT | 7 |
| WFC | 16 |

> Forward-fill for Type A (1-2 week gaps). Linear interpolation for Type B (3-8 week gaps). Type C gaps are not imputed.

---
## 3.4 — Robustness Check

Two versions of the weekly bias series have been saved:
- `weekly_bias_imputed.csv` — with imputed values filled in
- `weekly_bias_raw_complete.csv` — observed-only rows (gaps dropped)

Run key downstream tests (stationarity, Granger causality) on **both** versions to confirm results are not driven by imputation.

All outputs in `/home/cloud/project/media-bias/VAR/diagnostics/step3/`