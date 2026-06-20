# Step 3: Gap Analysis & Imputation
**Analysis window:** 2015-01-01 → 2025-12-31
**Frequency:** Weekly
**Bias metric:** `simple_mean_stance` (simple mean of daily stance scores)
**Companies:** 16 (ABNB, AMZN, T, BA, BAC, GM, GS, INTC, MCD, MSFT, MS, SBUX, UBER, V, WMT, WFC)

---
## 3.1 — Gap Classification

| Ticker | Total Weeks | Weeks w/ Data | Missing | % Covered | Gaps A (1-2w) | Gaps B (3-8w) | Gaps C (>8w) | Longest Gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ABNB | 574 | 394 | 180 | 68.6% | 84 | 18 | 0 | 7 |
| AMZN | 574 | 559 | 15 | 97.4% | 11 | 1 | 0 | 4 |
| T | 574 | 566 | 8 | 98.6% | 1 | 1 | 0 | 7 |
| BA | 574 | 547 | 27 | 95.3% | 27 | 0 | 0 | 1 |
| BAC | 574 | 408 | 166 | 71.1% | 104 | 10 | 0 | 5 |
| GM | 574 | 551 | 23 | 96.0% | 20 | 0 | 0 | 2 |
| GS | 574 | 543 | 31 | 94.6% | 29 | 0 | 0 | 2 |
| INTC | 574 | 533 | 41 | 92.9% | 34 | 2 | 0 | 3 |
| MCD | 574 | 458 | 116 | 79.8% | 82 | 4 | 0 | 6 |
| MSFT | 574 | 567 | 7 | 98.8% | 7 | 0 | 0 | 1 |
| MS | 574 | 547 | 27 | 95.3% | 27 | 0 | 0 | 1 |
| SBUX | 574 | 419 | 155 | 73.0% | 89 | 12 | 0 | 7 |
| UBER | 574 | 511 | 63 | 89.0% | 41 | 2 | 0 | 8 |
| V | 574 | 539 | 35 | 93.9% | 25 | 1 | 0 | 7 |
| WMT | 574 | 504 | 70 | 87.8% | 52 | 2 | 0 | 8 |
| WFC | 574 | 407 | 167 | 70.9% | 69 | 20 | 0 | 8 |

---
## 3.2 — Minimum Coverage Threshold

**Thresholds:** ≥60% weeks with data, no gap >8 weeks.

| Ticker | % Covered | Longest Gap | Passes | Reason |
|---|---:|---:|---|---|
| ABNB | 68.6% | 7 | ✅ | OK |
| AMZN | 97.4% | 4 | ✅ | OK |
| T | 98.6% | 7 | ✅ | OK |
| BA | 95.3% | 1 | ✅ | OK |
| BAC | 71.1% | 5 | ✅ | OK |
| GM | 96.0% | 2 | ✅ | OK |
| GS | 94.6% | 2 | ✅ | OK |
| INTC | 92.9% | 3 | ✅ | OK |
| MCD | 79.8% | 6 | ✅ | OK |
| MSFT | 98.8% | 1 | ✅ | OK |
| MS | 95.3% | 1 | ✅ | OK |
| SBUX | 73.0% | 7 | ✅ | OK |
| UBER | 89.0% | 8 | ✅ | OK |
| V | 93.9% | 7 | ✅ | OK |
| WMT | 87.8% | 8 | ✅ | OK |
| WFC | 70.9% | 8 | ✅ | OK |

> **16 companies pass**, **0 fail** the coverage threshold.

---
## 3.3 — Imputation Summary

| Method | Gap Type | Observations Imputed |
|---|---|---:|
| forward_fill | Type A | 822 |
| linear_interpolation | Type B | 309 |
| **Total** | — | **1,131** |

### Per-Company Imputation Counts

| Ticker | Imputed Weeks |
|---|---:|
| ABNB | 180 |
| AMZN | 15 |
| T | 8 |
| BA | 27 |
| BAC | 166 |
| GM | 23 |
| GS | 31 |
| INTC | 41 |
| MCD | 116 |
| MSFT | 7 |
| MS | 27 |
| SBUX | 155 |
| UBER | 63 |
| V | 35 |
| WMT | 70 |
| WFC | 167 |

> Forward-fill for Type A (1-2 week gaps). Linear interpolation for Type B (3-8 week gaps). Type C gaps are not imputed.

---
## 3.4 — Robustness Check

Two versions of the weekly bias series have been saved:
- `weekly_bias_imputed.csv` — with imputed values filled in
- `weekly_bias_raw_complete.csv` — observed-only rows (gaps dropped)

Run key downstream tests (stationarity, Granger causality) on **both** versions to confirm results are not driven by imputation.

All outputs in `/home/cloud/project/media-bias/VAR/diagnostics/step3/`