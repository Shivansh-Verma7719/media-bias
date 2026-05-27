# Step 4: Returns Series Construction & Alignment
**Analysis window:** 2015-01-01 → 2025-12-31
**Frequency:** Weekly (ISO weeks, Monday-anchored)
**Bias metric:** `simple_mean_stance` (simple mean)
**Min aligned weeks for VAR:** 100

---
## 4.1 — Weekly Log Returns Quality

| Ticker | Weeks | First | Last | Mean | Std | Min | Max | Extreme (>±15%) | Short Weeks (<4d) |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| ABNB | 522 | 2015-11-30 | 2025-11-24 | 0.00265 | 0.03562 | -0.1443 | 0.1399 | 0 | 1 |
| AMZN | 522 | 2015-11-30 | 2025-11-24 | 0.00235 | 0.0323 | -0.1829 | 0.1138 | 1 | 1 |
| T | 522 | 2015-11-30 | 2025-11-24 | 0.00147 | 0.06861 | -0.4093 | 0.3331 | 18 | 1 |
| BA | 522 | 2015-11-30 | 2025-11-24 | -0.00046 | 0.03924 | -0.2849 | 0.1939 | 4 | 1 |
| BAC | 260 | 2020-12-07 | 2025-11-24 | -0.0009 | 0.06274 | -0.1815 | 0.1896 | 5 | 2 |
| GM | 522 | 2015-11-30 | 2025-11-24 | 0.00503 | 0.05559 | -0.2923 | 0.1868 | 6 | 1 |
| GS | 522 | 2015-11-30 | 2025-11-24 | 0.00298 | 0.03935 | -0.2026 | 0.1322 | 2 | 1 |
| INTC | 522 | 2015-11-30 | 2025-11-24 | 0.00224 | 0.03584 | -0.1571 | 0.1397 | 1 | 1 |
| MCD | 522 | 2015-11-30 | 2025-11-24 | 0.00251 | 0.04381 | -0.2056 | 0.216 | 5 | 1 |
| MSFT | 522 | 2015-11-30 | 2025-11-24 | 0.00209 | 0.04499 | -0.2197 | 0.194 | 6 | 1 |
| MS | 522 | 2015-11-30 | 2025-11-24 | 0.00335 | 0.04431 | -0.2955 | 0.2533 | 6 | 1 |
| SBUX | 522 | 2015-11-30 | 2025-11-24 | 0.00264 | 0.04417 | -0.1883 | 0.1934 | 3 | 1 |
| UBER | 522 | 2015-11-30 | 2025-11-24 | 0.0025 | 0.04123 | -0.2416 | 0.2012 | 4 | 1 |
| V | 522 | 2015-11-30 | 2025-11-24 | 0.00154 | 0.05526 | -0.2859 | 0.1954 | 6 | 1 |
| WMT | 522 | 2015-11-30 | 2025-11-24 | 0.00154 | 0.05526 | -0.2859 | 0.1954 | 6 | 1 |
| WFC | 522 | 2015-11-30 | 2025-11-24 | 0.00012 | 0.04667 | -0.2215 | 0.168 | 10 | 1 |

> ⚠️ **AMZN** has 1 extreme weekly returns (>±15%). Inspect for data errors vs. real events (e.g., COVID crash, earnings).
> ⚠️ **T** has 18 extreme weekly returns (>±15%). Inspect for data errors vs. real events (e.g., COVID crash, earnings).
> ⚠️ **BA** has 4 extreme weekly returns (>±15%). Inspect for data errors vs. real events (e.g., COVID crash, earnings).
> ⚠️ **BAC** has 5 extreme weekly returns (>±15%). Inspect for data errors vs. real events (e.g., COVID crash, earnings).
> ⚠️ **GM** has 6 extreme weekly returns (>±15%). Inspect for data errors vs. real events (e.g., COVID crash, earnings).
> ⚠️ **GS** has 2 extreme weekly returns (>±15%). Inspect for data errors vs. real events (e.g., COVID crash, earnings).
> ⚠️ **INTC** has 1 extreme weekly returns (>±15%). Inspect for data errors vs. real events (e.g., COVID crash, earnings).
> ⚠️ **MCD** has 5 extreme weekly returns (>±15%). Inspect for data errors vs. real events (e.g., COVID crash, earnings).
> ⚠️ **MSFT** has 6 extreme weekly returns (>±15%). Inspect for data errors vs. real events (e.g., COVID crash, earnings).
> ⚠️ **MS** has 6 extreme weekly returns (>±15%). Inspect for data errors vs. real events (e.g., COVID crash, earnings).
> ⚠️ **SBUX** has 3 extreme weekly returns (>±15%). Inspect for data errors vs. real events (e.g., COVID crash, earnings).
> ⚠️ **UBER** has 4 extreme weekly returns (>±15%). Inspect for data errors vs. real events (e.g., COVID crash, earnings).
> ⚠️ **V** has 6 extreme weekly returns (>±15%). Inspect for data errors vs. real events (e.g., COVID crash, earnings).
> ⚠️ **WMT** has 6 extreme weekly returns (>±15%). Inspect for data errors vs. real events (e.g., COVID crash, earnings).
> ⚠️ **WFC** has 10 extreme weekly returns (>±15%). Inspect for data errors vs. real events (e.g., COVID crash, earnings).

---
## 4.2 — Alignment: Bias ↔ Returns

| Ticker | Weeks (bias) | Weeks (returns) | Weeks (both) | Bias-only | Returns-only | Sufficient (≥100)? |
|---|---:|---:|---:|---:|---:|---|
| ABNB | 574 | 522 | 522 | 52 | 0 | ✅ |
| AMZN | 574 | 522 | 522 | 52 | 0 | ✅ |
| T | 574 | 522 | 522 | 52 | 0 | ✅ |
| BA | 574 | 522 | 522 | 52 | 0 | ✅ |
| BAC | 574 | 260 | 260 | 314 | 0 | ✅ |
| GM | 574 | 522 | 522 | 52 | 0 | ✅ |
| GS | 574 | 522 | 522 | 52 | 0 | ✅ |
| INTC | 574 | 522 | 522 | 52 | 0 | ✅ |
| MCD | 574 | 522 | 522 | 52 | 0 | ✅ |
| MSFT | 574 | 522 | 522 | 52 | 0 | ✅ |
| MS | 574 | 522 | 522 | 52 | 0 | ✅ |
| SBUX | 574 | 522 | 522 | 52 | 0 | ✅ |
| UBER | 574 | 522 | 522 | 52 | 0 | ✅ |
| V | 574 | 522 | 522 | 52 | 0 | ✅ |
| WMT | 574 | 522 | 522 | 52 | 0 | ✅ |
| WFC | 574 | 522 | 522 | 52 | 0 | ✅ |

> **16 companies** have ≥100 aligned weeks. **0 companies** do not.
> Median aligned weeks: **522**

Final merged panel saved to `merged_weekly_panel.csv` (8,090 rows).

Plots saved to `/home/cloud/project/media-bias/VAR/diagnostics/step4/plots/`