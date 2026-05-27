# Step 6: Structural Break Testing Summary

## Empirical Break Dates
A Bai-Perron test (via Dynamic Programming, exactly 1 change point) was applied to the weekly bias index and weekly return series for each company. The algorithm identifies the single largest structural shift in the mean of the series (L2 cost).

| Ticker | Bias Break Date | Return Break Date | Pre-Break Obs | Post-Break Obs | Valid Split (>=60)? |
|---|---|---|---:|---:|---|
| ABNB | 2022-05-23 | 2021-09-06 | 338 | 184 | ✅ YES |
| AMZN | 2018-02-05 | 2022-01-03 | 114 | 408 | ✅ YES |
| T | 2022-05-02 | 2021-09-27 | 335 | 187 | ✅ YES |
| BA | 2024-01-08 | 2022-01-03 | 423 | 99 | ✅ YES |
| BAC | 2022-04-25 | 2022-12-26 | 72 | 188 | ✅ YES |
| GM | 2022-10-10 | 2017-10-30 | 358 | 164 | ✅ YES |
| GS | 2019-05-13 | 2021-11-22 | 180 | 342 | ✅ YES |
| INTC | 2024-03-25 | 2023-03-20 | 434 | 88 | ✅ YES |
| MCD | 2022-03-28 | 2023-10-30 | 330 | 192 | ✅ YES |
| MSFT | 2023-06-12 | 2021-09-27 | 393 | 129 | ✅ YES |
| MS | 2023-12-11 | 2023-10-30 | 419 | 103 | ✅ YES |
| SBUX | 2021-11-08 | 2022-10-03 | 310 | 212 | ✅ YES |
| UBER | 2018-04-02 | 2024-04-01 | 122 | 400 | ✅ YES |
| V | 2020-09-21 | 2020-03-23 | 251 | 271 | ✅ YES |
| WMT | 2024-07-22 | 2020-03-23 | 451 | 71 | ✅ YES |
| WFC | 2019-01-21 | 2021-09-06 | 164 | 358 | ✅ YES |

## Key Takeaways
1. **Bias Break Consistency**: Observe whether the empirical break dates for the bias index cluster around Q1 2020 (the COVID-19 shock) or vary by company.
2. **Returns Break**: Returns breaks are often noisier and harder to pinpoint exactly, but large volatility clusters (like March 2020) usually drive the algorithmic choice.
3. **Subsample Split Strategy**: The primary break date used for the `is_post_break` indicator in the panel is the **Bias Break Date**. This creates our Pre-period and Post-period for the subsequent Subsample Granger Causality analysis.
4. **Minimum Observations**: Any company failing the valid split criterion (fewer than 60 observations in either the pre or post period) should be interpreted with caution or excluded from split-sample VAR estimation.

Plots highlighting the identified breaks vs. the expected March 2020 shock are saved in `/home/cloud/project/media-bias/VAR/diagnostics/step6/plots/`.