# Step 6: Structural Break Testing Summary

## Empirical Break Dates
A Bai-Perron test (via Dynamic Programming, exactly 1 change point) was applied to the weekly bias index and weekly return series for each company. The algorithm identifies the single largest structural shift in the mean of the series (L2 cost).

| Ticker | Bias Break Date | Return Break Date | Pre-Break Obs | Post-Break Obs | Valid Split (>=60)? |
|---|---|---|---:|---:|---|
| ABNB | 2023-02-27 | 2021-09-06 | 378 | 144 | ✅ YES |
| AMZN | 2018-10-22 | 2022-01-03 | 151 | 371 | ✅ YES |
| T | 2022-11-28 | 2021-09-27 | 365 | 157 | ✅ YES |
| BA | 2019-03-04 | 2022-01-03 | 170 | 352 | ✅ YES |
| BAC | 2022-05-02 | 2022-12-26 | 73 | 187 | ✅ YES |
| GM | 2020-01-13 | 2017-10-30 | 215 | 307 | ✅ YES |
| GS | 2024-01-15 | 2021-11-22 | 424 | 98 | ✅ YES |
| INTC | 2024-07-22 | 2023-03-20 | 451 | 71 | ✅ YES |
| MCD | 2020-06-29 | 2023-10-30 | 239 | 283 | ✅ YES |
| MSFT | 2022-05-30 | 2021-09-27 | 339 | 183 | ✅ YES |
| MS | 2017-07-03 | 2023-10-30 | 83 | 439 | ✅ YES |
| SBUX | 2022-02-07 | 2022-10-03 | 323 | 199 | ✅ YES |
| UBER | 2023-01-16 | 2024-04-01 | 372 | 150 | ✅ YES |
| V | 2024-06-10 | 2020-03-23 | 445 | 77 | ✅ YES |
| WMT | 2024-04-22 | 2020-03-23 | 438 | 84 | ✅ YES |
| WFC | 2024-10-07 | 2021-09-06 | 462 | 60 | ✅ YES |

## Key Takeaways
1. **Bias Break Consistency**: Observe whether the empirical break dates for the bias index cluster around Q1 2020 (the COVID-19 shock) or vary by company.
2. **Returns Break**: Returns breaks are often noisier and harder to pinpoint exactly, but large volatility clusters (like March 2020) usually drive the algorithmic choice.
3. **Subsample Split Strategy**: The primary break date used for the `is_post_break` indicator in the panel is the **Bias Break Date**. This creates our Pre-period and Post-period for the subsequent Subsample Granger Causality analysis.
4. **Minimum Observations**: Any company failing the valid split criterion (fewer than 60 observations in either the pre or post period) should be interpreted with caution or excluded from split-sample VAR estimation.

Plots highlighting the identified breaks vs. the expected March 2020 shock are saved in `/home/cloud/project/media-bias/VAR/diagnostics/step6/plots/`.