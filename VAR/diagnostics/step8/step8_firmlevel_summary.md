# Step 8: Granger Causality Summary (Firm-level)

## Methodology
Granger Causality was tested using a direct OLS framework with **HAC-robust standard errors** (Newey-West kernel, `maxlags = p`). This controls for heteroskedasticity and autocorrelation, which are common in financial returns.

Significance: `***` $p<0.01$, `**` $p<0.05$, `*` $p<0.10$.

## Full Sample Results

| Company | Lag | N Obs | Bias → Returns (p-value) | Returns → Bias (p-value) |
|---|---:|---:|---|---|
| ABNB | 3 | 522 | 0.362 | 0.892 |
| AMZN | 2 | 522 | 0.944 | 0.277 |
| T | 6 | 522 | 0.570 | 0.367 |
| BA | 5 | 522 | 0.097* | 0.119 |
| BAC | 1 | 260 | 0.945 | 0.550 |
| GM | 4 | 522 | 0.629 | 0.543 |
| GS | 3 | 522 | 0.432 | 0.004*** |
| INTC | 3 | 522 | 0.611 | 0.363 |
| MCD | 3 | 522 | 0.276 | 0.518 |
| MSFT | 3 | 522 | 0.537 | 0.847 |
| MS | 4 | 522 | 0.389 | 0.959 |
| SBUX | 3 | 522 | 0.419 | 0.207 |
| UBER | 6 | 522 | 0.003*** | 0.375 |
| V | 3 | 522 | 0.281 | 0.860 |
| WFC | 1 | 522 | 0.303 | 0.148 |

## Pre vs Post Subsample Granger Comparison
The sample was split around the empirically identified structural break date for the bias index (Step 6).

| Company | Pre-period (Bias → Returns) | Pre-period (Returns → Bias) | Post-period (Bias → Returns) | Post-period (Returns → Bias) |
|---|---|---|---|---|
| ABNB | p = 0.165 | p = 0.884 | p = 0.416 | p = 0.936 |
| AMZN | p = 0.228 | p = 0.795 | p = 0.772 | p = 0.137 |
| T | p = 0.525 | p = 0.670 | p = 0.484 | p = 0.015** |
| BA | p = 0.091* | p = 0.247 | p = 0.258 | p = 0.010** |
| BAC | p = 0.578 | p = 0.407 | p = 0.778 | p = 0.849 |
| GM | p = 0.805 | p = 0.515 | p = 0.498 | p = 0.519 |
| GS | p = 0.397 | p = 0.026** | p = 0.919 | p = 0.210 |
| INTC | p = 0.668 | p = 0.535 | p = 0.074* | p = 0.157 |
| MCD | p = 0.178 | p = 0.456 | p = 0.920 | p = 0.461 |
| MSFT | p = 0.157 | p = 0.060* | p = 0.981 | p = 0.721 |
| MS | p = 0.019** | p = 0.514 | p = 0.462 | p = 0.938 |
| SBUX | p = 0.187 | p = 0.286 | p = 0.682 | p = 0.542 |
| UBER | p = 0.478 | p = 0.270 | p = 0.000*** | p = 0.131 |
| V | p = 0.253 | p = 0.871 | p = 0.854 | p = 0.268 |
| WFC | p = 0.593 | p = 0.497 | p = 0.138 | p = 0.030** |

## Key Takeaways
1. **Unidirectional Causality (Bias → Returns)**: This is the primary hypothesis. Check if significance emerges strongly in the Post-period.
2. **Bidirectional Causality**: Evidence of feedback loops between price discovery and media coverage.
3. **Structural Shift**: If pre-period results are insignificant but post-period results are significant, this validates the DiD result showing a structural break in the relationship.