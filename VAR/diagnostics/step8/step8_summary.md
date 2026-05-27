# Step 8: Granger Causality Summary

## Methodology
Granger Causality was tested using a direct OLS framework with **HAC-robust standard errors** (Newey-West kernel, `maxlags = p`). This controls for heteroskedasticity and autocorrelation, which are common in financial returns.

Significance: `***` $p<0.01$, `**` $p<0.05$, `*` $p<0.10$.

## Full Sample Results

| Company | Lag | N Obs | Bias → Returns (p-value) | Returns → Bias (p-value) |
|---|---:|---:|---|---|
| ABNB | 3 | 522 | 0.992 | 0.877 |
| AMZN | 2 | 522 | 0.732 | 0.759 |
| T | 4 | 522 | 0.947 | 0.345 |
| BA | 4 | 522 | 0.232 | 0.295 |
| BAC | 1 | 260 | 0.845 | 0.985 |
| GM | 3 | 522 | 0.218 | 0.765 |
| GS | 5 | 522 | 0.757 | 0.012** |
| INTC | 2 | 522 | 0.872 | 0.305 |
| MCD | 3 | 522 | 0.192 | 0.258 |
| MSFT | 3 | 522 | 0.680 | 0.134 |
| MS | 3 | 522 | 0.058* | 0.193 |
| SBUX | 2 | 522 | 0.971 | 0.379 |
| UBER | 3 | 522 | 0.750 | 0.192 |
| V | 3 | 522 | 0.981 | 0.174 |
| WFC | 5 | 522 | 0.023** | 0.307 |

## Pre vs Post Subsample Granger Comparison
The sample was split around the empirically identified structural break date for the bias index (Step 6).

| Company | Pre-period (Bias → Returns) | Pre-period (Returns → Bias) | Post-period (Bias → Returns) | Post-period (Returns → Bias) |
|---|---|---|---|---|
| ABNB | p = 0.667 | p = 0.690 | p = 0.629 | p = 0.623 |
| AMZN | p = 0.196 | p = 0.339 | p = 0.815 | p = 0.278 |
| T | p = 0.207 | p = 0.571 | p = 0.122 | p = 0.348 |
| BA | p = 0.196 | p = 0.775 | p = 0.543 | p = 0.014** |
| BAC | p = 0.836 | p = 0.862 | p = 0.676 | p = 0.778 |
| GM | p = 0.304 | p = 0.489 | p = 0.451 | p = 0.835 |
| GS | p = 0.699 | p = 0.833 | p = 0.902 | p = 0.007*** |
| INTC | p = 0.479 | p = 0.158 | p = 0.173 | p = 0.696 |
| MCD | p = 0.259 | p = 0.027** | p = 0.077* | p = 0.484 |
| MSFT | p = 0.137 | p = 0.234 | p = 0.309 | p = 0.477 |
| MS | p = 0.119 | p = 0.313 | p = 0.052* | p = 0.016** |
| SBUX | p = 0.328 | p = 0.582 | p = 0.243 | p = 0.414 |
| UBER | p = 0.489 | p = 0.980 | p = 0.856 | p = 0.190 |
| V | p = 0.873 | p = 0.157 | p = 0.827 | p = 0.629 |
| WFC | p = 0.345 | p = 0.247 | p = 0.013** | p = 0.636 |

## Key Takeaways
1. **Unidirectional Causality (Bias → Returns)**: This is the primary hypothesis. Check if significance emerges strongly in the Post-period.
2. **Bidirectional Causality**: Evidence of feedback loops between price discovery and media coverage.
3. **Structural Shift**: If pre-period results are insignificant but post-period results are significant, this validates the DiD result showing a structural break in the relationship.