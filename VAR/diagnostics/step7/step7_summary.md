# Step 7: VAR Specification Summary

## Lag Order Selection
Models were estimated up to a maximum of 8 lags. The optimal lag was chosen using the **Bayesian Information Criterion (BIC)** for parsimony. If BIC selected 0 lags, 1 lag was used to force a dynamic model.

| Ticker | AIC Lag | BIC Lag | Chosen Lag | N Obs |
|---|---:|---:|---:|---:|
| ABNB | 5 | 3 | 3 | 521 |
| AMZN | 7 | 2 | 2 | 521 |
| T | 7 | 6 | 6 | 521 |
| BA | 8 | 5 | 5 | 521 |
| BAC | 5 | 1 | 1 | 259 |
| GM | 8 | 4 | 4 | 521 |
| GS | 7 | 3 | 3 | 521 |
| INTC | 8 | 3 | 3 | 521 |
| MCD | 8 | 3 | 3 | 521 |
| MSFT | 8 | 3 | 3 | 521 |
| MS | 8 | 4 | 4 | 521 |
| SBUX | 6 | 3 | 3 | 521 |
| UBER | 8 | 6 | 6 | 521 |
| V | 7 | 3 | 3 | 521 |
| WFC | 8 | 1 | 1 | 521 |

## Model Diagnostics
Using the chosen lag, we performed tests on the VAR residuals.

| Ticker | Stable? | Min Root | Portmanteau $p$ | JB Normality $p$ | ARCH (Bias) $p$ | ARCH (Ret) $p$ |
|---|---|---:|---:|---:|---:|---:|
| ABNB | ✅ | 1.62 | 0.0356 | 0.0000 | 0.0097 | 0.0679 |
| AMZN | ✅ | 1.93 | 0.0002 | 0.0000 | 0.0202 | 0.0000 |
| T | ✅ | 1.24 | 0.2131 | 0.0000 | 0.2725 | 0.7046 |
| BA | ✅ | 1.29 | 0.2558 | 0.0000 | 0.0120 | 0.0000 |
| BAC | ✅ | 3.09 | 0.2267 | 0.0000 | 0.8870 | 0.2809 |
| GM | ✅ | 1.39 | 0.0416 | 0.0000 | 0.0662 | 0.0004 |
| GS | ✅ | 1.48 | 0.0038 | 0.0000 | 0.7429 | 0.0003 |
| INTC | ✅ | 1.58 | 0.0139 | 0.0000 | 0.0363 | 0.0000 |
| MCD | ✅ | 1.56 | 0.0016 | 0.0000 | 0.5908 | 0.0000 |
| MSFT | ✅ | 1.53 | 0.0114 | 0.0000 | 0.0134 | 0.2055 |
| MS | ✅ | 1.42 | 0.0024 | 0.0000 | 0.0645 | 0.0000 |
| SBUX | ✅ | 1.55 | 0.0475 | 0.0000 | 0.7787 | 0.0000 |
| UBER | ✅ | 1.19 | 0.0042 | 0.0000 | 0.0001 | 0.0000 |
| V | ✅ | 1.56 | 0.0033 | 0.0000 | 0.0032 | 0.0000 |
| WFC | ✅ | 2.57 | 0.0003 | 0.0000 | 0.0410 | 0.0000 |

## Key Takeaways
1. **Stability**: All selected models should be stable (min root > 1). If not, the model is explosive.
2. **Autocorrelation (Portmanteau)**: $p > 0.05$ indicates no remaining serial correlation in the residuals, meaning the chosen lag is sufficient.
3. **Normality (Jarque-Bera)**: Financial returns often exhibit heavy tails, leading to $p < 0.05$ (non-normality). VAR estimates remain consistent, but small-sample inference might be affected.
4. **Heteroskedasticity (ARCH)**: $p < 0.05$ indicates volatility clustering. If present, Granger causality tests might benefit from robust standard errors (though baseline VAR tests often ignore this).

> Note: **WMT** was excluded from this specification pass as requested.