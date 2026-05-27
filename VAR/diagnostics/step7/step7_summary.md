# Step 7: VAR Specification Summary

## Lag Order Selection
Models were estimated up to a maximum of 8 lags. The optimal lag was chosen using the **Bayesian Information Criterion (BIC)** for parsimony. If BIC selected 0 lags, 1 lag was used to force a dynamic model.

| Ticker | AIC Lag | BIC Lag | Chosen Lag | N Obs |
|---|---:|---:|---:|---:|
| ABNB | 6 | 3 | 3 | 521 |
| AMZN | 7 | 2 | 2 | 521 |
| T | 6 | 4 | 4 | 521 |
| BA | 4 | 4 | 4 | 521 |
| BAC | 4 | 1 | 1 | 259 |
| GM | 7 | 3 | 3 | 521 |
| GS | 8 | 5 | 5 | 521 |
| INTC | 8 | 2 | 2 | 521 |
| MCD | 8 | 3 | 3 | 521 |
| MSFT | 8 | 3 | 3 | 521 |
| MS | 7 | 3 | 3 | 521 |
| SBUX | 8 | 2 | 2 | 521 |
| UBER | 8 | 3 | 3 | 521 |
| V | 8 | 3 | 3 | 521 |
| WFC | 6 | 5 | 5 | 521 |

## Model Diagnostics
Using the chosen lag, we performed tests on the VAR residuals.

| Ticker | Stable? | Min Root | Portmanteau $p$ | JB Normality $p$ | ARCH (Bias) $p$ | ARCH (Ret) $p$ |
|---|---|---:|---:|---:|---:|---:|
| ABNB | ✅ | 1.56 | N/A | N/A | 0.0264 | 0.0301 |
| AMZN | ✅ | 1.99 | N/A | N/A | 0.1811 | 0.0000 |
| T | ✅ | 1.40 | N/A | N/A | 0.3270 | 0.8143 |
| BA | ✅ | 1.37 | N/A | N/A | 0.0784 | 0.0000 |
| BAC | ✅ | 2.02 | N/A | N/A | 0.0640 | 0.2924 |
| GM | ✅ | 1.55 | N/A | N/A | 0.0000 | 0.0003 |
| GS | ✅ | 1.28 | N/A | N/A | 0.5783 | 0.0007 |
| INTC | ✅ | 1.80 | N/A | N/A | 0.1713 | 0.0000 |
| MCD | ✅ | 1.56 | N/A | N/A | 0.0002 | 0.0000 |
| MSFT | ✅ | 1.65 | N/A | N/A | 0.0582 | 0.2155 |
| MS | ✅ | 1.53 | N/A | N/A | 0.6324 | 0.0000 |
| SBUX | ✅ | 1.76 | N/A | N/A | 0.0378 | 0.0000 |
| UBER | ✅ | 1.53 | N/A | N/A | 0.0000 | 0.0000 |
| V | ✅ | 1.65 | N/A | N/A | 0.0002 | 0.0000 |
| WFC | ✅ | 1.28 | N/A | N/A | 0.0340 | 0.0000 |

## Key Takeaways
1. **Stability**: All selected models should be stable (min root > 1). If not, the model is explosive.
2. **Autocorrelation (Portmanteau)**: $p > 0.05$ indicates no remaining serial correlation in the residuals, meaning the chosen lag is sufficient.
3. **Normality (Jarque-Bera)**: Financial returns often exhibit heavy tails, leading to $p < 0.05$ (non-normality). VAR estimates remain consistent, but small-sample inference might be affected.
4. **Heteroskedasticity (ARCH)**: $p < 0.05$ indicates volatility clustering. If present, Granger causality tests might benefit from robust standard errors (though baseline VAR tests often ignore this).

> Note: **WMT** was excluded from this specification pass as requested.