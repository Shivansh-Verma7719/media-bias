# MLR Assumption Checks

## Summary

| Metric | Value |
|---|---:|
| Observations | 63200 |
| Regressors | 2 |
| Time fixed effects | True |
| Dropped regressors | None |
| Residual mean | -6.745658883798419e-19 |
| Residual std | 0.2783023688910749 |
| Within R2 | 0.0017487316356180616 |
| Condition number | 157.45556336183196 |
| Max pairwise corr | 0.018888851685436768 |
| Durbin-Watson | 1.673398390076599 |

## Breusch-Pagan

| Metric | Value |
|---|---:|
| LM | 190.24973882749202 |
| df | 2 |
| p value | 4.872974310306326e-42 |

## Jarque-Bera

| Metric | Value |
|---|---:|
| JB | 4541.181103270138 |
| p value | 0.0 |
| Skew | -0.1983636522747903 |
| Kurtosis | 4.251841283456247 |

## VIF

| Variable | VIF |
|---|---:|
| post | 1.0003569160616184 |
| article_volume | 1.0003569160616184 |

## Notes

- Durbin-Watson is approximate for panel data; interpret cautiously.
- Breusch-Pagan and Jarque-Bera p-values require scipy if available.
