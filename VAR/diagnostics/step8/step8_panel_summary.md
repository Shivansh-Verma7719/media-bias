# Step 8: Panel VAR Granger Causality Summary

## Methodology
A **Pooled Panel VAR** was estimated covering 15 companies. To remove unobserved firm fixed effects, the **Helmert transformation** (forward mean-differencing) was applied. This approach preserves orthogonality between transformed variables and lagged regressors, avoiding the look-ahead bias inherent in standard mean-differencing.

- **Endogenous Variables**: $\Delta 	ext{Bias}$ and Weekly Log Returns.
- **Exogenous Controls**: Structural Break Dummy ($Post_t$), VIX, S&P 500 Weekly Log Returns, and Article Count ($N_{articles}$).
- **Lags**: $p=3$ (modal BIC choice).
- **Standard Errors**: HAC-robust (Newey-West, maxlags=3).

Significance: `***` $p<0.01$, `**` $p<0.05$, `*` $p<0.10$.

## Results

| Sample | N Obs | Bias → Returns ($p$-value) | Returns → Bias ($p$-value) |
|---|---:|---|---|
| Full | 7493 | 0.976 | 0.200 |
| Pre-break | 4168 | 0.245 | 0.801 |
| Post-break | 3265 | 0.617 | 0.187 |

## Key Takeaways
1. **Full Panel Analysis**: By pooling the cross-section, the test gains substantial statistical power.
2. **Pre vs Post Comparison**: If Bias → Returns is significant in the Post-break period but not the Pre-break period, it robustly confirms a structural shift in the influence of media bias on stock returns after 2020.
3. **Exogenous Factors Controlled**: Market-wide volatility (VIX) and returns (S&P 500) are explicitly partialed out.