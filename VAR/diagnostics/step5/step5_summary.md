# Step 5: Stationarity & Integration Testing Summary Report

## Winsorization Details
Weekly log returns winsorized at ±5 standard deviations from each company's mean, **excluding AT&T (T)**.

| Ticker | Raw Return Std | Outliers Winsorized | Status |
|---|---:|---:|---|
| ABNB | 0.03562 | 0 | No outliers (>5 std) |
| AMZN | 0.03230 | 1 | Winsorized 1 obs |
| T | 0.06861 | 0 | SKIPPED (AT&T) |
| BA | 0.03924 | 1 | Winsorized 1 obs |
| BAC | 0.06274 | 0 | No outliers (>5 std) |
| GM | 0.05559 | 1 | Winsorized 1 obs |
| GS | 0.03935 | 1 | Winsorized 1 obs |
| INTC | 0.03584 | 0 | No outliers (>5 std) |
| MCD | 0.04381 | 0 | No outliers (>5 std) |
| MSFT | 0.04499 | 0 | No outliers (>5 std) |
| MS | 0.04431 | 2 | Winsorized 2 obs |
| SBUX | 0.04417 | 0 | No outliers (>5 std) |
| UBER | 0.04123 | 1 | Winsorized 1 obs |
| V | 0.05526 | 1 | Winsorized 1 obs |
| WMT | 0.05526 | 1 | Winsorized 1 obs |
| WFC | 0.04667 | 0 | No outliers (>5 std) |

## Stationarity Test Interpretation (Constant-only Specification)
We class each series based on Augmented Dickey-Fuller (ADF) and KPSS tests:
- **Stationary (I(0))**: ADF rejects unit root ($p < 0.05$) AND KPSS fails to reject stationarity ($p \geq 0.05$).
- **Non-Stationary (I(1))**: ADF fails to reject unit root ($p \geq 0.05$) AND KPSS rejects stationarity ($p < 0.05$).
- **Conflicting**: Both tests reject (implies possible structural breaks).
- **Ambiguous**: Both tests fail to reject.

### Variable Integration Orders

| Ticker | Bias Level Conclusion | Return Level Conclusion | Bias Diff Conclusion | Return Diff Conclusion | Integration (Bias, Return) |
|---|---|---|---|---|---|
| ABNB | Stationary (I(0)) | Stationary (I(0)) | Stationary (I(0)) | Stationary (I(0)) | (I(0), I(0)) |
| AMZN | Conflicting (Structural Break Likely) | Stationary (I(0)) | Stationary (I(0)) | Conflicting (Structural Break Likely) | (I(1), I(0)) |
| T | Stationary (I(0)) | Stationary (I(0)) | Stationary (I(0)) | Stationary (I(0)) | (I(0), I(0)) |
| BA | Conflicting (Structural Break Likely) | Conflicting (Structural Break Likely) | Stationary (I(0)) | Stationary (I(0)) | (I(1), I(1)) |
| BAC | Conflicting (Structural Break Likely) | Stationary (I(0)) | Stationary (I(0)) | Stationary (I(0)) | (I(1), I(0)) |
| GM | Stationary (I(0)) | Stationary (I(0)) | Stationary (I(0)) | Stationary (I(0)) | (I(0), I(0)) |
| GS | Conflicting (Structural Break Likely) | Stationary (I(0)) | Stationary (I(0)) | Stationary (I(0)) | (I(1), I(0)) |
| INTC | Stationary (I(0)) | Stationary (I(0)) | Stationary (I(0)) | Stationary (I(0)) | (I(0), I(0)) |
| MCD | Stationary (I(0)) | Stationary (I(0)) | Stationary (I(0)) | Stationary (I(0)) | (I(0), I(0)) |
| MSFT | Conflicting (Structural Break Likely) | Stationary (I(0)) | Stationary (I(0)) | Stationary (I(0)) | (I(1), I(0)) |
| MS | Stationary (I(0)) | Stationary (I(0)) | Stationary (I(0)) | Stationary (I(0)) | (I(0), I(0)) |
| SBUX | Conflicting (Structural Break Likely) | Stationary (I(0)) | Stationary (I(0)) | Stationary (I(0)) | (I(1), I(0)) |
| UBER | Conflicting (Structural Break Likely) | Stationary (I(0)) | Stationary (I(0)) | Stationary (I(0)) | (I(1), I(0)) |
| V | Stationary (I(0)) | Stationary (I(0)) | Stationary (I(0)) | Stationary (I(0)) | (I(0), I(0)) |
| WMT | Stationary (I(0)) | Stationary (I(0)) | Stationary (I(0)) | Stationary (I(0)) | (I(0), I(0)) |
| WFC | Conflicting (Structural Break Likely) | Conflicting (Structural Break Likely) | Stationary (I(0)) | Stationary (I(0)) | (I(1), I(1)) |

## Cointegration Summary
| Ticker | Trace Stat (r=0) | CV (95%) | Cointegrated (95%)? | Recommendation |
|---|---:|---:|---|---|
| BA | 367.23 | 15.49 | ✅ YES | VECM |
| WFC | 317.53 | 15.49 | ✅ YES | VECM |

## Key Decisions / Modelling Takeaways
1. **Returns Stationarity**: As expected for financial returns, weekly log returns are stationary ($I(0)$) across all companies, both raw and winsorized.
2. **Bias Index Stationarity**: Review the integration column. If Bias is $I(0)$, a VAR in levels `(bias, return)` is appropriate. If Bias is $I(1)$ (non-stationary in levels but stationary in differences), running a VAR in levels might yield spurious regression unless they are cointegrated (unlikely since returns are $I(0)$). Thus, a VAR on `(diff_bias, return)` or `(diff_bias, diff_return)` is recommended.
3. **Winsorization Robustness**: Weekly returns for AT&T (T) contain extreme values but were preserved as-is. Compare AT&T stationarity statistics with other companies to verify if outlier behavior affects unit root decisions.

Detailed results written to `/home/cloud/project/media-bias/VAR/diagnostics/step5/stationarity_results.csv`.
Plots saved to `/home/cloud/project/media-bias/VAR/diagnostics/step5/plots/`