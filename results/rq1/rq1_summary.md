# RQ1 Regression Pipeline Summary

## Context

| Field | Value |
|---|---:|
| Source table | public.articles_no_title_deduped |
| Confidence threshold | 0.0 |
| Pre period | 2015-01-01 to 2019-12-31 |
| Post period | 2020-01-01 to 2025-12-31 |
| Time fixed effects | True |
| Dropped regressors | None |

## Data Flow Counts

| Stage | Count |
|---|---:|
| Raw article rows after SQL filters | 690,586 |
| Raw firms after SQL filters | 26 |
| Total article assignments in panel | 690,586 |
| Firm-day rows in panel | 63,200 |
| Firms in panel | 26 |

## Time Coverage

| Metric | Value |
|---|---:|
| Panel start date | 2015-01-01 |
| Panel end date | 2025-12-31 |
| Distinct days with at least one firm observed | 4,018 |
| Calendar days from min to max | 4,018 |
| Observed firm-day rows | 63,200 |
| Possible firm-day rows (firms x observed days) | 104,468 |
| Firm-day density | 0.6050 |

## Pre vs Post Panel Split

| Split | Firm-day rows | Distinct firms | Distinct days | Mean daily stance | Mean article volume |
|---|---:|---:|---:|---:|---:|
| Pre (2015-01-01 to 2019-12-31) | 30,025 | 26 | 1,826 | -0.018455 | 11.943 |
| Post (2020-01-01 to 2025-12-31) | 33,175 | 26 | 2,192 | -0.004776 | 10.007 |

## Volume Distribution

| Metric | Value |
|---|---:|
| Avg articles per firm-day | 10.927 |
| Median articles per firm-day | 6.000 |
| Min articles per firm-day | 1 |
| Max articles per firm-day | 539 |

## Main FE Result

| Variable | Coef | SE (HC1) | t-stat | p-value (normal approx) | 95% CI low | 95% CI high |
|---|---:|---:|---:|---:|---:|---:|
| post | 0.014130 | 0.013623 | 1.037218 | 0.299634 | -0.012571 | 0.040831 |
| article_volume | -0.000970 | 0.000076 | -12.772363 | 2.33942e-37 | -0.001119 | -0.000822 |

## Day-Clustered Inference

Clustered on: date (groups=4018)
Finite-sample correction: 1.0002647692211422

| Variable | Coef | SE (day-clustered) | t-stat | p-value | 95% CI low | 95% CI high |
|---|---:|---:|---:|---:|---:|---:|
| post | 0.014130 | 0.013432 | 1.051992 | 0.292803 | -0.012196 | 0.040456 |
| article_volume | -0.000970 | 0.000076 | -12.843754 | 9.32408e-38 | -0.001119 | -0.000822 |

## Model Fit

| Metric | Value |
|---|---:|
| Estimator | Entity fixed-effects OLS (within), HC1 robust SE |
| Observations | 63,200 |
| Regressors | 2 |
| Within R2 | 0.001749 |

## Pre-Period Mean Diagnostic

| Metric | Value |
|---|---:|
| Firms | 26 |
| Min firm pre mean | -0.22676551805496092 |
| Max firm pre mean | 0.43795649999999997 |
| Range | 0.6647220180549609 |
| Std | 0.1368625701860279 |
| Means equal (tolerance rule) | False |

## Time FE Test

| Metric | Value |
|---|---:|
| F stat | 1.1347819155326917 |
| df1 | 4017 |
| df2 | 59181 |
| p value | 1.1195540076012757e-08 |
| F critical (alpha=0.05) | 1.0382817131667614 |
