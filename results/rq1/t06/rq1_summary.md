# RQ1 Regression Pipeline Summary

## Context

| Field | Value |
|---|---:|
| Source table | public.articles_no_title_deduped |
| Confidence threshold | 0.6 |
| VIX source | finance.vix_daily |
| Pre period | 2015-01-01 to 2019-12-31 |
| Post period | 2020-01-01 to 2025-12-31 |

## Data Flow Counts

| Stage | Count |
|---|---:|
| Raw article rows after SQL filters | 378,573 |
| Raw firms after SQL filters | 22 |
| Total article assignments in panel | 378,573 |
| Firm-day rows in panel | 46,681 |
| Firms in panel | 22 |

## Time Coverage

| Metric | Value |
|---|---:|
| Panel start date | 2015-01-01 |
| Panel end date | 2025-11-12 |
| Distinct days with at least one firm observed | 3,969 |
| Calendar days from min to max | 3,969 |
| Observed firm-day rows | 46,681 |
| Possible firm-day rows (firms x observed days) | 87,318 |
| Firm-day density | 0.5346 |

## Pre vs Post Panel Split

| Split | Firm-day rows | Distinct firms | Distinct days | Mean daily stance | Mean article volume |
|---|---:|---:|---:|---:|---:|
| Pre (2015-01-01 to 2019-12-31) | 23,388 | 22 | 1,826 | -0.055104 | 9.269 |
| Post (2020-01-01 to 2025-12-31) | 23,293 | 22 | 2,143 | -0.030673 | 6.946 |

## Volume Distribution

| Metric | Value |
|---|---:|
| Avg articles per firm-day | 8.110 |
| Median articles per firm-day | 3.000 |
| Min articles per firm-day | 1 |
| Max articles per firm-day | 447 |

## Main FE Result

| Variable | Coef | SE (HC1) | t-stat | p-value (normal approx) | 95% CI low | 95% CI high |
|---|---:|---:|---:|---:|---:|---:|
| post | 0.023171 | 0.003602 | 6.432088 | 1.25863e-10 | 0.016111 | 0.030232 |
| article_volume | -0.000900 | 0.000106 | -8.488132 | 2.09985e-17 | -0.001108 | -0.000692 |
| vix | 0.000167 | 0.000246 | 0.680033 | 0.496484 | -0.000314 | 0.000648 |

## Model Fit

| Metric | Value |
|---|---:|
| Estimator | Entity fixed-effects OLS (within), HC1 robust SE |
| Observations | 46,681 |
| Regressors | 3 |
| Within R2 | 0.002041 |

## Pre-Period Mean Diagnostic

| Metric | Value |
|---|---:|
| Firms | 22 |
| Min firm pre mean | -0.2769503519717139 |
| Max firm pre mean | 0.2575101162790698 |
| Range | 0.5344604682507836 |
| Std | 0.11236348492285327 |
| Means equal (tolerance rule) | False |
