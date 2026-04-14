# RQ1 Regression Pipeline Summary

## Context

| Field | Value |
|---|---:|
| Source table | public.articles_no_title_deduped |
| Confidence threshold | 0.0 |
| VIX source | finance.vix_daily |
| Pre period | 2015-01-01 to 2019-12-31 |
| Post period | 2020-01-01 to 2025-12-31 |

## Data Flow Counts

| Stage | Count |
|---|---:|
| Raw article rows after SQL filters | 462,788 |
| Raw firms after SQL filters | 22 |
| Total article assignments in panel | 462,788 |
| Firm-day rows in panel | 49,136 |
| Firms in panel | 22 |

## Time Coverage

| Metric | Value |
|---|---:|
| Panel start date | 2015-01-01 |
| Panel end date | 2025-11-12 |
| Distinct days with at least one firm observed | 3,969 |
| Calendar days from min to max | 3,969 |
| Observed firm-day rows | 49,136 |
| Possible firm-day rows (firms x observed days) | 87,318 |
| Firm-day density | 0.5627 |

## Pre vs Post Panel Split

| Split | Firm-day rows | Distinct firms | Distinct days | Mean daily stance | Mean article volume |
|---|---:|---:|---:|---:|---:|
| Pre (2015-01-01 to 2019-12-31) | 24,487 | 22 | 1,826 | -0.037662 | 10.817 |
| Post (2020-01-01 to 2025-12-31) | 24,649 | 22 | 2,143 | -0.016790 | 8.029 |

## Volume Distribution

| Metric | Value |
|---|---:|
| Avg articles per firm-day | 9.419 |
| Median articles per firm-day | 4.000 |
| Min articles per firm-day | 1 |
| Max articles per firm-day | 537 |

## Main FE Result

| Variable | Coef | SE (HC1) | t-stat | p-value (normal approx) | 95% CI low | 95% CI high |
|---|---:|---:|---:|---:|---:|---:|
| post | 0.020757 | 0.003265 | 6.356834 | 2.05954e-10 | 0.014357 | 0.027158 |
| article_volume | -0.000632 | 0.000080 | -7.872202 | 3.48452e-15 | -0.000789 | -0.000475 |
| vix | 0.000157 | 0.000223 | 0.705265 | 0.480646 | -0.000280 | 0.000594 |

## Model Fit

| Metric | Value |
|---|---:|
| Estimator | Entity fixed-effects OLS (within), HC1 robust SE |
| Observations | 49,136 |
| Regressors | 3 |
| Within R2 | 0.001792 |

## Pre-Period Mean Diagnostic

| Metric | Value |
|---|---:|
| Firms | 22 |
| Min firm pre mean | -0.2248930623182298 |
| Max firm pre mean | 0.2991758181818182 |
| Range | 0.524068880500048 |
| Std | 0.10969725284228675 |
| Means equal (tolerance rule) | False |
