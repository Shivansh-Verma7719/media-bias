# RQ1 Regression Pipeline Summary

## Context

| Field | Value |
|---|---:|
| Source table | public.articles_no_title_deduped |
| Confidence threshold | 0.9 |
| VIX source | finance.vix_daily |
| Pre period | 2015-01-01 to 2019-12-31 |
| Post period | 2020-01-01 to 2025-12-31 |

## Data Flow Counts

| Stage | Count |
|---|---:|
| Raw article rows after SQL filters | 136,787 |
| Raw firms after SQL filters | 22 |
| Total article assignments in panel | 136,787 |
| Firm-day rows in panel | 31,575 |
| Firms in panel | 22 |

## Time Coverage

| Metric | Value |
|---|---:|
| Panel start date | 2015-01-01 |
| Panel end date | 2025-11-12 |
| Distinct days with at least one firm observed | 3,968 |
| Calendar days from min to max | 3,969 |
| Observed firm-day rows | 31,575 |
| Possible firm-day rows (firms x observed days) | 87,296 |
| Firm-day density | 0.3617 |

## Pre vs Post Panel Split

| Split | Firm-day rows | Distinct firms | Distinct days | Mean daily stance | Mean article volume |
|---|---:|---:|---:|---:|---:|
| Pre (2015-01-01 to 2019-12-31) | 16,342 | 22 | 1,826 | -0.179330 | 4.818 |
| Post (2020-01-01 to 2025-12-31) | 15,233 | 22 | 2,142 | -0.136062 | 3.811 |

## Volume Distribution

| Metric | Value |
|---|---:|
| Avg articles per firm-day | 4.332 |
| Median articles per firm-day | 2.000 |
| Min articles per firm-day | 1 |
| Max articles per firm-day | 153 |

## Main FE Result

| Variable | Coef | SE (HC1) | t-stat | p-value (normal approx) | 95% CI low | 95% CI high |
|---|---:|---:|---:|---:|---:|---:|
| post | 0.025762 | 0.005212 | 4.942536 | 7.71128e-07 | 0.015546 | 0.035979 |
| article_volume | -0.002767 | 0.000292 | -9.460264 | 3.07168e-21 | -0.003340 | -0.002193 |
| vix | 0.000530 | 0.000356 | 1.488556 | 0.136604 | -0.000168 | 0.001229 |

## Model Fit

| Metric | Value |
|---|---:|
| Estimator | Entity fixed-effects OLS (within), HC1 robust SE |
| Observations | 31,575 |
| Regressors | 3 |
| Within R2 | 0.002613 |

## Pre-Period Mean Diagnostic

| Metric | Value |
|---|---:|
| Firms | 22 |
| Min firm pre mean | -0.6321965036707683 |
| Max firm pre mean | 0.10032675 |
| Range | 0.7325232536707683 |
| Std | 0.15699772367775033 |
| Means equal (tolerance rule) | False |
