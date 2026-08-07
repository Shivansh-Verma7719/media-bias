# Panel Regressions (RQ1 and RQ2)

This directory contains the scripts for the static panel regression analysis that forms the basis for Research Questions 1 and 2 of the paper.

## Overview
While the `VAR/` directory handles the dynamic Vector Autoregression and structural break detection (RQ3), this directory contains the econometric models used to test for a static, market-wide level shift in both media stance and firm returns following the 2020 pandemic shock.

## Scripts

- **`rq-1-pipeline.py` & `rq-1_revised.py`**: These scripts estimate the panel regressions for **RQ1 (Did Media Stance Shift After the Shock?)**. They utilize firm and daily fixed effects to control for baseline differences across firms and common market-wide news shocks on a given day.
- **`rq-2-pipeline.py` & `rq-2_revised.py`**: These scripts estimate the panel regressions for **RQ2 (Did Firm-Level Returns Shift After the Shock?)**. They utilize firm and monthly fixed effects, controlling for broad market factors like the S&P 500 return and the VIX volatility index.
- **`mlr_assumptions_check.py`**: A diagnostic script to verify standard Multiple Linear Regression assumptions on the panel data models.

*Note: All intermediate result outputs and generated data files have been intentionally excluded from this directory to ensure the repository remains a clean codebase representation.*
