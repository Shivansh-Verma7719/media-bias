# VAR and Panel Regression Analysis

This directory contains the final econometric modeling for the research paper. Note that this analysis is in addition to the baseline regressions that have already been run before. The analysis is structured to answer three central research questions regarding the 2020 pandemic shock.

## Research Questions and Methodology

1. **RQ1: Did Media Stance Shift After the Shock?**
   - **Method**: Panel regression with firm and daily fixed effects, controlling for headline volume. Tests if the average tone of coverage shifted persistently after January 1, 2020.
   
2. **RQ2: Did Firm-Level Returns Shift After the Shock?**
   - **Method**: Panel regression with firm and monthly fixed effects, controlling for the S&P 500 index return and the VIX volatility index. Tests if stock returns shifted to a new level post-shock.

3. **RQ3: Did the Stance–Return Relationship Change?**
   - **Method**: Vector Autoregression (VAR) at the weekly level, augmented by Bai-Perron structural break detection. This allows the model to find data-driven breaks in the series instead of imposing the 2020 calendar date. Both single-firm and pooled (panel VAR) Granger causality tests are performed to see if stance leads returns or vice-versa.

## Script Workflow

The steps should be executed sequentially:

- `step1_data_inventory.py`: Initializes the panel data and checks completeness.
- `step2_bias_index.py`: Constructs the daily and weekly stance index.
- `step3_gap_analysis.py`: Identifies and imputes short missing periods in coverage.
- `step4_returns_construction.py`: Assembles the log returns from Yahoo Finance.
- `step5_stationarity.py`: Runs ADF and KPSS tests to ensure time series stationarity prior to VAR modeling.
- `step6_structural_breaks.py`: Estimates Bai-Perron structural break dates for both the stance and return series.
- `step7_var_specification.py`: Selects optimal lag orders for the VAR models using the Bayesian Information Criterion (BIC).
- `step8_granger_causality.py`: Runs firm-level Granger causality tests across full, pre-break, and post-break periods.
- `step8_panel_granger_causality.py`: Runs pooled Panel VAR Granger causality tests using Helmert (forward orthogonal) transformations.


