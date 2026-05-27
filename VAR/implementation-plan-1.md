# VAR Diagnostic Checklist Implementation Plan

This plan breaks down the execution of the VAR (Vector Autoregression) diagnostic pipeline into sequential, verifiable steps based on `VAR/VAR_Diagnostic_Checklist.md`. We will proceed step-by-step, validating the outputs at each stage before moving to the next.

## Status Updates
- **Relevance Filter**: Everything in the `articles_no_title_deduped` table is assumed to be relevant. We will not apply any column-based relevance filter (e.g. `predicted_label`).
- **Stock Price Data**: We will compute log returns directly from the `public.stock_prices` table (`close` prices) provided in `VAR/stocks.sql`.

## Proposed Changes

We will create a series of Python scripts (`VAR/step1_...` through `VAR/step10_...`), executing them one at a time and reviewing the results. 

### Step 1: Data Inventory & Structure Audit (Section 1)
#### [MODIFY] `VAR/step1_data_inventory.py`
- Refactor the script to import `run_query` and `get_connection` from `VAR/db.py`.
- Remove the `predicted_label = 'relevant'` filter from all SQL queries, as all rows are considered relevant.
- Update the stock data query to point directly to `public.stock_prices`.
- Run the script and generate `panel_structure.csv`, `stock_inventory.csv`, `temporal_frequency.csv`, and frequency histograms.
- **Review Checkpoint**: Decide on temporal frequency (daily vs. weekly) and whether to use firm-level or panel VAR based on sparsity.

---

### Step 2: Bias Index Construction & Quality (Section 2)
#### [NEW] `VAR/step2_bias_index.py`
- Query all articles and calculate raw stance scores: `pos_score - neg_score`.
- Generate distribution statistics and histograms of stance scores per company.
- Compute the daily bias index (mean, median, and article-count-weighted mean).
- Aggregate the daily index to the chosen temporal frequency (e.g., weekly).
- **Review Checkpoint**: Review index distributions and aggregation statistics.

---

### Step 3: Gap Analysis & Imputation (Section 3)
#### [NEW] `VAR/step3_gap_analysis.py`
- Classify gaps in the aggregated bias index into Type A (1-2 periods), Type B (3-8 periods), and Type C (>8 periods).
- Apply minimum coverage thresholds (e.g., >60% coverage, no gaps > 8 periods).
- Implement imputation: Forward-fill for Type A, ARIMA/Interpolation for Type B.
- Drop companies/periods failing the threshold (Type C gaps).
- **Review Checkpoint**: Review the number of imputed observations and dropped companies.

---

### Step 4: Returns Series Construction & Alignment (Section 4)
#### [NEW] `VAR/step4_returns_construction.py`
- Compute daily log returns from the `public.stock_prices` table: `ln(close_t) - ln(close_{t-1})`.
- Aggregate returns to the chosen temporal frequency (e.g., weekly).
- Merge the aggregated bias index with the returns dataset.
- Filter for overlapping periods (aligned weeks).
- **Review Checkpoint**: Verify the final aligned sample sizes (targeting ≥100 observations per company).

---

### Step 5: Stationarity Testing (Section 5)
#### [NEW] `VAR/step5_stationarity.py`
- Perform Augmented Dickey-Fuller (ADF) and KPSS tests on both bias index and returns for each company.
- Automatically apply first-differencing if series are non-stationary.
- (If both I(1)): Perform Johansen cointegration checks.
- **Review Checkpoint**: Confirm stationarity for all series entering the VAR (VAR vs. VECM decision).

---

### Step 6: Structural Break Testing (Section 6)
#### [NEW] `VAR/step6_structural_breaks.py`
- Run Bai-Perron or Chow test on the series around early 2020 (COVID-19 shock).
- Identify the empirical break date for each company.
- Split the dataset into Pre-period and Post-period subsamples.
- **Review Checkpoint**: Confirm structural break dates and subsample lengths.

---

### Step 7: VAR Specification (Section 7)
#### [NEW] `VAR/step7_var_specification.py`
- Estimate VAR models across a range of lags (1 to 8) to determine optimal lag order using BIC and AIC.
- Perform stability checks by computing eigenvalues of the companion matrix.
- Run residual diagnostics: Portmanteau (autocorrelation), Jarque-Bera (normality), ARCH (heteroskedasticity).
- **Review Checkpoint**: Finalize lag order and confirm model stability.

---

### Step 8: Granger Causality (Section 8)
#### [NEW] `VAR/step8_granger_causality.py`
- Run bidirectional Granger causality tests: `Bias -> Returns` and `Returns -> Bias`.
- Perform tests on the full sample, pre-period, and post-period subsamples.
- Generate comparison tables.
- **Review Checkpoint**: Discuss causal relationships found.

---

### Step 9: Impulse Response Functions (IRF) & FEVD (Section 9)
#### [NEW] `VAR/step9_irf_fevd.py`
- Estimate orthogonalized IRFs (Cholesky decomposition, assuming Bias is contemporaneously exogenous to Returns).
- Bootstrap 95% confidence bands for the IRFs.
- Compute Forecast Error Variance Decomposition (FEVD) at various horizons (e.g., 1, 4, 8, 12 periods).
- Generate visualizations.
- **Review Checkpoint**: Assess economic significance and persistence of shocks.

---

### Step 10: Panel VAR Considerations (Section 10) (Optional)
#### [NEW] `VAR/step10_panel_var.py`
- If we decide to use a Panel VAR in Step 1, perform tests for cross-sectional dependence (Pesaran CD test).
- Apply fixed effects (Helmert transformation) and use Driscoll-Kraay standard errors if necessary.
- **Review Checkpoint**: Evaluate Panel VAR results compared to firm-level models.

## Verification Plan
After executing each script, we will review the generated CSVs, terminal logs, and plots to ensure they align with the expected criteria defined in `VAR/VAR_Diagnostic_Checklist.md`. We will only proceed to the next step once the current step's output is verified.
