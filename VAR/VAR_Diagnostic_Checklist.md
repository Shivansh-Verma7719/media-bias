# VAR Pre-Modelling Diagnostic Checklist
### Media Bias × Stock Returns Study
*Work through each section in order. Each check tells you what to look for, what to produce, and what the result means for your modelling decisions.*

---

## SECTION 1: Data Inventory & Structure Audit

Before any statistical test, you need to know exactly what you have. This section is about mapping your data.

---

### 1.1 — Panel Structure Overview

**What to do:**
Produce a summary table with the following columns, one row per company:
- Company name / ticker
- Total number of articles (before relevance filter)
- Total number of articles (after relevance filter)
- Date of first article (post-filter)
- Date of last article (post-filter)
- Number of days with at least one article (post-filter)
- Number of days with zero articles (post-filter)
- % of trading days covered

**What to look for:**
- Companies where coverage drops below ~30% of trading days are likely too sparse for reliable time series modelling. Flag these.
- Large variation across companies means your panel is highly unbalanced — this affects whether you pool all companies into one VAR or run company-by-company VARs.
- Note whether the first/last article dates align across companies, or whether some companies only have data for sub-periods.

**Decision this informs:**
Whether to model as a pooled panel VAR, a company-level VAR, or to drop certain companies entirely.

---

### 1.2 — Stock Price Data Inventory

**What to do:**
For each company, confirm:
- Date range of stock data available
- Whether data is daily closing prices or already returns
- Number of trading days in your window (2015–2025 ≈ 2,520 trading days)
- Number of missing trading days (holidays are fine; unexpected gaps are not)
- Whether the stock data and article data share the same date range

**What to look for:**
- Any company where stock data starts later or ends earlier than article data — you need the overlap period only
- Any company with unexplained gaps in stock data (not holidays) — investigate these individually
- Confirm whether your stock data includes weekends or only trading days, as this affects alignment with the article data

**Decision this informs:**
The shared observation window for each company's VAR. The VAR can only be estimated on the intersection of dates where both series have data.

---

### 1.3 — Temporal Frequency Decision

**What to do:**
Count, for each company:
- Average number of relevant articles per day (across all days, including zero-article days)
- Average number of relevant articles per week
- Average number of relevant articles per month
- Distribution of articles per day (min, 25th pct, median, 75th pct, max) — a histogram per company is ideal here

**What to look for:**
- If the median articles-per-day is 0 or 1, daily frequency is too granular — weekly or monthly is more appropriate
- If median articles-per-week is still 0 or 1 for many companies, you may need monthly
- High variance in the distribution (some days with 50 articles, many days with 0) suggests aggregation will smooth things out significantly

**Decision this informs:**
The temporal resolution of your bias index. This is one of the most important decisions in the pipeline — it affects everything downstream. Weekly is usually the right choice for a dataset like this.

---

## SECTION 2: Bias Index Construction & Quality

---

### 2.1 — Stance Score Distribution

**What to do:**
For each company, plot the distribution of raw stance scores (prob_positive − prob_negative) across all qualifying articles. Produce:
- A histogram of raw scores (per company and aggregated across all companies)
- Summary statistics: mean, std dev, skewness, kurtosis
- Check what proportion of scores are exactly 0, or very close to 0 (within ±0.05)

**What to look for:**
- If the distribution is heavily skewed toward positive or negative, that is a finding (systematic outlet bias), not a data quality problem — but you should document it
- If an unusually high proportion of scores cluster at 0, this may indicate the model is frequently uncertain and defaulting to neutral — check whether your stance threshold (0.9 from the master doc) is filtering these out properly
- Multi-modal distributions (two peaks) at the company level can indicate two distinct types of coverage — worth flagging

**Decision this informs:**
Whether the raw mean is a sensible aggregation statistic, or whether you need median or trimmed mean to handle outliers.

---

### 2.2 — Daily Bias Index Construction

**What to do:**
For each company, for each day that has at least one qualifying article, compute:
- `bias_index(i,t)` = mean of (prob_positive − prob_negative) across all qualifying articles for company i on day t
- Also compute: median, and article-count-weighted mean as alternative versions
- Store the article count per company per day alongside the index

**What to look for:**
- Days with only 1 article driving the entire score — these are noisy. Check how many such days exist per company
- Verify there are no days where the index is outside [−1, +1] — this would indicate a bug in the construction
- Check for suspiciously volatile days (e.g., score swings from +0.8 to −0.8 on consecutive days) — trace these back to raw articles to see if they are real

**Decision this informs:**
Whether to use mean or a more robust aggregation. Also informs minimum article count thresholds for keeping a day's observation.

---

### 2.3 — Aggregation to Chosen Frequency

**What to do:**
Aggregate the daily bias index to your chosen frequency (e.g., weekly):
- For each company, for each week: compute the mean daily bias index across all days that have observations in that week, weighted by article count
- Store alongside: total article count for the week, number of days with coverage that week

**What to look for:**
- How many weeks are completely empty (zero articles) per company?
- Plot, per company: a calendar heatmap or simple time series showing which weeks have data and which are blank
- Identify structural gaps — e.g., a company with no data in 2017 at all, or consistently blank in Q1 each year

**Decision this informs:**
Which imputation strategy is needed and how extensive it is. Short random gaps get different treatment from long structural gaps.

---

## SECTION 3: Gap Analysis & Imputation

This section is the core of your data preparation problem. Work through it carefully.

---

### 3.1 — Gap Classification

**What to do:**
For each company's weekly bias series, classify all gaps into one of three types:
- **Type A — Short random gap:** 1–2 consecutive missing weeks, no obvious pattern. These are likely just weeks with no news.
- **Type B — Medium gap:** 3–8 consecutive missing weeks. Could be a data collection issue or a genuine quiet period.
- **Type C — Long structural gap:** More than 8 consecutive weeks missing. These are serious — likely a data availability problem for that company in that period.

Produce a table per company: number of gaps of each type, and total weeks missing.

**What to look for:**
- Companies with predominantly Type C gaps should be excluded from VAR analysis or their coverage window should be truncated
- If all companies share the same gap periods (e.g., everyone is missing 2016 data), that is a collection issue, not a company-specific issue — and you may need to truncate the full sample start date
- If gaps are random across companies and predominantly Type A, imputation is manageable

**Decision this informs:**
Which companies to include, what observation window to use, and which imputation method is appropriate per gap type.

---

### 3.2 — Minimum Coverage Threshold

**What to do:**
Define a threshold: a company-period is only included in the VAR if it meets a minimum coverage requirement. A reasonable starting point:
- At least 60% of weeks must have at least one article
- No single gap longer than 8 consecutive weeks

Apply this threshold and note which companies/periods fail it.

**What to look for:**
- How many companies are dropped entirely?
- For companies that pass overall but have a problematic sub-period (e.g., good 2018–2025 but sparse 2015–2017), consider truncating rather than dropping

**Decision this informs:**
Your final sample composition. Document every exclusion decision — reviewers will ask.

---

### 3.3 — Imputation Strategy Selection

**What to do:**
For each gap type that survives the coverage threshold, choose an imputation strategy. The options are:

**For Type A gaps (1–2 weeks missing):**
- Forward-fill: carry the last observed value forward
- Zero-fill: treat missing weeks as neutral sentiment (score = 0)
- Linear interpolation: fill the gap as a straight line between surrounding values
- Recommendation: forward-fill is most defensible for sentiment — it means "no new information arrived, the prior view persists"

**For Type B gaps (3–8 weeks missing):**
- ARIMA-based imputation: fit an AR(p) model on the observed data and predict missing values. This respects the autocorrelation structure of the series.
- Recommendation: use this for Type B. It requires fitting a simple AR model per company, which is lightweight.

**For Type C gaps (more than 8 weeks):**
- Do not impute. Truncate the series at the gap, or exclude the company from the analysis covering that period.
- Imputing long structural gaps with any method introduces too much manufactured data.

**What to document:**
For your methodology section, you must state: how many observations were imputed, which method was used, and the robustness check you ran (see Section 3.4).

---

### 3.4 — Imputation Robustness Check

**What to do:**
After imputation, run your key results (stationarity tests, Granger causality) twice: once on the imputed series and once on the raw series with gaps dropped (i.e., only using weeks where both bias and returns are genuinely observed). Compare the results.

**What to look for:**
- If Granger causality is significant on both versions, the imputation is not driving the result — this is the cleanest outcome
- If the result only appears in the imputed version, it may be an artifact of imputation — report this as a limitation
- If the result only appears in the raw version, imputation is attenuating the signal — investigate why

**Decision this informs:**
How confidently you can claim the main finding is robust to data preparation choices.

---

## SECTION 4: Returns Series Construction

---

### 4.1 — Compute Log Returns

**What to do:**
For each company, compute daily log returns:
- `r(i,t)` = ln(P_t) − ln(P_{t−1})
- where P_t is the closing price on day t

Then aggregate to your chosen frequency (e.g., weekly):
- Weekly return = sum of daily log returns within the week (this is equivalent to the log of the end-of-week price divided by the start-of-week price)
- Store alongside: number of trading days in that week (should be 5, flag weeks with fewer)

**What to look for:**
- Any extreme return values (e.g., > ±30% in a single week) — check whether these correspond to real events (earnings surprises, merger announcements, the COVID crash) or to data errors
- Any consecutive weeks where the price appears unchanged (return = 0 for multiple periods) — this may indicate a stale price feed
- Confirm that the 2020 crash is visible in the data as expected

**Decision this informs:**
Whether your returns series is clean and whether outliers need to be winsorized (capped at e.g. ±5 standard deviations from the mean).

---

### 4.2 — Alignment Check: Bias and Returns

**What to do:**
For each company, create a merged dataset where each row is a week with both:
- The weekly bias index (observed or imputed)
- The weekly log return

Check:
- How many weeks have both series present?
- How many weeks are dropped because one series is missing?
- Is the total number of aligned weeks sufficient for VAR estimation? (Rule of thumb: you need at least 100 observations per series for a reliable VAR)

Produce a simple count table: per company — total weeks in window, weeks with bias data, weeks with return data, weeks with both.

**What to look for:**
- Companies where the aligned sample falls below 100 observations — these are too short for VAR
- Whether the alignment gap is dominated by missing bias data (the more likely problem) or missing returns data

**Decision this informs:**
Final company inclusion list for VAR estimation.

---

## SECTION 5: Stationarity Testing

Stationarity is a prerequisite for VAR. Do not skip this section.

---

### 5.1 — Visual Inspection First

**What to do:**
For each company, plot side by side:
- The weekly bias index over time
- The weekly log return over time

**What to look for:**
- Returns: should look like white noise around zero with no visible trend. If it trends upward or downward persistently, something is wrong with the construction — returns should not trend.
- Bias index: may show a level shift around 2020 (which is your hypothesis). That is fine — the question is whether it has a unit root (random walk) or just a level break. These look similar visually, which is why you need formal tests.
- Obvious structural breaks visible in the plot should be noted — they may affect which version of the ADF test you use.

---

### 5.2 — ADF Test (Augmented Dickey-Fuller)

**What to do:**
For each company, run an ADF test on:
- The weekly bias index
- The weekly log return series

Use three specifications:
- No constant, no trend
- Constant only
- Constant and trend

Use automatic lag selection (e.g., AIC-based, up to a maximum of 12 lags for weekly data).

**What to look for:**
- For returns: you expect to reject the null of a unit root (p < 0.05). If you do not, something is structurally wrong.
- For bias index: the result here is your key diagnostic. If you reject the null, the series is stationary and you proceed with VAR directly. If you fail to reject, you need to difference the bias index (take first differences) or investigate further.
- Record: ADF statistic, p-value, number of lags chosen, and which specification was used, for every company.

---

### 5.3 — KPSS Test (as a complement to ADF)

**What to do:**
Run a KPSS test on the same series. The null hypothesis of KPSS is stationarity (the opposite of ADF), so KPSS and ADF together give you a clearer picture.

Interpret the combination as follows:
| ADF result | KPSS result | Conclusion |
|---|---|---|
| Reject unit root | Fail to reject stationarity | Stationary — proceed with VAR |
| Fail to reject unit root | Reject stationarity | Non-stationary — difference the series |
| Both reject | — | Structural break likely — investigate |
| Both fail to reject | — | Ambiguous — use KPSS with trend, inspect visually |

**What to look for:**
- If the bias index is non-stationary: first-difference it and re-test. The differenced bias index measures *changes* in sentiment rather than sentiment level. This is still meaningful — it just changes the interpretation.
- If differencing fixes stationarity, note the order of integration: I(1) means one difference was needed.

---

### 5.4 — Cointegration Check (if both series are I(1))

**What to do:**
If both the bias index and the returns series are integrated of order 1 (I(1)) — meaning both needed differencing — run a Johansen cointegration test to check whether there is a long-run equilibrium relationship between them.

**What to look for:**
- If cointegration is found (trace statistic exceeds critical value): use a Vector Error Correction Model (VECM) instead of a plain VAR. A VECM is essentially a VAR with an additional error correction term that captures the long-run relationship.
- If no cointegration: difference both series and run a standard VAR on the differenced series.
- In practice, returns are typically I(0) (stationary), so cointegration between returns and bias is unlikely — but check anyway.

**Decision this informs:**
Whether you use VAR or VECM. Document this decision explicitly in your methodology.

---

## SECTION 6: Structural Break Testing

Given that 2020 is central to your research questions, you need to formally test whether a structural break exists in your series.

---

### 6.1 — Chow Test / Bai-Perron Test

**What to do:**
Run a Bai-Perron test for structural breaks (or a Chow test with breakpoint set to January 2020) on:
- The weekly bias index per company
- The weekly return series per company

**What to look for:**
- Whether the model identifies January 2020 as a statistically significant break point, or whether it finds the break at a different date
- If the break is identified at a different date (say, March 2020 — when lockdowns began — rather than January), use that date as your Post(t) cutoff instead
- If no significant break is found for some companies, that is a meaningful result for RQ1 and RQ2 — it means the shock did not affect all companies equally

**Decision this informs:**
The exact cutoff date for your Post(t) binary variable in RQs 1–3. Using January 2020 is theoretically motivated but empirically it may be March 2020 — let the data decide and report what you find.

---

### 6.2 — Pre/Post Subsample Split

**What to do:**
After confirming the break date, split every series into:
- Pre-period: 2015 to the break date
- Post-period: break date to 2025

Check the length of each subsample per company:
- Pre-period: approximately 260 weeks (5 years)
- Post-period: approximately 260 weeks (5 years)

Confirm each subsample independently meets the minimum length requirement for VAR (at least 60–80 observations after accounting for lags).

**What to look for:**
- Any company where one subsample is too short (e.g., a company where data only starts in 2018 means the pre-period has ~100 weeks — still usable but tight)

**Decision this informs:**
Whether to run subsample VARs for Granger causality comparison (as planned in your RQ2 and RQ5 additional methods).

---

## SECTION 7: VAR Specification

---

### 7.1 — Lag Order Selection

**What to do:**
For each company (or each subsample if running pre/post), estimate a VAR with a range of lag lengths (1 through 8 for weekly data) and record:
- AIC (Akaike Information Criterion)
- BIC / SBC (Schwarz Bayesian Criterion)
- HQ (Hannan-Quinn)

**What to look for:**
- AIC tends to select more lags (favours fit), BIC tends to select fewer (penalises complexity more heavily). Use BIC as your primary criterion for a parsimonious model.
- If AIC selects 6 lags and BIC selects 2, run both and check whether your Granger results are stable across lag choices. Instability is a red flag.
- For cross-company consistency: if most companies select 2 lags and one company selects 6, investigate that company — it may have unusual autocorrelation structure.

**Decision this informs:**
The lag order p for each VAR. Report this in a table in your paper: one row per company, AIC-selected lag, BIC-selected lag, chosen lag.

---

### 7.2 — VAR Stability Check

**What to do:**
After estimating each VAR, compute the eigenvalues of the companion matrix and check whether all roots lie strictly inside the unit circle (modulus < 1).

**What to look for:**
- All eigenvalues should have modulus < 1 for the VAR to be stable. If any eigenvalue has modulus ≥ 1, the VAR is explosive and the model is misspecified.
- Common causes: non-stationarity in one or both series (go back to Section 5), incorrect lag order, or structural breaks within the estimation window.
- Plot the roots on a unit circle diagram — this is a standard diagnostic plot and is expected in papers using VAR.

---

### 7.3 — Residual Diagnostics

**What to do:**
On the residuals of each estimated VAR, run:
- Portmanteau test (or Ljung-Box test) for serial autocorrelation in residuals — you want to fail to reject the null of no autocorrelation
- Jarque-Bera test for normality of residuals — VAR does not require normality, but heavy non-normality affects the reliability of inference in small samples
- ARCH test for heteroskedasticity in residuals — financial returns often have ARCH effects (volatility clustering)

**What to look for:**
- If residuals are autocorrelated: your lag order is too low. Increase it and re-test.
- If ARCH effects are significant: your standard errors may be unreliable. Consider using heteroskedasticity-robust standard errors, or note it as a limitation.
- Normality violations are common with financial data — acceptable to flag and move on, but do not ignore severe non-normality (kurtosis > 10).

---

## SECTION 8: Granger Causality

---

### 8.1 — Granger Causality Tests

**What to do:**
For each company's estimated VAR, run Granger causality tests in both directions:
- Does past bias Granger-cause returns? (H₀: bias does not Granger-cause returns)
- Do past returns Granger-cause bias? (H₀: returns do not Granger-cause bias)

Report: F-statistic, p-value, lag order used, and whether H₀ is rejected at 1%, 5%, and 10% significance levels.

**What to look for:**
- Unidirectional causality (bias → returns but not returns → bias): supports your hypothesis that media drives prices
- Bidirectional causality: suggests a feedback loop — journalists react to price movements AND prices react to coverage. This is also an interesting finding.
- No causality in either direction: bias and returns are dynamically independent for this company. Possible, and worth reporting.
- Compare results across companies: are S&P 500 firms consistent, or is the result driven by a few large-cap companies with heavy media coverage?

---

### 8.2 — Pre vs Post Subsample Granger Comparison

**What to do:**
Run the Granger causality tests separately on the pre-2020 and post-2020 subsamples. Produce a comparison table:

| Company | Pre-period (bias→returns) | Pre-period (returns→bias) | Post-period (bias→returns) | Post-period (returns→bias) |
|---|---|---|---|---|
| Apple | p = 0.23 | p = 0.41 | p = 0.03* | p = 0.18 |
| ... | | | | |

**What to look for:**
- Cases where Granger causality is insignificant pre-2020 but significant post-2020: this is the cross-validation of your DiD result — the structural shift in the bias-return relationship is consistent across methods
- Cases where the result reverses post-2020: worth investigating. Could mean the causal direction changed — e.g., post-2020, prices may be driving coverage more than the other way around

---

## SECTION 9: Impulse Response Functions

---

### 9.1 — IRF Estimation

**What to do:**
For each company's estimated VAR, compute impulse response functions (IRFs) for:
- Response of returns to a one-standard-deviation shock in bias
- Response of bias to a one-standard-deviation shock in returns

Use a Cholesky decomposition for orthogonalisation (standard approach). Order: bias first, returns second. This ordering assumes that bias is contemporaneously exogenous to returns — i.e., today's media coverage is not instantly driven by today's stock move. This is a reasonable assumption for news published before market close.

Compute IRFs up to 8–12 periods (weeks) ahead. Include 95% confidence bands using bootstrap (1000 replications).

**What to look for:**
- How quickly does the return response to a bias shock die out? If it decays within 2 weeks, the effect is short-lived. If it persists for 6+ weeks, media bias has a lasting effect on price.
- Does the response cross zero (reverse sign)? A sign reversal suggests an initial overreaction followed by correction — this is interesting for behavioral finance.
- Are the confidence bands wide (crossing zero at all horizons)? If so, the IRF is not statistically different from zero and you cannot claim a meaningful dynamic response.

**What to visualise:**
Plot the IRF with confidence bands for each company. Then also plot the average IRF across all companies on one chart — this gives the aggregate picture. Run pre-period and post-period IRFs on the same chart to visually compare.

---

### 9.2 — Forecast Error Variance Decomposition (FEVD)

**What to do:**
Compute the FEVD for the returns equation: what percentage of the forecast error variance of returns at horizons 1, 4, 8, and 12 weeks is explained by shocks to the bias series?

**What to look for:**
- If bias explains less than 5% of return variance at all horizons: the effect is statistically detectable (Granger) but economically small — be honest about this in the paper
- If bias explains 15–20% or more of return variance: this is economically meaningful and strengthens the paper's contribution significantly
- Compare across companies and across pre/post periods

---

## SECTION 10: Panel VAR Considerations

If you are pooling across companies rather than running firm-level VARs:

---

### 10.1 — Fixed Effects in Panel VAR

**What to do:**
A panel VAR pools observations across all companies and adds firm fixed effects to control for time-invariant differences (e.g., Apple always gets more coverage than a smaller firm). The standard approach is Helmert transformation (forward mean differencing), which removes fixed effects without introducing the Nickell bias that comes from demeaning in dynamic models.

Confirm that your estimation package supports this (e.g., `pvar` in Stata, or `panelvar` in R).

**What to look for:**
- After applying fixed effects, re-run the stationarity tests on the demeaned series — fixed effects removal can affect the time series properties
- Check whether firm fixed effects are jointly significant — if they are, pooling without them would bias the estimates

---

### 10.2 — Cross-Sectional Dependence

**What to do:**
Test for cross-sectional dependence in your panel using a Pesaran CD test. This tests whether shocks are correlated across companies — which is likely, since a market-wide shock (like COVID) affects all firms simultaneously.

**What to look for:**
- If cross-sectional dependence is significant: standard panel VAR standard errors are too small (you are under-estimating uncertainty). Use Driscoll-Kraay standard errors, or bootstrap the standard errors.
- This is almost certainly going to be significant given the 2020 market shock affects all companies in your sample — plan for it.

---

## SECTION 11: Final Pre-Estimation Checklist

Before running the actual VAR, confirm every item below is satisfied:

- [ ] Observation window confirmed for each company (start date, end date, total aligned weeks)
- [ ] Temporal frequency chosen and documented (daily / weekly / monthly)
- [ ] Bias index construction documented (aggregation method, minimum article threshold per period)
- [ ] All gaps classified (Type A / B / C) and imputation strategy applied per type
- [ ] Imputation robustness check run and documented
- [ ] Log returns computed and verified (no unexplained gaps, outliers inspected)
- [ ] Alignment confirmed: both series have sufficient overlapping observations (≥100 weeks per company)
- [ ] Stationarity confirmed for all series entering the VAR (ADF + KPSS)
- [ ] If non-stationary: differencing applied, cointegration checked, VECM vs VAR decision made
- [ ] Structural break date confirmed (empirically, not just assumed to be January 2020)
- [ ] Lag order selected per company (BIC primary, AIC as sensitivity)
- [ ] VAR stability confirmed (all eigenvalues inside unit circle)
- [ ] Residual diagnostics passed (no serial correlation, ARCH effects noted)
- [ ] Companies failing minimum coverage threshold documented and excluded
- [ ] Granger causality tests run (both directions, both subsamples)
- [ ] IRF estimated with confidence bands
- [ ] FEVD computed and economic magnitude interpreted

---

## Appendix: What to Disclose in the Methodology Section

The following must be explicitly stated in your paper, regardless of which choices you made:

1. **Frequency choice** — why weekly (or whichever you chose) and what was lost by not using daily
2. **Relevance classifier impact** — how many articles were removed, what the pre- and post-filter counts were, and classifier accuracy metrics (precision, recall, F1 — which you have from your master doc)
3. **Imputation method and extent** — what % of observations were imputed, which method, per gap type
4. **Stationarity treatment** — whether the bias series required differencing and how this affects the interpretation of Granger causality results
5. **Structural break date** — whether January 2020 was empirically confirmed or assumed
6. **Company exclusions** — which companies were dropped and why (sparse coverage, short window, etc.)
7. **Ordering assumption in Cholesky decomposition** — why bias is ordered first (contemporaneous exogeneity assumption)
8. **Robustness checks run** — imputation sensitivity, lag order sensitivity, pre/post subsample comparison
