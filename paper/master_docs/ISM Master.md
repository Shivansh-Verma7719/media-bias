# Minutes

# Meeting Minutes

## 23rd March:

- Relevant articles are only those which have been annotated by the relevance classifier.   
- Acknowledge in the paper that there are big companies we’re missing out on, because data for them is not available. Because of gaps in data, most of our findings have to be taken with a grain of salt.   
- From SLM Prompt \- remove the articles that only talk about a single product or movie deal.  
- For confidence interval in stance score checking, we can try out multiple thresholds.

Tasks:

- Manually verify 100 articles each, report accuracy scores.   
- Verify the cutoff for BERT classifier. Change probability threshold.

## 

To Show next week:

- Show refined table for relevant and irrelevant classified samples  
- Some more indian data \- all 

## 16th Feb : 

\- per company per year/month \- need a balanced number of articles  
\- filter the 6 mil with top companies and do stratified sampling  
\- articles for company, with content   
\- company Id to URL hash  
\- JCSS \- compass equivalent

Stratified: equal no. Of articles for each company per year. Every company has some data in each month

Next week: get table for each year for each company. How many articles do we have for each company for each year \- tabulate this. Articles mean you have to scrape the article not just links

For 2015-2025 count the number of articles (overall) for each year. Take minimum, this becomes stratified sample size. Then make the stratified sample \- at least include all companies, try to keep equal proportion. For each year get a company wise count dictionary of articles.

Make a hash map of company ID to URL

Journal to submit to: JCSS journal

Discuss the deadline next week

## 23rd Feb

Use 3k per year per company \= 250 per month  
Data augmentation happens later, work on titles currently  
70% fall   
Figure out if a URL is a hit, NewsAPI  
Methodology \- 2 ppl  
Script for data \- 1 person keeps running

Augmentation : Link \+ text

Methodology :   
News MTSE : [fhamborg/NewsMTSC: Target-dependent sentiment classification in news articles reporting on political events. Includes a high-quality data set of over 11k sentences and a state-of-the-art classification model.](https://github.com/fhamborg/NewsMTSC)

Stance analysis : gives probabilities for positive, negative, neutral, filter probs \>0.9

Reject if neither are \>0.9

Fine tuning possible on newsMTSC

Task :   
Excel sheet : title, probabilities \- 100-150 articles

- VAR  
- Granger 

## 9th March : 

Company name as \*aspect\* , and then perform MTSE on that. Not sentiment; stance  
Article sampling methods needs to be changed, since there will be a lot of false positives. Cannot simply search the content for the term “apple” etc, as that is not reliable in terms of actually being associated with the company. Filter out articles only which have the company name in its title.   
De-duplicate based on titles. Same article should not come over and over again  
don’t actually remove, just make a separate table with the filtered out content  
Ideally we need the first paragraph of the article as well

\*NEW TECHNIQUES\*  
DiD (difference in differences) regression:  
Say question \= “effect of AI on student performance”. Split into control and treatment group (equally sized, assume). Plot means (in this case, mean marks of the students). Divide into pre and post periods for both groups. Record the scores for both before the trial. If these means in pre period are not equal, use SCDiD. DiD will capture the difference between the control (post \- pre) and the same for treatment, and then find the difference between the 2 differences. If this is significant, then that means a correlation effect.   
need to define the control and treatment. Not on the indian firm data. Right now, on the current data. Later, it will be indian vs non-indian.   
We can maybe see if recession as a shock (around 2020\) had any effect on media bias pre and post shock. This can be one of the research questions. “How has media reporting changed pre and post recession”.  
Check the same thing as ^ for stock prices instead of media bias. Both thru DID regression.  
Third question can then be to correlate the media bias to stock price using this DID regression. We’ll need to make up fixed effects for the DID regression, and have it vetted by an economics professor (potentially prof Parush). We’ll need some research for this.  
We’ll have about 5-6 research questions. Need supporting literature too. Best if that is qualitative, we can support it quantitatively.

\*DEADLINES\*  
stance scores in the next 2-3 days MAX  
Before next meeting: we need indian companies data  
Find similar hypothesis literature to our own. Have research questions properly framed and supported.  
Continue augmenting data (non-indian). Top 20 companies.

16th March

- Relevance Classifier on 3.8Mn (5 Fold)  
- 5 Fold

## 6th April

- Separate train and test dataset from relevance classfiier training.   
- 300 Manual annotations should be from the held out test dataset. Do not include this in the train dataset, and test accuracy only on this held out set  
- Create pipeline for RQ1 and RQ2


### 13th April

0.9,0.6,0 \<- try out these stance score thresholds for experiment

relveant \- articles that report about major shocks/mergers/econ activities that may have an impact on stock price  
irrelevant \- microimpact, not significant enough to affect stock price much. Not NO impact, just less

for the AT\&T articles specifically \- hardcode; check the actual name, not the ticker. For Visa and Intel also filter using hardcoding

Train BERT till convergence

Need at least 3k articles for training

For gap filling we can use a time series model using the previous x days

### 30th April

- Read literature  
- Try out old regressions for RQ1 and RQ2  
- Email OAA for an extension


### 20th May

- Log normalize volume in the regression equation and try again  
- Make a regression table with all the equations and the corresponding results  
- Try out timeseries modelling. Eg \- VAR for seeing impact of one time series on another  
- Bias and return timeseries for sanity checks \- try using VAR to see if you can use past 5 months to predict next 5 months somewhat reasonably for return data (sanity check)  
- Send mail about VM TOMORROW  
- Start writing a 2 column paper with everything we’ve done so far

# Publication

# **Tier 1: CORE A\* Journals**

Acceptance rates below 5–8%, These journals expect:

* Clear causal identification  
* Strong theoretical grounding  
* Advanced econometric rigor  
* Demonstrable economic significance  
* Large, high-quality datasets  
* Extensive robustness testing

## **Requirements** 

To be competitive at this level, the study would require:

1. **Causal Identification**  
   * Instrumental variable strategies  
   * Natural experiments (e.g., media blackouts, regulatory shocks)  
   * Difference-in-differences frameworks  
   * Structural VAR with identification restrictions  
2. **Full-Text Sentiment Construction**  
   * Full-article scraping  
   * Multiple sentiment models (FinBERT, Loughran–McDonald, transformer-based)  
   * Cross-method validation  
   * Manual classification subsample  
3. **Economic Significance**  
   * Tradable strategy  
   * Risk-adjusted alpha (Fama–French factors)  
   * Transaction cost adjustment  
   * Out-of-sample performance  
4. **Theoretical Contribution**  
   * Clear positioning within EMH, behavioral overreaction, or information diffusion theory  
5. **Advanced Econometrics**  
   * Dynamic panel GMM  
   * Fama–MacBeth regressions  
   * Double-clustered standard errors  
   * Multiple testing corrections

## **Tier 1 Journal Options**

### **1\. Journal of Finance**

[https://onlinelibrary.wiley.com/journal/15406261](https://onlinelibrary.wiley.com/journal/15406261)

Similar Papers:

* Tetlock (2007) — [https://doi.org/10.1111/j.1540-6261.2007.01232.x](https://doi.org/10.1111/j.1540-6261.2007.01232.x)  
* Engelberg & Parsons (2011) — [https://doi.org/10.1111/j.1540-6261.2011.01692.x](https://doi.org/10.1111/j.1540-6261.2011.01692.x)  
* Fang & Peress (2009) — [https://doi.org/10.1111/j.1540-6261.2009.01476.x](https://doi.org/10.1111/j.1540-6261.2009.01476.x)

### **2\. Journal of Financial Economics**

[https://www.sciencedirect.com/journal/journal-of-financial-economics](https://www.sciencedirect.com/journal/journal-of-financial-economics)

Similar Papers:

* Tetlock et al. (2008) — [https://doi.org/10.1016/j.jfineco.2008.02.002](https://doi.org/10.1016/j.jfineco.2008.02.002)  
* Dougal et al. (2012) — [https://doi.org/10.1016/j.jfineco.2011.10.004](https://doi.org/10.1016/j.jfineco.2011.10.004)  
* Loughran & McDonald (2011) — [https://doi.org/10.1016/j.jfineco.2010.09.004](https://doi.org/10.1016/j.jfineco.2010.09.004)

### **3\. Review of Financial Studies**

[https://academic.oup.com/rfs](https://academic.oup.com/rfs)

Similar Papers:

* Peress (2014) — [https://doi.org/10.1093/rfs/hht091](https://doi.org/10.1093/rfs/hht091)  
* Ahern & Sosyura (2014) — [https://doi.org/10.1093/rfs/hht053](https://doi.org/10.1093/rfs/hht053)  
* Boudoukh et al. (2019) — [https://doi.org/10.1093/rfs/hhy073](https://doi.org/10.1093/rfs/hhy073)

### **4\. Management Science**

[https://pubsonline.informs.org/journal/mnsc](https://pubsonline.informs.org/journal/mnsc)

Similar Papers:

* Chen et al. (2014) — [https://doi.org/10.1287/mnsc.2013.1800](https://doi.org/10.1287/mnsc.2013.1800)  
* Gurun & Butler (2012) — [https://doi.org/10.1287/mnsc.1110.1366](https://doi.org/10.1287/mnsc.1110.1366)

# **Tier 2: CORE A Journals**

These journals require strong robustness and methodological discipline but do not necessarily require groundbreaking theoretical innovation.

## **Requirements** 

1. Full-text sentiment analysis (strongly recommended)  
2. Panel regressions with firm and time fixed effects  
3. Clustered standard errors  
4. Economic magnitude interpretation  
5. Robustness across subsamples and horizons  
6. Optional but beneficial: dynamic panel or VAR analysis

## **Tier 2 Journal Options**

### **1\. Journal of Banking & Finance**

[https://www.sciencedirect.com/journal/journal-of-banking-and-finance](https://www.sciencedirect.com/journal/journal-of-banking-and-finance)

Similar Papers:

* Garcia (2013) — [https://doi.org/10.1016/j.jbankfin.2013.04.014](https://doi.org/10.1016/j.jbankfin.2013.04.014)  
* Uhl (2014) — [https://doi.org/10.1016/j.jbankfin.2014.01.006](https://doi.org/10.1016/j.jbankfin.2014.01.006)  
* Kearney & Liu (2014) — [https://doi.org/10.1016/j.jbankfin.2014.06.001](https://doi.org/10.1016/j.jbankfin.2014.06.001)

### **2\. Journal of Corporate Finance**

[https://www.sciencedirect.com/journal/journal-of-corporate-finance](https://www.sciencedirect.com/journal/journal-of-corporate-finance)

Similar Papers:

* Dyck et al. (2008) — [https://doi.org/10.1016/j.jcorpfin.2008.10.001](https://doi.org/10.1016/j.jcorpfin.2008.10.001)  
* Liu & McConnell (2013) — [https://doi.org/10.1016/j.jcorpfin.2013.01.003](https://doi.org/10.1016/j.jcorpfin.2013.01.003)

### **3\. Quantitative Finance**

[https://www.tandfonline.com/journals/rquf20](https://www.tandfonline.com/journals/rquf20)

Similar Papers:

* Hagenau et al. (2013) — [https://doi.org/10.1080/14697688.2012.698738](https://doi.org/10.1080/14697688.2012.698738)  
* Schumaker & Chen (2009) — [https://doi.org/10.1080/14697680902809466](https://doi.org/10.1080/14697680902809466)

### **4\. International Review of Finance**

[https://onlinelibrary.wiley.com/journal/14682443](https://onlinelibrary.wiley.com/journal/14682443)

Similar Papers:

* Cross-country sentiment studies  
* Media tone and volatility papers (various issues 2018–2023)

# **Tier 3: Q1 Journals** 

Tier 3 journals publish high-quality empirical work, including sentiment-based asset pricing and event studies. Acceptance rates are approximately 15–25%. These journals regularly publish ML and NLP-based finance research.

This tier represents a realistic and credible target for the present study.

---

## **Requirements**

1. Clear contribution statement  
2. Robust panel regressions with clustered standard errors  
3. Economic magnitude discussion  
4. Subsample stability tests  
5. Optional: basic trading strategy demonstration

Full-text sentiment is desirable but not always mandatory.

## **Tier 3 Journal Options**

### **1\. Finance Research Letters**

[https://www.sciencedirect.com/journal/finance-research-letters](https://www.sciencedirect.com/journal/finance-research-letters)

Similar Papers:

* Narayan et al. (2021) — [https://doi.org/10.1016/j.frl.2020.101732](https://doi.org/10.1016/j.frl.2020.101732)  
* Smales (2014) — [https://doi.org/10.1016/j.frl.2014.06.003](https://doi.org/10.1016/j.frl.2014.06.003)  
* Chen et al. (2022) — [https://doi.org/10.1016/j.frl.2022.102554](https://doi.org/10.1016/j.frl.2022.102554)

### **2\. Journal of Behavioral and Experimental Finance**

[https://www.sciencedirect.com/journal/journal-of-behavioral-and-experimental-finance](https://www.sciencedirect.com/journal/journal-of-behavioral-and-experimental-finance)

Similar Papers:

* Smales (2016) — [https://doi.org/10.1016/j.jbef.2016.06.001](https://doi.org/10.1016/j.jbef.2016.06.001)  
* Drakos et al. (2017) — [https://doi.org/10.1016/j.jbef.2017.03.001](https://doi.org/10.1016/j.jbef.2017.03.001)

### **3\. International Review of Financial Analysis**

[https://www.sciencedirect.com/journal/international-review-of-financial-analysis](https://www.sciencedirect.com/journal/international-review-of-financial-analysis)

Similar Papers:

* Das (2014) — [https://doi.org/10.1016/j.irfa.2014.02.004](https://doi.org/10.1016/j.irfa.2014.02.004)  
* Huang et al. (2021) — [https://doi.org/10.1016/j.irfa.2021.101865](https://doi.org/10.1016/j.irfa.2021.101865)

### **4\. Research in International Business and Finance**

[https://www.sciencedirect.com/journal/research-in-international-business-and-finance](https://www.sciencedirect.com/journal/research-in-international-business-and-finance)

Similar Papers:

* Media sentiment and emerging markets studies (various 2018–2023 issues)

# **Tier 4: Q2 / CORE B Journals (Applied Empirical Finance)**

Tier 4 journals publish applied empirical finance research with moderate econometric depth. Acceptance rates typically range from 25–35%. These journals are appropriate for empirical ML-driven studies without complex identification designs.

## **Requirements**

1. Clear empirical methodology  
2. Transparent sentiment construction  
3. Robust regression design  
4. Sensible economic interpretation  
5. Basic robustness checks

## **Tier 4 Journal Options**

### **1\. Review of Quantitative Finance and Accounting**

[https://www.springer.com/journal/11156](https://www.springer.com/journal/11156)

Similar Papers:

* Textual analysis and accounting sentiment studies  
* Earnings tone and returns research

### **2\. Journal of Forecasting**

[https://onlinelibrary.wiley.com/journal/1099131x](https://onlinelibrary.wiley.com/journal/1099131x)

Similar Papers:

* News-based VAR forecasting  
* Sentiment-based predictive models

### **3\. Applied Economics**

[https://www.tandfonline.com/journals/raec20](https://www.tandfonline.com/journals/raec20)

Similar Papers:

* Media tone and macro-financial relationships

### **4\. Emerging Markets Finance and Trade**

[https://www.tandfonline.com/journals/mree20](https://www.tandfonline.com/journals/mree20)

Similar Papers:

* News sentiment and emerging market stock predictability

## Tier 5 onwards : core B/C, scopus Q3 not very academically reputable, acceptance is 35-40%

# Resources

* Paper submitted : [STMA\_Final\_Report (Copy) \- Online LaTeX Editor Overleaf](https://www.overleaf.com/project/6947b677f12b74c92a7ec0e2)

* Presentation link : [ISM Presentation \- Online LaTeX Editor Overleaf](https://www.overleaf.com/project/698aeafa2256d7ed19f732a6)

# RQs

**RQ1 : How has media reporting (bias) toward S\&P 500 firms changed pre versus post the 2020 recession?**

**Explanation** The 2020 recession is used as an exogenous shock and the question asks whether media bias toward large-cap US firms shifted structurally and persistently after the shock. All selected firms are observed across both periods. The outcome variable is the daily stance score produced by the NewsMTSC pipeline.

a daily stance score is computed per firm per day as the average of (prob\_positive minus prob\_negative) across all qualifying articles for that firm on that day. This daily stance score is the independent variable used across all regression analyses 

**Method**

bias(i,t) \= α(i) \+ β × Post(t) \+ γ × X(i,t) \+ ε(i,t)

Where:

* bias(i,t) is the daily stance score for firm i on day t, computed as the average of (prob\_positive minus prob\_negative) across all high-confidence NewsMTSC outputs for that firm on that day  
* α(i) is the company fixed effect, which absorbs everything time-constant about a firm including its sector, size, and baseline level of media coverage  
* β is the main coefficient of interest and captures the average shift in daily stance scores across all firms after January 2020  
* Post(t) is a binary variable equal to 1 after January 2020 and 0 before  
* γ × X(i,t) is a vector of time-varying covariates including article volume per firm per day and VIX  
* ε(i,t) is the error term

Pre-period: 2015 to 2019\. Post-period: 2020 to 2025\.

If pre-period means are not equal across firms, the fallback is Synthetic Control DiD (SCDiD), which constructs a weighted synthetic counterfactual rather than relying on a simple group average.

**Additional Methods** Beyond the primary DiD regression, two supplementary analyses are useful here.

ARIMA on the stance score time series per firm tests whether media bias is itself autocorrelated — that is, whether a negative stance on one day predicts a negative stance the following day. If significant autocorrelation is found, it means negative coverage clusters over time rather than appearing randomly. This finding would strengthen the interpretation of the DiD result, as a persistent structural shift in bias post-2020 would be more credible if the bias series itself shows persistence within each period.

### **Alternative Methods:**

#### Panel Regression with Fixed Effects:

![][image1]

Measures average post-period shift. Does not model trend changes. Simple regression, but gives more info than OLS.

Where:

* (y\_{it}) \= daily stance score for firm (i) on day (t)  
* (Post\_t) \= 1 after the recession onset, 0 before  
* (Time\_t) \= running time trend  
* (Post\_t \\times Time\_t) \= change in trend after the shock  
* (\\alpha\_i) \= firm fixed effects  
* (X\_{it}) \= optional controls, like article volume, media outlet mix, industry dummies, volatility, returns, etc.

#### 

#### Segmented/Interrupted Time Series Regression: 

![][image2]

Upgrade on top of the Panel Regression. Models the time series trend overall and post the shock. More informative. This lets you test two things:

* **Level shift**: did stance jump immediately after the shock?  
* **Trend shift**: did the post-shock trajectory change persistently?


**Question : what is interrupted time series, online suggestions : sector wise split i.e low and highly affected firms???**  
**Validity** Without a within-sample treatment/control split, this design functions as an interrupted time series rather than a full DiD. To run a true DiD, a control group drawn from outside the sample is required, such as mid-cap firms not currently in the dataset. This should be resolved before finalising the design, either by collecting a small external control group or by relabelling this as an interrupted time series and reserving the DiD label for RQ3 and RQ4 where the control group structure is cleaner.

**Stance then Timeseries \- check methodology**

CHECK : political connectness of firms \- JDE journal

**Literature**

Blajer-Gołębiewska, Honecker and Nowak (2024) — Investor Sentiment Response to COVID-19: A Sectoral Analysis — North American Journal of Economics and Finance Documents that investor sentiment response to COVID-19 news changed structurally across the full US equity market and persisted across the pandemic cycle. Directly supports a market-wide pre-post framing for the stance score as outcome variable. https://www.sciencedirect.com/science/article/abs/pii/S1062940824000469

Haroon and Rizvi (2020) — COVID-19: Media Coverage and Financial Markets Behavior — Finance Research Letters COVID-19 caused a media panic shock that contributed to investment climate uncertainty across US markets broadly. Validates 2020 as a legitimate and documented market-wide media shock event, which is the precondition for using it as the structural break point in this design. https://pmc.ncbi.nlm.nih.gov/articles/PMC7227534/

News and Markets in the Time of COVID-19 (2024) — Journal of Financial and Quantitative Analysis Shows that the relationship between news coverage and market dynamics shifted structurally at the aggregate market level through COVID and did not fully revert afterward. Supports the claim that the 2020 shock produced a lasting change in how media covers firms rather than a temporary spike. https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/news-and-markets-in-the-time-of-covid19/C0EB2A55CF6A36CCBC5BCB3BAD99B9D4

**RQ2 : Did the 2020 recession shock affect stock prices differently pre versus post, at the market level?**

**Explanation** This runs the identical design as RQ1 but switches the outcome variable from stance score to stock returns. RQ1 and RQ2 are designed as a pair — one tracks whether the shock moved media coverage, the other tracks whether it moved prices. Both need to show a statistically significant shift before RQ3 can claim the two are structurally connected. The same control group question applies here as in RQ1.

**Method**

return(i,t) \= α(i) \+ β × Post(t) \+ γ × X(i,t) \+ ε(i,t)

return(i,t) \= α(i) \+ β × Post(t) \+ γ × X(i,t) \+ ε(i,t) → Try this out as well

Where:

* return(i,t) is the daily stock return for firm i on day t, computed from closing prices  
* α(i) is the company fixed effect  
* β is the main coefficient of interest and captures the average shift in daily returns across all firms after January 2020  
* Post(t) is a binary variable equal to 1 after January 2020 and 0 before  
* γ × X(i,t) is a vector of covariates including the S\&P 500 index return, which absorbs the post-2020 market-wide bull run, and VIX, which absorbs volatility regime changes affecting all firms simultaneously  
* ε(i,t) is the error term

Pre-period: 2015 to 2019\. Post-period: 2020 to 2025\.

**Additional Methods** Two supplementary analyses add robustness to the DiD result here.

A VAR (Vector Autoregression) model estimated on the returns series alongside the stance score series from RQ1 tests whether the two variables are dynamically linked over time, beyond what the DiD captures in levels. The VAR treats both bias and returns as jointly evolving and estimates how a shock to one propagates through the other across multiple lags. This is particularly useful for understanding the time structure of the relationship — whether returns respond to bias within one day, three days, or a week.

Granger causality tests derived from the VAR directly test the causal direction: does past bias improve prediction of future returns beyond what past returns alone predict, and does the reverse also hold? Running Granger tests separately for the pre-2020 and post-2020 subsamples also serves as a cross-validation of the DiD result. If the DiD shows a structural shift and the Granger relationship also changes between subsamples, the two methods are telling a consistent story.

**Literature**

Davis, Hansen and Seminaro (2020) — Firm-Level Risk Exposures and Stock Returns in the Wake of COVID-19 — NBER Documents a cross-firm return standard deviation of 6.6 percentage points on single COVID jump days, compared to 0.4 percentage points in 2019\. Establishes that firm-level return patterns shifted dramatically at the market level around 2020, validating the use of a pre-post design on returns. https://sekhansen.github.io/pdf\_files/risk\_exposures.pdf

Blajer-Gołębiewska et al. (2024) — same paper as RQ1 Also covers market-wide stock return findings across the full S\&P 500\. Relevant for both RQ1 and RQ2 as it addresses both sentiment and returns outcomes. https://www.sciencedirect.com/science/article/abs/pii/S1062940824000469

Fernandez-Cerezo et al. / CEPR (2022) — Heterogeneous Firm-Level Impact of COVID-19 Establishes that the COVID shock produced firm-level performance changes that persisted well into the recovery period and were not a transient 2020 event. Supports extending the post-period through 2025 rather than treating 2020 as an isolated year. [https://cepr.org/voxeu/columns/heterogeneous-firm-level-impact-covid-19-and-role-vaccine-developments-recovery](https://cepr.org/voxeu/columns/heterogeneous-firm-level-impact-covid-19-and-role-vaccine-developments-recovery)

**Similar to RQ1, we can try out panel regression here.** 

**RQ3 : Do the bias shift (RQ1) and the price shift (RQ2) correlate, does media bias drive return patterns through the shock?**

**Explanation** This is the linking question. RQ1 tests whether stance scores shifted post-2020. RQ2 tests whether returns shifted post-2020. RQ3 asks whether those two shifts are connected specifically, whether the sensitivity of stock returns to media stance increased after the shock. A significant triple interaction term (δ) would mean that after 2020, the same one-unit move in stance score produced a larger return response. This is the most complex regression in the paper and the fixed effects specification should be reviewed with an economics professor before running.

**Method**

return(i,t) \= α(i) \+ β × (Treatment(i) × Post(t)) \+ γ × bias(i,t) \+ δ × (Treatment(i) × Post(t) × bias(i,t)) \+ θ × X(i,t) \+ ε(i,t)

Where:

* return(i,t) is the daily stock return for firm i on day t  
* α(i) is the company fixed effect  
* β captures the average return difference between treatment and control firms in the post-period, independent of the stance score  
* Treatment(i) is a binary variable equal to 1 for firms designated as treated. To figure out \- One option is to use S\&P 500 firms as treatment and NIFTY 50 firms as control, which would also serve RQ4.  
* Post(t) is a binary variable equal to 1 after January 2020 and 0 before  
* Treatment(i) × Post(t) is the DiD interaction term, equal to 1 only for treated firms in the post-period and 0 otherwise  
* γ is the coefficient on the standalone stance score and captures the average stance-to-return relationship across all firms and both periods  
* bias(i,t) is the daily stance score for firm i on day t  
* δ is the key coefficient of interest and captures whether the stance-to-return sensitivity changed specifically for treated firms in the post-period  
* Treatment(i) × Post(t) × bias(i,t) is the triple interaction term, equal to the stance score only for treated firms after 2020 and zero in all other cases  
* θ × X(i,t) is a vector of covariates including index returns, VIX, and article volume per firm per day  
* ε(i,t) is the error term

**Additional Methods** The VAR and Granger causality framework from RQ2 is directly relevant here as well. Once the DiD establishes that the bias-to-return sensitivity changed post-2020 at the coefficient level, the VAR can characterise the time dynamics of that change — how quickly returns respond to a bias shock and whether the impulse response function differs between the pre and post subsamples. Plotting the impulse response functions side by side for the two periods is a clean and interpretable way to visualise what the δ coefficient captures statistically.

An event study design is also applicable as a robustness check. Identifying specific dates where a firm received a sharp and sudden shift in stance score, defined as a movement more than two standard deviations from its rolling mean, and then tracking abnormal returns in the window around that event produces a non-parametric complement to the DiD result. If the event study shows abnormal returns concentrated in the days following large stance shifts, it corroborates the DiD finding without relying on the same modelling assumptions.

**Validity** Variance Inflation Factors (VIFs) must be checked across all interaction terms before reporting results. The fixed effects specification — whether company fixed effects alone are sufficient or whether sector-by-year interaction terms are needed as additional controls — requires sign-off. Parallel trends must hold for both the stance score and returns series independently.

**Literature**

Costola, Nofer, Hinz and Pelizzon (2023) — Machine Learning Sentiment Analysis, COVID-19 News and Stock Market Reactions — Research in International Business and Finance During crisis regimes, ML-based sentiment from news becomes a significantly stronger predictor of both returns and volatility. Provides the empirical basis for expecting δ to be significant, as the stance-to-return sensitivity should increase post-2020. https://www.sciencedirect.com/science/article/abs/pii/S0275531922002355

Bai, Han, Pan and Zhang (2023) — Financial Market Sentiment and Returns — Finance Research Letters Negative financial market sentiment has a larger impact on stock returns than positive sentiment, and negative coverage amplifies the effect of a crisis on returns. Supports the theoretical expectation that δ should be negative and significant. https://ideas.repec.org/a/eee/finlet/v54y2023ics1544612323000831.html

Xu (2024) — News Bias in Financial Journalists' Social Networks — Journal of Accounting Research Establishes that firm-specific media bias is real and structurally varies across firms and time. Provides the theoretical grounding for why the stance score in the triple interaction carries a meaningful coefficient rather than capturing noise. https://onlinelibrary.wiley.com/doi/10.1111/1475-679X.12560

**RQ4 — Does media bias have a stronger effect on NIFTY 50 stock returns than S\&P 500 returns, and is this differential explained by lower financial literacy among Indian retail investors?**

**Phase 1 \- compare pre and post recession**

**Phase 2 \- check if media bias affects this** 

**Explanation** In markets where a larger share of investors make decisions based on news narratives rather than fundamental analysis — because they lack the financial background to evaluate earnings reports, discount rates, or analyst forecasts — media bias produces a larger and more immediate price impact. India versus the United States is a natural test of this hypothesis. The financial literacy gap between the two countries is large, externally documented, and measurable without collecting new data. The Indian retail investor base tripled in size post-2020, with the majority of new participants being first-generation investors with limited formal financial education, making them structurally more responsive to media narratives.

**Method** Two regressions. The first tests whether the same media stance signal produces a larger return response in Indian markets than in US markets.

return(i,t) \= α(i) \+ β × (Indian(i) × bias(i,t)) \+ γ × bias(i,t) \+ δ × X(i,t) \+ ε(i,t)

Where:

* return(i,t) is the daily stock return for firm i on day t  
* α(i) is the company fixed effect  
* β is the key coefficient of interest and captures whether the stance-to-return relationship is stronger for Indian firms than for US firms. A significant positive β means the same media signal produces a larger price response in India.  
* Indian(i) is a binary variable equal to 1 for NIFTY 50 firms and 0 for S\&P 500 firms  
* Indian(i) × bias(i,t) is the interaction term, equal to the stance score for Indian firms and zero for US firms  
* γ is the coefficient on the standalone stance score and captures the baseline stance-to-return relationship for US firms  
* bias(i,t) is the daily stance score for firm i on day t  
* δ × X(i,t) is a vector of covariates including article volume per firm per day, VIX, and local market index returns for each country separately  
* ε(i,t) is the error term

The second regression adds a financial literacy proxy as a moderating variable, either retail trading volume as a percentage of total market volume sourced from NSE and NYSE data, or the S\&P Global literacy rate as a fixed scalar assigned to each country, to test whether the literacy gap explains β rather than other structural differences between the two markets.

**Additional Methods** Two additional analyses are particularly valuable for this cross-market RQ.

A VAR model estimated separately on the Indian and US subsamples, with stance score and returns as the two endogenous variables, allows the impulse response functions of the two markets to be compared directly. If the Indian subsample shows a larger and longer-lasting return response to a one-unit shock in the stance score compared to the US subsample, this is direct dynamic evidence of the asymmetry that the DiD interaction term β captures in reduced form. The VAR result serves as a more granular characterisation of the same underlying phenomenon.

A quantile regression on returns — run separately for Indian and US firms — tests whether the asymmetry is concentrated in the tails of the return distribution. The noise trader hypothesis would predict that Indian markets overreact specifically to very negative media coverage, meaning the effect should be larger at the lower quantiles of the return distribution than at the median. If quantile regression confirms this pattern, it provides behavioral evidence that the mechanism is sentiment-driven overreaction rather than a general scaling of the bias signal.

**Validity** The main issue is article volume. Indian firms receive fewer English-language articles than US firms, meaning daily stance scores for Indian firms are computed from smaller samples and are therefore noisier. Noisier scores attenuate β toward zero, so any significant result found is a lower bound on the true effect. Article count per firm per day must be included as a covariate. Company fixed effects absorb time-constant firm-level differences, but compositional differences between Indian and US firms including sector distribution, market capitalisation, and accounting standards persist across the sample and should be disclosed as a limitation.

**Literature**

De Long, Shleifer, Summers and Waldmann (1990) — Noise Trader Risk in Financial Markets — Journal of Political Economy When irrational noise traders are prevalent in a market, prices deviate from fundamental values and rational arbitrageurs cannot fully correct the gap because doing so exposes them to the risk of noise trader sentiment moving further against them. This is the theoretical foundation for RQ4. The prediction being tested is that India's higher proportion of low-literacy retail investors creates more noise trader activity and therefore stronger media-to-price transmission than in the US. https://ms.mcmaster.ca/\~grasselli/DeLongShleiferSummersWaldmann90.pdf

Saleem, Sathyamoorthi and Jain (2025) — Financially Savvy or Swayed by Biases? — Journal of Risk and Financial Management The number of unique investors in Indian securities markets has nearly tripled since 2019, surpassing 130 million. Higher financial literacy leads to better diversification and more robust long-term investment strategies. The paper explicitly notes that the investment behaviour of this rapidly growing retail cohort is underexamined in the literature. https://www.mdpi.com/1911-8074/18/6/322

Advances in Consumer Research (2025) — Cognitive Biases and Retail Investors in India Indian retail investors are especially prone to overconfidence, availability anchoring, and confirmation biases. Many rely on peer pressure, social media trends, and emotional reactions to news rather than fundamental analysis. Qualitative support for the behavioral mechanism underpinning RQ4. https://acr-journal.com/article/studying-the-influence-of-cognitive-biases-on-the-investment-decision-making-of-retail-investors-in-india-1513/

SEBI Study (2024) — Study of Profit and Loss of Individual Traders in Equity Derivatives 93% of individual traders in India incurred net losses between FY22 and FY24. Hard regulatory data confirming that the majority of India's retail investor base trades against its own financial interest, which is the behavioral signature of sentiment-driven, low-sophistication decision-making at scale. https://www.sebi.gov.in

S\&P Global Financial Literacy Survey — Klapper, Lusardi and van Oudheusden (2015, updated 2022\) — World Bank / GFLEC Conducted across 140 countries with 150,000 respondents. India scores 24% financial literacy versus 57% for the United States. This makes the literacy differential between the two groups concrete, externally validated, and citable without requiring new data collection. https://gflec.org/wp-content/uploads/2015/11/Finlit\_paper\_16\_F2\_singles.pdf

**RQ5 — Does target-specific media stance toward a firm predict stock returns?**

**Explanation** This is the foundational question of the paper. It tests whether the daily stance score produced by the NewsMTSC pipeline has a statistically significant relationship with how a stock moves the following day. Every other research question depends on this signal existing. The methodological distinction from prior work is that the model evaluates stance toward a specific named company within the article rather than scoring the article as a whole. This produces a materially cleaner signal than the general sentiment approaches used in most existing literature, which cannot distinguish between an article that is positive about one firm and negative about another mentioned in the same piece.

**Method**

return(i,t+1) \= α(i) \+ β × daily\_bias(i,t) \+ γ × X(i,t) \+ ε(i,t)

Where:

* return(i,t+1) is the daily stock return for firm i on day t+1, meaning the next-day return  
* α(i) is the company fixed effect  
* β is the main coefficient of interest and captures whether a higher stance score on day t predicts higher returns on day t+1  
* daily\_bias(i,t) is the daily stance score for firm i on day t, computed as the average of (prob\_positive minus prob\_negative) across all high-confidence NewsMTSC outputs for that firm on that day  
* γ × X(i,t) is a vector of covariates including the S\&P 500 index return and VIX  
* ε(i,t) is the error term

Standard errors clustered at the firm level. Robustness checks run at t+2 and t+3 lags.

**Additional Methods** Three supplementary analyses are applicable here and together constitute a more complete picture of the bias-return relationship than OLS alone.

A VAR model with bias and returns as the two endogenous variables captures the dynamic, bidirectional relationship between the two series. The VAR does not assume that bias causes returns — it estimates both directions simultaneously across multiple lags. The resulting impulse response functions show how a shock to the bias series propagates through returns over the following days, and vice versa. This is valuable because the OLS regression in RQ5 only captures the contemporaneous next-day effect; the VAR captures the full decay structure.

Granger causality tests derived from the VAR formally test whether past values of the stance score improve prediction of future returns beyond what past returns alone predict, and whether the reverse also holds. If bias Granger-causes returns but returns do not Granger-cause bias, that is evidence for a one-directional information channel from media to markets. If the reverse is also true, it suggests journalists are partially reacting to price movements, which has implications for how the bias score should be interpreted in the DiD regressions.

ARIMA on the standalone bias time series per firm tests whether the stance scores are themselves autocorrelated. If the bias series has a significant AR(1) component, negative coverage on one day predicts negative coverage the next. This has two implications: first, it means the stance score is not white noise and carries temporal information beyond its same-day value; second, it motivates including lagged bias as an additional regressor in the OLS specification as a robustness check.

**Validity** Company fixed effects absorb everything time-constant about a firm. Using next-day returns rather than same-day returns creates a temporal lag that limits reverse causality concerns. The main remaining threat is that macroeconomic events can simultaneously drive negative coverage and falling prices. This is controlled for by including index returns and VIX as covariates.

**Literature**

Lopez-Lira and Tang (2023) — Can ChatGPT Forecast Stock Price Movements? — UCLA Anderson LLM-based sentiment extracted from news headlines significantly predicts next-day stock returns out-of-sample. Validates the core logic of RQ5 using a contemporary NLP pipeline and confirms the sentiment-to-return relationship holds with modern language models. https://papers.ssrn.com/sol3/papers.cfm?abstract\_id=4376881

Kengmegni (2024) — Limitations of News Sentiment Analysis in Short-term Stock Return Prediction — SSRN Benchmarks FinBERT, RoBERTa, and LLaMA 3 8B for sentiment-based return prediction and finds that high-coverage stocks show meaningfully different sentiment-return relationships than low-coverage stocks. Directly flags the article volume confound that this paper addresses through the stratified sampling design. https://papers.ssrn.com/sol3/papers.cfm?abstract\_id=5086825

Xu (2024) — News Bias in Financial Journalists' Social Networks — Journal of Accounting Research Journalist connections to company management introduce systematic media slant, with corporate ties associated with up to 47% less negativity in coverage relative to firm-initiated press releases. Establishes that firm-specific media bias is real, systematic, and varies across firms, which is the theoretical basis for using a firm-level stance score as a meaningful independent variable. https://onlinelibrary.wiley.com/doi/10.1111/1475-679X.12560

Hamborg and Donnay (2021) — NewsMTSC — EACL Introduces and benchmarks the exact model used in the pipeline. Establishes why target-dependent stance classification is a fundamentally different and superior task to general article-level sentiment classification for firm-specific analysis. https://aclanthology.org/2021.eacl-main.142/

## 

### 1\) Interrupted time series / segmented regression

![][image3]

This lets you test two things:

* Level shift: did stance jump immediately after the shock?  
* Trend shift: did the post-shock trajectory change persistently?

That matches your wording much better than plain OLS.

# Tab 5

Raw Data (\~700k articles)  
         │  
         │ Filter: S\&P 500 company tags  
         ▼  
Random Sample \- 1k  
         │  
         │   
         ▼  
  Human Annotation (930 articles)  
  Binary label: relevant / irrelevant  
         │  
         │ Split BEFORE any modeling  
         ├──────────────────────────────┐  
         │                         				      │  
         ▼                             				     ▼  
  Training Set (750 articles)    			Test Set (180 articles)  
                                  
         │  
        ▼  
  DeBERTa v3 Fine-tuning (v1)  
  microsoft/deberta-v3-base  
  6 epochs, batch=16, lr=2e-5  
Confidence Threshold for relevant \= 0.75  
         │  
         ▼  
  Evaluation on Test Set  
  Precision: 0.652  
  Recall:    0.811  
  F1:        0.723  
  Accuracy:  86.4%  
    
  Problem: Precision too low (target ≥0.90)  
  Model is too permissive

 Confusion Matrix:                                              │  
│            pred\_irr   pred\_rel                                  │  
│  true\_irr    116         16      ← 16 FPs                      │  
│  true\_rel      7         30      ← 7 FNs

         │  
         │ Need more data to improve precision  
         ▼  
  Semi-Supervised Expansion  
  Run v1 model on 14,420 unlabeled articles \- filtered to have an even distribution of companies  
  Keep only high-confidence predictions:  
    Relevant:   p\_rel ≥ 0.92  →  1,203 articles \-   
    Irrelevant: p\_rel ≤ 0.05  →  2,100 articles  
  Final training set: 4,032 articles  
         │  
         ▼  
  DeBERTa v3 Fine-tuning (v2)  
  Same architecture and hyperparameters  
  5.4x more training data  
  Hard negatives improve precision  
         │  
         ▼  
  Evaluation on Test Set  
  (results pending)  
    
  Targets:  
  Precision ≥ 0.90  
  Recall    ≥ 0.80  
  F1        ≥ 0.85  
  Accuracy  ≥ 90%

p\_rel range    TPs (relevant articles)    FPs (wrongly called relevant)  
\[0.00-0.50)    ███████  7                 0    
\[0.50-0.75)    █████    5                 0     
\[0.75-0.80)    ████     4                 some FPs here  
\[0.80-0.85)    █████    5                 some FPs here  
\[0.85-0.90)    ███████████ 11             ← most FPs live here too  
\[0.90-1.01)    ██████████ 10             ← only 0 FP here

# 

## **Base Model**

**DeBERTa-v3-base** (Microsoft)

* \~86M parameters  
* State-of-the-art encoder  
* Fine-tuned for **binary sequence classification** (relevant vs irrelevant) on article titles

## **Model 1:** 

**Training Data:**

* labeled data only (**750 articles**)

### **Test Performance (Best Threshold)**

| Threshold | Precision | Recall | F1 | FP | FN |
| ----- | ----- | ----- | ----- | ----- | ----- |
| 0.75 | 0.652 | 0.714 | 0.682 | 16 | 12 |

### **Error Analysis**

**Common Failure Cases:**

* Consumer promotions: *“Best Buy deals”, “50% off”*  
* Sports crossover articles:  
  * Athlete endorsements  
  * Stadium naming rights  
* Incidental mentions of companies (not primary subject)  
* Limited training data → poor coverage of edge cases

## **Model 2: model\_synthetic\_v2**

**Training Data:**

* manual (750) \+ All synthetic (**6,525 total**)  
* Includes targeted patterns:

### **Key Training Improvements**

* **Focal Loss (γ \= 2):**  
  * Focuses on hard boundary cases  
  * Down-weights easy predictions  
* **5× Gold Upweighting:**  
  * Gold data contributes 5× more to loss  
* **Targeted Synthetic Expansion:**  
  * Covers **additional error categories**

### **Test Performance (Best Threshold)**

| Threshold | Precision | Recall | F1 | FP | FN |
| ----- | ----- | ----- | ----- | ----- | ----- |
| 0.95 | 0.860 | 0.881 | 0.871 | 6 | 5 |

### **Errors**

* Informal earnings language:  
  * *“Finishes 2016 Strong”*  
* Employee-perspective layoffs  
* Crypto / new financial product launches  
* Cross-company investment framing

## **Ensemble Model (Both Models)**

**Method:**

* Equal-weight average of `p_relevant`  
* Predict **relevant if average ≥ 0.65**

### **Rationale**

* Models make **different errors**  
* Averaging:  
  * Reduces variance  
  * Requires consensus → improves precision  
  * No retraining required

---

## **Threshold Sweep (Test Data)**

| Threshold | Precision | Recall | F1 | FP | FN |
| ----- | ----- | ----- | ----- | ----- | ----- |
| 0.55 | 0.673 | 0.881 | 0.763 | 18 | 5 |
| 0.60 | 0.783 | 0.857 | 0.818 | 10 | 6 |
| 0.65 | 0.944 | 0.810 | 0.872 | 2 | 8 |
| 0.70 | 0.970 | 0.762 | 0.853 | 1 | 10 |
| 0.75 | 0.968 | 0.714 | 0.822 | 1 | 12 |

### **Selected Threshold: 0.65**

**Reason:**

* First point where:  
  * Precision ≥ 0.90  
  * Recall ≥ 0.80

Tweaked Single Model:

### **Test Performance (Best Threshold) \- DeBERTa 750**

| Threshold | Precision | Recall | F1 | FP | FN |
| ----- | ----- | ----- | ----- | ----- | ----- |
| 0.65 | 0.850 | 0.929 | 0.888 | 7 | 2 |

### **Test Performance (Best Threshold) \- Synthetic** 

| Threshold | Precision | Recall | F1 | FP | FN |
| ----- | ----- | ----- | ----- | ----- | ----- |
| 0.65 | 0.872 | 0.929 | 0.900 | 6 | 2 |

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnAAAAAzCAYAAAAXQDVIAAAV9ElEQVR4Xu2dBZPkNhOGv18TZma4YIWZmZmTCzMzM1S4klyY8cJ4YYYKXpiZ468ep94tbVvyyONZ8KW76qndaZnGlqVWq1vzv+mmm65wHMdxHMdxusP/rMJxHMdxHMcZ37gB5ziO4ziO0zHcgHMcx3Ecx+kYbsA5juM4juN0DDfgHMdxHMdxOoYbcI7jOI7jOB3DDTjHcRzHcZyO4Qac4ziO4zhOx3ADznEcx3Ecp2O4Aec4juM4jtMx3IBzHMdxHKdvJk2aVEw//fQVvTOyuAHnOI4zQL777ruKznFSPPTQQ8XLL79c0XeFjTbaqEBOOumkSpkzsrgBNw0xyyyzFJMnTy7++uuv4s4776yU5zLbbLMV88wzT7HAAgsUCy64YPl33nnnLeaaa65i1llnrWzvxLnxxhuLf/75p3j44YcrZc74Ybvttiu++uqr4uOPPy522223SnkudGS8e3POOWelDGacccbyvZp//vmH3q355puvmHvuuYvZZ5+9sr3z3+Gggw4qXnrppYq+C0yZMqU04KxeUNfpN6j/eOn4q8/ahvci3Ccsc9K4ATeNcOyxxxavvfbakBv7jDPOqH2p6nj//feLb775ptwf+fHHH4tPP/20+OSTT0rvguTXX3/1Fy3CxhtvXHz55ZdDxu5mm21W3i+7XR1vvfXW0H3Olb322qtynKeeeqr44IMPKnrnX3744YfiiCOOGPr8999/F9ddd11lu15ggCGHH354pUy88sorxeeffz70vDgXnzEcMSAlzz33XGXf/yL9SKyu77vvvsWff/5Z0Y833n333eKSSy6p6Mc7yLffflvRi48++qis61aOPPLIoW3oS0LhM44EeyxnOG7ATQOsssoqxWOPPVbRIyuvvHJFn4vE6kEdVqp80Cy33HLluca7p4Lrw9i1eiRssHqBHHLIIcN0O+64Y6mPNZZIzPMjsXpnumLq1KnFkksuOUyHB66f+4XgRbH6GHXPZL/99ivLYnVoJJg4cWLyWsYS6jKyyCKLVMpS9w9v90UXXVTRp7Yfb8wwwwyduE4Lsvzyy1f0FpwLEhsvp7Itttiisp+Txg24MQLPiNX1w/HHH5986ZGtt966os9h3XXXLffHO2DLhKSNkZjLs88+m/yebTn99NOLFVdcsaLvB0aaVgfIBRdcUNHHWGmllYr33nuvon/11VfL4xx22GGVstS9mTBhQkXXZfAMW10/rLPOOsUtt9xS0Z999tnJe5lioYUWarSPxOpzywfJH3/8MWrnasLdd99dGrNWz3Q38vXXX1fKHn300bLdsvqZZppp3A/8xPfff1+2dVbfhEG2Z71Q/JvVp5Cnjecb6pFFF120sv1IwrXH2tIu4QbcGPHMM89UdP2A7L///hU902lNXiwLgbUI0w+2TEhWWGGFStkgwQiV2LJBcNZZZw2kwVtqqaWKm2++uaJfbLHFymsn3smWxXjnnXcae9NS+mmNQRlweGusDhA6Gauvg+nyN954o6KPsdZaa5XnSA2M8DjVPedB08/3HQ1S358wEeTQQw+tlKXuaZfYaqutkt89l0G1Zzk8/vjjUa9nivXWW+/fyh18R+rf2muvXdl2pNlkk03cgHP6YxBxLkwrXn311RU9IG0SGexLZsFl3mubQSHv20glA5x77rkDafDoyK0OfvnllzKu0OpTpO5p3f1O6ac1BmXAEW9kdXiSEZIKbFkdyN57713Rx3jggQfK7VMDo9tvv70sJ+7Ulg0aYoyQE044oVLWD8Q67bnnnhV9yG+//VYOaKw+hLhakrGsHiRWrzKr6yJIG2/UoNqzHPq55xJihYmrzn13Bg3TtYMy4JZYYonyns8xxxyVspGkkQHHi4V88cUX0akihGB6qx/vkP6M4L4m2JXvFvOADJIXXnihomvKXXfdNXSdNIwSRjT33XdfZfsmSKwe8DTJFb7GGmtUyuHtt98uywncRlIZVkcfffRQOff9kUceKfW8CCnBILLHacP5558/kAYP4S/xSxLqVMxYaMpOO+1UHi9lJIaQ0YUwPWanmkisCAOKlXGMKHFCMXZKvpD+9ddfr5xLXHrppeU2TP0iOdfZD4Mw4Gi08XRo+gfhvUeWWWaZyvZ1YOwhVp9CYvU55ZyLuo/omdg4SaH3h7rIswiPmZIHH3ywcpym0DcccMABFT1QH9uGW0isPgZeVgyE2PaU8W7+/vvv5WdNzX744YfDzkH7Smax6jXfwR5LUHc4Lm2j2gCbXZkDctNNN1X0ubRtz8L4ZoRBh90m5cHO4dZbbx069oUXXlgpHy223HLL1gYcnveYbL755pVtR4JGBhzC3w033LD8n3R4lS277LKlbptttqns1y90NizF0JTU6DaGsi1DnR6KPo9EBtMgDLjwGq+44opy2pNAatYU+umnnyrb57L66quXx45ldFHhJRhytpyRCGIbIMQalUz9hgMBTZXaY0qsflC0bfCAIFx5Vc8555wyHueoo44qrrrqqoFcuzrsgw8+uFJm0flOPvnkyrll/KqMrOJw+m/NNdcs9WRi3nHHHUN6DAVk5plnjp7PGla8M3iI7bZtsefpBx2D6Urq5DXXXFOOxpHtt9++sn0dJKYgVp9CYvWKOWUgZgO84ZhjjinLN9hggyEdy/ogsc4UYekffY4F+I9U/BsGozUsORfJVnbbJuy8887l9WIk2jIL95G6qsSAXXbZZaiM91OxcjpemICCIYswqAynlxVof+aZZ1bOR0wzQjykdHgj+xnI6JqsPpc27ZkGc0i4EkEYS009isUa5hImM9iy0aStAcc7FUtMYzBLP2j1I0G2AbfaaqsVl112Wfn/iy++WLn5BCVa3SCgEUpBZWNUSkPGiJORAx6F3KUtFE9h9YsvvnipV2yZ9WIMgkEYcHWGJUJnbPU5aJonJldeeWW0ExcII1urp7IjdtvQSFS6ebiNOla77yBp0+AJlqLAY2X1gJBsYvVNyL0HNCh4E/hf3pqwnI6Nvxj5CN6FsJxBGWLrPI0dYo0yefNCHcR0g2AQBlzq2ljCIVWWQp5Hq4+hgVFMnnjiiWSjj2cHwfCwZRIMFenk1dZnMm0RG4OJDNqbLUIjDuON7263aYq8+r2yfWmf8PzzP8YWQh+hct0bnA2IvP6CwbDEHhu57bbbhul4LoidCiS85dRTT60coxepc+fSpj1D7CDmlFNOGXY9ba4NNBhErr/++kr5aNHGgOP+joclX7INOPsC2I5WEuroyGOGynj44mrUyOyzZYCQaEBQ+qqrrhotz5nvJiMoxmeffVbRiV4NFDAVQcac1QuEzkWf6XjxENlnFEPCwsC2rA4MMCQWvyGJ6TBECby1+4CuGY+WLROMdHMWGLb3WRBjh2Fq9SJnPaLY0h4CCacfmY7BQ8naX7xHOQMOidVbFl544WH72FhLeWUk66+//rByOhzE1m2Cle352RfB+8AzJxBZnVlsQdzTTjut9EpavWX33XevPAOBR8TqhD1ODEb/qSw/TVPrM9OsDFbxeiLhenHi/vvvr9yXFPLscB9sWR0Sqw/LwufI8SVc39JLL13ZT9NkeGJtWXgcq2sCRhzvdr8DSYvE6i28T2q7YvswW8Tf559/vlIGCg/BEAr1qu92qQsJC9ZybLJnmbqVERnCO9Jrmp6+IXZdFlv/RV17VmdQcl2E5Vg9qL/Hk7nHHntUynPBg8zai7qXOd9T4QBWz3e0uhj2Hgiy0HknrV7Y44QonGzXXXct278333yzNpmP9tPaTDl1oRfZBpzA8kQOPPDAYXqEjinUxUZdLHRpR/1jgbxvLNlgywChAUoFExM/Y3UxiBGLwZSY1YlYg2u55557alPjkRNPPLH8nyUTMOAUe2C3tUisvhd1+8XKwuxSiTWWJGQv2WOG29TdC2Hvs2DanRfR6oU9TgzE6sIyjfIZlYeeFgy/un1Ba5Ox6KstSyFDLBWQL7F6dV5WH9uebFmEhhQPI4Z0OG1nQaxBGYP7Y5+BYFBodU2eEx7KVHwKnpXwOyJayiKVHaikA6uPIanzYFvwmiExzyODFokdbFnhOsNyAt0Rltiwxw2PYXVNoMNiUJdjtOcgsfoUMlLPO++8Slnd8SQ2DlrxvKGOX9NAqJd0+vSLsX4vPLY1AC1a1NnqLbb+i7r2jJk0e5wcMOYx3nKuKwVOk9CIkeRkoFrjSHaI3S6GvQeCtoClnaxe2OOE8L4h4YLrMUePiCXuIL3qQi8aG3C4+u2N09RArwwkRr/E3KReqBj77LNPY+o6eyGxelue4x3ph5hnsgl1166pgTAeA3IMOIw9pOnSAopriMXeMTJFUmn+xFOQpYnYUSvC6MXuI3Req29CmykH4BrsdQvFlOBV4rM12Kivva6fJAgkxzMrJFYPddPSiB0pqiO8+OKLK9vGjhGD+CekbYMVM2SaUPc7pUpk0GcGb4qdkhfO7qMpOquPIbH6OmSIx6bgCWmpOyZ1jw4qto39rhY9L6vPhTqkDo3EgLpfqMiB9wdpMoiJ9VUhiF2PE+NaEtvehq1su+22pZ44Srt9jNhxLanz59K2PYvBoIp7FVsmKQeFbIQ6ZiGQpn0N3HvvvaUH1eqb0GYK1X6XfhjEMRobcMp2DHVkniLh1JmdggFZzay/Ysti4B3rRxj12WNZJFYfltuOTNR5GXIZSQPu6aefjpbnGHCKd7BxIb3QkgQxQ0adX/hjx5MnTx4WTyU3ebg+ngwHO5Wzww47DP3fZCSWom2DR2cXy9QCdfCa2uQnzsLAZqYjel2/RFM/OSAKhA4D30HeF2Xg2f3sM9Q1soYZn6kj/G0SBE8HzvlyPKV1tDXg6q4XCQcg4bUSqxPbl2nVmN5SZxjUIQNMcY0h+lmu0CjlPltD22ahgr0WPLWhd0bPy54zBwYlYUiBYortdk3AQEJow2xZijDuloF4OJWrtoX3MdynbmoPIVGM/zHcwu1zl2KJHdeSOn8ubduzGEqYyZ15ClHdt0te1d3rEDsrA7QDqbCbXPo14KhLTZJTUk6gXt87h8YGXOyG4z4OdXT+CryVjk5DqdXEycQCckcTvEH2ewh5jGLlBNJSccgutGVNaGPA1XmdVKbp05AcA04GOp44W9YLxI5opQ/T7/XrEXY0h4QeGu4zEsa3MQWnjlV1Cu8d9Sk2aMihbYPHaJC4DqsHpG79uti0jEVi9Snk1dN3svvK+2INY2WX21/v4LuFx9CyKMTL2GML1njCE45hwHOiM8WTynNqM6U2CAMuluVZ50nTDENsOprBaGq/EP02cdOBEZneiO04teCvNWgkoY5z2rAVJBwsKfEo9rzC/XrBAuKxZ4QRF8uWzUVik2jqQLQWJt81bEcUq2mnnp988slSb5c9Uhaw6gASnofEEXt+0CCK89F20Q7yfyp0R8drWk9C2rZnKcLvnIuygFMxypJYVivT+8Rw4pBQTC1xqszgIdTTpvUzpF8DjsGUBrExwnwBptXtr1U0qQu9aGzA2WxTpdGHOjwuscBDu91YoqktW3HolIjlC41SZeRpZEtjkPK45NLGgKMDZPkD+/unGiWl1tbpZcDpFwMQjFhb3gtlr+ozLy+dv40jZFqIaZVQx/exHk81mvrMSxwubQFIKvszl7YNHsI0QBiQKkO6riFWHEXMMABGbscdd1y5DRIu21NHmDVGxlfosQQET5ONf4olKmh76THewk4Pset+TZo0qYwxtcdo632DmHGQC1N6JMMQJB7q5eUinsnuw9SRGmB5XSxIXcwMsVTK1rXLa+SAwU1clD4TI4sQB2u3RcJsUxmAYaei7RQETtalneJEmj4vlvmoW/6C9qWpEce909I2SM5vboKCzDH4eK42WQ3vImL3k2y66abD9OFghWcY3i+cAdZA1uxRuA4cXsm6xDOBxDyuubRtz2LQlk+ZMqWiT8H31xJKtl0XtGf0rQh1PBZzyF/a1nB92ToHRhP6NeB0fs1ICJIGqQehI0EZy/Z6c+tCLxobcCCvAaIYDS0xIvDE2IUhkZwg5tEi/NkahKBs3XwekqaI7Np2iB25NaWNAacpk7BzR5iWrFs4MmXA0bH9/PPP5XHpIDFeeWlSL14dGtlKUjFPahAll19+eWUbUKA8YrPCALG6prRt8HSfyDSVkL0YWycvBIkZb6zBxv3X86CR4389k7oFdYXq7g033FApQ2wHBZwjFiOmeCjEZhhjBNJoSfCG2tjLQTW40MaAY5CAIUMSgwwqDIqc6S8lDMTWMkNskoBgmQ7eJxps7i2fOXfMWKxDSVcIA6KUMU9AeCgMRmOGGPHKErtuZr/PK2dKS8s+Wb2F58Igw74DtFO94veE+im7JA4gqdkCu16eUJ2Jrf0VLr2EcRjL9kRizyJEqyNYfRPatmcx8HqFa+nVQT9Enee5Ue95jqw1GXq+aTfCPoftGOxPnTq1cjx7PwhbsLGI/dCvAQeKFQ+FwSE2hd0W77ttV5FedSGHxgacnc9VGr0dzSOhkaMRSapD7wrWHdovbQy4fs+fMuC6iuqU1TelTYOH8Zv7I/UhGGNau6uf6eouQYNrPRT90saAa1pXYtOT1rMI1157beNjj2cG+bycf8k1ihnYppK9cmnTnqVAwrUGRwuMH7uuKMZek0TIFG0MuCYgYaZtbl3IoZEBJ7E6u3xIaOQo2I+p1zCuIDb66QKMoIkvYxRZN23Si37Xf+Hh95t9M60ZcGGdwhtpy3Oxrvsm4NVpGntnk2zCWKRpERrc0Mi1geNNaPPONTFK+LF0JJyyQ+x0cVhmvZNdJXxeeFNsudMcjGJ56qnDqUQ4JFWWS5v2LMVY9RsIA1xiauWxQvQdrWerCfSlTZb06YcwlEyhRLl1IYfGBlxorOEajaV1Mz2D+5qsQ3nsiLdg4Tz+j/1EU1fAgGOask3FaQPu+6ZeTGJm+FUJZaPROaXWwuoSqlO84LFpjdFAL2cuWszVit1uWoLEHwwfpu3aeNDawNQnQc9WXwfvumYRWByVKR+7jSDulDZhJDrP0SZ8XjZG2OkPph9JuCKmUokVFuK87OBuPDCITP9+0XkJpQl1GF8kh9kZwfEGjh6MNeK0J0yYUOpy6kIujQy4iRMnljdPhkDd1BFz+XZaFa9T7hIi4xmbETaajFUHOF6hTsXiyEaLQcRi/Bcg4HekR7t1MDXVT8wJgx2SMmwcbAyt12f1XWSsn9e0CMklsQVdgbrZNMFjtGC60masjyaxEJMuOSBicZ91daEJjQw4x3EcJw2dDauzW73jpODXPpCxiDHLYbx7uf7LuAHnOI7jOI7TMdyAcxzHcRzH6RhuwDmO4ziO43QMN+Acx3Ecx3E6hhtwjuM4juM4HcMNOMdxHMdxnI7hBpzjOI7jOE7HcAPOcRzHcRynY7gB5ziO4ziO0zHcgHMcx3Ecx+kYbsA5juM4juN0DDfgHMdxHMdxOoYbcI7jOI7jOB3DDTjHcRzHcZyO4Qac4ziO4zhOx3ADznEcx3Ecp2O4Aec4juM4jtMx/g8Gbm3rn+c8FAAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnAAAAAxCAYAAABaiJRDAAAT50lEQVR4Xu2dBZMrt9KG738JMzOfMDMzp8LMzMzMzFypcHJCFWZmZmZO5vueudV7tS3JnvF4155z3rfqqV33gO3RjNRqteT/TD311IUQQgghhGgP//EGIYQQQggx3MiBE0IIIYRoGXLghBBCCCFahhw4IYQQQoiWIQdOCCGEEKJlyIETQgghhGgZcuCEEEIIIVqGHDghhBBiMuP6668vpplmmsgu2oMcOCGE6MD3338f2YSABx98sFh44YUj+7CzwQYbFOiEE06Iton2IAeu5Rx44IHFb7/9Vrz99tvFqquuGm2vwswzz1zMOeecxbzzzlvMN9985d+55pqrmH322YuZZpop2l/8j5tuuqn4999/i4ceeijaJgbHNttsU3z99deNy+Xvv/8uZptttsgO0003XfmchM8NzxH7Tz/99NH+YtIEzTLLLJF9mHn66afLz+3tBvcydT/3OFG6GWaYYdR3xA7ha38OMfbIgWsxOA40VPw/7bTTlg/k7rvvHu3Xjffff7/49ttvy+PRTz/9VHz22WfFp59+WkYfTCeffHJ07OTKhhtuWHz11VcjDu4mm2xSXiO/X44333xz5LpWVaps0YQJEyL75MyPP/5YHHbYYeX/dcsl5LXXXisOPfTQyG68/PLLZefJhMP4ySefFJ9//vko+0orrRQdO7nQi/w54Ndffy3OOOOMyD4MzDrrrNnPPayg7777LrIbH330UfHPP/+ExTJqf9qCUNzvBAL8ecTYIgeupRAZ8NGxK6+8snyY/L5VMXk7vPPOO+W2gw8+ONrWb37//ffijz/+iOzDAj1RHFxvR4cffnhkT4EOOuigUbbtttuutKcqVpSKBKFdd901sk+ufPzxx8Viiy02ylanXIw111yzPM7bPeYg4rj5bWCaccYZo239Bj3zzDORfZCgBRdccJTt9ddfL+2+TFZbbbXS7s+x4447lvb33nsv2jYs0Al+/PHHI/uwgpZZZpnI7nn33XfLfT/88MNo2xNPPFFsttlmkV2MH3LgxhkiXd5Wl2OPPbbYf//9IzuVd6oCrMLaa69dHptriLbffvtye6/nrwN69NFHI3tTTj311GL55ZeP7HWhZ+ptgM4777zI7llhhRWSjdErr7xSnuOQQw6JtuWu+xJLLBHZ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>