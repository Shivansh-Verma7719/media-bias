# Quantitative Analysis of Media Bias and Stock Price Dynamics: The 2020 Shock

This repository contains the code, data pipelines, and analysis scripts for the research paper **"Quantitative Analysis of Media Bias and Stock Price Dynamics: The 2020 Shock"** by Shivansh Verma, Soham Tulsyan, Sashwat Dhanuka, and Anirban Sen. 

## Abstract & Motivation

Whether financial news influences stock prices or simply reflects information already incorporated into them remains an open question in financial economics. The COVID-19 pandemic provides a unique natural experiment to revisit this question due to its unprecedented disruption of both news coverage and financial markets. Existing studies largely approach the problem through aggregate sentiment measures, leaving it unclear whether the observed relationships hold at the level of individual firms.

We studied this question using 6.28 million news headlines covering 26 large United States firms between 2015 and 2025. After filtering the corpus to retain materially relevant firm-specific coverage, we construct daily stance measures and examine how their relationship with stock returns changed around the 2020 shock using panel regressions and vector autoregressions with data-driven structural breaks. Our findings indicate that the relationship between financial news and equity markets is heterogeneous across firms, and that no persistent market-wide change in media stance or stock returns occurred following the pandemic once common market shocks were accounted for.

## Architecture & Pipeline

The project is structured as a multi-stage data processing and analysis pipeline:

1. **Data Collection (`augmentation/`)**: We sourced raw news headlines from MediaCloud, gathering over 6 million headlines spanning 2015 to 2025 for heavily covered US firms.
2. **Relevance Classification (`relevance_classifier_v2/`)**: Since many headlines only mention firms in passing, we trained a custom DeBERTa-v3-base model to classify whether a headline is *materially relevant* to the firm. This filtered the dataset down to highly pertinent news.
3. **Sentiment/Stance Detection (`mtsc/`)**: For each relevant headline, a target-dependent sentiment classifier (NewsMTSC) determines the tone toward the specific firm mentioned, generating a signed stance score from -1 (unfavorable) to +1 (favorable).
4. **Market Data Integration (`finance/`)**: Daily stock prices for the firms, the S&P 500 index, and the VIX volatility index are collected and aligned with the daily stance scores.
5. **Statistical Analysis (`VAR/`)**: Panel regressions and Vector Autoregression (VAR) models with Bai-Perron structural break estimation are used to evaluate structural shifts and Granger causality between media stance and stock returns.

## Directory Structure

- **[`augmentation/`](file:///Users/sohamtulsyan/Documents/Coursework/ISM/media-bias/augmentation/README.md)**: Scripts for retrieving and processing raw headline data from MediaCloud.
- **[`relevance_classifier_v2/`](file:///Users/sohamtulsyan/Documents/Coursework/ISM/media-bias/relevance_classifier_v2/PIPELINE_DOCUMENTATION.md)**: Code, training data, and pipeline for the DeBERTa-v3-base model that predicts the financial relevance of a headline.
- **[`mtsc/`](file:///Users/sohamtulsyan/Documents/Coursework/ISM/media-bias/mtsc/README.md)**: Target-dependent sentiment classification pipeline for scoring headlines.
- **[`finance/`](file:///Users/sohamtulsyan/Documents/Coursework/ISM/media-bias/finance/README.md)**: Financial market data retrieval (Yahoo Finance).
- **[`VAR/`](file:///Users/sohamtulsyan/Documents/Coursework/ISM/media-bias/VAR/README.md)**: The dynamic Vector Autoregression and structural break scripts for RQ3 (Dynamic Lead-Lag Relationship).
- **[`panel_regressions/`](file:///Users/sohamtulsyan/Documents/Coursework/ISM/media-bias/panel_regressions/README.md)**: The static panel regression scripts for RQ1 (Stance Shift) and RQ2 (Return Shift).
- **[`sampling/`](file:///Users/sohamtulsyan/Documents/Coursework/ISM/media-bias/sampling/README.md)**: Utility scripts used for annotating and stratifying data samples.
- **[`scripts/`](file:///Users/sohamtulsyan/Documents/Coursework/ISM/media-bias/scripts/README.md)**: Standalone robustness check scripts.
- **[`helpers/`](file:///Users/sohamtulsyan/Documents/Coursework/ISM/media-bias/helpers/README.md)**: Shared database utilities.

## Note on Submissions

This repository contains only the relevant scripts, pipelines, and configuration necessary to reproduce the findings in the paper. Intermediate files, synthetic data sets for NLP training, and diagnostic checklists are included within their respective subdirectories.
