# MTSC Pipeline (Target-dependent Sentiment Classification)

This directory contains the sentiment extraction stage, where headline tone is evaluated.

## Overview

After the relevance classifier filters the corpus to only materially relevant headlines, those headlines must be scored for sentiment. To resolve instances where a headline might praise one firm but criticize another, a target-dependent model is used. 

## Implementation

The project relies on **NewsMTSC**, a model fine-tuned for multi-target dependent sentiment classification in political and financial news. For each relevant headline and its associated firm name, the model produces three probabilities:
- Positive ($`p_a^+`$)
- Negative ($`p_a^-`$)
- Neutral ($`p_a^0`$)

These are collapsed into a single daily firm-level stance score ($`bias_{i,t} = \frac{1}{N_{i,t}} \sum (p_a^+ - p_a^-)`$), resulting in a continuous variable bounded between -1 (wholly unfavorable) and +1 (wholly favorable).

## Scripts

- **`mtsc_pipeline.py`**: Executes the NewsMTSC model on the filtered headline corpus, writing the raw and derived stance scores to the database.
- **`export_results.py`**: Utilities for exporting the scored data for econometric analysis.
- **`helpers.py`**: Supporting functions for text preprocessing and handling.
