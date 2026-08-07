# Sampling Directory

This directory contains utility scripts and pipelines for generating representative samples of news articles for manual annotation.

## Overview
During the early phases of the project, a robust labeled dataset was required to train the relevance classifier. Because the full dataset consisted of millions of articles, these scripts were designed to intelligently sample articles across different firms, time periods, and sources, ensuring the resulting subset was stratified and balanced.

## Scripts

- **`create_stratified.py`**: Logic for performing stratified sampling across dimensions such as firm, time, and media outlet.
- **`load_filtered_articles_to_new_table.py`**: Database utility to move or isolate specific filtered samples into a new table for annotation tracking.
- **`main.py`**: The entry point for executing the sampling pipeline.
- **`pipelines/`, `matrix/`, `cdf/`, `diagnosis/`**: Submodules containing the specific mathematical and database routines needed to calculate the sampling distributions and verify the representativeness of the sampled data.
