# Augmentation Pipeline

This directory contains the scripts responsible for the primary data collection and preprocessing of news headlines.

## Overview

The data collection strategy targets the output of national news outlets to build a comprehensive corpus of firm-specific coverage. We use MediaCloud, a service that continuously indexes national news output, to query for mentions of 26 large-capitalization United States firms over the period 2015 to 2025.

## Scripts

- **`media_cloud_augmentation_pipeline.py`**: The main script to query MediaCloud for firm-specific headlines, filter out poorly covered firms, and retain only those firms that maintain a steady stream of coverage (at least 3,000 headlines in most years).
- **`run.sh` & `run_nohup.sh`**: Shell scripts used to run the pipeline in the background.

## Data Preprocessing Steps

1. **Raw Pull**: A query across the candidate firms returns millions of raw headlines.
2. **Firm Filtering**: Firms with intermittent or sparse coverage are dropped to ensure a continuous daily/weekly series.
3. **De-duplication**: Identical wire reports published across multiple outlets are collapsed into a single record per firm.
4. **Name Matching**: Only headlines carrying the firm's name or ticker in the title are kept for subsequent relevance classification.
