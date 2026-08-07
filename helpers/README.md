# Helpers Directory

This directory contains shared utility scripts used across various parts of the data pipeline to interact with external services and databases.

## Scripts

- **`supabase_helper.py`**: A utility module that provides a consistent interface for connecting to and querying the project's PostgreSQL database hosted on Supabase (via the PostgREST API). It includes functions to:
  - Fetch unprocessed companies for data collection pipelines.
  - Update the state of data processing tasks to manage failures and resumes.
  - Resolve and map media outlet domains to their respective database IDs.
