GDELT BigQuery Extraction Pipeline

A highly concurrent, cost-optimized pipeline to extract news metadata from the GDELT 2.0 Global Knowledge Graph (GKG) via Google BigQuery. Built specifically for stock movement/media bias analysis.

Core Features

Cost Optimized: Combines all your target companies into a single regex search and partitions the dataset by date. This minimizes BigQuery byte scans, heavily optimizing usage for the 1TB/month Sandbox limits.

Multithreaded: Utilizes concurrent workers to query different date chunks simultaneously to speed up retrieval.

Modular Storage: Uses a DataSink interface. Currently writes to CSV, but can be seamlessly replaced with a PostgreSQL/MongoDB insert logic.

Terminal UI: Tracks real-time progress using the rich library.

Prerequisites

Python 3.8+

Dependencies:

pip install google-cloud-bigquery rich


Google Cloud Authentication:
You must authenticate using your Service Account Key. Set the environment variable before running the script:

Linux/macOS:

export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"


Windows (CMD):

set GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\service-account-key.json"


Configuration

Open pipeline.py and modify the CONFIGURATION section at the top:

PROJECT_ID: Set this to your actual GCP Project ID.

START_DATE / END_DATE: The timeframe you are analyzing.

CHUNK_SIZE_DAYS: Number of days to process in a single BigQuery run (default: 3).

WORKER_THREADS: Number of concurrent requests (default: 5). Keep this below 10 for Sandbox accounts to avoid API rate limits.

Changing the Input CSV

The pipeline expects a CSV named companies.csv (configurable in script). It specifically looks for the company_name, aliases, and extra_terms columns to construct the search logic.

To scrape a different set of companies, simply overwrite companies.csv and run the script again.

Upgrading to a Database

When you are ready to move away from CSV files, update the DataSink implementation. In pipeline.py, create a new class:

class PostgresDataSink(DataSink):
    def __init__(self, connection_string):
        self.conn = connect_to_db(connection_string)
        
    def save(self, rows):
        # Execute your bulk insert logic here
        pass

# In main():
# data_sink = PostgresDataSink("your_db_url")


Important Note on BigQuery Free Tier (Sandbox)

The GDELT GKG dataset is massive (several GBs per day). The GCP Sandbox provides 1 TB of query data per month for free.

Do not query years of data at once without checking limits.

The "Cheeky Workaround": If you hit the 1TB limit on your primary project before the 7-10 day deadline, simply create a new GCP project, generate a new Service Account Key, update your PROJECT_ID in the code, export the new key path to your terminal, and resume the script where it left off (update START_DATE).