import os
import csv
import time
import logging
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError

# Rich UI imports
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.live import Live
from rich.panel import Panel

# ==========================================
# CONFIGURATION
# ==========================================
CSV_INPUT_FILE = "trial/nifty50.csv"
CSV_OUTPUT_FILE = "gdelt_articles.csv"
LOG_FILE = "pipeline.log"
PROJECT_ID = "media-bias-ism" # Replace with your project ID
WORKER_THREADS = 8 # BigQuery free tier handles 5-10 concurrent queries well
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
CHUNK_SIZE_DAYS = 3 # Number of days each thread will query at once

# ==========================================
# LOGGING SETUP
# ==========================================
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(threadName)s - %(message)s"
)

# ==========================================
# STORAGE ABSTRACTION
# ==========================================
class DataSink:
    """Base class for data storage to allow easy switching to a DB."""
    def save(self, rows):
        raise NotImplementedError("Save method must be implemented by subclasses.")

class CSVDataSink(DataSink):
    """CSV implementation of the DataSink with thread-safe writing."""
    def __init__(self, filename):
        self.filename = filename
        self.lock = threading.Lock()
        self.headers_written = False

    def save(self, rows):
        if not rows:
            return
            
        with self.lock:
            mode = 'a' if os.path.exists(self.filename) else 'w'
            with open(self.filename, mode, newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                if not self.headers_written and mode == 'w':
                    writer.writeheader()
                    self.headers_written = True
                writer.writerows(rows)

# ==========================================
# PIPELINE LOGIC
# ==========================================
def load_search_terms(csv_path):
    """Extracts company names, aliases, and extra terms to build a search regex."""
    terms = set()
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Add main company name
                if row.get('company_name'):
                    terms.add(row['company_name'].lower().strip())
                # Add aliases
                if row.get('aliases'):
                    for alias in row['aliases'].split('|'):
                        if alias.strip():
                            terms.add(alias.lower().strip())
                # Add extra terms
                if row.get('extra_terms'):
                    for term in row['extra_terms'].split('|'):
                        if term.strip():
                            terms.add(term.lower().strip())
    except Exception as e:
        logging.error(f"Failed to load CSV: {e}")
        
    # Clean up empty strings and create a regex pattern
    valid_terms = [t for t in terms if len(t) > 2] # Ignore tiny fragments
    regex_pattern = r'\b(' + '|'.join(valid_terms) + r')\b'
    return regex_pattern

def generate_date_chunks(start_date, end_date, chunk_days):
    """Yields tuple of (start, end) dates for partitioning queries."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    current = start
    while current <= end:
        chunk_end = current + timedelta(days=chunk_days - 1)
        if chunk_end > end:
            chunk_end = end
        yield (current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
        current = chunk_end + timedelta(days=1)

def query_gdelt_chunk(client, start_date, end_date, regex_pattern):
    """Executes the BigQuery search for a specific date chunk."""
    # We query the partitioned table to heavily restrict data scanned ($$$ saver)
    query = """
        SELECT 
            DATE as date,
            DocumentIdentifier as url,
            V2Organizations as organizations,
            V2Tone as tone
        FROM 
            `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE 
            _PARTITIONTIME >= TIMESTAMP(@start_date) 
            AND _PARTITIONTIME <= TIMESTAMP(@end_date)
            AND REGEXP_CONTAINS(LOWER(V2Organizations), @regex_pattern)
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "STRING", start_date),
            bigquery.ScalarQueryParameter("end_date", "STRING", end_date),
            bigquery.ScalarQueryParameter("regex_pattern", "STRING", regex_pattern),
        ]
    )
    
    try:
        start_time = time.time()
        query_job = client.query(query, job_config=job_config)
        results = query_job.result() # Waits for job to complete
        
        rows = []
        for row in results:
            rows.append({
                "date": row.date,
                "url": row.url,
                "organizations": row.organizations,
                "tone": row.tone
            })
            
        exec_time = time.time() - start_time
        speed = len(rows) / exec_time if exec_time > 0 else 0
        logging.info(f"Chunk {start_date} to {end_date}: Found {len(rows)} articles in {exec_time:.2f}s ({speed:.2f} articles/sec)")
        return rows
    except GoogleAPIError as e:
        logging.error(f"BigQuery API Error for {start_date} to {end_date}: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error for {start_date} to {end_date}: {e}")
        return []

def worker_task(chunk, regex_pattern, data_sink, progress, task_id, stats, start_time):
    """Worker function to process a chunk, save data, and update UI."""
    start_date, end_date = chunk
    client = bigquery.Client(project=PROJECT_ID)
    
    # Update UI to show processing
    progress.update(task_id, description=f"[cyan]Querying {start_date} to {end_date}...")
    
    rows = query_gdelt_chunk(client, start_date, end_date, regex_pattern)
    
    if rows:
        data_sink.save(rows)
        with stats["lock"]:
            stats["total_rows"] += len(rows)
            
    # Calculate live speed
    elapsed = time.time() - start_time
    current_speed = stats["total_rows"] / elapsed if elapsed > 0 else 0
            
    progress.update(task_id, advance=1, speed=f"{current_speed:.2f}", description=f"[green]Completed {start_date} to {end_date}")
    return len(rows)

def main():
    console = Console()
    console.print(Panel.fit("[bold magenta]GDELT BigQuery Pipeline[/bold magenta]\nInitializing...", border_style="cyan"))

    # 1. Init Data Sink
    data_sink = CSVDataSink(CSV_OUTPUT_FILE)
    
    # 2. Build Regex from CSV
    console.print("[yellow]Loading companies and building regex...[/yellow]")
    regex_pattern = load_search_terms(CSV_INPUT_FILE)
    logging.info(f"Loaded search regex. Length: {len(regex_pattern)} characters.")
    
    # 3. Create Date Chunks
    chunks = list(generate_date_chunks(START_DATE, END_DATE, CHUNK_SIZE_DAYS))
    total_chunks = len(chunks)
    
    # Shared stats for the UI
    stats = {"total_rows": 0, "lock": threading.Lock()}
    
    # 4. Setup Rich UI Progress
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[bold yellow]Speed: {task.fields[speed]} rows/s"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    )
    
    start_time = time.time()
    overall_task = progress.add_task("[bold blue]Overall Pipeline Progress", total=total_chunks, speed="0.00")
    
    # 5. Execute Thread Pool
    with Live(progress, refresh_per_second=10):
        with ThreadPoolExecutor(max_workers=WORKER_THREADS) as executor:
            futures = []
            for chunk in chunks:
                futures.append(
                    executor.submit(worker_task, chunk, regex_pattern, data_sink, progress, overall_task, stats, start_time)
                )
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Thread failed: {e}")

    # 6. Final Report
    elapsed = time.time() - start_time
    speed = stats["total_rows"] / elapsed if elapsed > 0 else 0
    
    summary = (
        f"Pipeline Completed!\n"
        f"Time Taken: {elapsed:.2f} seconds\n"
        f"Total Articles Found: {stats['total_rows']}\n"
        f"Speed: {speed:.2f} rows/second"
    )
    console.print(Panel.fit(summary, title="[bold green]Done", border_style="green"))

if __name__ == "__main__":
    # Ensure GCP credentials are set in the environment
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        print("ERROR: GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
        print("Please set it pointing to your service account JSON key file.")
        exit(1)
        
    main()