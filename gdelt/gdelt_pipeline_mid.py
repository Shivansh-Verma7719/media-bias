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
CHECKPOINT_FILE = "gdelt/pipeline_checkpoint.json"  # Tracks last completed chunk end date
PROJECT_ID = "media-bias-ism" # Replace with your project ID
WORKER_THREADS = 6 # BigQuery free tier handles 5-10 concurrent queries well
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
CHUNK_SIZE_DAYS = 1 # Number of days each thread will query at once
MAXIMUM_BYTES_BILLED = 10**9

# ==========================================
# LOGGING SETUP
# ==========================================
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(threadName)s - %(message)s"
)

import json
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL must be set in .env")

# ==========================================
# CHECKPOINT HELPERS
# ==========================================
def load_checkpoint():
    """Returns the last completed chunk end date string, or None if no checkpoint exists."""
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
            return data.get("last_completed_end")  # e.g. "2021-06-03"
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def save_checkpoint(end_date: str, lock: threading.Lock):
    """Thread-safe write of the last completed chunk end date to the checkpoint file."""
    with lock:
        try:
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump({
                    "last_completed_end": end_date,
                    "updated_at": datetime.utcnow().isoformat()
                }, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save checkpoint: {e}")

# ==========================================
# STORAGE ABSTRACTION
# ==========================================
def get_domain(url):
    try:
        return urlparse(url).netloc.replace('www.', '')
    except:
        return None

class DataSink:
    """Base class for data storage to allow easy switching to a DB."""
    def save(self, rows):
        raise NotImplementedError("Save method must be implemented by subclasses.")

class PostgresDataSink(DataSink):
    """PostgreSQL implementation of DataSink for indian_cos schema."""
    def __init__(self, db_url, companies_cache):
        self.db_url = db_url
        self.companies_cache = companies_cache # List of dicts with id, name, terms

    def _get_connection(self):
        conn = psycopg2.connect(self.db_url)
        conn.autocommit = True
        return conn

    def save(self, rows):
        if not rows:
            return
            
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # 1. Map articles to companies based on terms in organizations
            articles_to_insert = []
            domains_to_insert = set()
            
            for row in rows:
                url = row['url']
                domain = get_domain(url)
                if domain:
                    domains_to_insert.add(domain)
                
                # Convert GDELT date (YYYYMMDDHHMMSS) to timestamp string or use as is
                # BigQuery DATE from gkg_partitioned is usually YYYY-MM-DD for the partition, but BigQuery timestamp string
                published_at = None
                try:
                    # Depending on the exact BigQuery output format, we might need to parse it. 
                    # If it's a date object from BQ client, we can convert it. 
                    if isinstance(row['date'], str):
                        # Assuming it's already a suitable string if it came directly from DATE cast
                        published_at = row['date']
                    else:
                        published_at = row['date'].strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    pass

                orgs = row['organizations'].lower() if row['organizations'] else ""
                
                # Find matching company
                matched_company_id = None
                for company in self.companies_cache:
                    for term in company['terms']:
                        # The regex already matched, but we do a simple string match here to find WHICH company
                        # Orgs string from GDELT is usually comma separated or semicolon separated
                        if f"{term}" in orgs:
                            matched_company_id = company['id']
                            break
                    if matched_company_id:
                        break
                        
                if matched_company_id:
                    articles_to_insert.append({
                        'title': url.split('/')[-1][:200], # Fallback title
                        'url': url,
                        'source': domain,
                        'published_at': published_at,
                        'company_id': matched_company_id,
                        'domain': domain # For media_outlet_id lookup later
                    })

            if not articles_to_insert:
                return

            # 2. Insert/Resolve Media Outlets
            if domains_to_insert:
                media_query = """
                    INSERT INTO indian_cos.media_outlets (domain, name) 
                    VALUES %s 
                    ON CONFLICT (domain) DO NOTHING;
                """
                psycopg2.extras.execute_values(
                    cursor,
                    media_query,
                    [(d, d) for d in domains_to_insert],
                    page_size=100
                )

            # Re-fetch media outlet mappings
            cursor.execute("SELECT domain, id FROM indian_cos.media_outlets WHERE domain = ANY(%s)", (list(domains_to_insert),))
            domain_to_id = {row[0]: row[1] for row in cursor.fetchall()}

            # 3. Insert Articles
            records = []
            for art in articles_to_insert:
                records.append((
                    art['title'],
                    None, # content
                    art['url'],
                    art['source'],
                    art['published_at'],
                    domain_to_id.get(art['domain']),
                    art['company_id']
                ))

            article_query = """
                INSERT INTO indian_cos.articles (
                    title, content, url, source, published_at, 
                    media_outlet_id, company_id
                ) VALUES %s
                ON CONFLICT (url) DO NOTHING;
            """
            
            psycopg2.extras.execute_values(
                cursor,
                article_query,
                records,
                page_size=100
            )

        except Exception as e:
            logging.error(f"Error saving to database: {e}")
        finally:
            cursor.close()
            conn.close()

# ==========================================
# PIPELINE LOGIC
# ==========================================
def load_search_terms_from_db():
    """Extracts unprocessed companies, aliases, and extra terms from DB to build a regex pattern and cache."""
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    try:
        # Fetch fully from indian_cos.companies where is_processed = false
        cursor.execute("SELECT id, name, symbol FROM indian_cos.companies WHERE is_processed = False;")
        companies = cursor.fetchall()
        
        all_regex_terms = set()
        companies_cache = []
        
        for row in companies:
            c_id = row['id']
            name = row['name'].lower().strip() if row['name'] else ""
            symbol = row['symbol'].lower().strip() if row['symbol'] else ""
            
            company_terms = set([name, symbol])
            company_terms = [t for t in company_terms if len(t) > 2] # Clean up tiny fragments
            
            if company_terms:
                companies_cache.append({
                    'id': c_id,
                    'name': name,
                    'terms': company_terms
                })
                all_regex_terms.update(company_terms)
                
        # Create a combined regex pattern
        valid_terms = list(all_regex_terms)
        if not valid_terms:
             return r'\b(NO_MATCHING_DATA_FORCE_FAIL)\b', []
             
        regex_pattern = r'\b(' + '|'.join(valid_terms) + r')\b'
        return regex_pattern, companies_cache
        
    except Exception as e:
        logging.error(f"Failed to load search terms from DB: {e}")
        return r'', []
    finally:
        cursor.close()
        conn.close()

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
        ],
        maximum_bytes_billed= 5 * 10**9
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

    # ---- Checkpoint: mark this chunk as done regardless of row count ----
    # Even empty chunks count as processed — we never want to re-scan them.
    save_checkpoint(end_date, stats["checkpoint_lock"])
    logging.debug(f"Checkpoint saved: last_completed_end={end_date}")

    # Calculate live speed
    elapsed = time.time() - start_time
    current_speed = stats["total_rows"] / elapsed if elapsed > 0 else 0
            
    progress.update(task_id, advance=1, speed=f"{current_speed:.2f}", description=f"[green]Completed {start_date} to {end_date}")
    return len(rows)

def main():
    console = Console()
    console.print(Panel.fit("[bold magenta]GDELT BigQuery Pipeline[/bold magenta]\nInitializing...", border_style="cyan"))

    # ---- Checkpoint: determine resume point --------------------------------
    last_completed = load_checkpoint()  # e.g. "2021-06-03" or None
    if last_completed:
        console.print(f"[bold yellow]Checkpoint found — resuming after {last_completed}[/bold yellow]")
        logging.info(f"Resuming from checkpoint. Last completed chunk end: {last_completed}")
    else:
        console.print("[yellow]No checkpoint found — starting from the beginning.[/yellow]")
        logging.info("No checkpoint found. Starting fresh from START_DATE.")

    # 2. Build Regex and load cache from DB
    console.print("[yellow]Loading unprocessed companies from DB and building regex...[/yellow]")
    regex_pattern, companies_cache = load_search_terms_from_db()
    logging.info(f"Loaded search regex. Length: {len(regex_pattern)} characters. Cached {len(companies_cache)} companies.")
    
    if not companies_cache:
        console.print("[bold red]No unprocessed companies found in the database. Exiting.[/bold red]")
        return
        
    # 1. Init Data Sink
    data_sink = PostgresDataSink(DATABASE_URL, companies_cache)
    
    # 3. Create Date Chunks — skip any already completed
    all_chunks = list(generate_date_chunks(START_DATE, END_DATE, CHUNK_SIZE_DAYS))
    if last_completed:
        chunks = [(s, e) for s, e in all_chunks if e > last_completed]
        skipped = len(all_chunks) - len(chunks)
        console.print(f"[dim]Skipping {skipped} already-completed chunks. {len(chunks)} remaining.[/dim]")
        logging.info(f"Skipping {skipped} chunks already completed. {len(chunks)} remaining.")
    else:
        chunks = all_chunks
    total_chunks = len(chunks)

    if not chunks:
        console.print("[bold green]All chunks already completed! Nothing to do.[/bold green]")
        logging.info("All chunks already completed per checkpoint. Pipeline exiting normally.")
        return
    
    # Shared stats for the UI (separate lock for checkpoint writes)
    stats = {
        "total_rows": 0,
        "lock": threading.Lock(),
        "checkpoint_lock": threading.Lock()
    }
    
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