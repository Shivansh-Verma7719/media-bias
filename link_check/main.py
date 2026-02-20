import os
import sys
import time
import queue
import threading
import requests
from requests.exceptions import RequestException
from supabase import create_client, Client
from dotenv import load_dotenv
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console, Group
from rich import box
from collections import deque

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: Missing Supabase credentials in .env file.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

NUM_WORKERS = 50
FETCH_BATCH_SIZE = 1000
PUSH_BATCH_SIZE = 100
REQUEST_TIMEOUT = 5

task_queue = queue.Queue()
push_buffer = []
buffer_lock = threading.Lock()
metrics_lock = threading.Lock()

global_metrics = {
    "total_fetched": 0,
    "total_processed": 0,
    "total_pushed": 0,
    "successful_links": 0,
    "failed_links": 0,
    "current_status": "Starting up..."
}

worker_states = {}
recent_errors = deque(maxlen=5)

def update_worker_state(worker_id, status, url="-"):
    if worker_id not in worker_states:
        worker_states[worker_id] = {"status": "Idle", "url": "-", "processed": 0}
    worker_states[worker_id]["status"] = status
    worker_states[worker_id]["url"] = url

def check_link(url: str) -> (int, bool):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.head(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if response.status_code in [405, 403, 400]:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, stream=True)
            response.close()
            
        status = response.status_code
        ok = 200 <= status < 400
        return status, ok
    except RequestException:
        return 0, False
    except Exception:
        return 0, False

def push_thread_worker():
    global push_buffer
    while True:
        with buffer_lock:
            should_push = len(push_buffer) >= PUSH_BATCH_SIZE or (len(push_buffer) > 0 and push_buffer[-1] is None)
            
        if should_push:
            with buffer_lock:
                batch_to_push = push_buffer.copy()
                push_buffer.clear()
            
            is_final = False
            if None in batch_to_push:
                is_final = True
                batch_to_push.remove(None)
                
            if len(batch_to_push) > 0:
                try:
                    # Deduplicate batch by ID to prevent ON CONFLICT DO UPDATE constraint errors
                    unique_batch = {item['id']: item for item in batch_to_push}.values()
                    
                    supabase.table("articles_sample").upsert(list(unique_batch)).execute()
                    with metrics_lock:
                        global_metrics["total_pushed"] += len(unique_batch)
                        global_metrics["current_status"] = f"Pushed {len(unique_batch)} to DB"
                except Exception as e:
                    with metrics_lock:
                        global_metrics["current_status"] = f"DB Push Error: {e}"
                    recent_errors.append(f"DB Push Error: {e}")
                    with buffer_lock:
                        push_buffer = batch_to_push + push_buffer
                    time.sleep(2)
            
            if is_final:
                break
        else:
            time.sleep(1)

def worker_thread(worker_id: int):
    update_worker_state(worker_id, "Idle")
    
    while True:
        try:
            task = task_queue.get(timeout=1)
        except queue.Empty:
            time.sleep(0.5)
            continue
            
        if task is None:
            task_queue.task_done()
            update_worker_state(worker_id, "Done")
            break
            
        try:
            article_id = task.get("id")
            url = task.get("url")
            
            if not url or not isinstance(url, str) or not url.startswith("http"):
                status_code, is_ok = 0, False
                update_worker_state(worker_id, "Skipped", "Invalid URL")
            else:
                update_worker_state(worker_id, "Checking", url)
                status_code, is_ok = check_link(url)
                
            with buffer_lock:
                push_buffer.append({
                    "id": article_id,
                    "link_health": status_code,
                    "link_ok": is_ok,
                    "url": url
                })
                
            with metrics_lock:
                global_metrics["total_processed"] += 1
                if is_ok:
                    global_metrics["successful_links"] += 1
                else:
                    global_metrics["failed_links"] += 1
                    
            if worker_id in worker_states:
                worker_states[worker_id]["processed"] += 1
                
        except Exception as e:
            recent_errors.append(f"Worker {worker_id} Error: {e}")
        finally:
            task_queue.task_done()

def generate_layout() -> Group:
    with metrics_lock:
        metrics_text = f"""[bold cyan]Total Fetched:[/bold cyan] {global_metrics['total_fetched']:,}
[bold magenta]Total Processed:[/bold magenta] {global_metrics['total_processed']:,}
[bold green]Successful Links:[/bold green] {global_metrics['successful_links']:,}
[bold red]Failed Links:[/bold red] {global_metrics['failed_links']:,}
[bold blue]Total DB Pushed:[/bold blue] {global_metrics['total_pushed']:,}
[bold yellow]Status:[/bold yellow] {global_metrics['current_status']}"""
    
    metrics_panel = Panel(metrics_text, title="Global Metrics", style="bold white", box=box.DOUBLE)
    
    with buffer_lock:
        pending_count = len(push_buffer)
    buffer_panel = Panel(f"Pending items in DB buffer: {pending_count} / {PUSH_BATCH_SIZE} before flush", title="Buffer Status", style="blue")
    
    table = Table(title="Worker Status", box=box.ROUNDED)
    table.add_column("Worker", justify="center", style="cyan", no_wrap=True)
    table.add_column("Status", style="yellow")
    table.add_column("Current URL", style="magenta", max_width=60, overflow="ellipsis")
    table.add_column("Processed", justify="right", style="green")
    
    for w_id in sorted(worker_states.keys()):
        state = worker_states[w_id]
        table.add_row(
            str(w_id),
            state.get("status", "Idle"),
            state.get("url", "-"),
            str(state.get("processed", 0))
        )
        
    error_panel = Panel(
        "\n".join(recent_errors) if recent_errors else "No errors",
        title="Recent Errors",
        style="red",
        box=box.ROUNDED,
        height=7
    )
        
    return Group(metrics_panel, buffer_panel, table, error_panel)

def fetch_batch():
    try:
        response = supabase.table("articles_sample") \
            .select("id, url") \
            .is_("link_ok", "null") \
            .limit(FETCH_BATCH_SIZE) \
            .execute()
        return response.data
    except Exception as e:
        recent_errors.append(f"Fetch Error: {e}")
        return []

def main():
    console = Console()
    console.print("[bold green]Starting Supabase Link Checking Pipeline...[/bold green]")
    
    for i in range(NUM_WORKERS):
        update_worker_state(i, "Starting...")
        
    workers = []
    for i in range(NUM_WORKERS):
        t = threading.Thread(target=worker_thread, args=(i,), daemon=True)
        t.start()
        workers.append(t)
        
    push_t = threading.Thread(target=push_thread_worker, daemon=True)
    push_t.start()
        
    with Live(generate_layout(), refresh_per_second=4) as live:
        try:
            while True:
                with metrics_lock:
                    global_metrics["current_status"] = "Fetching batch from DB..."
                live.update(generate_layout())
                
                batch = fetch_batch()
                
                if not batch:
                    # Let queue drain
                    while task_queue.unfinished_tasks > 0:
                        live.update(generate_layout())
                        time.sleep(0.5)
                        
                    batch = fetch_batch()
                    if not batch:
                        with metrics_lock:
                            global_metrics["current_status"] = "No more articles to process."
                        live.update(generate_layout())
                        break
                    
                with metrics_lock:
                    global_metrics["total_fetched"] += len(batch)
                    global_metrics["current_status"] = f"Queuing {len(batch)} tasks..."
                live.update(generate_layout())
                
                for item in batch:
                    if isinstance(item, dict):
                        task_queue.put(item)
                    elif hasattr(item, "id"):
                        task_queue.put({"id": getattr(item, "id", None), "url": getattr(item, "url", None)})
                    elif hasattr(item, "__dict__"):
                        task_queue.put(item.__dict__)
                    else:
                        task_queue.put(dict(item))
                    
                with metrics_lock:
                    global_metrics["current_status"] = "Waiting for workers to process batch..."
                
                # Polling instead of join prevents the main thread from blocking the Live update
                while task_queue.unfinished_tasks > 0:
                    live.update(generate_layout())
                    time.sleep(0.5)
                
        except KeyboardInterrupt:
            with metrics_lock:
                global_metrics["current_status"] = "Interrupted! Shutting down..."
            
    for _ in range(NUM_WORKERS):
        task_queue.put(None)
    for w in workers:
        w.join()
        
    with buffer_lock:
        push_buffer.append(None)
    push_t.join()
    
    console.print("[bold green]Pipeline finished successfully.[/bold green]")

if __name__ == "__main__":
    main()
