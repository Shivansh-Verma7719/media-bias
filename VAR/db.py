import os
from pathlib import Path
import psycopg2
import pandas as pd
from dotenv import load_dotenv

# Find the root .env file by traversing upwards
def load_project_env():
    current_dir = Path(__file__).resolve().parent
    for parent in [current_dir] + list(current_dir.parents):
        env_path = parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            return env_path
    load_dotenv()  # Fallback to default search
    return None

# Load the env variables on import
env_loaded_path = load_project_env()

def resolve_db_url() -> str:
    db_url = os.getenv("POOLER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("Missing POOLER_DATABASE_URL/DATABASE_URL in environment.")
    return db_url

def get_connection(connect_timeout: int = 10):
    url = resolve_db_url()
    # Add connect_timeout to connection parameters if not already present
    if "connect_timeout" not in url and "?" not in url:
        conn = psycopg2.connect(url, connect_timeout=connect_timeout)
    elif "connect_timeout" not in url:
        conn = psycopg2.connect(f"{url}&connect_timeout={connect_timeout}")
    else:
        conn = psycopg2.connect(url)
    return conn

def run_query(sql: str, params=None, connect_timeout: int = 10) -> pd.DataFrame:
    """Executes a SQL query and returns a pandas DataFrame."""
    with get_connection(connect_timeout=connect_timeout) as conn:
        return pd.read_sql(sql, conn, params=params)

if __name__ == "__main__":
    print(f"Loaded .env from: {env_loaded_path}")
    print("Testing DB connection...")
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                res = cur.fetchone()
                print(f"Connection Successful! Test Query Result: {res}")
    except Exception as e:
        print(f"Connection Failed: {e}")
