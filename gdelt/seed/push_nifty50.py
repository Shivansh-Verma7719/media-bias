import os
import csv
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL must be set in .env")

CSV_FILE = os.path.join(os.path.dirname(__file__), "nifty50.csv")

def main():
    print(f"Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cursor = conn.cursor()

    print(f"Reading {CSV_FILE}...")
    with open(CSV_FILE, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        records = []
        for i, row in enumerate(reader):
            # Parse aliases and extra_terms
            aliases = [a.strip() for a in row.get('aliases', '').split('|') if a.strip()]
            extra_terms = [t.strip() for t in row.get('extra_terms', '').split('|') if t.strip()]
            
            # Map index_weight_pct and market_cap_usd, handling empty strings
            index_weight_pct = float(row['index_weight_pct']) if row.get('index_weight_pct') else None
            market_cap_usd = float(row['market_cap_usd']) if row.get('market_cap_usd') else None
            
            # Source verified is technically a string in CSV (e.g. "NSE Indices Factsheet..."), 
            # schema expects a boolean. We'll set to True if it's non-empty.
            source_verified = bool(row.get('source_verified'))
            
            # ID will be i+1 (1-indexed)
            records.append((
                i + 1,
                row['company_name'],
                row['ticker'],
                False, # is_processed
                None,  # last_error
                row.get('bse_code'),
                row.get('isin'),
                index_weight_pct,
                source_verified,
                market_cap_usd,
                aliases if aliases else None,
                extra_terms if extra_terms else None
            ))

    print(f"Inserting {len(records)} records into indian_cos.companies...")
    
    insert_query = """
        INSERT INTO indian_cos.companies (
            id, name, symbol, is_processed, last_error,
            bse_code, isin, index_weight_pct, source_verified, 
            market_cap_usd, aliases, extra_terms
        ) VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            symbol = EXCLUDED.symbol,
            bse_code = EXCLUDED.bse_code,
            isin = EXCLUDED.isin,
            index_weight_pct = EXCLUDED.index_weight_pct,
            source_verified = EXCLUDED.source_verified,
            market_cap_usd = EXCLUDED.market_cap_usd,
            aliases = EXCLUDED.aliases,
            extra_terms = EXCLUDED.extra_terms;
    """
    
    psycopg2.extras.execute_values(
        cursor,
        insert_query,
        records,
        template=None,
        page_size=100
    )
    
    print("Done!")

if __name__ == "__main__":
    main()
