"""
Fetch a random sample of titles from indian_cos.articles_stratified via psycopg2.
Used as input for zero-shot LLM annotation for the NIFTY pipeline.
"""
import os
import random
import argparse
import pandas as pd
import psycopg2
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

TABLE   = 'indian_cos.articles_stratified'
TOTAL   = 703_097


def get_conn():
    tenant_id = os.getenv('POOLER_TENANT_ID', 'your-tenant-id')
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=int(os.getenv('DB_PORT', 5432)),
        dbname=os.getenv('DB_NAME', 'postgres'),
        user=f"postgres.{tenant_id}",
        password=os.getenv('DB_PASSWORD'),
        connect_timeout=15,
    )


def main():
    parser = argparse.ArgumentParser(description="Fetch random NIFTY sample via psycopg2")
    parser.add_argument("--output", "-o", type=str, default="relevance_classifier/00_filtered_nifty_sample.csv")
    parser.add_argument("--target", "-t", type=int, default=4000)
    parser.add_argument("--batch_size", "-b", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    n_batches = (args.target + args.batch_size - 1) // args.batch_size
    offsets = sorted(random.sample(range(TOTAL - args.batch_size), n_batches))

    print(f"Connecting to {os.getenv('DB_HOST')} → {TABLE}")
    conn = get_conn()
    cur = conn.cursor()

    # Load company map for company_name
    print("Loading company map...")
    cur.execute("SELECT id, name FROM indian_cos.companies")
    company_map = {row[0]: row[1] for row in cur.fetchall()}
    print(f"  {len(company_map)} companies loaded.")

    print(f"Fetching {args.target} titles in {n_batches} batches of {args.batch_size}...")
    collected = []

    for offset in tqdm(offsets, desc="Fetching batches"):
        cur.execute(f"""
            SELECT id, title, pos_score, neutral_score, neg_score, company_id
            FROM {TABLE}
            WHERE title IS NOT NULL
            LIMIT %s OFFSET %s
        """, (args.batch_size, offset))
        rows = cur.fetchall()
        for row in rows:
            collected.append({
                'id':           row[0],
                'title':        row[1],
                'pos_score':    row[2],
                'neutral_score': row[3],
                'neg_score':    row[4],
                'company_id':   row[5],
                'company_name': company_map.get(row[5], ''),
            })

    cur.close()
    conn.close()

    random.shuffle(collected)
    collected = collected[:args.target]

    df = pd.DataFrame(collected)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} titles to {args.output}")
    print("\nSample titles:")
    for t in df['title'].head(10).tolist():
        print(f"  {t}")


if __name__ == "__main__":
    main()
