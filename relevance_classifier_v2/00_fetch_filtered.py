"""
Stage 0: Fetch articles from Supabase with quality filters.

Fixes the three systematic DB tagging problems:
  1. Company-name-in-title filter  — drops articles where the company
     name (or a known alias) does not appear in the headline.
     Rationale: Media Cloud returns articles that mention the company
     anywhere in the body; we store only the title, so the only
     reliable signal is the headline.
  2. Excluded companies            — removes companies whose DB tags
     are known to be corrupted (e.g. Boeing → Indian politics).
  3. Near-duplicate collapse       — clusters titles by TF-IDF cosine
     similarity and keeps one article per cluster, preventing a
     single event (e.g. an AWS outage) from dominating the sample.

Output mirrors the format expected by 01_annotate.py:
  id, title, company_id, company_name

Usage:
  python 00_fetch_filtered.py -o 00_filtered_clean.csv [options]
"""
import os
import re
import argparse
import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# ── Company name → title-match keywords ──────────────────────────────────────
# An article is kept only if at least one keyword appears in the title
# (case-insensitive). Add aliases as needed.
# Keys must EXACTLY match the `name` column in the companies table.
# Values are title-level keywords — at least one must appear in the headline.
COMPANY_KEYWORDS: dict[str, list[str]] = {
    "Microsoft":                  ["microsoft"],
    "Apple Inc.":                 ["apple"],
    "Amazon":                     ["amazon", "aws", "whole foods"],
    "Alphabet Inc. (Class A)":    ["alphabet", "google", "youtube", "deepmind"],
    "Meta Platforms":             ["meta platforms", "facebook", "instagram", "whatsapp", "oculus"],
    "Nvidia":                     ["nvidia"],
    "Tesla, Inc.":                ["tesla"],
    "Berkshire Hathaway":         ["berkshire", "warren buffett"],
    "Lilly (Eli)":                ["eli lilly", "lilly"],
    "Broadcom":                   ["broadcom"],
    "Visa Inc.":                  ["visa"],
    "Mastercard":                 ["mastercard", "master card"],
    "ExxonMobil":                 ["exxonmobil", "exxon mobil"],
    "UnitedHealth Group":         ["unitedhealth", "united health"],
    "Johnson & Johnson":          ["johnson & johnson", "j&j", "janssen"],
    "Walmart":                    ["walmart", "wal-mart"],
    "Procter & Gamble":           ["procter & gamble", "procter and gamble", "p&g"],
    "Home Depot (The)":           ["home depot"],
    "Netflix":                    ["netflix"],
    "Salesforce":                 ["salesforce"],
    "Uber":                       ["uber"],
    "AT&T":                       ["at&t", "time warner", "directv"],
    "Intel":                      ["intel"],
    "General Motors":             ["general motors"],
    "Airbnb":                     ["airbnb"],
    "FedEx":                      ["fedex"],
    "Starbucks":                  ["starbucks"],
    "Wells Fargo":                ["wells fargo"],
    "Nike, Inc.":                 ["nike"],
    "Walt Disney Company (The)":  ["disney"],
    "Pfizer":                     ["pfizer"],
    "Coca-Cola Company (The)":    ["coca-cola", "coca cola"],
    "JPMorgan Chase":             ["jpmorgan", "jp morgan", "jamie dimon"],
    "Ford Motor Company":         ["ford motor", "ford "],
}

# ── Companies to exclude entirely (known DB tagging corruption or analyst-firm bias) ──
# Goldman Sachs, Morgan Stanley, JPMorgan are analyst firms — most of their
# article coverage is about them rating OTHER companies, not corporate events
# about themselves. They're in the DB but unsuitable as training subjects.
# Boeing: DB tagging corrupted (Indian politics/Bollywood articles tagged to Boeing).
EXCLUDED_COMPANIES: set[str] = {
    "Boeing",
    "Goldman Sachs",
    "Morgan Stanley",
}


def get_db_connection():
    host     = os.getenv("DB_HOST", "10.2.94.119")
    port     = int(os.getenv("DB_PORT", 5432))
    database = os.getenv("DB_NAME", "postgres")
    user     = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "")
    tenant   = os.getenv("POOLER_TENANT_ID", "")

    # When connecting from outside the docker network, Supavisor intercepts
    # port 5432 and requires username in the format "user.tenant_id".
    # Try plain first (works inside docker network / from VM directly),
    # fall back to Supavisor format if needed.
    try:
        return psycopg2.connect(
            host=host, port=port, database=database,
            user=user, password=password,
        )
    except psycopg2.OperationalError as e:
        if "Tenant or user not found" in str(e) and tenant:
            pooler_user = f"{user}.{tenant}"
            print(f"  Retrying via Supavisor with user='{pooler_user}'...")
            return psycopg2.connect(
                host=host, port=port, database=database,
                user=pooler_user, password=password,
            )
        raise


def fetch_articles(conn, company_name: str, limit_per_company: int) -> pd.DataFrame:
    """Fetch articles for a company, applying title-keyword filter in SQL."""
    keywords = COMPANY_KEYWORDS.get(company_name, [company_name.lower()])
    # Build ILIKE clauses
    ilike_clauses = " OR ".join(
        f"LOWER(a.title) LIKE '%%{kw.lower()}%%'" for kw in keywords
    )
    query = f"""
        SELECT a.id, a.title, a.company_id, c.name AS company_name
        FROM articles a
        JOIN companies c ON a.company_id = c.id
        WHERE c.name = %s
          AND a.title IS NOT NULL
          AND a.title != ''
          AND ({ilike_clauses})
        ORDER BY RANDOM()
        LIMIT %s
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, (company_name, limit_per_company * 5))  # oversample, then dedup
        rows = cur.fetchall()
    return pd.DataFrame(rows)


def deduplicate(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """
    Remove near-duplicate headlines using TF-IDF cosine similarity.
    Articles with pairwise similarity > threshold are collapsed to one.
    """
    if len(df) < 2:
        return df

    titles = df["title"].tolist()
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000, sublinear_tf=True)
    try:
        mat = vec.fit_transform(titles)
    except ValueError:
        return df  # too sparse to vectorize, keep all

    keep = [True] * len(titles)
    # Greedy: mark later duplicates as dropped
    # Process in chunks to avoid OOM on large sets
    chunk = 500
    for i in range(0, len(titles), chunk):
        sub = mat[i : i + chunk]
        sims = cosine_similarity(sub, mat)
        for local_idx, global_idx in enumerate(range(i, min(i + chunk, len(titles)))):
            if not keep[global_idx]:
                continue
            for j in range(global_idx + 1, len(titles)):
                if sims[local_idx, j] >= threshold:
                    keep[j] = False

    return df[keep].reset_index(drop=True)


def sample_company(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Sample up to n rows from a company's article pool."""
    if len(df) <= n:
        return df
    return df.sample(n, random_state=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",      "-o", default="00_filtered_clean.csv")
    parser.add_argument("--per_company", "-n", type=int, default=500,
                        help="Max articles to keep per company after dedup (default 500)")
    parser.add_argument("--dedup_threshold", "-d", type=float, default=0.85,
                        help="TF-IDF cosine threshold for near-duplicate removal (default 0.85)")
    parser.add_argument("--companies",   "-c", nargs="*", default=None,
                        help="Restrict to specific company names (default: all in COMPANY_KEYWORDS)")
    args = parser.parse_args()

    target_companies = args.companies or list(COMPANY_KEYWORDS.keys())
    target_companies = [c for c in target_companies if c not in EXCLUDED_COMPANIES]

    print(f"Connecting to DB at {os.getenv('DB_HOST', '10.2.94.119')}...")
    conn = get_db_connection()
    print(f"Connected. Fetching for {len(target_companies)} companies.\n")

    all_parts = []
    for company in target_companies:
        df = fetch_articles(conn, company, args.per_company)
        if df.empty:
            print(f"  {company}: 0 articles found — skipping")
            continue

        before_dedup = len(df)
        df = deduplicate(df, threshold=args.dedup_threshold)
        after_dedup = len(df)
        df = sample_company(df, args.per_company)

        print(f"  {company}: {before_dedup} fetched → {after_dedup} after dedup → {len(df)} sampled")
        all_parts.append(df)

    conn.close()

    if not all_parts:
        print("No articles fetched. Check DB connection and company names.")
        return

    combined = pd.concat(all_parts, ignore_index=True)
    combined = combined.drop_duplicates(subset="id")

    print(f"\nTotal: {len(combined)} articles across {combined['company_name'].nunique()} companies")
    print(combined["company_name"].value_counts().to_string())

    combined.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
