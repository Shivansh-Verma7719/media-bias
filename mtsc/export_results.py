"""
Export MTSC analysis from the database to an Excel file.
Does aggregation in SQL to avoid loading 3M+ rows into memory.
Sheets: Summary by Company, Stance Distribution, Sample (1 lakh random rows).
"""

import os, sys
import psycopg2, psycopg2.extras
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.environ.get("POOLER_DATABASE_URL")
if not DB_URL:
    print("ERROR: POOLER_DATABASE_URL not found in environment.")
    sys.exit(1)

CONFIDENCE_THRESHOLD = 0.9
OUTPUT_EXCEL = "results/mtsc_results.xlsx"
SAMPLE_SIZE  = 100_000

os.makedirs("results", exist_ok=True)

conn = psycopg2.connect(DB_URL)
cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# ── 1. Overall stats ──────────────────────────────────────────────
print("Fetching overall stats...")
cur.execute("""
    SELECT
        COUNT(*)                                                     AS total_scored,
        COUNT(*) FILTER (WHERE GREATEST(pos_score, neutral_score, neg_score) >= %s)
                                                                     AS high_confidence,
        COUNT(*) FILTER (WHERE pos_score >= neutral_score AND pos_score >= neg_score)
                                                                     AS positive_count,
        COUNT(*) FILTER (WHERE neg_score >= pos_score AND neg_score >= neutral_score)
                                                                     AS negative_count,
        COUNT(*) FILTER (WHERE neutral_score >= pos_score AND neutral_score >= neg_score)
                                                                     AS neutral_count,
        ROUND(AVG(pos_score)::numeric, 6)                            AS avg_pos,
        ROUND(AVG(neg_score)::numeric, 6)                            AS avg_neg,
        ROUND(AVG(neutral_score)::numeric, 6)                        AS avg_neutral
    FROM articles_stratified
    WHERE pos_score IS NOT NULL
""", (CONFIDENCE_THRESHOLD,))
overall = cur.fetchone()
print(f"  Total scored: {overall['total_scored']:,}")

# ── 2. Summary by company ────────────────────────────────────────
print("Fetching per-company summary...")
cur.execute("""
    SELECT
        c.symbol                                                     AS ticker,
        c.name                                                       AS company_name,
        COUNT(*)                                                     AS total_articles,
        COUNT(*) FILTER (WHERE GREATEST(a.pos_score, a.neutral_score, a.neg_score) >= %s)
                                                                     AS high_confidence,
        COUNT(*) FILTER (WHERE a.pos_score >= a.neutral_score AND a.pos_score >= a.neg_score)
                                                                     AS positive,
        COUNT(*) FILTER (WHERE a.neg_score >= a.pos_score AND a.neg_score >= a.neutral_score)
                                                                     AS negative,
        COUNT(*) FILTER (WHERE a.neutral_score >= a.pos_score AND a.neutral_score >= a.neg_score)
                                                                     AS neutral,
        ROUND(AVG(a.pos_score)::numeric, 6)                          AS avg_pos_score,
        ROUND(AVG(a.neg_score)::numeric, 6)                          AS avg_neg_score,
        ROUND(AVG(a.neutral_score)::numeric, 6)                      AS avg_neutral_score
    FROM articles_stratified a
    JOIN top_companies c ON c.id = a.company_id
    WHERE a.pos_score IS NOT NULL
    GROUP BY c.symbol, c.name
    ORDER BY total_articles DESC
""", (CONFIDENCE_THRESHOLD,))
df_summary = pd.DataFrame(cur.fetchall())

# ── 3. High-confidence stance breakdown per company ───────────────
print("Fetching high-confidence stance breakdown...")
cur.execute("""
    WITH scored AS (
        SELECT
            c.symbol AS ticker,
            a.pos_score, a.neg_score, a.neutral_score,
            GREATEST(a.pos_score, a.neutral_score, a.neg_score) AS max_conf,
            CASE
                WHEN a.pos_score >= a.neutral_score AND a.pos_score >= a.neg_score THEN 'positive'
                WHEN a.neg_score >= a.pos_score AND a.neg_score >= a.neutral_score THEN 'negative'
                ELSE 'neutral'
            END AS stance
        FROM articles_stratified a
        JOIN top_companies c ON c.id = a.company_id
        WHERE a.pos_score IS NOT NULL
    )
    SELECT ticker, stance, COUNT(*) AS count
    FROM scored
    WHERE max_conf >= %s
    GROUP BY ticker, stance
    ORDER BY ticker, stance
""", (CONFIDENCE_THRESHOLD,))
rows = cur.fetchall()
df_stance_raw = pd.DataFrame(rows)
if not df_stance_raw.empty:
    df_stance = df_stance_raw.pivot(index="ticker", columns="stance", values="count").fillna(0).astype(int)
    df_stance["total"] = df_stance.sum(axis=1)
    df_stance = df_stance.sort_values("total", ascending=False)
else:
    df_stance = pd.DataFrame()

# ── 4. Random sample of 1 lakh rows ──────────────────────────────
print(f"Fetching {SAMPLE_SIZE:,} random sample rows...")
cur.execute("""
    SELECT
        a.id            AS original_id,
        c.symbol        AS ticker,
        c.name          AS company_name,
        LEFT(a.published_at::text, 10) AS published_at,
        a.title,
        a.source,
        a.url,
        ROUND(a.pos_score::numeric, 6)     AS prob_positive,
        ROUND(a.neg_score::numeric, 6)     AS prob_negative,
        ROUND(a.neutral_score::numeric, 6) AS prob_neutral,
        ROUND(GREATEST(a.pos_score, a.neutral_score, a.neg_score)::numeric, 6) AS max_confidence,
        CASE
            WHEN a.pos_score >= a.neutral_score AND a.pos_score >= a.neg_score THEN 'positive'
            WHEN a.neg_score >= a.pos_score AND a.neg_score >= a.neutral_score THEN 'negative'
            ELSE 'neutral'
        END AS stance
    FROM articles_stratified a
    JOIN top_companies c ON c.id = a.company_id
    WHERE a.pos_score IS NOT NULL
    ORDER BY RANDOM()
    LIMIT %s
""", (SAMPLE_SIZE,))
df_sample = pd.DataFrame(cur.fetchall())
print(f"  Sample fetched: {len(df_sample):,} rows")

cur.close()
conn.close()

# ── Write Excel ───────────────────────────────────────────────────
print(f"Writing Excel to {OUTPUT_EXCEL}...")
with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as w:
    # Sheet 1: Overall summary (single row)
    pd.DataFrame([overall]).to_excel(w, sheet_name="Overall Stats", index=False)

    # Sheet 2: Per-company summary
    df_summary.to_excel(w, sheet_name="Summary by Company", index=False)

    # Sheet 3: High-confidence stance pivot
    if not df_stance.empty:
        df_stance.to_excel(w, sheet_name=f"HC Stance (>{CONFIDENCE_THRESHOLD})")

    # Sheet 4: Random sample of 1 lakh rows
    df_sample.to_excel(w, sheet_name="Sample (1 Lakh)", index=False)

print(f"Done! Excel saved to {OUTPUT_EXCEL}")

# Print summary to console
print(f"\n{'='*50}")
print(f"Total scored:              {overall['total_scored']:,}")
print(f"High confidence (>{CONFIDENCE_THRESHOLD}):   {overall['high_confidence']:,}")
print(f"Positive / Negative / Neutral:  {overall['positive_count']:,} / {overall['negative_count']:,} / {overall['neutral_count']:,}")
print(f"Avg scores — pos: {overall['avg_pos']}  neg: {overall['avg_neg']}  neutral: {overall['avg_neutral']}")
print(f"{'='*50}")
