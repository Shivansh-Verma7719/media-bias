import os, re, sys, argparse
import psycopg2, psycopg2.extras
from psycopg2 import sql
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from NewsSentiment import TargetSentimentClassifier
tsc = TargetSentimentClassifier()

from helpers import get_first_paragraph, split_on_aspect, choose_input_text

DB_URL = os.environ.get("POOLER_DATABASE_URL")
if not DB_URL:
    print("ERROR: POOLER_DATABASE_URL not found in environment.")
    sys.exit(1)

CONFIDENCE_THRESHOLD = 0.9

DB_SCHEMA = os.environ.get("DB_SCHEMA", "public")
TOP_COMPANIES_TABLE = os.environ.get("TOP_COMPANIES_TABLE", "top_companies")
ARTICLES_TABLE = os.environ.get("ARTICLES_TABLE", "articles_no_title_deduped")


# Step 1: Pull & filter articles from DB
def fetch_articles(sample_n: int | None = None) -> pd.DataFrame:
    print("STEP 1: Fetching & filtering articles from DB")

    conn = psycopg2.connect(DB_URL)
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Table is already title-checked and deduplicated upstream; fetch only unscored rows.
    query = sql.SQL("""
        SELECT
            a.id,
            a.company_id,
            COALESCE(c.symbol, '') AS ticker,
            COALESCE(c.name, '')   AS company_name,
            a.title,
            a.content,
            a.url,
            a.source,
            a.published_at
        FROM {}.{} a
        LEFT JOIN {}.{} c ON c.id = a.company_id
        WHERE a.pos_score IS NULL
        ORDER BY a.published_at DESC
    """).format(
        sql.Identifier(DB_SCHEMA),
        sql.Identifier(ARTICLES_TABLE),
        sql.Identifier(DB_SCHEMA),
        sql.Identifier(TOP_COMPANIES_TABLE),
    )

    params = []
    if sample_n:
        query += sql.SQL(" LIMIT %s")
        params.append(sample_n)

    cur.execute(query, tuple(params))
    fetched = cur.fetchall()

    rows_out = []
    for row in fetched:
        rows_out.append({
            "original_id":     row["id"],
            "company_id":      row["company_id"],
            "ticker":          row["ticker"],
            "company_name":    row["company_name"],
            "title":           (row["title"] or "").strip(),
            "first_paragraph": get_first_paragraph(row.get("content") or ""),
            "url":             row.get("url"),
            "source":          row.get("source"),
            "published_at":    str(row.get("published_at") or "")[:10],
        })

    conn.close()

    print(f"\n  Unscored rows fetched: {len(rows_out):,}")
    return pd.DataFrame(rows_out)


# Step 2: Run NewsMTSC
def run_mtsc(df: pd.DataFrame) -> pd.DataFrame:
    print("Step 2: Running NewsMTSC stance analysis")

    results = []
    errors  = 0

    for i, row in enumerate(df.itertuples(index=False), 1):
        company_name = row.company_name
        title        = row.title
        first_para   = str(row.first_paragraph or "")

        # Choose best input text
        text = choose_input_text(title, first_para, company_name)

        # Split around aspect
        left, target, right = split_on_aspect(text, company_name)

        # If context on either side is too short, pad with full text
        if len(left.strip()) < 3 or len(right.strip()) < 3:
            left, right = text, text

        try:
            probs_raw = tsc.infer_from_text(left, target, right)
        except Exception:
            # Fallback: use full title as both contexts
            try:
                title_text = str(row.title)
                probs_raw  = tsc.infer_from_text(title_text, target, title_text)
            except Exception as e2:
                errors += 1
                if errors <= 5:
                    print(f"  ⚠️  Row {i} ({row.ticker}): {e2}")
                continue

        try:
            probs = {p["class_label"]: round(float(p["class_prob"]), 6)
                     for p in probs_raw}

            pos    = probs.get("positive", 0.0)
            neg    = probs.get("negative", 0.0)
            neu    = probs.get("neutral",  0.0)
            conf   = max(pos, neg, neu)
            stance = max(probs, key=probs.get)

            results.append({
                "original_id":    row.original_id,
                "company_id":     row.company_id,
                "ticker":         row.ticker,
                "company_name":   row.company_name,
                "published_at":   row.published_at,
                "title":          title,
                "text_used":      text[:300],
                "left_context":   left[:150],
                "right_context":  right[:150],
                "source":         row.source,
                "url":            row.url,
                "prob_positive":  pos,
                "prob_negative":  neg,
                "prob_neutral":   neu,
                "max_confidence": conf,
                "stance":         stance,
                "high_confidence": conf >= CONFIDENCE_THRESHOLD,
            })
        except Exception as e:
            errors += 1
            if errors <= 5:   # only print first 5
                print(f"Row {i} error: {e}")
            continue

        if i % 50 == 0:
            print(f"  {i:,}/{len(df):,} ({i/len(df)*100:.0f}%) scored...")

    print(f"\n  Scored: {len(results):,} | Errors: {errors}")
    return pd.DataFrame(results)


# ── Step 3: Populate DB scores only ────────────────────────────────
def update_db_scores(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("STEP 3: Updating DB with MTSC scores")
    print("=" * 60)

    # 1. Save to DB
    print("  Updating pipeline scores in database...")
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    update_query = sql.SQL("""
        UPDATE {}.{}
        SET pos_score = %s, neutral_score = %s, neg_score = %s
        WHERE id = %s
    """).format(
        sql.Identifier(DB_SCHEMA),
        sql.Identifier(ARTICLES_TABLE),
    )
    
    updates = []
    for row in df.itertuples(index=False):
        updates.append((row.prob_positive, row.prob_neutral, row.prob_negative, row.original_id))
        
    try:
        psycopg2.extras.execute_batch(cur, update_query, updates)
        conn.commit()
        print(f"Database updated with {len(updates)} records.")
    except Exception as e:
        conn.rollback()
        print(f"Error updating database: {e}")
    finally:
        cur.close()
        conn.close()

    print(f"Results Summary")
    print(f"Total scored:            {len(df):,}")
    df_high = df[df["high_confidence"]]
    print(f"High confidence (>{CONFIDENCE_THRESHOLD}): {len(df_high):,}")
    print(f"Low confidence:          {len(df) - len(df_high):,}")
    print(f"\nStance (high confidence):")
    if df_high.empty:
        print("None")
    else:
        print(df_high["stance"].value_counts().to_string())
    print(f"\nPer company:")
    if df_high.empty:
        print("None")
    else:
        print(df_high.groupby("ticker")["stance"].value_counts().to_string())


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Fetch only N articles per company (for testing)")
    args = parser.parse_args()

    if args.sample:
        print(f"⚡ SAMPLE MODE: {args.sample} articles per company\n")

    df_clean  = fetch_articles(sample_n=args.sample)
    if df_clean.empty:
        print("No articles passed the filter. Try a larger --sample or check DB.")
        sys.exit(1)
    df_scored = run_mtsc(df_clean)
    if df_scored.empty:
        print("No articles were scored successfully. Check the error messages above.")
        sys.exit(1)
    update_db_scores(df_scored)
