import os, re, sys, argparse
import psycopg2, psycopg2.extras
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from NewsSentiment import TargetSentimentClassifier
tsc = TargetSentimentClassifier()

from helpers import find_company_in_title, get_first_paragraph, split_on_aspect, choose_input_text

DB_URL = os.environ.get("POOLER_DATABASE_URL")
if not DB_URL:
    print("ERROR: POOLER_DATABASE_URL not found in environment.")
    sys.exit(1)

CONFIDENCE_THRESHOLD = 0.9
OUTPUT_CSV   = "results/mtsc_results.csv"
OUTPUT_EXCEL = "results/mtsc_results.xlsx"


# Step 1: Pull & filter articles from DB
def fetch_articles(sample_n: int | None = None) -> pd.DataFrame:
    print("STEP 1: Fetching & filtering articles from DB")

    conn = psycopg2.connect(DB_URL)
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("SELECT id, name, symbol FROM top_companies")
    companies = cur.fetchall()

    rows_out  = []
    seen      = set()   # (company_id, title_key) for dedup
    stats     = dict(fetched=0, dupes=0, kept=0)

    for comp in companies:
        company_id   = comp["id"]
        company_name = comp["name"]
        ticker       = comp["symbol"]

        limit_clause = f"LIMIT {sample_n}" if sample_n else ""
        # Only fetch articles that haven't been scored yet
        cur.execute(f"""
            SELECT id, title, content, url, source, published_at
            FROM   articles_stratified
            WHERE  company_id = %s
              AND  pos_score IS NULL
            ORDER  BY published_at DESC
            {limit_clause}
        """, (company_id,))

        fetched = cur.fetchall()
        stats["fetched"] += len(fetched)

        kept = dupes = 0
        for row in fetched:
            title = (row["title"] or "").strip()

            # Deduplicate
            key = (company_id, re.sub(r'\s+', ' ', title.lower().strip()))
            if key in seen:
                dupes += 1
                continue
            seen.add(key)

            rows_out.append({
                "original_id":     row["id"],
                "company_id":      company_id,
                "ticker":          ticker,
                "company_name":    company_name,
                "title":           title,
                "first_paragraph": get_first_paragraph(row.get("content") or ""),
                "url":             row.get("url"),
                "source":          row.get("source"),
                "published_at":    str(row.get("published_at") or "")[:10],
            })
            kept += 1

        stats["dupes"]    += dupes
        stats["kept"]     += kept
        print(f"  [{ticker}] fetched={len(fetched):,} | kept={kept:,} | "
              f"dupes={dupes}")

    conn.close()

    print(f"\n  Total fetched:       {stats['fetched']:,}")
    print(f"  Duplicates removed:  {stats['dupes']:,}")
    print(f"  Clean articles:      {stats['kept']:,}")
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


# ── Step 3: Save outputs & Populate DB ────────────────────────────
def save_outputs(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("STEP 3: Saving outputs & Updating DB")
    print("=" * 60)

    # 1. Save to DB
    print("  Updating pipeline scores in database...")
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    update_query = """
        UPDATE articles_stratified 
        SET pos_score = %s, neutral_score = %s, neg_score = %s
        WHERE id = %s
    """
    
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


    # 2. Save CSV / Excel
    df_high = df[df["high_confidence"]].copy()

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Full CSV:  {OUTPUT_CSV}  ({len(df):,} rows)")

    display_cols = [
        "ticker", "company_name", "published_at", "title",
        "prob_positive", "prob_negative", "prob_neutral",
        "max_confidence", "stance", "source", "url",
    ]

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df_high[display_cols].to_excel(
            writer, sheet_name=f"High Confidence (>{CONFIDENCE_THRESHOLD})", index=False)
        df[display_cols].to_excel(
            writer, sheet_name="All Scored", index=False)
        summary = (df_high.groupby(["ticker", "stance"])
                   .size().unstack(fill_value=0))
        summary["total"] = summary.sum(axis=1)
        summary.to_excel(writer, sheet_name="Summary by Company")

    print(f"Excel:     {OUTPUT_EXCEL}")

    print(f"Results Summary")
    print(f"Total scored:            {len(df):,}")
    print(f"High confidence (>{CONFIDENCE_THRESHOLD}): {len(df_high):,}")
    print(f"Ambiguous (rejected):    {len(df) - len(df_high):,}")
    print(f"\nStance (high confidence):")
    print(df_high["stance"].value_counts().to_string())
    print(f"\nPer company:")
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
    save_outputs(df_scored)
