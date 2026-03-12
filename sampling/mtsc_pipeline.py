#!/usr/bin/env python3
"""
NewsMTSC Pipeline — full end-to-end

1. Pulls articles from DB (read-only)
2. Keeps ONLY articles where company alias appears in the title (word boundary)
3. Deduplicates by (company, normalized title)
4. Runs NewsMTSC with company alias as the ASPECT (target entity)
5. Saves all results + high-confidence filtered results to CSV + Excel

Usage:
  Run full pipeline:
    python sampling/mtsc_pipeline.py

  Run on small sample (N articles per company, for testing):
    python sampling/mtsc_pipeline.py --sample 50
"""
import os, re, sys, argparse
import psycopg2, psycopg2.extras
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────
DB_URL = os.getenv(
    "POOLER_DATABASE_URL",
    "postgresql://postgres.your-tenant-id:d6c612f32a9a01ed8106f3cebf526f7f@10.2.94.119:6543/postgres"
)
CONFIDENCE_THRESHOLD = 0.9
OUTPUT_CSV   = "sampling/mtsc_results.csv"
OUTPUT_EXCEL = "sampling/mtsc_results.xlsx"

# Company aliases: what must appear in the title (word-boundary matched)
COMPANY_ALIASES = {
    "AAPL":  ["Apple"],
    "ABNB":  ["Airbnb"],
    "BAC":   ["Bank of America", "BofA"],
    "FDX":   ["FedEx", "Federal Express"],
    "GM":    ["General Motors"],
    "GS":    ["Goldman Sachs", "Goldman"],
    "INTC":  ["Intel"],
    "MCD":   ["McDonald"],
    "MS":    ["Morgan Stanley"],
    "MSFT":  ["Microsoft"],
    "NDAQ":  ["Nasdaq"],
    "NFLX":  ["Netflix"],
    "NKE":   ["Nike"],
    "ORCL":  ["Oracle"],
    "PGR":   ["Progressive"],
    "SBUX":  ["Starbucks"],
    "T":     ["AT&T"],
    "TSLA":  ["Tesla"],
    "UBER":  ["Uber"],
    "V":     ["Visa"],
    "WFC":   ["Wells Fargo"],
    "WMT":   ["Walmart", "Wal-Mart"],
}

# ── Helpers ───────────────────────────────────────────────────────
def find_alias_in_title(title: str, aliases: list) -> str | None:
    """Return first alias found in title via word-boundary match, or None."""
    title_lower = title.lower()
    for alias in aliases:
        if re.search(r'\b' + re.escape(alias.lower()) + r'\b', title_lower):
            return alias
    return None


def get_first_paragraph(content: str) -> str:
    """Extract first substantial paragraph from content (capped at 1000 chars)."""
    if not content:
        return ""
    for para in re.split(r'\n{2,}|\r\n\r\n', content.strip()):
        para = para.strip()
        if len(para) > 80:
            return para[:1000]
    return content.strip()[:500]


def split_on_aspect(text: str, aspect: str):
    """
    Split text into (left, aspect, right) around the aspect string.
    NewsMTSC requires all three to be non-empty strings — uses single space if empty.
    """
    m = re.search(re.escape(aspect), text, re.IGNORECASE)
    if m:
        left  = text[:m.start()].strip() or " "
        mid   = text[m.start():m.end()]
        right = text[m.end():].strip()  or " "
        return left, mid, right
    # Aspect not found in text — use full text as left context
    return text.strip() or " ", aspect, " "


def choose_input_text(title: str, first_para: str, alias: str) -> str:
    """
    Prefer first_paragraph if it contains the alias and is long enough.
    Fall back to title (which is guaranteed to contain the alias).
    """
    if (first_para and len(first_para) > 100 and
            re.search(re.escape(alias), first_para, re.IGNORECASE)):
        return first_para
    return title


# ── Step 1: Pull & filter articles from DB ────────────────────────
def fetch_articles(sample_n: int | None = None) -> pd.DataFrame:
    print("=" * 60)
    print("STEP 1: Fetching & filtering articles from DB")
    print("=" * 60)

    conn = psycopg2.connect(DB_URL)
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute(
        "SELECT id, name, symbol FROM top_companies WHERE symbol = ANY(%s)",
        (list(COMPANY_ALIASES.keys()),)
    )
    company_map   = {row["symbol"]: row for row in cur.fetchall()}
    symbol_to_id  = {sym: info["id"] for sym, info in company_map.items()}

    rows_out  = []
    seen      = set()   # (company_id, title_key) for dedup
    stats     = dict(fetched=0, no_match=0, dupes=0, kept=0)

    for ticker, aliases in COMPANY_ALIASES.items():
        if ticker not in symbol_to_id:
            continue
        company_id   = symbol_to_id[ticker]
        company_name = company_map[ticker]["name"]

        limit_clause = f"LIMIT {sample_n}" if sample_n else ""
        cur.execute(f"""
            SELECT id, title, content, url, source, published_at
            FROM   articles_stratified
            WHERE  company_id = %s
              AND  title IS NOT NULL
              AND  length(title) > 10
            ORDER  BY published_at DESC
            {limit_clause}
        """, (company_id,))

        fetched = cur.fetchall()
        stats["fetched"] += len(fetched)

        kept = no_match = dupes = 0
        for row in fetched:
            title = (row["title"] or "").strip()

            # ① Company alias must be in the title
            alias = find_alias_in_title(title, aliases)
            if not alias:
                no_match += 1
                continue

            # ② Deduplicate
            key = (company_id, re.sub(r'\s+', ' ', title.lower().strip()))
            if key in seen:
                dupes += 1
                continue
            seen.add(key)

            rows_out.append({
                "original_id":     row["id"],
                "ticker":          ticker,
                "company_name":    company_name,
                "matched_alias":   alias,
                "title":           title,
                "first_paragraph": get_first_paragraph(row.get("content") or ""),
                "url":             row.get("url"),
                "source":          row.get("source"),
                "published_at":    str(row.get("published_at") or "")[:10],
            })
            kept += 1

        stats["no_match"] += no_match
        stats["dupes"]    += dupes
        stats["kept"]     += kept
        print(f"  [{ticker}] fetched={len(fetched):,} | kept={kept:,} | "
              f"no_title_match={no_match:,} | dupes={dupes}")

    conn.close()

    print(f"\n  Total fetched:       {stats['fetched']:,}")
    print(f"  No title match:      {stats['no_match']:,}")
    print(f"  Duplicates removed:  {stats['dupes']:,}")
    print(f"  Clean articles:      {stats['kept']:,}")
    return pd.DataFrame(rows_out)


# ── Step 2: Run NewsMTSC ──────────────────────────────────────────
def run_mtsc(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 2: Running NewsMTSC stance analysis")
    print(f"  Articles to score: {len(df):,}")
    print("=" * 60)

    from NewsSentiment import TargetSentimentClassifier
    tsc = TargetSentimentClassifier()
    print("  Model loaded.\n")

    results = []
    errors  = 0

    for i, row in enumerate(df.itertuples(index=False), 1):
        alias      = row.matched_alias
        title      = row.title
        first_para = str(row.first_paragraph or "")

        # Choose best input text
        text = choose_input_text(title, first_para, alias)

        # Split around aspect
        left, target, right = split_on_aspect(text, alias)

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
                "ticker":         row.ticker,
                "company_name":   row.company_name,
                "matched_alias":  alias,
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
                print(f"  ⚠️  Row {i} error: {e}")
            continue

        if i % 50 == 0:
            print(f"  {i:,}/{len(df):,} ({i/len(df)*100:.0f}%) scored...")

    print(f"\n  Scored: {len(results):,} | Errors: {errors}")
    return pd.DataFrame(results)


# ── Step 3: Save outputs ──────────────────────────────────────────
def save_outputs(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("STEP 3: Saving outputs")
    print("=" * 60)

    df_high = df[df["high_confidence"]].copy()

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  ✅ Full CSV:  {OUTPUT_CSV}  ({len(df):,} rows)")

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

    print(f"  ✅ Excel:     {OUTPUT_EXCEL}")

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total scored:            {len(df):,}")
    print(f"  High confidence (>{CONFIDENCE_THRESHOLD}): {len(df_high):,}")
    print(f"  Ambiguous (rejected):    {len(df) - len(df_high):,}")
    print(f"\n  Stance (high confidence):")
    print(df_high["stance"].value_counts().to_string())
    print(f"\n  Per company:")
    print(df_high.groupby("ticker")["stance"].value_counts().to_string())


# ── Main ──────────────────────────────────────────────────────────
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
