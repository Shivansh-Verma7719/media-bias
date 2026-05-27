#!/usr/bin/env python3
"""
STEP 1: Data Inventory & Structure Audit
=========================================
Covers Sections 1.1, 1.2, and 1.3 of VAR_Diagnostic_Checklist.md

Modes:
  Default (no flags):
    All rows in articles_no_title_deduped are considered relevant.
    "Before filter" = all rows; "After filter" = rows with pos_score IS NOT NULL.

  --relevance-csv <path>:
    Uses the ensemble predictions CSV (e.g. relevance_classifier_v2/full_predictions_ensemble.csv)
    to determine which articles are relevant (predicted_label = 'relevant').
    Relevant IDs are uploaded to a temp table and JOINed into queries.
    "Before filter" = all rows; "After filter" = rows marked relevant in CSV.

Outputs (written to diagnostics/step1/):
  - panel_structure.csv       (1.1: per-company article coverage)
  - stock_inventory.csv       (1.2: per-company stock data coverage)
  - temporal_frequency.csv    (1.3: articles per day/week/month per company)
  - freq_distribution.csv     (1.3: per-company daily article distribution stats)
  - step1_summary.md          (human-readable summary report)
  - plots/freq_hist_<ticker>.png  (1.3: histogram of articles/day per company)

Usage:
  # Default: all articles are relevant
  python step1_data_inventory.py

  # Use relevance CSV as source of truth
  python step1_data_inventory.py --relevance-csv ../relevance_classifier_v2/full_predictions_ensemble.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent dir so we can import db module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import run_query, get_connection  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ANALYSIS_START = "2015-01-01"
ANALYSIS_END   = "2025-12-31"

OUTPUT_DIR = Path(__file__).resolve().parent / "diagnostics" / "step1"
PLOT_DIR   = OUTPUT_DIR / "plots"

TRADING_DAYS_2015_2025 = 2520   # approximate trading days 2015–2025


# ---------------------------------------------------------------------------
# Relevance helpers
# ---------------------------------------------------------------------------

def load_relevant_ids(csv_path: str) -> set[int]:
    """Load the relevance CSV and return the set of article IDs marked relevant."""
    print(f"  Loading relevance CSV: {csv_path}")
    df = pd.read_csv(csv_path, usecols=["id", "predicted_label"])
    relevant = df[df["predicted_label"] == "relevant"]["id"]
    ids = set(relevant.astype(int).tolist())
    print(f"  → {len(ids):,} relevant articles out of {len(df):,} total in CSV")
    return ids


def create_relevant_temp_table(conn, relevant_ids: set[int]) -> None:
    """Create a temporary table _relevant_ids containing the relevant article IDs.

    This table lives for the duration of the connection and is used to filter
    article queries via JOIN.
    """
    print(f"  Uploading {len(relevant_ids):,} relevant IDs to temp table...")
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS _relevant_ids")
        cur.execute("CREATE TEMP TABLE _relevant_ids (id INTEGER PRIMARY KEY)")

        # Batch insert for speed
        batch_size = 10_000
        ids_list = list(relevant_ids)
        for i in range(0, len(ids_list), batch_size):
            batch = ids_list[i : i + batch_size]
            args = ",".join(cur.mogrify("(%s)", (aid,)).decode() for aid in batch)
            cur.execute(f"INSERT INTO _relevant_ids (id) VALUES {args}")

        conn.commit()
    print("  → Temp table _relevant_ids created.")


# ---------------------------------------------------------------------------
# SQL query builder — injects the relevance filter when needed
# ---------------------------------------------------------------------------

def _relevance_join(use_csv: bool) -> str:
    """Return the JOIN clause for relevance filtering (empty string if not using CSV)."""
    if use_csv:
        return "JOIN _relevant_ids rel ON rel.id = a.id"
    return ""


def _relevance_filter(use_csv: bool) -> str:
    """Return the WHERE-clause filter for the 'after filter' FILTER() expressions.

    Default mode:  pos_score IS NOT NULL  (articles that have been sentiment-scored)
    CSV mode:      always true (the JOIN already limits to relevant articles)
    """
    if use_csv:
        return "TRUE"  # JOIN already filters
    return "a.pos_score IS NOT NULL"


# ---------------------------------------------------------------------------
# 1.1 — Panel Structure Overview
# ---------------------------------------------------------------------------

def fetch_panel_structure(conn, use_csv: bool) -> pd.DataFrame:
    """Per-company article counts.

    "Before filter" = every row in articles_no_title_deduped for that company.
    "After filter"  = depends on mode (see _relevance_filter).
    """
    print("  [1.1] Fetching panel structure from DB...")

    rel_join   = _relevance_join(use_csv)
    rel_filter = _relevance_filter(use_csv)

    sql = f"""
        SELECT
            a.company_id,
            MAX(c.name)                 AS company_name,
            MAX(c.symbol)               AS ticker,
            COUNT(*)                    AS total_articles_before_filter,
            COUNT(*) FILTER (
                WHERE {rel_filter}
            )                           AS total_articles_after_filter,
            MIN(a.published_at::date) FILTER (
                WHERE {rel_filter}
            )                           AS first_article_date,
            MAX(a.published_at::date) FILTER (
                WHERE {rel_filter}
            )                           AS last_article_date,
            COUNT(DISTINCT a.published_at::date) FILTER (
                WHERE {rel_filter}
            )                           AS days_with_articles
        FROM public.articles_no_title_deduped a
        JOIN public.top_companies c ON c.id = a.company_id
        {rel_join}
        WHERE a.published_at::date BETWEEN %(start)s AND %(end)s
          AND a.company_id IS NOT NULL
        GROUP BY a.company_id
        ORDER BY company_name
    """
    df = pd.read_sql(sql, conn, params={"start": ANALYSIS_START, "end": ANALYSIS_END})

    # Derived columns
    df["first_article_date"] = pd.to_datetime(df["first_article_date"])
    df["last_article_date"]  = pd.to_datetime(df["last_article_date"])

    df["window_calendar_days"] = (df["last_article_date"] - df["first_article_date"]).dt.days + 1
    df["approx_trading_days"] = (df["window_calendar_days"] * 5 / 7).round().astype(int)
    df["days_zero_articles"]  = (df["approx_trading_days"] - df["days_with_articles"]).clip(lower=0)
    df["pct_trading_days_covered"] = (
        df["days_with_articles"] / df["approx_trading_days"] * 100
    ).round(1)

    df["flag_sparse"] = df["pct_trading_days_covered"] < 30

    return df


# ---------------------------------------------------------------------------
# 1.2 — Stock Price Data Inventory
# ---------------------------------------------------------------------------

def fetch_stock_inventory(conn) -> pd.DataFrame:
    """Per-company stock data coverage check using public.stock_prices."""
    print("  [1.2] Fetching stock price data inventory...")

    sql = """
        SELECT
            c.id            AS company_id,
            c.symbol        AS company_symbol,
            c.name          AS company_name,
            MIN(s.date::date)   AS stock_start,
            MAX(s.date::date)   AS stock_end,
            COUNT(*)            AS trading_day_rows
        FROM public.stock_prices s
        JOIN public.top_companies c ON c.symbol = s.ticker
        WHERE s.date::date BETWEEN %(start)s AND %(end)s
        GROUP BY c.id, c.symbol, c.name
        ORDER BY c.name
    """
    try:
        stock_df = pd.read_sql(sql, conn, params={"start": ANALYSIS_START, "end": ANALYSIS_END})
    except Exception as e:
        print(f"    ⚠️  Failed to query public.stock_prices: {e}")
        return pd.DataFrame(columns=[
            "company_id", "company_symbol", "company_name",
            "stock_start", "stock_end", "trading_day_rows",
            "note",
        ])

    stock_df["stock_start"] = pd.to_datetime(stock_df["stock_start"])
    stock_df["stock_end"]   = pd.to_datetime(stock_df["stock_end"])
    stock_df["window_calendar_days"] = (
        stock_df["stock_end"] - stock_df["stock_start"]
    ).dt.days + 1
    stock_df["approx_trading_days"] = (stock_df["window_calendar_days"] * 5 / 7).round().astype(int)
    stock_df["gap_rate_pct"] = (
        (1 - stock_df["trading_day_rows"] / stock_df["approx_trading_days"]) * 100
    ).round(1).clip(lower=0)

    return stock_df


# ---------------------------------------------------------------------------
# 1.3 — Temporal Frequency Decision
# ---------------------------------------------------------------------------

def fetch_temporal_frequency(conn, panel_df: pd.DataFrame, use_csv: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-company article counts by day/week/month and distribution stats."""
    print("  [1.3] Fetching temporal frequency data...")

    rel_join   = _relevance_join(use_csv)
    rel_filter = _relevance_filter(use_csv)

    # In CSV mode the JOIN already limits rows; in default mode we add a WHERE clause
    where_extra = f"AND {rel_filter}" if not use_csv else ""

    sql = f"""
        SELECT
            a.company_id,
            MAX(c.symbol)           AS ticker,
            a.published_at::date    AS article_date,
            COUNT(*)                AS articles_on_day
        FROM public.articles_no_title_deduped a
        JOIN public.top_companies c ON c.id = a.company_id
        {rel_join}
        WHERE a.published_at::date BETWEEN %(start)s AND %(end)s
          AND a.company_id IS NOT NULL
          {where_extra}
        GROUP BY a.company_id, a.published_at::date
        ORDER BY a.company_id, a.published_at::date
    """
    daily = pd.read_sql(sql, conn, params={"start": ANALYSIS_START, "end": ANALYSIS_END})
    daily["article_date"] = pd.to_datetime(daily["article_date"])

    # Build full date grid per company and fill zeros
    all_dates = pd.date_range(ANALYSIS_START, ANALYSIS_END, freq="D")
    companies = panel_df[["company_id", "ticker"]].drop_duplicates()

    rows = []
    for _, crow in companies.iterrows():
        cid    = crow["company_id"]
        ticker = crow["ticker"]
        cdata  = daily[daily["company_id"] == cid].set_index("article_date")["articles_on_day"]
        full   = cdata.reindex(all_dates, fill_value=0)

        avg_per_day   = full.mean()
        avg_per_week  = full.resample("W").sum().mean()
        avg_per_month = full.resample("ME").sum().mean()

        dist = full[full > 0]
        rows.append({
            "company_id":                     cid,
            "ticker":                         ticker,
            "avg_articles_per_day":            round(avg_per_day,   3),
            "avg_articles_per_week":           round(avg_per_week,  3),
            "avg_articles_per_month":          round(avg_per_month, 3),
            "median_articles_per_day_all":     round(full.median(), 3),
            "median_articles_per_day_nonzero": round(dist.median() if len(dist) else 0, 3),
            "p25_articles_per_day":            round(full.quantile(0.25), 3),
            "p75_articles_per_day":            round(full.quantile(0.75), 3),
            "max_articles_per_day":            int(full.max()),
            "days_with_zero_articles":         int((full == 0).sum()),
            "pct_zero_days":                   round((full == 0).sum() / len(full) * 100, 1),
        })

    freq_df = pd.DataFrame(rows)
    return daily, freq_df


def plot_frequency_histograms(daily_raw: pd.DataFrame, panel_df: pd.DataFrame) -> None:
    """Save one histogram of articles/day per company (zero-article days excluded)."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    companies = panel_df[["company_id", "ticker"]].drop_duplicates()

    for _, crow in companies.iterrows():
        cid    = crow["company_id"]
        ticker = str(crow["ticker"])
        cdata  = daily_raw[daily_raw["company_id"] == cid]["articles_on_day"]
        if cdata.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(cdata, bins=30, color="#4C72B0", edgecolor="white", linewidth=0.5)
        ax.set_title(f"{ticker} — Daily Article Count Distribution\n(zero-article days excluded)", fontsize=11)
        ax.set_xlabel("Articles per day")
        ax.set_ylabel("Frequency")
        ax.axvline(cdata.median(), color="crimson", linestyle="--", linewidth=1.2,
                   label=f"Median = {cdata.median():.0f}")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"freq_hist_{ticker}.png", dpi=100)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Write summary report
# ---------------------------------------------------------------------------

def write_summary(
    panel_df:  pd.DataFrame,
    stock_df:  pd.DataFrame,
    freq_df:   pd.DataFrame,
    mode_label: str,
) -> None:
    lines = []

    lines += [
        "# Step 1: Data Inventory & Structure Audit",
        f"**Analysis window:** {ANALYSIS_START} → {ANALYSIS_END}",
        f"**Approximate trading days in full window:** {TRADING_DAYS_2015_2025:,}",
        f"**Relevance source:** {mode_label}",
        "",
    ]

    # ---------- 1.1
    lines += [
        "---",
        "## 1.1 — Panel Structure Overview",
        "",
        "| Company | Ticker | Total Articles (all) | Articles (filtered) | First Article | Last Article | Days w/ Articles | Approx Trading Days | % Coverage | Sparse? |",
        "|---|---|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for _, r in panel_df.iterrows():
        flag = "⚠️ YES" if r["flag_sparse"] else "—"
        lines.append(
            f"| {r['company_name']} | {r['ticker']} "
            f"| {r['total_articles_before_filter']:,} "
            f"| {r['total_articles_after_filter']:,} "
            f"| {str(r['first_article_date'])[:10]} "
            f"| {str(r['last_article_date'])[:10]} "
            f"| {r['days_with_articles']:,} "
            f"| {r['approx_trading_days']:,} "
            f"| {r['pct_trading_days_covered']}% "
            f"| {flag} |"
        )

    n_sparse = int(panel_df["flag_sparse"].sum())
    lines += [
        "",
        f"> **{n_sparse} companies** flagged as sparse (<30% trading-day coverage). "
        f"These may need to be reviewed before modelling.",
        "",
    ]

    # ---------- 1.2
    lines += [
        "---",
        "## 1.2 — Stock Price Data Inventory",
        "",
    ]
    if stock_df.empty or "note" in stock_df.columns:
        lines += [
            "> ⚠️ **Stock price table not found.** Check DB schema/table names.",
            "  Available tables printed to console during script execution.",
            "",
        ]
    else:
        lines += [
            "| Company | Ticker | Stock Start | Stock End | Trading Day Rows | Est. Gap Rate |",
            "|---|---|---|---|---:|---:|",
        ]
        for _, r in stock_df.iterrows():
            lines.append(
                f"| {r.get('company_name', '—')} | {r.get('company_symbol', '—')} "
                f"| {str(r['stock_start'])[:10]} | {str(r['stock_end'])[:10]} "
                f"| {r['trading_day_rows']:,} | {r['gap_rate_pct']}% |"
            )
        lines.append("")

    # ---------- 1.3
    lines += [
        "---",
        "## 1.3 — Temporal Frequency Decision",
        "",
        "| Company | Avg/Day | Avg/Week | Avg/Month | Median/Day (all) | Median/Day (non-zero) | P75/Day | Max/Day | % Zero Days |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in freq_df.iterrows():
        lines.append(
            f"| {r['ticker']} "
            f"| {r['avg_articles_per_day']} "
            f"| {r['avg_articles_per_week']} "
            f"| {r['avg_articles_per_month']} "
            f"| {r['median_articles_per_day_all']} "
            f"| {r['median_articles_per_day_nonzero']} "
            f"| {r['p75_articles_per_day']} "
            f"| {r['max_articles_per_day']} "
            f"| {r['pct_zero_days']}% |"
        )

    # Frequency recommendation
    median_all = freq_df["median_articles_per_day_all"].median()
    if median_all < 1:
        rec = "**WEEKLY** (median daily article count across companies < 1 — daily is too granular)"
    elif median_all < 3:
        rec = "**WEEKLY** (median daily article count is low; weekly aggregation smooths sparsity)"
    else:
        rec = "**DAILY** (median daily article count is sufficient for daily VAR)"

    lines += [
        "",
        f"> **Frequency recommendation based on data:** {rec}",
        f"> Median articles/day across all companies (including zero days): **{median_all:.2f}**",
        "",
        f"Histograms saved to `{PLOT_DIR}/freq_hist_<TICKER>.png`",
    ]

    (OUTPUT_DIR / "step1_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  → Summary written to {OUTPUT_DIR / 'step1_summary.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1: Data Inventory & Structure Audit for VAR pipeline.",
    )
    parser.add_argument(
        "--relevance-csv",
        type=str,
        default=None,
        help=(
            "Path to the ensemble predictions CSV "
            "(e.g. relevance_classifier_v2/full_predictions_ensemble.csv). "
            "When provided, only articles with predicted_label='relevant' are used. "
            "Default: all articles in the table are treated as relevant."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_csv = args.relevance_csv is not None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== STEP 1: Data Inventory & Structure Audit ===\n")

    if use_csv:
        mode_label = f"Relevance CSV ({args.relevance_csv})"
        print(f"  Mode: RELEVANCE CSV\n")
        relevant_ids = load_relevant_ids(args.relevance_csv)
    else:
        mode_label = "Database (all articles treated as relevant)"
        print(f"  Mode: DATABASE (all articles = relevant)\n")

    # Use a single connection so the temp table persists across queries
    conn = get_connection()
    try:
        if use_csv:
            create_relevant_temp_table(conn, relevant_ids)

        # 1.1
        panel_df = fetch_panel_structure(conn, use_csv)
        panel_df.to_csv(OUTPUT_DIR / "panel_structure.csv", index=False)
        print(f"  → panel_structure.csv written ({len(panel_df)} companies)")

        # 1.2
        stock_df = fetch_stock_inventory(conn)
        if not stock_df.empty:
            stock_df.to_csv(OUTPUT_DIR / "stock_inventory.csv", index=False)
            print(f"  → stock_inventory.csv written ({len(stock_df)} companies)")

        # 1.3
        daily_raw, freq_df = fetch_temporal_frequency(conn, panel_df, use_csv)
        freq_df.to_csv(OUTPUT_DIR / "freq_distribution.csv", index=False)
        print("  → freq_distribution.csv written")
        print("  → Plotting histograms...")
        plot_frequency_histograms(daily_raw, panel_df)
        print(f"  → Histograms saved to {PLOT_DIR}/")

        # Summary report
        write_summary(panel_df, stock_df, freq_df, mode_label)
    finally:
        conn.close()

    print("\n=== STEP 1 COMPLETE ===")
    print(f"All outputs in: {OUTPUT_DIR.resolve()}")

    # Quick console preview
    print("\n--- Panel Structure (first 5 rows) ---")
    print(panel_df[["ticker", "total_articles_before_filter", "total_articles_after_filter",
                     "pct_trading_days_covered", "flag_sparse"]].to_string(index=False))

    print("\n--- Frequency Summary ---")
    print(freq_df[["ticker", "avg_articles_per_day", "avg_articles_per_week",
                   "median_articles_per_day_all", "pct_zero_days"]].to_string(index=False))


if __name__ == "__main__":
    main()
