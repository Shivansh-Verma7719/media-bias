#!/usr/bin/env python3
"""
STEP 2: Bias Index Construction & Quality
==========================================
Covers Sections 2.1, 2.2, and 2.3 of VAR_Diagnostic_Checklist.md

Modes:
  Default (no flags):
    All rows in articles_no_title_deduped with pos_score IS NOT NULL are used.

  --relevance-csv <path>:
    Only articles marked relevant in the ensemble predictions CSV are used.

Company filter:
  Only the 17 companies selected after Step 1 are processed:
    ABNB, AMZN, T, BA, BAC, GM, GS, INTC, MCD, MSFT, MS,
    NFLX, SBUX, UBER, V, WMT, WFC

Outputs (written to diagnostics/step2/):
  2.1 — Stance Score Distribution
    - stance_distribution_stats.csv    (per-company summary stats)
    - plots/stance_hist_<ticker>.png   (histogram per company)
    - plots/stance_hist_all.png        (aggregated histogram)

  2.2 — Daily Bias Index
    - daily_bias_index.csv             (company_id, date, mean, median, weighted_mean, n_articles)
    - daily_bias_quality.csv           (per-company quality flags)

  2.3 — Weekly Aggregation
    - weekly_bias_index.csv            (company_id, week, weighted_mean, n_articles, n_days_covered)
    - weekly_coverage.csv              (per-company: total weeks, empty weeks, etc.)
    - plots/weekly_ts_<ticker>.png     (time-series plot per company)

  - step2_summary.md                   (human-readable report)

Usage:
  python step2_bias_index.py
  python step2_bias_index.py --relevance-csv ../relevance_classifier_v2/full_predictions_ensemble.csv
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
from scipy import stats as sp_stats

# Add parent dir so we can import db module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_connection  # noqa: E402

# Re-use the relevance helpers from step1
from step1_data_inventory import (  # noqa: E402
    load_relevant_ids,
    create_relevant_temp_table,
    _relevance_join,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ANALYSIS_START = "2015-01-01"
ANALYSIS_END   = "2025-12-31"

OUTPUT_DIR = Path(__file__).resolve().parent / "diagnostics" / "step2"
PLOT_DIR   = OUTPUT_DIR / "plots"

# 17 companies selected after Step 1 review
SELECTED_TICKERS = [
    "ABNB", "AMZN", "T", "BA", "BAC", "GM", "GS", "INTC",
    "MCD", "MSFT", "MS", "NFLX", "SBUX", "UBER", "V", "WMT", "WFC",
]


# ---------------------------------------------------------------------------
# 2.1 — Stance Score Distribution
# ---------------------------------------------------------------------------

def fetch_stance_scores(conn, use_csv: bool) -> pd.DataFrame:
    """Fetch raw stance scores (pos_score - neg_score) for selected companies."""
    print("  [2.1] Fetching stance scores...")

    rel_join = _relevance_join(use_csv)

    # In CSV mode the JOIN limits rows; in default mode we require scores to exist
    where_extra = "" if use_csv else "AND a.pos_score IS NOT NULL"

    tickers_placeholder = ",".join(["%s"] * len(SELECTED_TICKERS))

    sql = f"""
        SELECT
            a.id              AS article_id,
            a.company_id,
            c.symbol          AS ticker,
            a.published_at::date AS article_date,
            a.pos_score,
            a.neg_score,
            (a.pos_score - a.neg_score) AS stance_score
        FROM public.articles_no_title_deduped a
        JOIN public.top_companies c ON c.id = a.company_id
        {rel_join}
        WHERE a.published_at::date BETWEEN %s AND %s
          AND a.company_id IS NOT NULL
          AND a.pos_score IS NOT NULL
          AND c.symbol IN ({tickers_placeholder})
        ORDER BY c.symbol, a.published_at
    """
    params = [ANALYSIS_START, ANALYSIS_END] + SELECTED_TICKERS
    df = pd.read_sql(sql, conn, params=params)
    df["article_date"] = pd.to_datetime(df["article_date"])
    print(f"  → {len(df):,} articles with stance scores fetched")
    return df


def compute_stance_distribution(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Per-company summary statistics for stance scores."""
    print("  [2.1] Computing distribution statistics...")

    rows = []
    for ticker in SELECTED_TICKERS:
        s = scores_df[scores_df["ticker"] == ticker]["stance_score"]
        if s.empty:
            continue
        rows.append({
            "ticker":      ticker,
            "n_articles":  len(s),
            "mean":        round(s.mean(), 4),
            "std":         round(s.std(), 4),
            "skewness":    round(sp_stats.skew(s, nan_policy="omit"), 4),
            "kurtosis":    round(sp_stats.kurtosis(s, nan_policy="omit"), 4),
            "min":         round(s.min(), 4),
            "p25":         round(s.quantile(0.25), 4),
            "median":      round(s.median(), 4),
            "p75":         round(s.quantile(0.75), 4),
            "max":         round(s.max(), 4),
            "pct_near_zero":  round((s.abs() <= 0.05).sum() / len(s) * 100, 1),
            "pct_exactly_zero": round((s == 0).sum() / len(s) * 100, 1),
        })
    return pd.DataFrame(rows)


def plot_stance_histograms(scores_df: pd.DataFrame) -> None:
    """Per-company and aggregate stance score histograms."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-company
    for ticker in SELECTED_TICKERS:
        s = scores_df[scores_df["ticker"] == ticker]["stance_score"]
        if s.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(s, bins=50, color="#4C72B0", edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.axvline(s.mean(), color="crimson", linestyle="--", linewidth=1.2,
                   label=f"Mean = {s.mean():.3f}")
        ax.axvline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"{ticker} — Stance Score Distribution (n={len(s):,})", fontsize=11)
        ax.set_xlabel("Stance score (pos − neg)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"stance_hist_{ticker}.png", dpi=100)
        plt.close(fig)

    # Aggregate
    all_scores = scores_df["stance_score"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_scores, bins=80, color="#2E86AB", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.axvline(all_scores.mean(), color="crimson", linestyle="--", linewidth=1.2,
               label=f"Mean = {all_scores.mean():.3f}")
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_title(f"All Companies — Stance Score Distribution (n={len(all_scores):,})", fontsize=11)
    ax.set_xlabel("Stance score (pos − neg)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "stance_hist_all.png", dpi=100)
    plt.close(fig)
    print(f"  → Stance histograms saved to {PLOT_DIR}/")


# ---------------------------------------------------------------------------
# 2.2 — Daily Bias Index
# ---------------------------------------------------------------------------

def compute_daily_bias_index(scores_df: pd.DataFrame) -> pd.DataFrame:
    """For each company × day, compute mean, median, weighted-mean stance."""
    print("  [2.2] Computing daily bias index...")

    grouped = scores_df.groupby(["company_id", "ticker", "article_date"])

    daily = grouped.agg(
        n_articles=("stance_score", "size"),
        mean_stance=("stance_score", "mean"),
        median_stance=("stance_score", "median"),
    ).reset_index()

    # Weighted mean is same as mean here (weight = 1 per article), but we keep
    # the column for consistency with the weekly aggregation where it matters.
    daily["weighted_mean_stance"] = daily["mean_stance"]

    # Sort
    daily = daily.sort_values(["ticker", "article_date"]).reset_index(drop=True)
    print(f"  → {len(daily):,} company-day observations")
    return daily


def compute_daily_quality_flags(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Per-company quality diagnostics for the daily index."""
    print("  [2.2] Computing daily quality flags...")

    rows = []
    for ticker in SELECTED_TICKERS:
        d = daily_df[daily_df["ticker"] == ticker]
        if d.empty:
            continue

        n_single = int((d["n_articles"] == 1).sum())
        pct_single = round(n_single / len(d) * 100, 1)
        out_of_range = int(((d["mean_stance"] < -1) | (d["mean_stance"] > 1)).sum())

        # Consecutive-day swings > 1.0
        sorted_d = d.sort_values("article_date")
        diff = sorted_d["mean_stance"].diff().abs()
        big_swings = int((diff > 1.0).sum())

        rows.append({
            "ticker":                 ticker,
            "total_days":             len(d),
            "days_single_article":    n_single,
            "pct_single_article":     pct_single,
            "out_of_range_days":      out_of_range,
            "big_swing_days":         big_swings,
            "mean_daily_articles":    round(d["n_articles"].mean(), 2),
            "median_daily_articles":  int(d["n_articles"].median()),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2.3 — Weekly Aggregation
# ---------------------------------------------------------------------------

def compute_weekly_bias_index(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily bias index to ISO weeks, weighted by article count."""
    print("  [2.3] Aggregating to weekly frequency...")

    df = daily_df.copy()
    # ISO year-week label for grouping
    df["year"]    = df["article_date"].dt.isocalendar().year.astype(int)
    df["week"]    = df["article_date"].dt.isocalendar().week.astype(int)
    df["iso_week"] = df["article_date"].dt.to_period("W")

    # Article-count-weighted mean per week
    def _weighted_mean(group):
        weights = group["n_articles"]
        return np.average(group["mean_stance"], weights=weights)

    weekly = df.groupby(["company_id", "ticker", "iso_week"]).apply(
        lambda g: pd.Series({
            "weighted_mean_stance": np.average(g["mean_stance"], weights=g["n_articles"]),
            "simple_mean_stance":   g["mean_stance"].mean(),
            "median_stance":        g["median_stance"].median(),
            "n_articles":           int(g["n_articles"].sum()),
            "n_days_covered":       len(g),
            "week_start":           g["article_date"].min(),
            "week_end":             g["article_date"].max(),
        }),
        include_groups=False,
    ).reset_index()

    weekly["iso_week"] = weekly["iso_week"].astype(str)
    weekly = weekly.sort_values(["ticker", "iso_week"]).reset_index(drop=True)
    print(f"  → {len(weekly):,} company-week observations")
    return weekly


def compute_weekly_coverage(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Per-company summary of weekly coverage (how many weeks have data)."""
    print("  [2.3] Computing weekly coverage stats...")

    # Total weeks in the analysis window
    all_weeks = pd.date_range(ANALYSIS_START, ANALYSIS_END, freq="W")
    total_weeks = len(all_weeks)

    rows = []
    for ticker in SELECTED_TICKERS:
        w = weekly_df[weekly_df["ticker"] == ticker]
        if w.empty:
            continue

        n_weeks = len(w)
        n_empty = total_weeks - n_weeks
        rows.append({
            "ticker":            ticker,
            "total_weeks_window": total_weeks,
            "weeks_with_data":   n_weeks,
            "weeks_empty":       n_empty,
            "pct_covered":       round(n_weeks / total_weeks * 100, 1),
            "mean_articles_per_week": round(w["n_articles"].mean(), 1),
            "median_articles_per_week": int(w["n_articles"].median()),
        })

    return pd.DataFrame(rows)


def plot_weekly_time_series(weekly_df: pd.DataFrame) -> None:
    """Time-series plot of weekly bias index per company."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in SELECTED_TICKERS:
        w = weekly_df[weekly_df["ticker"] == ticker].copy()
        if w.empty:
            continue

        w["week_dt"] = pd.to_datetime(w["week_start"])
        w = w.sort_values("week_dt")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})

        # Top: bias index
        ax1.plot(w["week_dt"], w["weighted_mean_stance"], color="#2E86AB",
                 linewidth=0.8, alpha=0.9)
        ax1.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax1.axvline(pd.Timestamp("2020-01-01"), color="red", linestyle="--",
                    linewidth=0.8, alpha=0.6, label="Jan 2020")
        ax1.set_ylabel("Weighted mean stance")
        ax1.set_title(f"{ticker} — Weekly Bias Index", fontsize=11)
        ax1.legend(fontsize=8)

        # Bottom: article count
        ax2.bar(w["week_dt"], w["n_articles"], width=5, color="#4C72B0", alpha=0.7)
        ax2.set_ylabel("Articles")
        ax2.set_xlabel("Date")

        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"weekly_ts_{ticker}.png", dpi=100)
        plt.close(fig)

    print(f"  → Weekly time-series plots saved to {PLOT_DIR}/")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_summary(
    dist_df:     pd.DataFrame,
    quality_df:  pd.DataFrame,
    weekly_df:   pd.DataFrame,
    coverage_df: pd.DataFrame,
    mode_label:  str,
) -> None:
    lines = []

    lines += [
        "# Step 2: Bias Index Construction & Quality",
        f"**Analysis window:** {ANALYSIS_START} → {ANALYSIS_END}",
        f"**Frequency:** Weekly (ISO weeks)",
        f"**Relevance source:** {mode_label}",
        f"**Companies:** {', '.join(SELECTED_TICKERS)}",
        "",
    ]

    # ---------- 2.1 Stance distribution
    lines += [
        "---",
        "## 2.1 — Stance Score Distribution",
        "",
        "| Ticker | N | Mean | Std | Skew | Kurt | % Near Zero (±0.05) | % Exactly 0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in dist_df.iterrows():
        lines.append(
            f"| {r['ticker']} | {r['n_articles']:,} | {r['mean']} | {r['std']} "
            f"| {r['skewness']} | {r['kurtosis']} | {r['pct_near_zero']}% | {r['pct_exactly_zero']}% |"
        )
    lines.append("")

    # ---------- 2.2 Daily quality
    lines += [
        "---",
        "## 2.2 — Daily Bias Index Quality",
        "",
        "| Ticker | Total Days | Days w/ 1 Article | % Single-Article | Out-of-Range | Big Swings (>1.0) | Mean Articles/Day | Median Articles/Day |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in quality_df.iterrows():
        lines.append(
            f"| {r['ticker']} | {r['total_days']:,} | {r['days_single_article']:,} "
            f"| {r['pct_single_article']}% | {r['out_of_range_days']} "
            f"| {r['big_swing_days']} | {r['mean_daily_articles']} "
            f"| {r['median_daily_articles']} |"
        )
    lines.append("")

    # ---------- 2.3 Weekly coverage
    lines += [
        "---",
        "## 2.3 — Weekly Aggregation Coverage",
        "",
        "| Ticker | Total Weeks | Weeks w/ Data | Weeks Empty | % Covered | Mean Art/Week | Median Art/Week |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in coverage_df.iterrows():
        lines.append(
            f"| {r['ticker']} | {r['total_weeks_window']} | {r['weeks_with_data']} "
            f"| {r['weeks_empty']} | {r['pct_covered']}% "
            f"| {r['mean_articles_per_week']} | {r['median_articles_per_week']} |"
        )

    # Recommend aggregation method
    median_coverage = coverage_df["pct_covered"].median()
    if median_coverage > 90:
        agg_note = "**Excellent coverage** — most companies have data for >90% of weeks."
    elif median_coverage > 70:
        agg_note = "**Good coverage** — some gaps exist but are manageable with imputation."
    else:
        agg_note = "**Sparse coverage** — significant gaps will require careful imputation or company exclusion."

    lines += [
        "",
        f"> {agg_note}",
        f"> Median weekly coverage across companies: **{median_coverage:.1f}%**",
        "",
        f"Plots saved to `{PLOT_DIR}/`",
    ]

    (OUTPUT_DIR / "step2_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  → Summary written to {OUTPUT_DIR / 'step2_summary.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2: Bias Index Construction & Quality.",
    )
    parser.add_argument(
        "--relevance-csv",
        type=str,
        default=None,
        help=(
            "Path to the ensemble predictions CSV. "
            "When provided, only articles with predicted_label='relevant' are used. "
            "Default: all scored articles in the table are used."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_csv = args.relevance_csv is not None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== STEP 2: Bias Index Construction & Quality ===\n")

    if use_csv:
        mode_label = f"Relevance CSV ({args.relevance_csv})"
        print(f"  Mode: RELEVANCE CSV\n")
        relevant_ids = load_relevant_ids(args.relevance_csv)
    else:
        mode_label = "Database (all scored articles)"
        print(f"  Mode: DATABASE (all scored articles)\n")

    conn = get_connection()
    try:
        if use_csv:
            create_relevant_temp_table(conn, relevant_ids)

        # ---- 2.1: Stance Score Distribution ----
        scores_df = fetch_stance_scores(conn, use_csv)

        dist_df = compute_stance_distribution(scores_df)
        dist_df.to_csv(OUTPUT_DIR / "stance_distribution_stats.csv", index=False)
        print(f"  → stance_distribution_stats.csv written")

        plot_stance_histograms(scores_df)

        # ---- 2.2: Daily Bias Index ----
        daily_df = compute_daily_bias_index(scores_df)
        daily_df.to_csv(OUTPUT_DIR / "daily_bias_index.csv", index=False)
        print(f"  → daily_bias_index.csv written ({len(daily_df):,} rows)")

        quality_df = compute_daily_quality_flags(daily_df)
        quality_df.to_csv(OUTPUT_DIR / "daily_bias_quality.csv", index=False)
        print(f"  → daily_bias_quality.csv written")

        # ---- 2.3: Weekly Aggregation ----
        weekly_df = compute_weekly_bias_index(daily_df)
        weekly_df.to_csv(OUTPUT_DIR / "weekly_bias_index.csv", index=False)
        print(f"  → weekly_bias_index.csv written ({len(weekly_df):,} rows)")

        coverage_df = compute_weekly_coverage(weekly_df)
        coverage_df.to_csv(OUTPUT_DIR / "weekly_coverage.csv", index=False)
        print(f"  → weekly_coverage.csv written")

        plot_weekly_time_series(weekly_df)

        # ---- Summary ----
        write_summary(dist_df, quality_df, weekly_df, coverage_df, mode_label)

    finally:
        conn.close()

    print("\n=== STEP 2 COMPLETE ===")
    print(f"All outputs in: {OUTPUT_DIR.resolve()}")

    # Console preview
    print("\n--- Stance Distribution Summary ---")
    print(dist_df[["ticker", "n_articles", "mean", "std", "skewness",
                    "pct_near_zero"]].to_string(index=False))

    print("\n--- Weekly Coverage Summary ---")
    print(coverage_df[["ticker", "weeks_with_data", "weeks_empty",
                        "pct_covered"]].to_string(index=False))


if __name__ == "__main__":
    main()
