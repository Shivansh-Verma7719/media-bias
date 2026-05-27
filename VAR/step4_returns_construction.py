#!/usr/bin/env python3
"""
STEP 4: Returns Series Construction & Alignment
=================================================
Covers Sections 4.1 and 4.2 of VAR_Diagnostic_Checklist.md

4.1 — Compute daily log returns from public.stock_prices (close prices),
      aggregate to weekly (sum of daily log returns within each ISO week).
4.2 — Merge weekly returns with the imputed weekly bias index from Step 3,
      check alignment, and produce the final merged panel.

Companies (16):
  ABNB, AMZN, T, BA, BAC, GM, GS, INTC,
  MCD, MSFT, MS, SBUX, UBER, V, WMT, WFC

Outputs (written to diagnostics/step4/):
  - daily_log_returns.csv           (company, date, close, log_return)
  - weekly_log_returns.csv          (company, iso_week, weekly_return, n_trading_days)
  - alignment_check.csv             (per-company: weeks bias, weeks returns, weeks both)
  - merged_weekly_panel.csv         (final aligned panel: bias + returns per company-week)
  - plots/returns_ts_<ticker>.png   (weekly returns time-series per company)
  - plots/returns_hist_<ticker>.png (return distribution per company)
  - step4_summary.md

Usage:
  python step4_returns_construction.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent dir so we can import db module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_connection  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ANALYSIS_START = "2015-01-01"
ANALYSIS_END   = "2025-12-31"

STEP3_DIR  = Path(__file__).resolve().parent / "diagnostics" / "step3"
OUTPUT_DIR = Path(__file__).resolve().parent / "diagnostics" / "step4"
PLOT_DIR   = OUTPUT_DIR / "plots"

BIAS_COL = "simple_mean_stance"

# 16 companies (same as Step 3)
SELECTED_TICKERS = [
    "ABNB", "AMZN", "T", "BA", "BAC", "GM", "GS", "INTC",
    "MCD", "MSFT", "MS", "SBUX", "UBER", "V", "WMT", "WFC",
]

MIN_ALIGNED_WEEKS = 100  # minimum for reliable VAR estimation


# ---------------------------------------------------------------------------
# 4.1 — Compute Log Returns
# ---------------------------------------------------------------------------

def fetch_daily_prices(conn) -> pd.DataFrame:
    """Fetch daily close prices for selected tickers from public.stock_prices."""
    print("  [4.1] Fetching daily close prices...")

    tickers_placeholder = ",".join(["%s"] * len(SELECTED_TICKERS))

    sql = f"""
        SELECT
            s.ticker,
            s.date::date   AS trade_date,
            s.close
        FROM public.stock_prices s
        WHERE s.date::date BETWEEN %s AND %s
          AND s.ticker IN ({tickers_placeholder})
        ORDER BY s.ticker, s.date
    """
    params = [ANALYSIS_START, ANALYSIS_END] + SELECTED_TICKERS
    df = pd.read_sql(sql, conn, params=params)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["close"] = df["close"].astype(float)
    print(f"  → {len(df):,} daily price rows for {df['ticker'].nunique()} tickers")
    return df


def compute_daily_log_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns: ln(P_t) - ln(P_{t-1})."""
    print("  [4.1] Computing daily log returns...")

    df = prices_df.sort_values(["ticker", "trade_date"]).copy()
    df["log_close"] = np.log(df["close"])
    df["log_return"] = df.groupby("ticker")["log_close"].diff()

    # Drop the first row per ticker (NaN return)
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)
    print(f"  → {len(df):,} daily return observations")
    return df


def aggregate_weekly_returns(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily log returns to ISO weeks (sum within each week)."""
    print("  [4.1] Aggregating to weekly returns...")

    df = daily_returns.copy()
    # Monday-anchored week
    df["week_monday"] = df["trade_date"] - pd.to_timedelta(
        df["trade_date"].dt.weekday, unit="D"
    )

    weekly = df.groupby(["ticker", "week_monday"]).agg(
        weekly_log_return=("log_return", "sum"),
        n_trading_days=("log_return", "size"),
        week_close=("close", "last"),
        week_open=("close", "first"),
    ).reset_index()

    weekly = weekly.sort_values(["ticker", "week_monday"]).reset_index(drop=True)

    # Flag short weeks (< 4 trading days) and extreme returns
    weekly["flag_short_week"] = weekly["n_trading_days"] < 4
    weekly["flag_extreme"] = weekly["weekly_log_return"].abs() > 0.15  # >±15% weekly

    print(f"  → {len(weekly):,} company-week return observations")
    return weekly


def check_returns_quality(weekly_returns: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker quality summary for returns."""
    print("  [4.1] Returns quality check...")

    rows = []
    for ticker in SELECTED_TICKERS:
        w = weekly_returns[weekly_returns["ticker"] == ticker]
        if w.empty:
            continue

        r = w["weekly_log_return"]
        rows.append({
            "ticker":            ticker,
            "n_weeks":           len(w),
            "first_week":        str(w["week_monday"].min())[:10],
            "last_week":         str(w["week_monday"].max())[:10],
            "mean_return":       round(r.mean(), 5),
            "std_return":        round(r.std(), 5),
            "min_return":        round(r.min(), 4),
            "max_return":        round(r.max(), 4),
            "n_extreme":         int(w["flag_extreme"].sum()),
            "n_short_weeks":     int(w["flag_short_week"].sum()),
            "pct_zero_return":   round((r.abs() < 1e-8).sum() / len(r) * 100, 1),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4.2 — Alignment Check
# ---------------------------------------------------------------------------

def load_imputed_bias() -> pd.DataFrame:
    """Load the imputed weekly bias index from Step 3."""
    path = STEP3_DIR / "weekly_bias_imputed.csv"
    if not path.exists():
        print(f"  ERROR: {path} not found. Run step3_gap_analysis.py first.")
        sys.exit(1)

    df = pd.read_csv(path)
    df["week_monday"] = pd.to_datetime(df["week_monday"])
    df = df[df["ticker"].isin(SELECTED_TICKERS)].copy()
    print(f"  Loaded {len(df):,} rows from imputed bias series")
    return df


def merge_and_align(
    bias_df: pd.DataFrame,
    returns_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge bias and returns on (ticker, week_monday); produce alignment stats."""
    print("  [4.2] Merging bias and returns...")

    # Full outer join to see what's available
    merged = bias_df.merge(
        returns_df[["ticker", "week_monday", "weekly_log_return", "n_trading_days",
                     "flag_short_week", "flag_extreme"]],
        on=["ticker", "week_monday"],
        how="outer",
        indicator=True,
    )

    # Alignment summary per company
    alignment_rows = []
    for ticker in SELECTED_TICKERS:
        m = merged[merged["ticker"] == ticker]
        n_total    = len(m)
        n_bias     = int((m["_merge"].isin(["both", "left_only"])).sum())
        n_returns  = int((m["_merge"].isin(["both", "right_only"])).sum())
        n_both     = int((m["_merge"] == "both").sum())
        n_bias_only    = int((m["_merge"] == "left_only").sum())
        n_returns_only = int((m["_merge"] == "right_only").sum())

        alignment_rows.append({
            "ticker":              ticker,
            "total_weeks_union":   n_total,
            "weeks_with_bias":     n_bias,
            "weeks_with_returns":  n_returns,
            "weeks_both":          n_both,
            "weeks_bias_only":     n_bias_only,
            "weeks_returns_only":  n_returns_only,
            "sufficient":          n_both >= MIN_ALIGNED_WEEKS,
        })

    alignment_df = pd.DataFrame(alignment_rows)

    # Keep only rows where BOTH series are present
    panel = merged[merged["_merge"] == "both"].drop(columns=["_merge"]).copy()
    panel = panel.sort_values(["ticker", "week_monday"]).reset_index(drop=True)

    print(f"  → Aligned panel: {len(panel):,} company-week rows")
    return panel, alignment_df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_returns(weekly_returns: pd.DataFrame) -> None:
    """Time-series and histogram of weekly returns per company."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in SELECTED_TICKERS:
        w = weekly_returns[weekly_returns["ticker"] == ticker].copy()
        if w.empty:
            continue
        w = w.sort_values("week_monday")

        # Time-series plot
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.plot(w["week_monday"], w["weekly_log_return"], color="#2E86AB",
                linewidth=0.6, alpha=0.9)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.7)
        ax.axvline(pd.Timestamp("2020-03-01"), color="red", linestyle="--",
                   linewidth=0.8, alpha=0.6, label="Mar 2020")
        ax.set_title(f"{ticker} — Weekly Log Returns", fontsize=11)
        ax.set_ylabel("Log return")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"returns_ts_{ticker}.png", dpi=100)
        plt.close(fig)

        # Histogram
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(w["weekly_log_return"], bins=60, color="#4C72B0",
                edgecolor="white", linewidth=0.4, alpha=0.85)
        ax.axvline(w["weekly_log_return"].mean(), color="crimson",
                   linestyle="--", linewidth=1.2,
                   label=f"Mean = {w['weekly_log_return'].mean():.4f}")
        ax.set_title(f"{ticker} — Weekly Return Distribution (n={len(w)})", fontsize=11)
        ax.set_xlabel("Weekly log return")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"returns_hist_{ticker}.png", dpi=100)
        plt.close(fig)

    print(f"  → Return plots saved to {PLOT_DIR}/")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_summary(
    quality_df:    pd.DataFrame,
    alignment_df:  pd.DataFrame,
    panel_df:      pd.DataFrame,
) -> None:
    lines = []

    lines += [
        "# Step 4: Returns Series Construction & Alignment",
        f"**Analysis window:** {ANALYSIS_START} → {ANALYSIS_END}",
        f"**Frequency:** Weekly (ISO weeks, Monday-anchored)",
        f"**Bias metric:** `{BIAS_COL}` (simple mean)",
        f"**Min aligned weeks for VAR:** {MIN_ALIGNED_WEEKS}",
        "",
    ]

    # 4.1 Returns quality
    lines += [
        "---",
        "## 4.1 — Weekly Log Returns Quality",
        "",
        "| Ticker | Weeks | First | Last | Mean | Std | Min | Max | Extreme (>±15%) | Short Weeks (<4d) |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in quality_df.iterrows():
        lines.append(
            f"| {r['ticker']} | {r['n_weeks']} "
            f"| {r['first_week']} | {r['last_week']} "
            f"| {r['mean_return']} | {r['std_return']} "
            f"| {r['min_return']} | {r['max_return']} "
            f"| {r['n_extreme']} | {r['n_short_weeks']} |"
        )
    lines.append("")

    # Flag extreme events
    for _, r in quality_df.iterrows():
        if r["n_extreme"] > 0:
            lines.append(
                f"> ⚠️ **{r['ticker']}** has {r['n_extreme']} extreme weekly returns (>±15%). "
                f"Inspect for data errors vs. real events (e.g., COVID crash, earnings)."
            )
    lines.append("")

    # 4.2 Alignment
    lines += [
        "---",
        "## 4.2 — Alignment: Bias ↔ Returns",
        "",
        "| Ticker | Weeks (bias) | Weeks (returns) | Weeks (both) | Bias-only | Returns-only | Sufficient (≥100)? |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in alignment_df.iterrows():
        icon = "✅" if r["sufficient"] else "❌"
        lines.append(
            f"| {r['ticker']} | {r['weeks_with_bias']} "
            f"| {r['weeks_with_returns']} | {r['weeks_both']} "
            f"| {r['weeks_bias_only']} | {r['weeks_returns_only']} "
            f"| {icon} |"
        )

    n_sufficient = int(alignment_df["sufficient"].sum())
    n_insuf = len(alignment_df) - n_sufficient
    median_both = int(alignment_df["weeks_both"].median())

    lines += [
        "",
        f"> **{n_sufficient} companies** have ≥{MIN_ALIGNED_WEEKS} aligned weeks. "
        f"**{n_insuf} companies** do not.",
        f"> Median aligned weeks: **{median_both}**",
        "",
        f"Final merged panel saved to `merged_weekly_panel.csv` ({len(panel_df):,} rows).",
        "",
        f"Plots saved to `{PLOT_DIR}/`",
    ]

    (OUTPUT_DIR / "step4_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  → Summary written to {OUTPUT_DIR / 'step4_summary.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== STEP 4: Returns Series Construction & Alignment ===\n")

    # --- 4.1: Fetch prices & compute returns ---
    conn = get_connection()
    try:
        prices_df = fetch_daily_prices(conn)
    finally:
        conn.close()

    daily_returns = compute_daily_log_returns(prices_df)
    daily_returns.to_csv(OUTPUT_DIR / "daily_log_returns.csv", index=False)
    print(f"  → daily_log_returns.csv written")

    weekly_returns = aggregate_weekly_returns(daily_returns)
    weekly_returns.to_csv(OUTPUT_DIR / "weekly_log_returns.csv", index=False)
    print(f"  → weekly_log_returns.csv written")

    quality_df = check_returns_quality(weekly_returns)
    quality_df.to_csv(OUTPUT_DIR / "returns_quality.csv", index=False)
    print(f"  → returns_quality.csv written")

    plot_returns(weekly_returns)

    # --- 4.2: Alignment ---
    bias_df = load_imputed_bias()
    panel_df, alignment_df = merge_and_align(bias_df, weekly_returns)

    alignment_df.to_csv(OUTPUT_DIR / "alignment_check.csv", index=False)
    print(f"  → alignment_check.csv written")

    panel_df.to_csv(OUTPUT_DIR / "merged_weekly_panel.csv", index=False)
    print(f"  → merged_weekly_panel.csv written ({len(panel_df):,} rows)")

    # Summary
    write_summary(quality_df, alignment_df, panel_df)

    print("\n=== STEP 4 COMPLETE ===")
    print(f"All outputs in: {OUTPUT_DIR.resolve()}")

    # Console preview
    print("\n--- Returns Quality ---")
    print(quality_df[["ticker", "n_weeks", "mean_return", "std_return",
                       "min_return", "max_return", "n_extreme"]].to_string(index=False))

    print("\n--- Alignment ---")
    print(alignment_df[["ticker", "weeks_with_bias", "weeks_with_returns",
                          "weeks_both", "sufficient"]].to_string(index=False))


if __name__ == "__main__":
    main()
