#!/usr/bin/env python3
"""
STEP 3: Gap Analysis & Imputation
==================================
Covers Sections 3.1, 3.2, 3.3, and 3.4 of VAR_Diagnostic_Checklist.md

Reads the weekly_bias_index.csv produced by Step 2 and:
  3.1 — Classifies all gaps (missing weeks) into Type A (1-2), B (3-8), C (>8)
  3.2 — Applies minimum coverage threshold (≥60% weeks, no gap >8 weeks)
  3.3 — Imputes:  forward-fill (Type A), linear interpolation (Type B), drop (Type C)
  3.4 — Outputs both imputed and raw (gap-dropped) versions for robustness

Primary bias metric: simple_mean_stance (per user decision).

Companies (16 — NFLX dropped after Step 2 review):
  ABNB, AMZN, T, BA, BAC, GM, GS, INTC,
  MCD, MSFT, MS, SBUX, UBER, V, WMT, WFC

Outputs (written to diagnostics/step3/):
  - gap_classification.csv         (every gap: company, start/end week, length, type)
  - gap_summary.csv                (per-company: count of A/B/C gaps, total missing weeks)
  - coverage_threshold.csv         (per-company: pass/fail, reason)
  - weekly_bias_imputed.csv        (imputed weekly series for companies that pass)
  - weekly_bias_raw_complete.csv   (raw series with gaps dropped — observed-only rows)
  - imputation_log.csv             (every imputed observation: company, week, method, value)
  - plots/gap_map_<ticker>.png     (visual gap map per company)
  - step3_summary.md               (human-readable report)

Usage:
  python step3_gap_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ANALYSIS_START = "2015-01-01"
ANALYSIS_END   = "2025-12-31"

STEP2_DIR  = Path(__file__).resolve().parent / "diagnostics" / "step2"
OUTPUT_DIR = Path(__file__).resolve().parent / "diagnostics" / "step3"
PLOT_DIR   = OUTPUT_DIR / "plots"

# Primary bias column from Step 2
BIAS_COL = "simple_mean_stance"

# 16 companies (NFLX dropped)
SELECTED_TICKERS = [
    "ABNB", "AMZN", "T", "BA", "BAC", "GM", "GS", "INTC",
    "MCD", "MSFT", "MS", "SBUX", "UBER", "V", "WMT", "WFC",
]

# Coverage thresholds (Section 3.2)
MIN_COVERAGE_PCT = 60          # at least 60% of weeks must have data
MAX_CONSEC_GAP_WEEKS = 8       # no single gap longer than 8 weeks

# Gap type boundaries
TYPE_A_MAX = 2   # 1-2 consecutive missing weeks
TYPE_B_MAX = 8   # 3-8 consecutive missing weeks
                  # >8 = Type C


# ---------------------------------------------------------------------------
# Load Step 2 output
# ---------------------------------------------------------------------------

def load_weekly_index() -> pd.DataFrame:
    """Load weekly_bias_index.csv from Step 2."""
    path = STEP2_DIR / "weekly_bias_index.csv"
    if not path.exists():
        print(f"  ERROR: {path} not found. Run step2_bias_index.py first.")
        sys.exit(1)

    df = pd.read_csv(path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["week_end"]   = pd.to_datetime(df["week_end"])

    # Filter to selected tickers
    df = df[df["ticker"].isin(SELECTED_TICKERS)].copy()
    print(f"  Loaded {len(df):,} company-week rows from Step 2 ({df['ticker'].nunique()} companies)")
    return df


# ---------------------------------------------------------------------------
# Build full weekly grid
# ---------------------------------------------------------------------------

def build_full_weekly_grid(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Create a complete week grid per company, marking present/missing weeks."""

    # Generate all ISO weeks in the analysis window
    all_mondays = pd.date_range(ANALYSIS_START, ANALYSIS_END, freq="W-MON")

    company_info = (
        weekly_df[["company_id", "ticker"]]
        .drop_duplicates()
        .set_index("ticker")
    )

    frames = []
    for ticker in SELECTED_TICKERS:
        if ticker not in company_info.index:
            continue
        cid = int(company_info.loc[ticker, "company_id"])

        # This company's observed weeks (keyed by the Monday of each ISO week)
        obs = weekly_df[weekly_df["ticker"] == ticker].copy()
        obs["week_monday"] = obs["week_start"] - pd.to_timedelta(
            obs["week_start"].dt.weekday, unit="D"
        )

        full = pd.DataFrame({"week_monday": all_mondays})
        full["company_id"] = cid
        full["ticker"]     = ticker

        merged = full.merge(
            obs[["week_monday", BIAS_COL, "n_articles", "n_days_covered"]],
            on="week_monday",
            how="left",
        )
        merged["has_data"] = merged[BIAS_COL].notna()
        frames.append(merged)

    grid = pd.concat(frames, ignore_index=True).sort_values(
        ["ticker", "week_monday"]
    ).reset_index(drop=True)

    return grid


# ---------------------------------------------------------------------------
# 3.1 — Gap Classification
# ---------------------------------------------------------------------------

def classify_gaps(grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify and classify every contiguous gap per company.

    Returns:
        gap_detail: one row per gap (company, start, end, length, type)
        gap_summary: one row per company (counts by type)
    """
    print("  [3.1] Classifying gaps...")

    detail_rows = []
    summary_rows = []

    for ticker in SELECTED_TICKERS:
        cdf = grid[grid["ticker"] == ticker].sort_values("week_monday").reset_index(drop=True)
        cid = int(cdf["company_id"].iloc[0])

        # Find contiguous runs of missing weeks
        missing_mask = ~cdf["has_data"]
        gap_starts = []
        i = 0
        while i < len(cdf):
            if missing_mask.iloc[i]:
                start_idx = i
                while i < len(cdf) and missing_mask.iloc[i]:
                    i += 1
                end_idx = i - 1
                gap_len = end_idx - start_idx + 1
                gap_starts.append((start_idx, end_idx, gap_len))
            else:
                i += 1

        counts = {"A": 0, "B": 0, "C": 0}
        total_missing = 0

        for start_idx, end_idx, gap_len in gap_starts:
            total_missing += gap_len
            if gap_len <= TYPE_A_MAX:
                gtype = "A"
            elif gap_len <= TYPE_B_MAX:
                gtype = "B"
            else:
                gtype = "C"
            counts[gtype] += 1

            detail_rows.append({
                "company_id":  cid,
                "ticker":      ticker,
                "gap_start":   cdf["week_monday"].iloc[start_idx],
                "gap_end":     cdf["week_monday"].iloc[end_idx],
                "gap_weeks":   gap_len,
                "gap_type":    gtype,
            })

        total_weeks = len(cdf)
        weeks_with_data = int(cdf["has_data"].sum())

        summary_rows.append({
            "ticker":         ticker,
            "total_weeks":    total_weeks,
            "weeks_with_data": weeks_with_data,
            "weeks_missing":  total_missing,
            "pct_covered":    round(weeks_with_data / total_weeks * 100, 1),
            "gaps_type_a":    counts["A"],
            "gaps_type_b":    counts["B"],
            "gaps_type_c":    counts["C"],
            "longest_gap":    max((g[2] for g in gap_starts), default=0),
        })

    gap_detail  = pd.DataFrame(detail_rows)
    gap_summary = pd.DataFrame(summary_rows)

    print(f"  → {len(gap_detail)} total gaps identified across {len(gap_summary)} companies")
    return gap_detail, gap_summary


# ---------------------------------------------------------------------------
# 3.2 — Minimum Coverage Threshold
# ---------------------------------------------------------------------------

def apply_coverage_threshold(gap_summary: pd.DataFrame) -> pd.DataFrame:
    """Evaluate each company against thresholds; return pass/fail table."""
    print("  [3.2] Applying coverage thresholds...")

    rows = []
    for _, r in gap_summary.iterrows():
        reasons = []
        passes = True

        if r["pct_covered"] < MIN_COVERAGE_PCT:
            passes = False
            reasons.append(f"Coverage {r['pct_covered']}% < {MIN_COVERAGE_PCT}%")

        if r["longest_gap"] > MAX_CONSEC_GAP_WEEKS:
            passes = False
            reasons.append(f"Longest gap {r['longest_gap']} weeks > {MAX_CONSEC_GAP_WEEKS}")

        rows.append({
            "ticker":       r["ticker"],
            "pct_covered":  r["pct_covered"],
            "longest_gap":  r["longest_gap"],
            "gaps_type_c":  r["gaps_type_c"],
            "passes":       passes,
            "reason":       "; ".join(reasons) if reasons else "OK",
        })

    threshold_df = pd.DataFrame(rows)
    n_pass = int(threshold_df["passes"].sum())
    n_fail = len(threshold_df) - n_pass
    print(f"  → {n_pass} companies pass, {n_fail} fail")
    return threshold_df


# ---------------------------------------------------------------------------
# 3.3 — Imputation
# ---------------------------------------------------------------------------

def impute_series(
    grid: pd.DataFrame,
    gap_detail: pd.DataFrame,
    threshold_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply imputation to companies that pass the threshold.

    Type A (1-2 weeks): forward-fill
    Type B (3-8 weeks): linear interpolation
    Type C (>8 weeks):  not imputed (company may be truncated)

    Returns:
        imputed_df: the full imputed weekly series (only passing companies)
        imputation_log: one row per imputed observation
    """
    print("  [3.3] Imputing gaps...")

    passing_tickers = set(threshold_df[threshold_df["passes"]]["ticker"])
    log_rows = []
    imputed_frames = []

    for ticker in SELECTED_TICKERS:
        if ticker not in passing_tickers:
            continue

        cdf = grid[grid["ticker"] == ticker].sort_values("week_monday").copy()
        cdf = cdf.set_index("week_monday")

        series = cdf[BIAS_COL].copy()
        n_articles = cdf["n_articles"].copy()

        # Get this company's gaps
        cgaps = gap_detail[gap_detail["ticker"] == ticker]

        for _, gap in cgaps.iterrows():
            gtype = gap["gap_type"]
            gstart = gap["gap_start"]
            gend   = gap["gap_end"]

            if gtype == "C":
                # Don't impute Type C — leave NaN (will be handled in truncation)
                continue

            gap_mask = (series.index >= gstart) & (series.index <= gend)
            gap_indices = series.index[gap_mask]

            if gtype == "A":
                # Forward-fill
                method = "forward_fill"
                filled = series.ffill()
                for idx in gap_indices:
                    if pd.isna(series[idx]):
                        series[idx] = filled[idx]
                        n_articles[idx] = 0  # mark as imputed
                        log_rows.append({
                            "ticker": ticker,
                            "week_monday": idx,
                            "gap_type": "A",
                            "method": method,
                            "imputed_value": series[idx],
                        })

            elif gtype == "B":
                # Linear interpolation
                method = "linear_interpolation"
                interp = series.interpolate(method="linear")
                for idx in gap_indices:
                    if pd.isna(series[idx]):
                        series[idx] = interp[idx]
                        n_articles[idx] = 0
                        log_rows.append({
                            "ticker": ticker,
                            "week_monday": idx,
                            "gap_type": "B",
                            "method": method,
                            "imputed_value": series[idx],
                        })

        cdf[BIAS_COL] = series
        cdf["n_articles"] = n_articles
        cdf["is_imputed"] = cdf["n_articles"] == 0
        imputed_frames.append(cdf.reset_index())

    imputed_df = pd.concat(imputed_frames, ignore_index=True) if imputed_frames else pd.DataFrame()
    imputation_log = pd.DataFrame(log_rows)

    n_imputed = len(imputation_log)
    print(f"  → {n_imputed:,} observations imputed across {len(passing_tickers)} companies")
    return imputed_df, imputation_log


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_gap_maps(grid: pd.DataFrame, gap_detail: pd.DataFrame) -> None:
    """Visual gap map for each company: green = data, red = gap, colored by type."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    type_colors = {"A": "#FFC107", "B": "#FF5722", "C": "#B71C1C"}

    for ticker in SELECTED_TICKERS:
        cdf = grid[grid["ticker"] == ticker].sort_values("week_monday")
        if cdf.empty:
            continue

        fig, ax = plt.subplots(figsize=(14, 2.5))

        # Plot data presence as green bars
        data_weeks = cdf[cdf["has_data"]]["week_monday"]
        ax.bar(data_weeks, 1, width=6, color="#4CAF50", alpha=0.6, label="Data")

        # Overlay gaps colored by type
        cgaps = gap_detail[gap_detail["ticker"] == ticker]
        for _, gap in cgaps.iterrows():
            gap_weeks = cdf[
                (cdf["week_monday"] >= gap["gap_start"]) &
                (cdf["week_monday"] <= gap["gap_end"])
            ]["week_monday"]
            color = type_colors.get(gap["gap_type"], "gray")
            label = f"Type {gap['gap_type']}" if gap["gap_type"] not in [
                l.get_label() for l in ax.get_children() if hasattr(l, "get_label")
            ] else None
            ax.bar(gap_weeks, 1, width=6, color=color, alpha=0.8, label=label)

        ax.axvline(pd.Timestamp("2020-01-01"), color="blue", linestyle="--",
                   linewidth=0.8, alpha=0.6)
        ax.set_yticks([])
        ax.set_title(f"{ticker} — Weekly Data Coverage & Gaps", fontsize=10)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # De-duplicate legend labels
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8, loc="upper right")

        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"gap_map_{ticker}.png", dpi=100)
        plt.close(fig)

    print(f"  → Gap maps saved to {PLOT_DIR}/")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_summary(
    gap_summary:    pd.DataFrame,
    gap_detail:     pd.DataFrame,
    threshold_df:   pd.DataFrame,
    imputation_log: pd.DataFrame,
) -> None:
    lines = []

    lines += [
        "# Step 3: Gap Analysis & Imputation",
        f"**Analysis window:** {ANALYSIS_START} → {ANALYSIS_END}",
        f"**Frequency:** Weekly",
        f"**Bias metric:** `{BIAS_COL}` (simple mean of daily stance scores)",
        f"**Companies:** {len(SELECTED_TICKERS)} ({', '.join(SELECTED_TICKERS)})",
        "",
    ]

    # ---------- 3.1 Gap Summary
    lines += [
        "---",
        "## 3.1 — Gap Classification",
        "",
        "| Ticker | Total Weeks | Weeks w/ Data | Missing | % Covered | Gaps A (1-2w) | Gaps B (3-8w) | Gaps C (>8w) | Longest Gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in gap_summary.iterrows():
        lines.append(
            f"| {r['ticker']} | {r['total_weeks']} | {r['weeks_with_data']} "
            f"| {r['weeks_missing']} | {r['pct_covered']}% "
            f"| {r['gaps_type_a']} | {r['gaps_type_b']} | {r['gaps_type_c']} "
            f"| {r['longest_gap']} |"
        )

    # Shared gaps check
    if not gap_detail.empty:
        type_c = gap_detail[gap_detail["gap_type"] == "C"]
        if not type_c.empty:
            lines += [
                "",
                "### Type C Gaps (>8 weeks) — Detail",
                "",
                "| Ticker | Start | End | Weeks |",
                "|---|---|---|---:|",
            ]
            for _, g in type_c.iterrows():
                lines.append(
                    f"| {g['ticker']} | {str(g['gap_start'])[:10]} "
                    f"| {str(g['gap_end'])[:10]} | {g['gap_weeks']} |"
                )
    lines.append("")

    # ---------- 3.2 Coverage Threshold
    lines += [
        "---",
        "## 3.2 — Minimum Coverage Threshold",
        f"",
        f"**Thresholds:** ≥{MIN_COVERAGE_PCT}% weeks with data, no gap >{MAX_CONSEC_GAP_WEEKS} weeks.",
        "",
        "| Ticker | % Covered | Longest Gap | Passes | Reason |",
        "|---|---:|---:|---|---|",
    ]
    for _, r in threshold_df.iterrows():
        icon = "✅" if r["passes"] else "❌"
        lines.append(
            f"| {r['ticker']} | {r['pct_covered']}% | {r['longest_gap']} "
            f"| {icon} | {r['reason']} |"
        )

    n_pass = int(threshold_df["passes"].sum())
    n_fail = len(threshold_df) - n_pass
    lines += [
        "",
        f"> **{n_pass} companies pass**, **{n_fail} fail** the coverage threshold.",
        "",
    ]

    # ---------- 3.3 Imputation
    lines += [
        "---",
        "## 3.3 — Imputation Summary",
        "",
        "| Method | Gap Type | Observations Imputed |",
        "|---|---|---:|",
    ]
    if not imputation_log.empty:
        for (gtype, method), g in imputation_log.groupby(["gap_type", "method"]):
            lines.append(f"| {method} | Type {gtype} | {len(g):,} |")
        total_imp = len(imputation_log)
        lines.append(f"| **Total** | — | **{total_imp:,}** |")
    else:
        lines.append("| — | — | 0 |")

    per_company_imp = {}
    if not imputation_log.empty:
        per_company_imp = imputation_log.groupby("ticker").size().to_dict()
    if per_company_imp:
        lines += [
            "",
            "### Per-Company Imputation Counts",
            "",
            "| Ticker | Imputed Weeks |",
            "|---|---:|",
        ]
        for t in SELECTED_TICKERS:
            if t in per_company_imp:
                lines.append(f"| {t} | {per_company_imp[t]} |")

    lines += [
        "",
        "> Forward-fill for Type A (1-2 week gaps). Linear interpolation for Type B (3-8 week gaps). Type C gaps are not imputed.",
        "",
    ]

    # ---------- 3.4 Robustness note
    lines += [
        "---",
        "## 3.4 — Robustness Check",
        "",
        "Two versions of the weekly bias series have been saved:",
        "- `weekly_bias_imputed.csv` — with imputed values filled in",
        "- `weekly_bias_raw_complete.csv` — observed-only rows (gaps dropped)",
        "",
        "Run key downstream tests (stationarity, Granger causality) on **both** versions to confirm results are not driven by imputation.",
        "",
        f"All outputs in `{OUTPUT_DIR}/`",
    ]

    (OUTPUT_DIR / "step3_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  → Summary written to {OUTPUT_DIR / 'step3_summary.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== STEP 3: Gap Analysis & Imputation ===\n")

    # Load Step 2 weekly data
    weekly_df = load_weekly_index()

    # Build complete weekly grid
    grid = build_full_weekly_grid(weekly_df)
    print(f"  → Full weekly grid: {len(grid):,} rows")

    # 3.1: Gap classification
    gap_detail, gap_summary = classify_gaps(grid)
    gap_detail.to_csv(OUTPUT_DIR / "gap_classification.csv", index=False)
    gap_summary.to_csv(OUTPUT_DIR / "gap_summary.csv", index=False)
    print(f"  → gap_classification.csv and gap_summary.csv written")

    # 3.2: Coverage threshold
    threshold_df = apply_coverage_threshold(gap_summary)
    threshold_df.to_csv(OUTPUT_DIR / "coverage_threshold.csv", index=False)
    print(f"  → coverage_threshold.csv written")

    # 3.3: Imputation
    imputed_df, imputation_log = impute_series(grid, gap_detail, threshold_df)

    if not imputed_df.empty:
        imputed_df.to_csv(OUTPUT_DIR / "weekly_bias_imputed.csv", index=False)
        print(f"  → weekly_bias_imputed.csv written ({len(imputed_df):,} rows)")

    if not imputation_log.empty:
        imputation_log.to_csv(OUTPUT_DIR / "imputation_log.csv", index=False)
        print(f"  → imputation_log.csv written ({len(imputation_log):,} imputed observations)")

    # Raw complete (observed-only, for robustness comparison)
    passing_tickers = set(threshold_df[threshold_df["passes"]]["ticker"])
    raw_complete = grid[
        (grid["ticker"].isin(passing_tickers)) & (grid["has_data"])
    ].copy()
    raw_complete.to_csv(OUTPUT_DIR / "weekly_bias_raw_complete.csv", index=False)
    print(f"  → weekly_bias_raw_complete.csv written ({len(raw_complete):,} rows)")

    # Plots
    plot_gap_maps(grid, gap_detail)

    # Summary
    write_summary(gap_summary, gap_detail, threshold_df, imputation_log)

    print("\n=== STEP 3 COMPLETE ===")
    print(f"All outputs in: {OUTPUT_DIR.resolve()}")

    # Console preview
    print("\n--- Gap Summary ---")
    print(gap_summary[["ticker", "weeks_with_data", "weeks_missing",
                        "pct_covered", "gaps_type_a", "gaps_type_b",
                        "gaps_type_c", "longest_gap"]].to_string(index=False))

    print("\n--- Coverage Threshold ---")
    print(threshold_df[["ticker", "pct_covered", "longest_gap",
                         "passes", "reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
