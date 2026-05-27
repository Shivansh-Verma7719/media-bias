#!/usr/bin/env python3
"""
STEP 6: Structural Break Testing
================================
Covers Sections 6.1 and 6.2 of VAR_Diagnostic_Checklist.md

6.1 — Bai-Perron Test (using ruptures library) for structural breaks on:
      - The weekly bias index per company
      - The weekly return series per company
6.2 — Pre/Post Subsample Split:
      - Pre-period: 2015 to the break date
      - Post-period: break date to 2025
      - Check subsample lengths (at least 60-80 observations).

Outputs (written to diagnostics/step6/):
  - structural_breaks.csv      (per-company break dates and subsample lengths)
  - panel_with_breaks.csv      (panel data with `is_post_break` boolean indicator)
  - plots/breaks_<ticker>.png  (time-series plot highlighting the break date)
  - step6_summary.md           (summary of empirical break dates)

Usage:
  python step6_structural_breaks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt

# Config
STEP5_DIR  = Path(__file__).resolve().parent / "diagnostics" / "step5"
OUTPUT_DIR = Path(__file__).resolve().parent / "diagnostics" / "step6"
PLOT_DIR   = OUTPUT_DIR / "plots"

BIAS_COL = "simple_mean_stance"
RETURN_COL = "weekly_log_return"

MIN_SUBSAMPLE_OBS = 60  # Minimum weeks for a subsample to be valid for VAR

SELECTED_TICKERS = [
    "ABNB", "AMZN", "T", "BA", "BAC", "GM", "GS", "INTC",
    "MCD", "MSFT", "MS", "SBUX", "UBER", "V", "WMT", "WFC",
]

def find_break_point(series: pd.Series, model="l2") -> int | None:
    """
    Find exactly 1 structural break point using Dynamic Programming.
    Returns the integer index of the break point.
    """
    signal = series.values.reshape(-1, 1)
    # min_size=52 ensures breaks are at least 1 year from the edges
    try:
        algo = rpt.Dynp(model=model, min_size=52, jump=1).fit(signal)
        result = algo.predict(n_bkps=1)
        # result returns a list of break indices, the last one is the length of the array
        break_idx = result[0]
        return break_idx
    except Exception as e:
        print(f"      Warning: Dynp failed ({e})")
        return None

def plot_breaks(df: pd.DataFrame, ticker: str, bias_break_date, return_break_date):
    """Plot time series with structural break lines."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    
    # 1. Bias Index Break
    axes[0].plot(df["week_monday"], df[BIAS_COL], color="#2CA02C", linewidth=1.0)
    if bias_break_date is not None:
        axes[0].axvline(bias_break_date, color="red", linestyle="--", linewidth=1.5, label=f"Break: {bias_break_date.date()}")
    axes[0].axvline(pd.Timestamp("2020-03-01"), color="gray", linestyle=":", linewidth=1.0, label="Mar 2020 (Expected)")
    axes[0].set_title(f"{ticker} — Bias Index Structural Break")
    axes[0].legend()
    
    # 2. Returns Break
    axes[1].plot(df["week_monday"], df[RETURN_COL], color="#1F77B4", linewidth=1.0, alpha=0.8)
    if return_break_date is not None:
        axes[1].axvline(return_break_date, color="red", linestyle="--", linewidth=1.5, label=f"Break: {return_break_date.date()}")
    axes[1].axvline(pd.Timestamp("2020-03-01"), color="gray", linestyle=":", linewidth=1.0, label="Mar 2020 (Expected)")
    axes[1].set_title(f"{ticker} — Weekly Returns Structural Break")
    axes[1].legend()
    
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"breaks_{ticker}.png", dpi=120)
    plt.close(fig)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=== STEP 6: Structural Break Testing ===\n")
    
    panel_path = STEP5_DIR / "stationarity_checked_panel.csv"
    if not panel_path.exists():
        print(f"ERROR: {panel_path} not found. Run step5_stationarity.py first.")
        sys.exit(1)
        
    df_panel = pd.read_csv(panel_path)
    df_panel["week_monday"] = pd.to_datetime(df_panel["week_monday"])
    
    break_results = []
    # We will append an is_post_break boolean to the panel.
    # The convention is that the post-period starts AT the break date.
    df_panel["is_post_break"] = False 
    
    for ticker in SELECTED_TICKERS:
        ticker_mask = df_panel["ticker"] == ticker
        w_df = df_panel[ticker_mask].sort_values("week_monday").reset_index(drop=True)
        if w_df.empty:
            continue
            
        print(f"  Testing {ticker} (n={len(w_df)} weeks)...")
        
        # Bias Break
        bias_break_idx = find_break_point(w_df[BIAS_COL])
        bias_break_date = w_df.loc[bias_break_idx, "week_monday"] if bias_break_idx is not None and bias_break_idx < len(w_df) else None
        
        # Return Break
        return_break_idx = find_break_point(w_df[RETURN_COL])
        return_break_date = w_df.loc[return_break_idx, "week_monday"] if return_break_idx is not None and return_break_idx < len(w_df) else None
        
        # Let's use the bias break date as the primary structural break for the relationship
        primary_break_date = bias_break_date
        
        pre_len = bias_break_idx if bias_break_idx is not None else len(w_df)
        post_len = len(w_df) - bias_break_idx if bias_break_idx is not None else 0
        
        is_valid = (pre_len >= MIN_SUBSAMPLE_OBS) and (post_len >= MIN_SUBSAMPLE_OBS)
        
        break_results.append({
            "ticker": ticker,
            "bias_break_date": bias_break_date.date() if bias_break_date else None,
            "return_break_date": return_break_date.date() if return_break_date else None,
            "primary_break_date": primary_break_date.date() if primary_break_date else None,
            "pre_obs": pre_len,
            "post_obs": post_len,
            "is_valid_split": is_valid
        })
        
        print(f"    → Bias Break: {bias_break_date.date() if bias_break_date else 'None'} | Return Break: {return_break_date.date() if return_break_date else 'None'}")
        
        # Update panel
        if primary_break_date is not None:
            df_panel.loc[ticker_mask & (df_panel["week_monday"] >= primary_break_date), "is_post_break"] = True
            
        # Plot
        plot_breaks(w_df, ticker, bias_break_date, return_break_date)

    breaks_df = pd.DataFrame(break_results)
    breaks_df.to_csv(OUTPUT_DIR / "structural_breaks.csv", index=False)
    print(f"\n  → Saved break results to: {OUTPUT_DIR / 'structural_breaks.csv'}")
    
    df_panel.to_csv(OUTPUT_DIR / "panel_with_breaks.csv", index=False)
    print(f"  → Saved updated panel to: {OUTPUT_DIR / 'panel_with_breaks.csv'}")
    
    # ---------------------------------------------------------------------------
    # Write Step 6 Summary
    # ---------------------------------------------------------------------------
    summary_lines = [
        "# Step 6: Structural Break Testing Summary",
        "",
        "## Empirical Break Dates",
        "A Bai-Perron test (via Dynamic Programming, exactly 1 change point) was applied to the weekly bias index and weekly return series for each company. The algorithm identifies the single largest structural shift in the mean of the series (L2 cost).",
        "",
        "| Ticker | Bias Break Date | Return Break Date | Pre-Break Obs | Post-Break Obs | Valid Split (>=60)? |",
        "|---|---|---|---:|---:|---|",
    ]
    
    for _, r in breaks_df.iterrows():
        valid_icon = "✅ YES" if r['is_valid_split'] else "❌ NO"
        summary_lines.append(f"| {r['ticker']} | {r['bias_break_date']} | {r['return_break_date']} | {r['pre_obs']} | {r['post_obs']} | {valid_icon} |")
        
    summary_lines += [
        "",
        "## Key Takeaways",
        "1. **Bias Break Consistency**: Observe whether the empirical break dates for the bias index cluster around Q1 2020 (the COVID-19 shock) or vary by company.",
        "2. **Returns Break**: Returns breaks are often noisier and harder to pinpoint exactly, but large volatility clusters (like March 2020) usually drive the algorithmic choice.",
        "3. **Subsample Split Strategy**: The primary break date used for the `is_post_break` indicator in the panel is the **Bias Break Date**. This creates our Pre-period and Post-period for the subsequent Subsample Granger Causality analysis.",
        "4. **Minimum Observations**: Any company failing the valid split criterion (fewer than 60 observations in either the pre or post period) should be interpreted with caution or excluded from split-sample VAR estimation.",
        "",
        f"Plots highlighting the identified breaks vs. the expected March 2020 shock are saved in `{PLOT_DIR}/`."
    ]
    
    (OUTPUT_DIR / "step6_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"  → Written summary report to: {OUTPUT_DIR / 'step6_summary.md'}")
    print("\n=== STEP 6 COMPLETE ===")

if __name__ == "__main__":
    main()
