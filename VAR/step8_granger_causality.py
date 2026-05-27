#!/usr/bin/env python3
"""
STEP 8: Granger Causality
=========================
Covers Section 8 of VAR_Diagnostic_Checklist.md

8.1 — Granger Causality Tests (Full Sample)
8.2 — Pre vs Post Subsample Granger Comparison

Implementation Details:
- Granger Causality is tested in both directions (Bias -> Returns, Returns -> Bias).
- Tests are performed using OLS with HAC-robust standard errors (Newey-West kernel).
  - cov_type='HAC', cov_kwds={'maxlags': chosen_lag}
- Lags are read from `diagnostics/step7/var_lag_selection.csv`.
- The dataset is split into Pre and Post periods using `is_post_break` from Step 6.

Outputs (written to diagnostics/step8/):
  - granger_results.csv   (Detailed F-statistics and p-values for all samples)
  - step8_summary.md      (Comparison tables formatted per the checklist)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Config
STEP6_DIR  = Path(__file__).resolve().parent / "diagnostics" / "step6"
STEP7_DIR  = Path(__file__).resolve().parent / "diagnostics" / "step7"
OUTPUT_DIR = Path(__file__).resolve().parent / "diagnostics" / "step8"

BIAS_COL = "d_simple_mean_stance" # First difference (I(1) logic from Step 5/7)
RETURN_COL = "weekly_log_return"  # Levels

def format_p(p_val: float | None) -> str:
    """Format p-value with significance stars."""
    if p_val is None or pd.isna(p_val):
        return "N/A"
    if p_val < 0.01:
        return f"{p_val:.3f}***"
    elif p_val < 0.05:
        return f"{p_val:.3f}**"
    elif p_val < 0.10:
        return f"{p_val:.3f}*"
    else:
        return f"{p_val:.3f}"

def robust_granger(data: pd.DataFrame, caused: str, causing: str, p: int) -> tuple[float | None, float | None]:
    """
    Perform a Granger causality test using OLS with HAC robust standard errors.
    H0: 'causing' does not Granger-cause 'caused'.
    """
    # Create lagged features
    df = pd.DataFrame()
    df["y"] = data[caused]
    for i in range(1, p + 1):
        df[f"y_L{i}"] = data[caused].shift(i)
        df[f"x_L{i}"] = data[causing].shift(i)
        
    df = df.dropna()
    
    if len(df) <= p * 2 + 5: # Need sufficient degrees of freedom
        return None, None
        
    y = df["y"]
    X = df.drop(columns=["y"])
    X = sm.add_constant(X)
    
    try:
        model = sm.OLS(y, X)
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": p})
        
        # Joint Wald/F-test on all x_L* coefficients being 0
        x_cols = [f"x_L{i}" for i in range(1, p + 1)]
        hypothesis = " = ".join(x_cols) + " = 0"
        
        f_res = res.f_test(hypothesis)
        f_stat = float(np.squeeze(f_res.statistic))
        p_val = float(np.squeeze(f_res.pvalue))
        
        return f_stat, p_val
    except Exception as e:
        print(f"      Granger Test Error: {e}")
        return None, None

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=== STEP 8: Granger Causality (HAC-Robust) ===\n")
    
    # 1. Load Data
    panel_path = STEP6_DIR / "panel_with_breaks.csv"
    if not panel_path.exists():
        print(f"ERROR: {panel_path} not found.")
        sys.exit(1)
    df_panel = pd.read_csv(panel_path)
    df_panel["week_monday"] = pd.to_datetime(df_panel["week_monday"])
    
    # 2. Load Lags
    lags_path = STEP7_DIR / "var_lag_selection.csv"
    if not lags_path.exists():
        print(f"ERROR: {lags_path} not found.")
        sys.exit(1)
    lags_df = pd.read_csv(lags_path)
    ticker_lags = dict(zip(lags_df["ticker"], lags_df["chosen_lag"]))
    
    results = []
    
    for ticker, chosen_lag in ticker_lags.items():
        w_df = df_panel[df_panel["ticker"] == ticker].sort_values("week_monday")
        
        # Split into samples
        samples = {
            "Full": w_df,
            "Pre": w_df[~w_df["is_post_break"]],
            "Post": w_df[w_df["is_post_break"]]
        }
        
        print(f"  Testing {ticker} (Lag = {chosen_lag})...")
        
        for sample_name, s_df in samples.items():
            if len(s_df) < 40: # Arbitrary minimum to avoid crashes
                print(f"    - {sample_name}: Skipped (n={len(s_df)} too small)")
                continue
                
            # Bias -> Returns
            f_b2r, p_b2r = robust_granger(s_df, caused=RETURN_COL, causing=BIAS_COL, p=chosen_lag)
            
            # Returns -> Bias
            f_r2b, p_r2b = robust_granger(s_df, caused=BIAS_COL, causing=RETURN_COL, p=chosen_lag)
            
            results.append({
                "ticker": ticker,
                "sample": sample_name,
                "n_obs": len(s_df),
                "lag": chosen_lag,
                "f_stat_bias2ret": f_b2r,
                "p_val_bias2ret": p_b2r,
                "f_stat_ret2bias": f_r2b,
                "p_val_ret2bias": p_r2b
            })
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_DIR / "step8_firmlevel_granger.csv", index=False)
    print(f"\n  → Saved test results to: {OUTPUT_DIR / 'step8_firmlevel_granger.csv'}")
    
    # ---------------------------------------------------------------------------
    # Write Step 8 Summary
    # ---------------------------------------------------------------------------
    summary_lines = [
        "# Step 8: Granger Causality Summary (Firm-level)",
        "",
        "## Methodology",
        "Granger Causality was tested using a direct OLS framework with **HAC-robust standard errors** (Newey-West kernel, `maxlags = p`). This controls for heteroskedasticity and autocorrelation, which are common in financial returns.",
        "",
        "Significance: `***` $p<0.01$, `**` $p<0.05$, `*` $p<0.10$.",
        "",
        "## Full Sample Results",
        "",
        "| Company | Lag | N Obs | Bias → Returns (p-value) | Returns → Bias (p-value) |",
        "|---|---:|---:|---|---|",
    ]
    
    full_df = res_df[res_df["sample"] == "Full"]
    for _, r in full_df.iterrows():
        summary_lines.append(f"| {r['ticker']} | {r['lag']} | {r['n_obs']} | {format_p(r['p_val_bias2ret'])} | {format_p(r['p_val_ret2bias'])} |")
        
    summary_lines += [
        "",
        "## Pre vs Post Subsample Granger Comparison",
        "The sample was split around the empirically identified structural break date for the bias index (Step 6).",
        "",
        "| Company | Pre-period (Bias → Returns) | Pre-period (Returns → Bias) | Post-period (Bias → Returns) | Post-period (Returns → Bias) |",
        "|---|---|---|---|---|",
    ]
    
    for ticker in ticker_lags.keys():
        pre_row = res_df[(res_df["ticker"] == ticker) & (res_df["sample"] == "Pre")]
        post_row = res_df[(res_df["ticker"] == ticker) & (res_df["sample"] == "Post")]
        
        pre_b2r = format_p(pre_row["p_val_bias2ret"].values[0]) if not pre_row.empty else "N/A"
        pre_r2b = format_p(pre_row["p_val_ret2bias"].values[0]) if not pre_row.empty else "N/A"
        
        post_b2r = format_p(post_row["p_val_bias2ret"].values[0]) if not post_row.empty else "N/A"
        post_r2b = format_p(post_row["p_val_ret2bias"].values[0]) if not post_row.empty else "N/A"
        
        summary_lines.append(f"| {ticker} | p = {pre_b2r} | p = {pre_r2b} | p = {post_b2r} | p = {post_r2b} |")
        
    summary_lines += [
        "",
        "## Key Takeaways",
        "1. **Unidirectional Causality (Bias → Returns)**: This is the primary hypothesis. Check if significance emerges strongly in the Post-period.",
        "2. **Bidirectional Causality**: Evidence of feedback loops between price discovery and media coverage.",
        "3. **Structural Shift**: If pre-period results are insignificant but post-period results are significant, this validates the DiD result showing a structural break in the relationship.",
    ]
    
    (OUTPUT_DIR / "step8_firmlevel_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"  → Written summary report to: {OUTPUT_DIR / 'step8_firmlevel_summary.md'}")
    print("\n=== STEP 8 COMPLETE ===")

if __name__ == "__main__":
    main()
