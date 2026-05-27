#!/usr/bin/env python3
"""
STEP 7: VAR Specification
=========================
Covers Section 7 of VAR_Diagnostic_Checklist.md

7.1 — Lag Order Selection: Estimate VAR up to 8 lags. Select lag using BIC (primary) and AIC.
7.2 — VAR Stability Check: Verify roots of companion matrix are outside the unit circle.
7.3 — Residual Diagnostics:
      - Portmanteau (autocorrelation)
      - Jarque-Bera (normality)
      - ARCH (heteroskedasticity)

Data input: `diagnostics/step6/panel_with_breaks.csv`
Since Step 5 indicated that Bias is I(1) and Returns are I(0) for most companies,
we will use `d_simple_mean_stance` (first difference) and `weekly_log_return` (level).

Outputs (written to diagnostics/step7/):
  - var_lag_selection.csv      (Lag criteria per company)
  - var_diagnostics.csv        (Stability and residual test results)
  - step7_summary.md           (Summary report of model fits)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import het_arch

# Suppress minor warnings
warnings.filterwarnings("ignore")

# Config
STEP6_DIR  = Path(__file__).resolve().parent / "diagnostics" / "step6"
OUTPUT_DIR = Path(__file__).resolve().parent / "diagnostics" / "step7"

BIAS_COL = "d_simple_mean_stance" # Use first differences for Bias (I(1))
RETURN_COL = "weekly_log_return"  # Use levels for Returns (I(0), winsorized)
MAX_LAGS = 8

SELECTED_TICKERS = [
    "ABNB", "AMZN", "T", "BA", "BAC", "GM", "GS", "INTC",
    "MCD", "MSFT", "MS", "SBUX", "UBER", "V", 
    # "WMT", # WMT commented out per instructions
    "WFC",
]

def run_arch_test(resid: pd.Series, maxlag: int = 4) -> float | None:
    """Run univariate Engle's ARCH test on a residual series."""
    try:
        # het_arch returns: LM stat, LM pval, F stat, F pval
        res = het_arch(resid, nlags=maxlag)
        return float(res[1]) # Return LM p-value
    except Exception as e:
        return None

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=== STEP 7: VAR Specification ===\n")
    
    panel_path = STEP6_DIR / "panel_with_breaks.csv"
    if not panel_path.exists():
        print(f"ERROR: {panel_path} not found. Run step6_structural_breaks.py first.")
        sys.exit(1)
        
    df_panel = pd.read_csv(panel_path)
    df_panel["week_monday"] = pd.to_datetime(df_panel["week_monday"])
    
    lag_rows = []
    diag_rows = []
    
    for ticker in SELECTED_TICKERS:
        w_df = df_panel[df_panel["ticker"] == ticker].sort_values("week_monday")
        
        # We need both columns to be non-null
        endog_df = w_df[[BIAS_COL, RETURN_COL]].dropna()
        
        if len(endog_df) < 50:
            print(f"  [{ticker}] Skipped: Insufficient data ({len(endog_df)} rows).")
            continue
            
        print(f"  Processing {ticker} (n={len(endog_df)})...")
        
        # 1. Fit VAR and select lag
        model = VAR(endog_df)
        try:
            # We use trend="c" by default (constant only)
            lag_order_res = model.select_order(maxlags=MAX_LAGS)
            best_aic = lag_order_res.aic
            best_bic = lag_order_res.bic
            
            # Use BIC as primary, fallback to 1 if it chooses 0 (we need a dynamic model)
            chosen_lag = best_bic if best_bic > 0 else 1
            
            lag_rows.append({
                "ticker": ticker,
                "aic_lag": best_aic,
                "bic_lag": best_bic,
                "chosen_lag": chosen_lag,
                "n_obs": len(endog_df)
            })
            
            # 2. Fit the final model with chosen lag
            results = model.fit(chosen_lag)
            
            # 3. Stability check (roots > 1 means stable in statsmodels)
            is_stable = results.is_stable(verbose=False)
            roots = np.abs(results.roots)
            min_root = np.min(roots) if len(roots) > 0 else np.nan
            
            # 4. Residual Diagnostics
            # Portmanteau for autocorrelation (nlags > chosen_lag)
            try:
                # Add enough lags beyond p
                port_res = results.test_whiteness(nlags=chosen_lag + 10)
                port_pval = port_res.pvalue
            except Exception:
                port_pval = np.nan
                
            # Jarque-Bera for normality
            try:
                jb_res = results.test_normality()
                jb_pval = jb_res.pvalue
            except Exception:
                jb_pval = np.nan
                
            # ARCH test on individual equation residuals
            resid_bias = results.resid[BIAS_COL]
            resid_ret = results.resid[RETURN_COL]
            arch_pval_bias = run_arch_test(resid_bias)
            arch_pval_ret = run_arch_test(resid_ret)
            
            diag_rows.append({
                "ticker": ticker,
                "chosen_lag": chosen_lag,
                "is_stable": is_stable,
                "min_root_modulus": min_root,
                "portmanteau_p": port_pval,
                "jb_normality_p": jb_pval,
                "arch_bias_p": arch_pval_bias,
                "arch_return_p": arch_pval_ret,
            })
            
        except Exception as e:
            print(f"    Error fitting VAR for {ticker}: {e}")
            
    # Save CSVs
    lag_df = pd.DataFrame(lag_rows)
    diag_df = pd.DataFrame(diag_rows)
    
    lag_df.to_csv(OUTPUT_DIR / "var_lag_selection.csv", index=False)
    diag_df.to_csv(OUTPUT_DIR / "var_diagnostics.csv", index=False)
    
    print(f"\n  → Saved results to {OUTPUT_DIR}/var_lag_selection.csv")
    print(f"  → Saved diagnostics to {OUTPUT_DIR}/var_diagnostics.csv")
    
    # ---------------------------------------------------------------------------
    # Write Step 7 Summary
    # ---------------------------------------------------------------------------
    summary_lines = [
        "# Step 7: VAR Specification Summary",
        "",
        "## Lag Order Selection",
        "Models were estimated up to a maximum of 8 lags. The optimal lag was chosen using the **Bayesian Information Criterion (BIC)** for parsimony. If BIC selected 0 lags, 1 lag was used to force a dynamic model.",
        "",
        "| Ticker | AIC Lag | BIC Lag | Chosen Lag | N Obs |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in lag_df.iterrows():
        summary_lines.append(f"| {r['ticker']} | {r['aic_lag']} | {r['bic_lag']} | {r['chosen_lag']} | {r['n_obs']} |")
        
    summary_lines += [
        "",
        "## Model Diagnostics",
        "Using the chosen lag, we performed tests on the VAR residuals.",
        "",
        "| Ticker | Stable? | Min Root | Portmanteau $p$ | JB Normality $p$ | ARCH (Bias) $p$ | ARCH (Ret) $p$ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in diag_df.iterrows():
        stable_icon = "✅" if r['is_stable'] else "❌"
        port_p = f"{r['portmanteau_p']:.4f}" if not pd.isna(r['portmanteau_p']) else "N/A"
        jb_p = f"{r['jb_normality_p']:.4f}" if not pd.isna(r['jb_normality_p']) else "N/A"
        arch_b = f"{r['arch_bias_p']:.4f}" if not pd.isna(r['arch_bias_p']) else "N/A"
        arch_r = f"{r['arch_return_p']:.4f}" if not pd.isna(r['arch_return_p']) else "N/A"
        
        summary_lines.append(f"| {r['ticker']} | {stable_icon} | {r['min_root_modulus']:.2f} | {port_p} | {jb_p} | {arch_b} | {arch_r} |")
        
    summary_lines += [
        "",
        "## Key Takeaways",
        "1. **Stability**: All selected models should be stable (min root > 1). If not, the model is explosive.",
        "2. **Autocorrelation (Portmanteau)**: $p > 0.05$ indicates no remaining serial correlation in the residuals, meaning the chosen lag is sufficient.",
        "3. **Normality (Jarque-Bera)**: Financial returns often exhibit heavy tails, leading to $p < 0.05$ (non-normality). VAR estimates remain consistent, but small-sample inference might be affected.",
        "4. **Heteroskedasticity (ARCH)**: $p < 0.05$ indicates volatility clustering. If present, Granger causality tests might benefit from robust standard errors (though baseline VAR tests often ignore this).",
        "",
        "> Note: **WMT** was excluded from this specification pass as requested.",
    ]
    
    (OUTPUT_DIR / "step7_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"  → Written summary report to: {OUTPUT_DIR / 'step7_summary.md'}")
    print("\n=== STEP 7 COMPLETE ===")

if __name__ == "__main__":
    main()
