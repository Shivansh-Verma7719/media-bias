#!/usr/bin/env python3
"""
STEP 8: Panel VAR Granger Causality
===================================
Re-writes Step 8 using a pooled Panel VAR with Helmert transformation (forward mean-differencing).

Data inputs:
  - diagnostics/step6/panel_with_breaks.csv
  - yfinance (S&P 500 log returns and VIX)

Methodology:
  - Exogenous variables: Post(t), VIX, market_return (S&P 500), n_articles
  - Lags: p=3 (selected modal BIC from Step 7)
  - Transformation: Arellano-Bover / Helmert transformation applied to endogenous, 
    lagged, and exogenous variables per firm to remove fixed effects without look-ahead bias.
  - Estimation: Pooled OLS on transformed data (without constant), using HAC-robust SEs.
  - Granger Tests: Joint Wald (F) tests on the pooled coefficients.
  - Subsamples: Full, Pre-break, Post-break.

Outputs (written to diagnostics/step8/):
  - step8_panel_granger.csv
  - step8_panel_summary.md
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf

warnings.filterwarnings("ignore")

# Config
STEP6_DIR  = Path(__file__).resolve().parent / "diagnostics" / "step6"
OUTPUT_DIR = Path(__file__).resolve().parent / "diagnostics" / "step8"

BIAS_COL = "d_simple_mean_stance"
RETURN_COL = "weekly_log_return"
LAG_P = 3

def fetch_market_data() -> pd.DataFrame:
    """Download and resample VIX and S&P 500 from yfinance."""
    print("  Downloading ^GSPC (S&P 500) and ^VIX from yfinance...")
    
    # S&P 500
    spy = yf.download("^GSPC", start="2015-01-01", end="2026-01-01", auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
        
    spy["log_ret"] = np.log(spy["Close"] / spy["Close"].shift(1))
    spy_weekly = spy["log_ret"].resample("W-MON").apply(lambda x: x.sum())
    spy_weekly.name = "market_return"

    # VIX
    vix = yf.download("^VIX", start="2015-01-01", end="2026-01-01", progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
        
    vix_weekly = vix["Close"].resample("W-MON").last()
    vix_weekly.name = "VIX"

    market_df = pd.concat([spy_weekly, vix_weekly], axis=1).reset_index()
    # Handle possible "Date" or "Datetime" column naming from yfinance
    date_col = market_df.columns[0]
    market_df = market_df.rename(columns={date_col: "week_monday"})
    
    # Ensure timezone naive for merging
    if market_df["week_monday"].dt.tz is not None:
        market_df["week_monday"] = market_df["week_monday"].dt.tz_convert(None)
    
    return market_df

def apply_helmert_transformation(group: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Apply Arellano-Bover / Helmert transformation (forward mean-differencing)
    to a group of time-series data.
    """
    T = len(group)
    transformed = pd.DataFrame(index=group.index, columns=cols, dtype=float)
    
    # We need at least 2 observations to do forward differencing
    if T < 2:
        return transformed

    # Vectorized Helmert transformation
    # values: (T, K) array
    values = group[cols].values
    
    for t in range(T - 1):
        c = np.sqrt((T - t - 1) / (T - t))
        future_mean = np.mean(values[t + 1:], axis=0)
        transformed.iloc[t] = c * (values[t] - future_mean)
        
    # The last row cannot be transformed because there are no future observations
    # It will remain NaN.
    return transformed

def prepare_and_test(df: pd.DataFrame, sample_name: str) -> dict | None:
    """
    Create lags, apply Helmert transformation, run Panel VAR, and Granger causality tests.
    """
    print(f"\n  --- Running Panel VAR for: {sample_name} ---")
    
    # 1. Create lagged variables for Endogenous variables
    df = df.sort_values(["ticker", "week_monday"]).copy()
    
    for i in range(1, LAG_P + 1):
        df[f"{BIAS_COL}_L{i}"] = df.groupby("ticker")[BIAS_COL].shift(i)
        df[f"{RETURN_COL}_L{i}"] = df.groupby("ticker")[RETURN_COL].shift(i)
        
    # Drop rows with NaN due to lagging
    df = df.dropna(subset=[f"{BIAS_COL}_L{LAG_P}", f"{RETURN_COL}_L{LAG_P}", "VIX", "market_return"])
    
    if len(df) < 50:
        print(f"  Skipping {sample_name} (Insufficient observations: {len(df)})")
        return None

    # 2. Define all variables that need Helmert transformation
    y_cols = [BIAS_COL, RETURN_COL]
    x_lag_cols = [f"{BIAS_COL}_L{i}" for i in range(1, LAG_P + 1)] + \
                 [f"{RETURN_COL}_L{i}" for i in range(1, LAG_P + 1)]
    exog_cols = ["is_post_break", "VIX", "market_return", "n_articles"]
    
    # Ensure float type for all transformation columns
    df["is_post_break"] = df["is_post_break"].astype(float)
    df["n_articles"] = df["n_articles"].astype(float)
    
    all_transform_cols = y_cols + x_lag_cols + exog_cols
    
    # 3. Apply Helmert transformation per ticker
    print("    Applying Helmert transformation...")
    transformed_list = []
    for ticker, group in df.groupby("ticker"):
        t_group = apply_helmert_transformation(group, all_transform_cols)
        transformed_list.append(t_group)
        
    df_transformed = pd.concat(transformed_list)
    df_transformed = df_transformed.dropna() # Drops the last observation per firm
    
    N_obs = len(df_transformed)
    print(f"    Pooled estimation N = {N_obs}")
    
    if N_obs < (len(x_lag_cols) + len(exog_cols) + 10):
        print("    Not enough observations after transformation.")
        return None

    # 4. Estimation using HAC Robust SEs (no constant because of Helmert)
    # Predictors: Lags of Endogenous + Exogenous
    X = df_transformed[x_lag_cols + exog_cols]
    
    results = {"sample": sample_name, "n_obs": N_obs}
    
    # Hypotheses
    # To test if Returns Granger-cause Bias: Wald test on all RETURN_COL_L* = 0 in Bias Eq
    ret_lags = [f"{RETURN_COL}_L{i}" for i in range(1, LAG_P + 1)]
    hypothesis_r2b = " = ".join(ret_lags) + " = 0"
    
    # To test if Bias Granger-causes Returns: Wald test on all BIAS_COL_L* = 0 in Return Eq
    bias_lags = [f"{BIAS_COL}_L{i}" for i in range(1, LAG_P + 1)]
    hypothesis_b2r = " = ".join(bias_lags) + " = 0"

    try:
        # Eq 1: Bias Equation
        model_bias = sm.OLS(df_transformed[BIAS_COL], X)
        res_bias = model_bias.fit(cov_type="HAC", cov_kwds={"maxlags": LAG_P})
        f_r2b = res_bias.f_test(hypothesis_r2b)
        results["f_stat_ret2bias"] = float(np.squeeze(f_r2b.statistic))
        results["p_val_ret2bias"] = float(np.squeeze(f_r2b.pvalue))
        
        # Eq 2: Return Equation
        model_ret = sm.OLS(df_transformed[RETURN_COL], X)
        res_ret = model_ret.fit(cov_type="HAC", cov_kwds={"maxlags": LAG_P})
        f_b2r = res_ret.f_test(hypothesis_b2r)
        results["f_stat_bias2ret"] = float(np.squeeze(f_b2r.statistic))
        results["p_val_bias2ret"] = float(np.squeeze(f_b2r.pvalue))
        
    except Exception as e:
        print(f"    Error during estimation: {e}")
        return None
        
    return results

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

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== STEP 8: Panel VAR Granger Causality ===\n")
    
    # 1. Load Panel Data
    panel_path = STEP6_DIR / "panel_with_breaks.csv"
    if not panel_path.exists():
        print(f"ERROR: {panel_path} not found.")
        sys.exit(1)
        
    df_panel = pd.read_csv(panel_path)
    df_panel["week_monday"] = pd.to_datetime(df_panel["week_monday"])
    
    # Ensure WMT is excluded
    df_panel = df_panel[df_panel["ticker"] != "WMT"].copy()
    
    # 2. Fetch Market Data and Merge
    market_df = fetch_market_data()
    df_panel = pd.merge(df_panel, market_df, on="week_monday", how="left")
    
    # 3. Create Subsamples
    # We define subsamples BEFORE lagging and Helmert transformation
    samples = {
        "Full": df_panel.copy(),
        "Pre-break": df_panel[~df_panel["is_post_break"]].copy(),
        "Post-break": df_panel[df_panel["is_post_break"]].copy()
    }
    
    # 4. Run Estimation
    all_results = []
    for s_name, s_df in samples.items():
        res = prepare_and_test(s_df, sample_name=s_name)
        if res:
            all_results.append(res)
            
    if not all_results:
        print("No valid results were generated.")
        sys.exit(1)
        
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(OUTPUT_DIR / "step8_panel_granger.csv", index=False)
    print(f"\n  → Saved panel test results to: {OUTPUT_DIR / 'step8_panel_granger.csv'}")
    
    # 5. Write Summary Markdown
    summary_lines = [
        "# Step 8: Panel VAR Granger Causality Summary",
        "",
        "## Methodology",
        "A **Pooled Panel VAR** was estimated covering 15 companies. To remove unobserved firm fixed effects, the **Helmert transformation** (forward mean-differencing) was applied. This approach preserves orthogonality between transformed variables and lagged regressors, avoiding the look-ahead bias inherent in standard mean-differencing.",
        "",
        "- **Endogenous Variables**: $\Delta \text{Bias}$ and Weekly Log Returns.",
        "- **Exogenous Controls**: Structural Break Dummy ($Post_t$), VIX, S&P 500 Weekly Log Returns, and Article Count ($N_{articles}$).",
        f"- **Lags**: $p={LAG_P}$ (modal BIC choice).",
        "- **Standard Errors**: HAC-robust (Newey-West, maxlags=3).",
        "",
        "Significance: `***` $p<0.01$, `**` $p<0.05$, `*` $p<0.10$.",
        "",
        "## Results",
        "",
        "| Sample | N Obs | Bias → Returns ($p$-value) | Returns → Bias ($p$-value) |",
        "|---|---:|---|---|",
    ]
    
    for _, r in res_df.iterrows():
        b2r = format_p(r.get('p_val_bias2ret', None))
        r2b = format_p(r.get('p_val_ret2bias', None))
        summary_lines.append(f"| {r['sample']} | {r['n_obs']} | {b2r} | {r2b} |")
        
    summary_lines += [
        "",
        "## Key Takeaways",
        "1. **Full Panel Analysis**: By pooling the cross-section, the test gains substantial statistical power.",
        "2. **Pre vs Post Comparison**: If Bias → Returns is significant in the Post-break period but not the Pre-break period, it robustly confirms a structural shift in the influence of media bias on stock returns after 2020.",
        "3. **Exogenous Factors Controlled**: Market-wide volatility (VIX) and returns (S&P 500) are explicitly partialed out."
    ]
    
    (OUTPUT_DIR / "step8_panel_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"  → Written summary report to: {OUTPUT_DIR / 'step8_panel_summary.md'}")
    print("\n=== STEP 8 COMPLETE ===")

if __name__ == "__main__":
    main()
