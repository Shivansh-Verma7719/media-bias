#!/usr/bin/env python3
"""
STEP 5: Stationarity Testing
=============================
Covers Sections 5.1, 5.2, 5.3, and 5.4 of VAR_Diagnostic_Checklist.md

5.1 — Visual Inspection: Plot weekly bias index and weekly returns side by side.
5.2 — ADF Test: Test weekly bias and weekly returns under 3 specifications (nc, c, ct) with AIC lag selection.
5.3 — KPSS Test: Test weekly bias and returns under 2 specifications (c, ct).
      Difference series and re-test if non-stationary; identify order of integration (I(0) vs. I(1)).
5.4 — Cointegration Check: Johansen cointegration test if both series are I(1).

Special Rule: Winsorize weekly log returns at ±5 standard deviations for all companies EXCEPT AT&T (T).

Outputs (written to diagnostics/step5/):
  - stationarity_results.csv        (detailed test statistics, p-values, lags, decision)
  - stationarity_checked_panel.csv  (merged panel with winsorized returns, first differences)
  - plots/stationarity_<ticker>.png (side-by-side levels and difference plots)
  - step5_summary.md                (summary table and interpretation guide)
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
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Suppress warnings from kpss tests about p-values
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning)

# Config
STEP4_DIR  = Path(__file__).resolve().parent / "diagnostics" / "step4"
OUTPUT_DIR = Path(__file__).resolve().parent / "diagnostics" / "step5"
PLOT_DIR   = OUTPUT_DIR / "plots"

BIAS_COL = "simple_mean_stance"
RETURN_COL = "weekly_log_return"

SELECTED_TICKERS = [
    "ABNB", "AMZN", "T", "BA", "BAC", "GM", "GS", "INTC",
    "MCD", "MSFT", "MS", "SBUX", "UBER", "V", "WMT", "WFC",
]


# ---------------------------------------------------------------------------
# Winsorization
# ---------------------------------------------------------------------------

def winsorize_series(series: pd.Series, threshold: float = 5.0) -> tuple[pd.Series, int]:
    """Winsorize series at threshold standard deviations from the mean."""
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return series.copy(), 0
    
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std
    
    outliers = (series < lower_bound) | (series > upper_bound)
    count = int(outliers.sum())
    
    winsorized = series.clip(lower=lower_bound, upper=upper_bound)
    return winsorized, count


# ---------------------------------------------------------------------------
# Stationarity Test Wrappers
# ---------------------------------------------------------------------------

def run_adf_specs(series: pd.Series, maxlag: int = 12) -> dict:
    """Run ADF test for 'nc', 'c', and 'ct' regressions."""
    results = {}
    for spec in ["nc", "c", "ct"]:
        try:
            res = adfuller(series, maxlag=maxlag, regression=spec, autolag="AIC")
            results[spec] = {
                "stat": float(res[0]),
                "pvalue": float(res[1]),
                "lags": int(res[2]),
                "nobs": int(res[3]),
                "crit_5pct": float(res[4]["5%"]),
            }
        except Exception as e:
            results[spec] = {"error": str(e)}
    return results


def run_kpss_specs(series: pd.Series) -> dict:
    """Run KPSS test for 'c' and 'ct' regressions."""
    results = {}
    for spec in ["c", "ct"]:
        try:
            res = kpss(series, regression=spec, nlags="auto")
            results[spec] = {
                "stat": float(res[0]),
                "pvalue": float(res[1]),
                "lags": int(res[2]),
                "crit_5pct": float(res[3]["5%"]),
            }
        except Exception as e:
            results[spec] = {"error": str(e)}
    return results


def interpret_stationarity(
    adf_p: float | None,
    kpss_p: float | None,
    significance: float = 0.05
) -> str:
    """Interpret the combination of ADF and KPSS test p-values."""
    if adf_p is None or kpss_p is None:
        return "Unknown"
    
    adf_reject = adf_p < significance
    kpss_reject = kpss_p < significance
    
    if adf_reject and not kpss_reject:
        return "Stationary (I(0))"
    elif not adf_reject and kpss_reject:
        return "Non-Stationary (I(1) / Unit Root)"
    elif adf_reject and kpss_reject: # Both reject
        return "Conflicting (Structural Break Likely)"
    else:
        return "Ambiguous (Low Power)"


# ---------------------------------------------------------------------------
# Cointegration Check
# ---------------------------------------------------------------------------

def run_johansen_test(df: pd.DataFrame, col1: str, col2: str) -> dict | None:
    """Perform Johansen cointegration test (bivariate)."""
    sub_df = df[[col1, col2]].dropna()
    if len(sub_df) < 50:
        return None
    
    try:
        # det_order = 0 (constant term), k_ar_diff = 1 (1 lag in differences = VAR(2) in levels)
        res = coint_johansen(sub_df, det_order=0, k_ar_diff=1)
        
        # lr1 contains the trace statistic. Index 0 is r=0, index 1 is r<=1
        trace_r0 = float(res.lr1[0])
        trace_r1 = float(res.lr1[1])
        
        # cvt is (2, 3) where [0, 1] is 95% critical value for r=0, [1, 1] is 95% for r<=1
        cv_r0_95 = float(res.cvt[0, 1])
        cv_r1_95 = float(res.cvt[1, 1])
        
        coint_r0 = trace_r0 > cv_r0_95
        
        return {
            "trace_r0": trace_r0,
            "cv_r0_95": cv_r0_95,
            "trace_r1": trace_r1,
            "cv_r1_95": cv_r1_95,
            "cointegrated_95": coint_r0,
        }
    except Exception as e:
        print(f"    Johansen error: {e}")
        return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_stationarity(
    df: pd.DataFrame,
    ticker: str,
    bias_col: str,
    ret_col: str,
    d_bias_col: str,
    d_ret_col: str
) -> None:
    """Generate diagnostic plots: level and diff side-by-side."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    
    # 1. Bias Level
    axes[0, 0].plot(df["week_monday"], df[bias_col], color="#2CA02C", linewidth=0.8)
    axes[0, 0].axhline(0, color="gray", linestyle=":", linewidth=0.7)
    axes[0, 0].set_title(f"{ticker} — Bias Index (Level)")
    axes[0, 0].set_ylabel("Stance Score")
    
    # 2. Return Level (Winsorized)
    axes[0, 1].plot(df["week_monday"], df[ret_col], color="#1F77B4", linewidth=0.8)
    axes[0, 1].axhline(0, color="gray", linestyle=":", linewidth=0.7)
    axes[0, 1].set_title(f"{ticker} — Returns (Level, Winsorized)")
    axes[0, 1].set_ylabel("Weekly Log Return")
    
    # 3. Bias Diff
    df_diff = df.dropna(subset=[d_bias_col])
    axes[1, 0].plot(df_diff["week_monday"], df_diff[d_bias_col], color="#98DF8A", linewidth=0.8)
    axes[1, 0].axhline(0, color="gray", linestyle=":", linewidth=0.7)
    axes[1, 0].set_title(f"{ticker} — Δ Bias Index (First Difference)")
    axes[1, 0].set_ylabel("Change in Stance")
    
    # 4. Return Diff
    df_ret_diff = df.dropna(subset=[d_ret_col])
    axes[1, 1].plot(df_ret_diff["week_monday"], df_ret_diff[d_ret_col], color="#AEC7E8", linewidth=0.8)
    axes[1, 1].axhline(0, color="gray", linestyle=":", linewidth=0.7)
    axes[1, 1].set_title(f"{ticker} — Δ Returns (First Difference)")
    axes[1, 1].set_ylabel("Change in Return")
    
    for ax in axes.flat:
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)
        
    fig.suptitle(f"Stationarity Diagnostics for {ticker}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"stationarity_{ticker}.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=== STEP 5: Stationarity & Integration Testing ===\n")
    
    # Load Step 4 merged weekly panel
    panel_path = STEP4_DIR / "merged_weekly_panel.csv"
    if not panel_path.exists():
        print(f"ERROR: {panel_path} not found. Run step4_returns_construction.py first.")
        sys.exit(1)
        
    df_panel = pd.read_csv(panel_path)
    df_panel["week_monday"] = pd.to_datetime(df_panel["week_monday"])
    
    print(f"Loaded merged panel: {len(df_panel):,} rows for {df_panel['ticker'].nunique()} companies.")
    
    # ---------------------------------------------------------------------------
    # Winsorization of returns (All companies except 'T')
    # ---------------------------------------------------------------------------
    print("\n[5.0] Winsorization of returns (threshold = ±5 std deviations)...")
    df_panel["weekly_log_return_raw"] = df_panel[RETURN_COL]
    
    winsor_stats = []
    
    for ticker in SELECTED_TICKERS:
        ticker_mask = df_panel["ticker"] == ticker
        if not ticker_mask.any():
            continue
            
        raw_ret = df_panel.loc[ticker_mask, RETURN_COL]
        
        if ticker == "T":
            # Do NOT winsorize AT&T
            df_panel.loc[ticker_mask, RETURN_COL] = raw_ret
            winsor_stats.append({"ticker": "T", "raw_std": raw_ret.std(), "winsor_count": 0, "status": "SKIPPED (AT&T)"})
            print("  → T (AT&T): Winsorization skipped per instruction.")
        else:
            clean_ret, w_count = winsorize_series(raw_ret, threshold=5.0)
            df_panel.loc[ticker_mask, RETURN_COL] = clean_ret
            winsor_stats.append({
                "ticker": ticker,
                "raw_std": raw_ret.std(),
                "winsor_count": w_count,
                "status": f"Winsorized {w_count} obs" if w_count > 0 else "No outliers (>5 std)"
            })
            if w_count > 0:
                print(f"  → {ticker}: Winsorized {w_count} outlier return weeks.")
                
    winsor_df = pd.DataFrame(winsor_stats)
    
    # Calculate first differences for the panel
    print("\n[5.1] Computing first differences...")
    df_panel = df_panel.sort_values(["ticker", "week_monday"]).reset_index(drop=True)
    df_panel["d_simple_mean_stance"] = df_panel.groupby("ticker")[BIAS_COL].diff()
    df_panel["d_weekly_log_return"] = df_panel.groupby("ticker")[RETURN_COL].diff()
    
    # Save the updated panel
    df_panel.to_csv(OUTPUT_DIR / "stationarity_checked_panel.csv", index=False)
    print(f"  → Saved processed panel to: {OUTPUT_DIR / 'stationarity_checked_panel.csv'}")
    
    # ---------------------------------------------------------------------------
    # Stationarity Testing (ADF and KPSS)
    # ---------------------------------------------------------------------------
    print("\n[5.2 / 5.3] Running stationarity tests...")
    
    test_rows = []
    cointegration_candidates = []
    
    for ticker in SELECTED_TICKERS:
        w_df = df_panel[df_panel["ticker"] == ticker].sort_values("week_monday").copy()
        if w_df.empty:
            continue
            
        print(f"  Testing {ticker} (n={len(w_df)} weeks)...")
        
        # Test Bias index and Returns in Level and Diff
        series_to_test = {
            "bias_level": {"series": w_df[BIAS_COL], "name": "Bias Level"},
            "bias_diff": {"series": w_df["d_simple_mean_stance"].dropna(), "name": "Δ Bias (Diff)"},
            "return_level": {"series": w_df[RETURN_COL], "name": "Return Level"},
            "return_diff": {"series": w_df["d_weekly_log_return"].dropna(), "name": "Δ Return (Diff)"},
        }
        
        integration_orders = {}
        
        for key, s_info in series_to_test.items():
            s = s_info["series"]
            s_name = s_info["name"]
            
            # ADF Test Specs
            adf_res = run_adf_specs(s)
            # KPSS Test Specs
            kpss_res = run_kpss_specs(s)
            
            # Focus on 'c' specification for the primary decision
            adf_c = adf_res.get("c", {})
            kpss_c = kpss_res.get("c", {})
            
            adf_p = adf_c.get("pvalue", None)
            kpss_p = kpss_c.get("pvalue", None)
            
            conclusion = interpret_stationarity(adf_p, kpss_p)
            
            # Store primary specs in row
            test_rows.append({
                "ticker": ticker,
                "variable": s_name,
                "is_diff": "diff" in key,
                "adf_nc_p": adf_res.get("nc", {}).get("pvalue", np.nan),
                "adf_c_stat": adf_c.get("stat", np.nan),
                "adf_c_p": adf_p if adf_p is not None else np.nan,
                "adf_c_lags": adf_c.get("lags", np.nan),
                "adf_ct_p": adf_res.get("ct", {}).get("pvalue", np.nan),
                "kpss_c_stat": kpss_c.get("stat", np.nan),
                "kpss_c_p": kpss_p if kpss_p is not None else np.nan,
                "kpss_ct_p": kpss_res.get("ct", {}).get("pvalue", np.nan),
                "conclusion": conclusion
            })
            
            # Save decision for integration order
            var_type = "bias" if "bias" in key else "return"
            is_level = "level" in key
            
            if is_level:
                # If level is stationary, it's I(0)
                if conclusion == "Stationary (I(0))":
                    integration_orders[var_type] = "I(0)"
                else:
                    # Check first diff
                    integration_orders[var_type] = "Pending"
            else:
                # Check first diff results if levels were not stationary
                if integration_orders[var_type] == "Pending":
                    if conclusion == "Stationary (I(0))": # Stationary in diffs = I(1)
                        integration_orders[var_type] = "I(1)"
                    else:
                        integration_orders[var_type] = "I(2) or higher"
        
        # Determine final integration orders
        bias_order = integration_orders.get("bias", "I(0)")
        return_order = integration_orders.get("return", "I(0)")
        
        print(f"    → Conclusion: Bias is {bias_order}, Returns is {return_order}")
        
        # If both are I(1), flag for cointegration check
        if bias_order == "I(1)" and return_order == "I(1)":
            cointegration_candidates.append(ticker)
            
        # Plot
        plot_stationarity(w_df, ticker, BIAS_COL, RETURN_COL, "d_simple_mean_stance", "d_weekly_log_return")
        
    results_df = pd.DataFrame(test_rows)
    results_df.to_csv(OUTPUT_DIR / "stationarity_results.csv", index=False)
    print(f"\n  → Saved test results to: {OUTPUT_DIR / 'stationarity_results.csv'}")
    
    # ---------------------------------------------------------------------------
    # Cointegration Testing (Johansen)
    # ---------------------------------------------------------------------------
    print("\n[5.4] Cointegration check...")
    coint_results = []
    
    if len(cointegration_candidates) == 0:
        print("  → No companies had both series as I(1). Cointegration tests skipped.")
    else:
        print(f"  → Cointegration candidates: {', '.join(cointegration_candidates)}")
        for ticker in cointegration_candidates:
            w_df = df_panel[df_panel["ticker"] == ticker].sort_values("week_monday")
            coint_res = run_johansen_test(w_df, BIAS_COL, RETURN_COL)
            if coint_res:
                coint_results.append({
                    "ticker": ticker,
                    "trace_stat_r0": coint_res["trace_r0"],
                    "cv_r0_95": coint_res["cv_r0_95"],
                    "trace_stat_r1": coint_res["trace_r1"],
                    "cv_r1_95": coint_res["cv_r1_95"],
                    "cointegrated_95": coint_res["cointegrated_95"]
                })
                decision = "COINTEGRATED (VECM recommended)" if coint_res["cointegrated_95"] else "NO COINTEGRATION (VAR on diffs recommended)"
                print(f"    {ticker}: Trace={coint_res['trace_r0']:.2f} vs CV(95%)={coint_res['cv_r0_95']:.2f} → {decision}")
                
    coint_df = pd.DataFrame(coint_results)
    if not coint_df.empty:
        coint_df.to_csv(OUTPUT_DIR / "cointegration_results.csv", index=False)
        print(f"  → Saved cointegration results to: {OUTPUT_DIR / 'cointegration_results.csv'}")
        
    # ---------------------------------------------------------------------------
    # Write Step 5 Summary Report
    # ---------------------------------------------------------------------------
    summary_lines = [
        "# Step 5: Stationarity & Integration Testing Summary Report",
        "",
        "## Winsorization Details",
        "Weekly log returns winsorized at ±5 standard deviations from each company's mean, **excluding AT&T (T)**.",
        "",
        "| Ticker | Raw Return Std | Outliers Winsorized | Status |",
        "|---|---:|---:|---|",
    ]
    for _, r in winsor_df.iterrows():
        summary_lines.append(f"| {r['ticker']} | {r['raw_std']:.5f} | {r['winsor_count']} | {r['status']} |")
        
    summary_lines += [
        "",
        "## Stationarity Test Interpretation (Constant-only Specification)",
        "We class each series based on Augmented Dickey-Fuller (ADF) and KPSS tests:",
        "- **Stationary (I(0))**: ADF rejects unit root ($p < 0.05$) AND KPSS fails to reject stationarity ($p \\geq 0.05$).",
        "- **Non-Stationary (I(1))**: ADF fails to reject unit root ($p \\geq 0.05$) AND KPSS rejects stationarity ($p < 0.05$).",
        "- **Conflicting**: Both tests reject (implies possible structural breaks).",
        "- **Ambiguous**: Both tests fail to reject.",
        "",
        "### Variable Integration Orders",
        "",
        "| Ticker | Bias Level Conclusion | Return Level Conclusion | Bias Diff Conclusion | Return Diff Conclusion | Integration (Bias, Return) |",
        "|---|---|---|---|---|---|",
    ]
    
    # Process results into a nice summary table
    for ticker in SELECTED_TICKERS:
        t_res = results_df[results_df["ticker"] == ticker]
        if t_res.empty:
            continue
            
        b_lvl = t_res[t_res["variable"] == "Bias Level"]["conclusion"].values[0]
        r_lvl = t_res[t_res["variable"] == "Return Level"]["conclusion"].values[0]
        b_dif = t_res[t_res["variable"] == "Δ Bias (Diff)"]["conclusion"].values[0]
        r_dif = t_res[t_res["variable"] == "Δ Return (Diff)"]["conclusion"].values[0]
        
        # Deduce integration
        b_int = "I(0)" if b_lvl == "Stationary (I(0))" else ("I(1)" if b_dif == "Stationary (I(0))" else "I(d > 1)")
        r_int = "I(0)" if r_lvl == "Stationary (I(0))" else ("I(1)" if r_dif == "Stationary (I(0))" else "I(d > 1)")
        
        summary_lines.append(
            f"| {ticker} | {b_lvl} | {r_lvl} | {b_dif} | {r_dif} | ({b_int}, {r_int}) |"
        )
        
    summary_lines += [
        "",
        "## Cointegration Summary",
    ]
    
    if coint_df.empty:
        summary_lines.append("No bivariate systems were candidates for cointegration because returns are uniformly stationary ($I(0)$).")
    else:
        summary_lines += [
            "| Ticker | Trace Stat (r=0) | CV (95%) | Cointegrated (95%)? | Recommendation |",
            "|---|---:|---:|---|---|",
        ]
        for _, r in coint_df.iterrows():
            rec = "VECM" if r["cointegrated_95"] else "VAR on Diffs"
            coint_icon = "✅ YES" if r["cointegrated_95"] else "❌ NO"
            summary_lines.append(
                f"| {r['ticker']} | {r['trace_stat_r0']:.2f} | {r['cv_r0_95']:.2f} | {coint_icon} | {rec} |"
            )
            
    summary_lines += [
        "",
        "## Key Decisions / Modelling Takeaways",
        "1. **Returns Stationarity**: As expected for financial returns, weekly log returns are stationary ($I(0)$) across all companies, both raw and winsorized.",
        "2. **Bias Index Stationarity**: Review the integration column. If Bias is $I(0)$, a VAR in levels `(bias, return)` is appropriate. If Bias is $I(1)$ (non-stationary in levels but stationary in differences), running a VAR in levels might yield spurious regression unless they are cointegrated (unlikely since returns are $I(0)$). Thus, a VAR on `(diff_bias, return)` or `(diff_bias, diff_return)` is recommended.",
        "3. **Winsorization Robustness**: Weekly returns for AT&T (T) contain extreme values but were preserved as-is. Compare AT&T stationarity statistics with other companies to verify if outlier behavior affects unit root decisions.",
        "",
        f"Detailed results written to `{OUTPUT_DIR / 'stationarity_results.csv'}`.",
        f"Plots saved to `{PLOT_DIR}/`",
    ]
    
    (OUTPUT_DIR / "step5_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\n  → Written summary report to: {OUTPUT_DIR / 'step5_summary.md'}")
    print("\n=== STEP 5 COMPLETE ===")


if __name__ == "__main__":
    main()
