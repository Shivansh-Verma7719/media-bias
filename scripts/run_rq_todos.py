#!/usr/bin/env python3
"""
Script to run the RQ1 log-volume re-estimate and RQ2 two-way FE re-estimate
using saved panel CSVs (no DB access required).
Writes outputs to results/rq1/log_volume_* and results/rq2/two_way_fe_*
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RQ_DIR = ROOT / "rq"
RESULTS = ROOT / "results"

def _normal_2sided_p(t_stat: np.ndarray) -> np.ndarray:
    abs_t = np.abs(t_stat)
    return np.array([np.math.erfc(float(v) / np.sqrt(2.0)) for v in abs_t])


def _cluster_robust_inference(x_tilde: np.ndarray, resid: np.ndarray, clusters: pd.Series) -> dict | None:
    cluster_codes, unique_clusters = pd.factorize(clusters, sort=True)
    n, k = x_tilde.shape
    g = int(len(unique_clusters))
    if g <= 1 or n <= k:
        return None

    xtx = x_tilde.T @ x_tilde
    xtx_inv = np.linalg.pinv(xtx)
    meat = np.zeros((k, k), dtype=float)

    for g_code in range(g):
        mask = cluster_codes == g_code
        xg = x_tilde[mask]
        ug = resid[mask]
        score = xg.T @ ug
        meat += np.outer(score, score)

    finite_sample = (g / (g - 1)) * ((n - 1) / (n - k)) if n > k + 1 else 1.0
    vcov = finite_sample * (xtx_inv @ meat @ xtx_inv)
    se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))

    return {
        "cluster_col": str(clusters.name or "cluster"),
        "n_groups": g,
        "finite_sample_correction": float(finite_sample),
        "se": se,
    }


def _within_transform(panel: pd.DataFrame, y_col: str, x_cols: list[str], entity_col: str, time_col: str, time_fe: bool):
    y = panel[y_col].astype(float)
    x = panel[x_cols].astype(float)

    entity_mean_y = y.groupby(panel[entity_col]).transform("mean")
    y_tilde = y - entity_mean_y

    x_tilde = np.column_stack([(x[c] - x[c].groupby(panel[entity_col]).transform("mean")) .to_numpy() for c in x_cols])

    if time_fe:
        time_mean_y = y.groupby(panel[time_col]).transform("mean")
        overall_y = float(y.mean())
        y_tilde = y_tilde - (time_mean_y - overall_y)

        for i, c in enumerate(x_cols):
            time_mean_x = x[c].groupby(panel[time_col]).transform("mean")
            overall_x = float(x[c].mean())
            x_tilde[:, i] = x_tilde[:, i] - (time_mean_x - overall_x).to_numpy()

    col_std = np.std(x_tilde, axis=0)
    keep_cols = col_std > 1e-12
    kept_names = [name for name, keep in zip(x_cols, keep_cols) if keep]
    dropped_names = [name for name, keep in zip(x_cols, keep_cols) if not keep]

    if not np.all(keep_cols):
        x_tilde = x_tilde[:, keep_cols]

    return y_tilde.to_numpy(), x_tilde, kept_names, dropped_names


def fit_within_ols(panel: pd.DataFrame, y_col: str, x_cols: list[str], entity_col: str, time_col: str, time_fe: bool, cluster_col: str | None = None):
    y_tilde, x_tilde, kept_cols, dropped_cols = _within_transform(panel=panel, y_col=y_col, x_cols=x_cols, entity_col=entity_col, time_col=time_col, time_fe=time_fe)

    keep = np.isfinite(y_tilde) & np.all(np.isfinite(x_tilde), axis=1)
    y_tilde = y_tilde[keep]
    x_tilde = x_tilde[keep]
    cluster_series = None
    if cluster_col is not None:
        cluster_series = pd.Series(panel[cluster_col].to_numpy()[keep], name=cluster_col)

    n, k = x_tilde.shape
    if n <= k:
        raise RuntimeError("Not enough observations for FE regression.")

    xtx = x_tilde.T @ x_tilde
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (x_tilde.T @ y_tilde)

    resid = y_tilde - (x_tilde @ beta)

    meat = np.zeros((k, k), dtype=float)
    for i in range(n):
        xi = x_tilde[i : i + 1, :].T
        meat += (resid[i] ** 2) * (xi @ xi.T)

    scale = n / max(n - k, 1)
    vcov = scale * (xtx_inv @ meat @ xtx_inv)
    se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))

    t_stat = beta / np.where(se == 0, np.nan, se)
    p_vals = _normal_2sided_p(np.nan_to_num(t_stat, nan=0.0))
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se

    sse = float(np.sum(resid**2))
    sst = float(np.sum((y_tilde - np.mean(y_tilde)) ** 2))
    r2 = 1.0 - (sse / sst if sst > 0 else np.nan)

    class OLSResult:
        def __init__(self, beta, se, t_stat, p_value, ci_low, ci_high, r2, n_obs, n_regressors):
            self.beta = beta
            self.se = se
            self.t_stat = t_stat
            self.p_value = p_value
            self.ci_low = ci_low
            self.ci_high = ci_high
            self.r2 = r2
            self.n_obs = n_obs
            self.n_regressors = n_regressors

    res = OLSResult(beta=beta, se=se, t_stat=t_stat, p_value=p_vals, ci_low=ci_low, ci_high=ci_high, r2=r2, n_obs=n, n_regressors=k)

    cluster_summary = None
    if cluster_series is not None:
        cluster_fit = _cluster_robust_inference(x_tilde=x_tilde, resid=resid, clusters=cluster_series)
        if cluster_fit is not None:
            cluster_se = cluster_fit["se"]
            cluster_t = beta / np.where(cluster_se == 0, np.nan, cluster_se)
            cluster_p = _normal_2sided_p(np.nan_to_num(cluster_t, nan=0.0))
            cluster_summary = {
                "cluster_col": cluster_col,
                "n_groups": int(cluster_fit["n_groups"]),
                "finite_sample_correction": float(cluster_fit["finite_sample_correction"]),
                "se": cluster_se.tolist(),
                "t_stat": cluster_t.tolist(),
                "p_value": cluster_p.tolist(),
                "ci_low": (beta - 1.96 * cluster_se).tolist(),
                "ci_high": (beta + 1.96 * cluster_se).tolist(),
            }

    diagnostics = {
        "time_fixed_effects": bool(time_fe),
        "dropped_regressors": dropped_cols,
        "clustered_inference": cluster_summary,
    }
    return res, kept_cols, diagnostics


def run_rq1_log_volume():
    panel_path = RESULTS / "rq1" / "rq1_daily_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(panel_path)
    panel = pd.read_csv(panel_path, parse_dates=["date"]) 
    # compute log-normalised volume
    panel["article_volume_log"] = np.log1p(panel["article_volume"].astype(float))

    res, kept, diag = fit_within_ols(
        panel=panel,
        y_col="daily_stance",
        x_cols=["post", "article_volume_log"],
        entity_col="company_id",
        time_col="date",
        time_fe=True,
        cluster_col="date",
    )

    out_dir = RESULTS / "rq1" / "log_volume"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save coefficients
    coef_rows = []
    for i, name in enumerate(kept):
        coef_rows.append(
            {
                "variable": name,
                "coef": float(res.beta[i]),
                "se_hc1": float(res.se[i]),
                "t_stat": float(res.t_stat[i]),
                "p_value": float(res.p_value[i]),
                "ci95_low": float(res.ci_low[i]),
                "ci95_high": float(res.ci_high[i]),
            }
        )
    pd.DataFrame(coef_rows).to_csv(out_dir / "rq1_fe_coefficients_log_volume.csv", index=False)

    summary = {
        "n_obs": res.n_obs,
        "n_regressors": res.n_regressors,
        "r2_within": res.r2,
        "coefficients": coef_rows,
        "clustered_inference": diag.get("clustered_inference", None),
    }
    with open(out_dir / "rq1_summary_log_volume.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("RQ1 log-volume re-estimate complete. Wrote:", out_dir)
    return summary


def run_rq2_two_way_fe():
    panel_path = RESULTS / "rq2" / "rq2_daily_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(panel_path)
    panel = pd.read_csv(panel_path, parse_dates=["date"]) 

    # run with time fixed effects using local FE function
    res, kept, diag = fit_within_ols(
        panel=panel,
        y_col="daily_return",
        x_cols=["post", "sp500_return", "vix"],
        entity_col="company_id",
        time_col="date",
        time_fe=True,
        cluster_col="date",
    )

    out_dir = RESULTS / "rq2" / "two_way_fe"
    out_dir.mkdir(parents=True, exist_ok=True)

    coef_rows = []
    for i, name in enumerate(kept):
        coef_rows.append(
            {
                "variable": name,
                "coef": float(res.beta[i]),
                "se_hc1": float(res.se[i]),
                "t_stat": float(res.t_stat[i]),
                "p_value": float(res.p_value[i]),
                "ci95_low": float(res.ci_low[i]),
                "ci95_high": float(res.ci_high[i]),
            }
        )
    pd.DataFrame(coef_rows).to_csv(out_dir / "rq2_fe_coefficients_two_way.csv", index=False)

    summary = {
        "n_obs": res.n_obs,
        "n_regressors": res.n_regressors,
        "r2_within": res.r2,
        "coefficients": coef_rows,
        "clustered_inference": diag.get("clustered_inference", None),
    }
    with open(out_dir / "rq2_summary_two_way.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("RQ2 two-way FE re-estimate complete. Wrote:", out_dir)
    return summary


if __name__ == "__main__":
    print("Running RQ TODOs: RQ1 log-volume and RQ2 two-way FE")
    s1 = run_rq1_log_volume()
    s2 = run_rq2_two_way_fe()
    print("Both tasks complete.")
