#!/usr/bin/env python3
"""
RQ1 regression pipeline.

Research question:
How has media reporting (bias) toward S&P 500 firms changed pre vs post the 2020 recession?

Model:
    bias(i,t) = alpha(i) + delta_t + beta * Post(t) + gamma * article_volume(i,t) + epsilon(i,t)

Where bias(i,t) is firm-day stance score computed as:
    mean(prob_positive - prob_negative)
across high-confidence NewsMTSC outputs for firm i on day t.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv


# Default DB target requested by user
DEFAULT_SCHEMA = "public"
DEFAULT_TABLE = "articles_no_title_deduped"
# Analysis window
PRE_START = "2015-01-01"
PRE_END = "2019-12-31"
POST_START = "2020-01-01"
POST_END = "2025-12-31"


@dataclass
class OLSResult:
    beta: np.ndarray
    se: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    r2: float
    n_obs: int
    n_regressors: int


@dataclass
class MonthlyEventStudyResult:
    coefficients: pd.DataFrame
    n_obs: int
    n_regressors: int
    window_lower: int
    window_upper: int
    reference_month: int


def resolve_db_url() -> str:
    load_dotenv()
    db_url = os.getenv("POOLER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("Missing POOLER_DATABASE_URL/DATABASE_URL in environment.")
    return db_url


def validate_identifier(name: str, label: str) -> str:
    if not name or not name.replace("_", "").isalnum():
        raise ValueError(f"Invalid {label}: {name}")
    return name


def fetch_article_scores(
    db_url: str,
    schema: str,
    table: str,
    confidence_threshold: float,
) -> pd.DataFrame:
    schema = validate_identifier(schema, "schema")
    table = validate_identifier(table, "table")

    query = f"""
        SELECT
            company_id,
            published_at::date AS date,
            pos_score,
            neg_score,
            neutral_score
        FROM {schema}.{table}
        WHERE published_at::date BETWEEN %s AND %s
          AND company_id IS NOT NULL
          AND pos_score IS NOT NULL
          AND neg_score IS NOT NULL
          AND neutral_score IS NOT NULL
          AND GREATEST(pos_score, neg_score, neutral_score) >= %s
    """

    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (PRE_START, POST_END, confidence_threshold))
            rows = cur.fetchall()
            col_names = [desc[0] for desc in cur.description]

    df = pd.DataFrame(rows, columns=col_names)

    if df.empty:
        raise RuntimeError("No rows returned from DB after applying filters.")

    df["date"] = pd.to_datetime(df["date"])
    df["company_id"] = df["company_id"].astype(str)
    return df


def fetch_article_filter_diagnostics(
    db_url: str,
    schema: str,
    table: str,
    confidence_threshold: float,
) -> dict[str, Any]:
    schema = validate_identifier(schema, "schema")
    table = validate_identifier(table, "table")

    query = f"""
        SELECT
            COUNT(*) AS total_rows_date_window,
            COUNT(*) FILTER (WHERE company_id IS NOT NULL) AS rows_with_company,
            COUNT(*) FILTER (
                WHERE pos_score IS NOT NULL
                  AND neg_score IS NOT NULL
                  AND neutral_score IS NOT NULL
            ) AS rows_with_all_scores,
            COUNT(*) FILTER (
                WHERE company_id IS NOT NULL
                  AND pos_score IS NOT NULL
                  AND neg_score IS NOT NULL
                  AND neutral_score IS NOT NULL
            ) AS rows_company_and_scores,
            COUNT(*) FILTER (
                WHERE company_id IS NOT NULL
                  AND pos_score IS NOT NULL
                  AND neg_score IS NOT NULL
                  AND neutral_score IS NOT NULL
                  AND GREATEST(pos_score, neg_score, neutral_score) >= %s
            ) AS rows_after_threshold,
            COUNT(DISTINCT company_id) FILTER (
                WHERE company_id IS NOT NULL
                  AND pos_score IS NOT NULL
                  AND neg_score IS NOT NULL
                  AND neutral_score IS NOT NULL
                  AND GREATEST(pos_score, neg_score, neutral_score) >= %s
            ) AS firms_after_threshold
        FROM {schema}.{table}
        WHERE published_at::date BETWEEN %s AND %s
    """

    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (confidence_threshold, confidence_threshold, PRE_START, POST_END))
            row = cur.fetchone()

    return {
        "total_rows_date_window": int(row[0]),
        "rows_with_company": int(row[1]),
        "rows_with_all_scores": int(row[2]),
        "rows_company_and_scores": int(row[3]),
        "rows_after_threshold": int(row[4]),
        "firms_after_threshold": int(row[5]),
    }


def build_daily_panel(raw_df: pd.DataFrame) -> pd.DataFrame:
    working = raw_df.copy()
    working["stance_component"] = working["pos_score"] - working["neg_score"]

    daily = (
        working.groupby(["company_id", "date"], as_index=False)
        .agg(
            daily_stance=("stance_component", "mean"),
            article_volume=("stance_component", "size"),
        )
        .sort_values(["company_id", "date"])
    )

    daily["post"] = (daily["date"] >= pd.Timestamp(POST_START)).astype(int)
    return daily


def prepare_regression_data(daily_df: pd.DataFrame) -> pd.DataFrame:
    panel = daily_df.copy()
    panel = panel.dropna(subset=["daily_stance", "article_volume", "post"])
    panel = panel[(panel["date"] >= PRE_START) & (panel["date"] <= POST_END)]

    pre = panel[(panel["date"] >= PRE_START) & (panel["date"] <= PRE_END)]
    post = panel[(panel["date"] >= POST_START) & (panel["date"] <= POST_END)]
    if pre.empty or post.empty:
        raise RuntimeError("Insufficient pre/post observations after merge and filtering.")

    return panel


def _normal_2sided_p(t_stat: np.ndarray) -> np.ndarray:
    abs_t = np.abs(t_stat)
    return np.array([math.erfc(float(v) / math.sqrt(2.0)) for v in abs_t])


def _cluster_robust_inference(
    x_tilde: np.ndarray,
    resid: np.ndarray,
    clusters: pd.Series,
) -> dict[str, Any] | None:
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


def _format_month_label(month_index: int) -> str:
    if month_index == 0:
        return "m0"
    sign = "p" if month_index > 0 else "m"
    return f"{sign}{abs(month_index)}"


def _build_monthly_event_study_panel(
    panel: pd.DataFrame,
    lower_month: int,
    upper_month: int,
    reference_month: int,
) -> tuple[pd.DataFrame, list[str]]:
    event_panel = panel.copy()
    event_panel["event_month"] = (
        (event_panel["date"].dt.year - 2020) * 12 + (event_panel["date"].dt.month - 1)
    )
    event_panel = event_panel[
        (event_panel["event_month"] >= lower_month) & (event_panel["event_month"] <= upper_month)
    ].copy()

    dummy_cols: list[str] = []
    for month_value in range(lower_month, upper_month + 1):
        if month_value == reference_month:
            continue
        col_name = f"event_{_format_month_label(month_value)}"
        event_panel[col_name] = (event_panel["event_month"] == month_value).astype(int)
        dummy_cols.append(col_name)

    return event_panel, dummy_cols


def fit_within_ols(
    panel: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    entity_col: str,
    time_col: str,
    time_fe: bool,
    cluster_col: str | None = None,
) -> tuple[OLSResult, list[str], dict[str, Any]]:
    y_tilde, x_tilde, kept_cols, dropped_cols = _within_transform(
        panel=panel,
        y_col=y_col,
        x_cols=x_cols,
        entity_col=entity_col,
        time_col=time_col,
        time_fe=time_fe,
    )

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

    res = OLSResult(
        beta=beta,
        se=se,
        t_stat=t_stat,
        p_value=p_vals,
        ci_low=ci_low,
        ci_high=ci_high,
        r2=r2,
        n_obs=n,
        n_regressors=k,
    )

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


def _within_transform(
    panel: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    entity_col: str,
    time_col: str,
    time_fe: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    y = panel[y_col].astype(float)
    x = panel[x_cols].astype(float)

    entity_mean_y = y.groupby(panel[entity_col]).transform("mean")
    y_tilde = y - entity_mean_y

    x_tilde = np.column_stack(
        [
            (x[c] - x[c].groupby(panel[entity_col]).transform("mean")).to_numpy()
            for c in x_cols
        ]
    )

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


def test_time_fixed_effects(
    panel: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    entity_col: str,
    time_col: str,
) -> dict[str, Any]:
    y_tilde, x_tilde, kept_cols, dropped_cols = _within_transform(
        panel=panel,
        y_col=y_col,
        x_cols=x_cols,
        entity_col=entity_col,
        time_col=time_col,
        time_fe=False,
    )

    keep = np.isfinite(y_tilde) & np.all(np.isfinite(x_tilde), axis=1)
    y_tilde = y_tilde[keep]
    x_tilde = x_tilde[keep]

    n, k = x_tilde.shape
    if n <= k:
        return {
            "f_stat": None,
            "df1": None,
            "df2": None,
            "p_value": None,
            "dropped_regressors": dropped_cols,
            "note": "Not enough observations for time FE test.",
        }

    xtx = x_tilde.T @ x_tilde
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (x_tilde.T @ y_tilde)
    resid = y_tilde - (x_tilde @ beta)

    time_series = panel[time_col].to_numpy()[keep]
    resid_series = pd.Series(resid)
    time_mean = resid_series.groupby(pd.Series(time_series)).transform("mean").to_numpy()

    ssr_restricted = float(np.sum(resid**2))
    ssr_unrestricted = float(np.sum((resid - time_mean) ** 2))

    n_times = int(pd.Series(time_series).nunique())
    df1 = n_times - 1
    df2 = n - k - df1
    if df1 <= 0 or df2 <= 0 or ssr_unrestricted <= 0:
        return {
            "f_stat": None,
            "df1": df1,
            "df2": df2,
            "p_value": None,
            "dropped_regressors": dropped_cols,
            "note": "Insufficient degrees of freedom for time FE test.",
        }

    f_stat = ((ssr_restricted - ssr_unrestricted) / df1) / (ssr_unrestricted / df2)
    if f_stat < 0:
        f_stat = 0.0

    p_value = None
    try:
        from scipy import stats  # type: ignore

        p_value = float(stats.f.sf(f_stat, df1, df2))
        # critical value at alpha=0.05 (upper tail)
        f_critical = float(stats.f.ppf(1.0 - 0.05, df1, df2))
    except Exception:
        p_value = None
        # F critical fallback using chi-square approximation for large df:
        # F_{0.95}(d1,d2) ~ chi2_{0.95,d1} / d1, and
        # chi2_{0.95,d1} ~ d1 + z_{0.95}*sqrt(2*d1)
        try:
            z_95 = 1.6448536269514722
            chi2_95 = float(df1) + z_95 * ((2.0 * float(df1)) ** 0.5)
            f_critical = float(chi2_95 / float(df1))
        except Exception:
            f_critical = None

    return {
        "f_stat": float(f_stat),
        "df1": int(df1),
        "df2": int(df2),
        "p_value": p_value,
        "f_critical_0.05": f_critical,
        "dropped_regressors": dropped_cols,
    }


def run_entity_fe_ols(
    panel: pd.DataFrame,
    time_fe: bool,
) -> tuple[OLSResult, list[str], dict[str, Any]]:
    return fit_within_ols(
        panel=panel,
        y_col="daily_stance",
        x_cols=["post", "article_volume"],
        entity_col="company_id",
        time_col="date",
        time_fe=time_fe,
        cluster_col="date",
    )


def run_monthly_event_study(
    panel: pd.DataFrame,
    lower_month: int = -24,
    upper_month: int = 24,
    reference_month: int = -1,
) -> MonthlyEventStudyResult:
    event_panel, dummy_cols = _build_monthly_event_study_panel(
        panel=panel,
        lower_month=lower_month,
        upper_month=upper_month,
        reference_month=reference_month,
    )

    if event_panel.empty:
        raise RuntimeError("No observations available for monthly event study.")

    res, kept_cols, diagnostics = fit_within_ols(
        panel=event_panel,
        y_col="daily_stance",
        x_cols=dummy_cols + ["article_volume"],
        entity_col="company_id",
        time_col="date",
        time_fe=False,
        cluster_col="date",
    )

    cluster_summary = diagnostics.get("clustered_inference")
    rows = []
    for i, name in enumerate(kept_cols):
        row: dict[str, Any] = {
            "variable": name,
            "coef": float(res.beta[i]),
            "se_hc1": float(res.se[i]),
            "t_stat": float(res.t_stat[i]),
            "p_value_hc1": float(res.p_value[i]),
            "ci95_low_hc1": float(res.ci_low[i]),
            "ci95_high_hc1": float(res.ci_high[i]),
            "month_label": name.replace("event_", "") if name.startswith("event_") else name,
        }
        if cluster_summary is not None:
            row.update(
                {
                    "se_day_cluster": float(cluster_summary["se"][i]),
                    "t_stat_day_cluster": float(cluster_summary["t_stat"][i]),
                    "p_value_day_cluster": float(cluster_summary["p_value"][i]),
                    "ci95_low_day_cluster": float(cluster_summary["ci_low"][i]),
                    "ci95_high_day_cluster": float(cluster_summary["ci_high"][i]),
                }
            )
        rows.append(row)

    return MonthlyEventStudyResult(
        coefficients=pd.DataFrame(rows),
        n_obs=res.n_obs,
        n_regressors=res.n_regressors,
        window_lower=lower_month,
        window_upper=upper_month,
        reference_month=reference_month,
    )


def pre_period_mean_diagnostic(
    panel: pd.DataFrame,
    tolerance: float,
) -> dict[str, Any]:
    pre = panel[(panel["date"] >= PRE_START) & (panel["date"] <= PRE_END)]
    pre_means = pre.groupby("company_id")["daily_stance"].mean()

    if pre_means.empty:
        return {
            "n_firms": 0,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "std": np.nan,
            "means_equal": False,
        }

    spread = float(pre_means.max() - pre_means.min())
    std = float(pre_means.std(ddof=1)) if len(pre_means) > 1 else 0.0
    means_equal = spread <= tolerance

    return {
        "n_firms": int(pre_means.shape[0]),
        "min": float(pre_means.min()),
        "max": float(pre_means.max()),
        "range": spread,
        "std": std,
        "means_equal": bool(means_equal),
    }


def write_outputs(
    output_dir: Path,
    panel: pd.DataFrame,
    raw_articles: pd.DataFrame,
    fe_result: OLSResult,
    regressor_names: list[str],
    diagnostics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    panel.to_csv(output_dir / "rq1_daily_panel.csv", index=False)

    coef_rows = []
    for i, name in enumerate(regressor_names):
        coef_rows.append(
            {
                "variable": name,
                "coef": float(fe_result.beta[i]),
                "se_hc1": float(fe_result.se[i]),
                "t_stat": float(fe_result.t_stat[i]),
                "p_value_norm_approx": float(fe_result.p_value[i]),
                "ci95_low": float(fe_result.ci_low[i]),
                "ci95_high": float(fe_result.ci_high[i]),
            }
        )

    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(output_dir / "rq1_fe_coefficients.csv", index=False)

    formula = "daily_stance_it = alpha_i + beta*post_t + gamma1*article_volume_it + eps_it"
    if metadata.get("time_fixed_effects"):
        formula = "daily_stance_it = alpha_i + delta_t + beta*post_t + gamma1*article_volume_it + eps_it"

    post_idx = regressor_names.index("post") if "post" in regressor_names else None
    main_beta_post = float(fe_result.beta[post_idx]) if post_idx is not None else None
    main_beta_post_se = float(fe_result.se[post_idx]) if post_idx is not None else None
    main_beta_post_p = float(fe_result.p_value[post_idx]) if post_idx is not None else None

    payload: dict[str, Any] = {
        "metadata": metadata,
        "model": {
            "formula": formula,
            "estimator": "Entity fixed-effects OLS (within), HC1 robust SE",
            "n_obs": fe_result.n_obs,
            "n_regressors": fe_result.n_regressors,
            "r2_within": fe_result.r2,
            "main_beta_post": main_beta_post,
            "main_beta_post_se": main_beta_post_se,
            "main_beta_post_p": main_beta_post_p,
        },
        "diagnostics": {
            "pre_period_means": diagnostics,
            "time_fe_test": metadata.get("time_fe_test"),
            "clustered_inference": metadata.get("clustered_inference"),
        },
        "coefficients": coef_rows,
    }

    with open(output_dir / "rq1_summary.json", "w", encoding="ascii") as f:
        json.dump(payload, f, indent=2)

    # Build expanded report stats.
    panel_start = pd.Timestamp(panel["date"].min())
    panel_end = pd.Timestamp(panel["date"].max())
    panel_days_present = int(panel["date"].nunique())
    panel_calendar_days = int((panel_end - panel_start).days + 1)
    panel_firms = int(panel["company_id"].nunique())
    firm_day_possible = int(panel_firms * panel_days_present)
    firm_day_observed = int(len(panel))
    firm_day_density = (
        float(firm_day_observed / firm_day_possible) if firm_day_possible > 0 else float("nan")
    )

    total_articles = int(len(raw_articles))
    panel_article_assignments = int(panel["article_volume"].sum())
    avg_articles_per_firm_day = (
        float(panel["article_volume"].mean()) if firm_day_observed > 0 else float("nan")
    )
    median_articles_per_firm_day = (
        float(panel["article_volume"].median()) if firm_day_observed > 0 else float("nan")
    )

    pre_mask = (panel["date"] >= pd.Timestamp(PRE_START)) & (panel["date"] <= pd.Timestamp(PRE_END))
    post_mask = (panel["date"] >= pd.Timestamp(POST_START)) & (panel["date"] <= pd.Timestamp(POST_END))
    pre_panel = panel.loc[pre_mask]
    post_panel = panel.loc[post_mask]

    def _safe_mean(series: pd.Series) -> float:
        return float(series.mean()) if len(series) > 0 else float("nan")

    md_lines = [
        "# RQ1 Regression Pipeline Summary",
        "",
        "## Context",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Source table | {metadata.get('schema')}.{metadata.get('table')} |",
        f"| Confidence threshold | {metadata.get('confidence_threshold')} |",
        f"| Pre period | {PRE_START} to {PRE_END} |",
        f"| Post period | {POST_START} to {POST_END} |",
        f"| Time fixed effects | {metadata.get('time_fixed_effects')} |",
        f"| Dropped regressors | {', '.join(metadata.get('dropped_regressors', [])) or 'None'} |",
        "",
        "## Data Flow Counts",
        "",
        "| Stage | Count |",
        "|---|---:|",
        f"| Raw article rows after SQL filters | {total_articles:,} |",
        f"| Raw firms after SQL filters | {raw_articles['company_id'].nunique():,} |",
        f"| Total article assignments in panel | {panel_article_assignments:,} |",
        f"| Firm-day rows in panel | {firm_day_observed:,} |",
        f"| Firms in panel | {panel_firms:,} |",
        "",
        "## Time Coverage",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Panel start date | {panel_start.date()} |",
        f"| Panel end date | {panel_end.date()} |",
        f"| Distinct days with at least one firm observed | {panel_days_present:,} |",
        f"| Calendar days from min to max | {panel_calendar_days:,} |",
        f"| Observed firm-day rows | {firm_day_observed:,} |",
        f"| Possible firm-day rows (firms x observed days) | {firm_day_possible:,} |",
        f"| Firm-day density | {firm_day_density:.4f} |",
        "",
        "## Pre vs Post Panel Split",
        "",
        "| Split | Firm-day rows | Distinct firms | Distinct days | Mean daily stance | Mean article volume |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Pre ({PRE_START} to {PRE_END}) | {len(pre_panel):,} | {pre_panel['company_id'].nunique():,} | {pre_panel['date'].nunique():,} | {_safe_mean(pre_panel['daily_stance']):.6f} | {_safe_mean(pre_panel['article_volume']):.3f} |",
        f"| Post ({POST_START} to {POST_END}) | {len(post_panel):,} | {post_panel['company_id'].nunique():,} | {post_panel['date'].nunique():,} | {_safe_mean(post_panel['daily_stance']):.6f} | {_safe_mean(post_panel['article_volume']):.3f} |",
        "",
        "## Volume Distribution",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Avg articles per firm-day | {avg_articles_per_firm_day:.3f} |",
        f"| Median articles per firm-day | {median_articles_per_firm_day:.3f} |",
        f"| Min articles per firm-day | {int(panel['article_volume'].min()):,} |",
        f"| Max articles per firm-day | {int(panel['article_volume'].max()):,} |",
        "",
        "## Main FE Result",
        "",
        "| Variable | Coef | SE (HC1) | t-stat | p-value (normal approx) | 95% CI low | 95% CI high |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for i, name in enumerate(regressor_names):
        md_lines.append(
            f"| {name} | {fe_result.beta[i]:.6f} | {fe_result.se[i]:.6f} | {fe_result.t_stat[i]:.6f} | {fe_result.p_value[i]:.6g} | {fe_result.ci_low[i]:.6f} | {fe_result.ci_high[i]:.6f} |"
        )

    cluster_info = metadata.get("clustered_inference")
    if cluster_info:
        md_lines.extend(
            [
                "",
                "## Day-Clustered Inference",
                "",
                f"Clustered on: {cluster_info.get('cluster_col')} (groups={cluster_info.get('n_groups')})",
                f"Finite-sample correction: {cluster_info.get('finite_sample_correction')}",
                "",
                "| Variable | Coef | SE (day-clustered) | t-stat | p-value | 95% CI low | 95% CI high |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for i, name in enumerate(regressor_names):
            md_lines.append(
                f"| {name} | {fe_result.beta[i]:.6f} | {cluster_info['se'][i]:.6f} | {cluster_info['t_stat'][i]:.6f} | {cluster_info['p_value'][i]:.6g} | {cluster_info['ci_low'][i]:.6f} | {cluster_info['ci_high'][i]:.6f} |"
            )

    md_lines.extend(
        [
            "",
            "## Model Fit",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Estimator | Entity fixed-effects OLS (within), HC1 robust SE |",
            f"| Observations | {fe_result.n_obs:,} |",
            f"| Regressors | {fe_result.n_regressors:,} |",
            f"| Within R2 | {fe_result.r2:.6f} |",
            "",
            "## Pre-Period Mean Diagnostic",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Firms | {diagnostics.get('n_firms')} |",
            f"| Min firm pre mean | {diagnostics.get('min')} |",
            f"| Max firm pre mean | {diagnostics.get('max')} |",
            f"| Range | {diagnostics.get('range')} |",
            f"| Std | {diagnostics.get('std')} |",
            f"| Means equal (tolerance rule) | {diagnostics.get('means_equal')} |",
        ]
    )

    time_fe_test = metadata.get("time_fe_test")
    if time_fe_test:
        md_lines.extend(
            [
                "",
                "## Time FE Test",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| F stat | {time_fe_test.get('f_stat')} |",
                f"| df1 | {time_fe_test.get('df1')} |",
                f"| df2 | {time_fe_test.get('df2')} |",
                f"| p value | {time_fe_test.get('p_value')} |",
                f"| F critical (alpha=0.05) | {time_fe_test.get('f_critical_0.05')} |",
            ]
        )

    with open(output_dir / "rq1_summary.md", "w", encoding="ascii") as f:
        f.write("\n".join(md_lines) + "\n")


def write_event_study_outputs(output_dir: Path, result: MonthlyEventStudyResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    result.coefficients.to_csv(output_dir / "rq1_event_study_monthly.csv", index=False)
    with open(output_dir / "rq1_event_study_monthly.json", "w", encoding="ascii") as f:
        json.dump(
            {
                "n_obs": result.n_obs,
                "n_regressors": result.n_regressors,
                "window_lower": result.window_lower,
                "window_upper": result.window_upper,
                "reference_month": result.reference_month,
                "n_coefficients": int(len(result.coefficients)),
            },
            f,
            indent=2,
        )

    md_lines = [
        "# RQ1 Monthly Event Study",
        "",
        "Firm fixed effects with monthly event-time dummies relative to January 2020. Month -1 (December 2019) is the omitted reference period.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Observations | {result.n_obs:,} |",
        f"| Regressors | {result.n_regressors:,} |",
        f"| Window lower month | {result.window_lower} |",
        f"| Window upper month | {result.window_upper} |",
        f"| Reference month | {result.reference_month} |",
        "",
        "| Variable | Coef | SE (HC1) | t-stat | p-value | 95% CI low | 95% CI high |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in result.coefficients.itertuples(index=False):
        md_lines.append(
            f"| {row.variable} | {row.coef:.6f} | {row.se_hc1:.6f} | {row.t_stat:.6f} | {row.p_value_hc1:.6g} | {row.ci95_low_hc1:.6f} | {row.ci95_high_hc1:.6f} |"
        )

    if "se_day_cluster" in result.coefficients.columns:
        md_lines.extend(
            [
                "",
                "## Day-Clustered Inference",
                "",
                "| Variable | SE (day-clustered) | t-stat | p-value | 95% CI low | 95% CI high |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in result.coefficients.itertuples(index=False):
            md_lines.append(
                f"| {row.variable} | {row.se_day_cluster:.6f} | {row.t_stat_day_cluster:.6f} | {row.p_value_day_cluster:.6g} | {row.ci95_low_day_cluster:.6f} | {row.ci95_high_day_cluster:.6f} |"
            )

    with open(output_dir / "rq1_event_study_monthly.md", "w", encoding="ascii") as f:
        f.write("\n".join(md_lines) + "\n")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQ1 FE regression pipeline for media bias shift around 2020 recession.")

    parser.add_argument("--schema", type=str, default=DEFAULT_SCHEMA, help="DB schema name")
    parser.add_argument("--table", type=str, default=DEFAULT_TABLE, help="DB table name")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="High-confidence threshold based on max(pos,neg,neutral)",
    )
    parser.add_argument(
        "--pre-mean-tolerance",
        type=float,
        default=0.02,
        help="Tolerance used to flag unequal pre-period firm means in diagnostics",
    )
    parser.add_argument(
        "--time-fe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include time fixed effects (delta_t) in the regression.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/rq1",
        help="Directory to save outputs",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_url = resolve_db_url()

    print("Computing SQL filter diagnostics...")
    filter_diag = fetch_article_filter_diagnostics(
        db_url=db_url,
        schema=args.schema,
        table=args.table,
        confidence_threshold=args.confidence_threshold,
    )

    print("Fetching article scores from database...")
    raw = fetch_article_scores(
        db_url=db_url,
        schema=args.schema,
        table=args.table,
        confidence_threshold=args.confidence_threshold,
    )

    print("Building firm-day stance panel...")
    daily = build_daily_panel(raw)

    print("Preparing regression dataset...")
    panel = prepare_regression_data(daily)

    print("Running entity FE regression...")
    fe_result, regressor_names, fe_diag = run_entity_fe_ols(panel, time_fe=args.time_fe)

    print("Running monthly event study...")
    event_study = run_monthly_event_study(panel)

    time_fe_test = None
    if args.time_fe:
        print("Testing time fixed effects...")
        time_fe_test = test_time_fixed_effects(
            panel=panel,
            y_col="daily_stance",
            x_cols=["post", "article_volume"],
            entity_col="company_id",
            time_col="date",
        )

    print("Running pre-period diagnostic...")
    diag = pre_period_mean_diagnostic(panel, tolerance=args.pre_mean_tolerance)

    metadata = {
        "schema": args.schema,
        "table": args.table,
        "confidence_threshold": args.confidence_threshold,
        "pre_period": [PRE_START, PRE_END],
        "post_period": [POST_START, POST_END],
        "sql_filter_diagnostics": filter_diag,
        "time_fixed_effects": bool(args.time_fe),
        "time_fe_test": time_fe_test,
        "dropped_regressors": fe_diag.get("dropped_regressors", []),
        "clustered_inference": fe_diag.get("clustered_inference"),
    }

    print("Writing outputs...")
    write_outputs(
        output_dir=Path(args.output_dir),
        panel=panel,
        raw_articles=raw,
        fe_result=fe_result,
        regressor_names=regressor_names,
        diagnostics=diag,
        metadata=metadata,
    )

    write_event_study_outputs(Path(args.output_dir), event_study)

    print("Done.")
    if "post" in regressor_names:
        post_idx = regressor_names.index("post")
        print(f"Main coefficient beta(Post): {fe_result.beta[post_idx]:.6f}")
    else:
        print("Post regressor dropped (collinear with time fixed effects).")


if __name__ == "__main__":
    main()
