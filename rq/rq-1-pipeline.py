#!/usr/bin/env python3
"""
RQ1 regression pipeline.

Research question:
How has media reporting (bias) toward S&P 500 firms changed pre vs post the 2020 recession?

Model:
    bias(i,t) = alpha(i) + beta * Post(t) + gamma * X(i,t) + epsilon(i,t)

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
DEFAULT_VIX_SCHEMA = "finance"
DEFAULT_VIX_TABLE = "vix_daily"

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


def load_vix(
    db_url: str,
    start_date: str,
    end_date: str,
    vix_csv: str | None,
    vix_schema: str,
    vix_table: str,
) -> pd.DataFrame:
    if vix_csv:
        vix_df = pd.read_csv(vix_csv)
        cols_lower = {c.lower(): c for c in vix_df.columns}
        date_col = cols_lower.get("date")
        close_col = cols_lower.get("close")

        if not date_col or not close_col:
            raise ValueError(
                "VIX CSV must have Date and Close columns (case-insensitive)."
            )

        out = vix_df[[date_col, close_col]].copy()
        out.columns = ["date", "vix"]
    else:
        vix_schema = validate_identifier(vix_schema, "vix_schema")
        vix_table = validate_identifier(vix_table, "vix_table")

        query = f"""
            SELECT
                trade_date::date AS date,
                close AS vix
            FROM {vix_schema}.{vix_table}
            WHERE trade_date::date BETWEEN %s AND %s
              AND close IS NOT NULL
            ORDER BY trade_date
        """

        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (start_date, end_date))
                rows = cur.fetchall()

        out = pd.DataFrame(rows, columns=["date", "vix"])

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["vix"] = pd.to_numeric(out["vix"], errors="coerce")
    out = out.dropna(subset=["date", "vix"])

    mask = (out["date"] >= pd.Timestamp(start_date)) & (out["date"] <= pd.Timestamp(end_date))
    out = out.loc[mask].sort_values("date")

    if out.empty:
        raise RuntimeError("No VIX values available for the requested date range.")

    return out


def prepare_regression_data(daily_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    panel = daily_df.merge(vix_df, on="date", how="left")
    panel["vix"] = panel["vix"].ffill().bfill()

    panel = panel.dropna(subset=["daily_stance", "article_volume", "post", "vix"])
    panel = panel[(panel["date"] >= PRE_START) & (panel["date"] <= POST_END)]

    pre = panel[(panel["date"] >= PRE_START) & (panel["date"] <= PRE_END)]
    post = panel[(panel["date"] >= POST_START) & (panel["date"] <= POST_END)]
    if pre.empty or post.empty:
        raise RuntimeError("Insufficient pre/post observations after merge and filtering.")

    return panel


def _normal_2sided_p(t_stat: np.ndarray) -> np.ndarray:
    abs_t = np.abs(t_stat)
    return np.array([math.erfc(float(v) / math.sqrt(2.0)) for v in abs_t])


def run_entity_fe_ols(panel: pd.DataFrame) -> tuple[OLSResult, list[str]]:
    # Within transformation to absorb company fixed effects alpha(i)
    cols = ["post", "article_volume", "vix"]
    y = panel["daily_stance"].astype(float)
    x = panel[cols].astype(float)

    group_mean_y = y.groupby(panel["company_id"]).transform("mean")
    y_tilde = (y - group_mean_y).to_numpy()

    x_tilde = np.column_stack(
        [
            (x[c] - x[c].groupby(panel["company_id"]).transform("mean")).to_numpy()
            for c in cols
        ]
    )

    # Drop numerically empty rows
    keep = np.isfinite(y_tilde) & np.all(np.isfinite(x_tilde), axis=1)
    y_tilde = y_tilde[keep]
    x_tilde = x_tilde[keep]

    n, k = x_tilde.shape
    if n <= k:
        raise RuntimeError("Not enough observations for FE regression.")

    xtx = x_tilde.T @ x_tilde
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (x_tilde.T @ y_tilde)

    resid = y_tilde - (x_tilde @ beta)

    # White HC1 robust variance
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
    return res, cols


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

    payload: dict[str, Any] = {
        "metadata": metadata,
        "model": {
            "formula": "daily_stance_it = alpha_i + beta*post_t + gamma1*article_volume_it + gamma2*vix_t + eps_it",
            "estimator": "Entity fixed-effects OLS (within), HC1 robust SE",
            "n_obs": fe_result.n_obs,
            "n_regressors": fe_result.n_regressors,
            "r2_within": fe_result.r2,
            "main_beta_post": float(fe_result.beta[0]),
            "main_beta_post_se": float(fe_result.se[0]),
            "main_beta_post_p": float(fe_result.p_value[0]),
        },
        "diagnostics": {
            "pre_period_means": diagnostics,
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
        f"| VIX source | {metadata.get('vix_source')} |",
        f"| Pre period | {PRE_START} to {PRE_END} |",
        f"| Post period | {POST_START} to {POST_END} |",
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

    with open(output_dir / "rq1_summary.md", "w", encoding="ascii") as f:
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
        "--vix-csv",
        type=str,
        default=None,
        help="Optional local CSV path for VIX (Date and Close columns). If omitted, VIX is loaded from DB.",
    )
    parser.add_argument(
        "--vix-schema",
        type=str,
        default=DEFAULT_VIX_SCHEMA,
        help="DB schema for VIX table (used when --vix-csv is not provided)",
    )
    parser.add_argument(
        "--vix-table",
        type=str,
        default=DEFAULT_VIX_TABLE,
        help="DB table for VIX data (used when --vix-csv is not provided)",
    )
    parser.add_argument(
        "--pre-mean-tolerance",
        type=float,
        default=0.02,
        help="Tolerance used to flag unequal pre-period firm means in diagnostics",
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

    print("Loading VIX series...")
    vix = load_vix(
        db_url=db_url,
        start_date=PRE_START,
        end_date=POST_END,
        vix_csv=args.vix_csv,
        vix_schema=args.vix_schema,
        vix_table=args.vix_table,
    )

    print("Preparing regression dataset...")
    panel = prepare_regression_data(daily, vix)

    print("Running entity FE regression...")
    fe_result, regressor_names = run_entity_fe_ols(panel)

    print("Running pre-period diagnostic...")
    diag = pre_period_mean_diagnostic(panel, tolerance=args.pre_mean_tolerance)

    metadata = {
        "schema": args.schema,
        "table": args.table,
        "confidence_threshold": args.confidence_threshold,
        "vix_source": f"{args.vix_schema}.{args.vix_table}" if not args.vix_csv else args.vix_csv,
        "pre_period": [PRE_START, PRE_END],
        "post_period": [POST_START, POST_END],
        "sql_filter_diagnostics": filter_diag,
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

    print("Done.")
    print(f"Main coefficient beta(Post): {fe_result.beta[0]:.6f}")


if __name__ == "__main__":
    main()
