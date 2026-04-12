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
            "trigger_scdid": True,
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
        "trigger_scdid": bool(not means_equal),
    }


def _synthetic_weights(x_pre: np.ndarray, y_pre: np.ndarray) -> np.ndarray:
    # Unconstrained least squares, then project to simplex-like nonnegative weights.
    w, *_ = np.linalg.lstsq(x_pre, y_pre, rcond=None)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0:
        return np.ones(x_pre.shape[1]) / x_pre.shape[1]
    return w / s


def run_scdid_fallback(
    panel: pd.DataFrame,
    min_pre_days: int = 180,
    min_post_days: int = 180,
) -> dict[str, Any]:
    pivot = panel.pivot_table(index="date", columns="company_id", values="daily_stance", aggfunc="mean")
    pre_dates = pivot.index[(pivot.index >= pd.Timestamp(PRE_START)) & (pivot.index <= pd.Timestamp(PRE_END))]
    post_dates = pivot.index[(pivot.index >= pd.Timestamp(POST_START)) & (pivot.index <= pd.Timestamp(POST_END))]

    effects: list[float] = []
    used_firms = 0

    for firm in pivot.columns:
        y_pre_full = pivot.loc[pre_dates, firm]
        y_post_full = pivot.loc[post_dates, firm]

        if y_pre_full.notna().sum() < min_pre_days or y_post_full.notna().sum() < min_post_days:
            continue

        donors = [c for c in pivot.columns if c != firm]
        x_pre = pivot.loc[pre_dates, donors]
        x_post = pivot.loc[post_dates, donors]

        # Keep pre rows where treated is observed and enough donor signal exists.
        valid_pre = y_pre_full.notna() & (x_pre.notna().sum(axis=1) >= 5)
        if valid_pre.sum() < min_pre_days:
            continue

        x_pre_use = x_pre.loc[valid_pre].copy()
        y_pre_use = y_pre_full.loc[valid_pre].to_numpy(dtype=float)

        donor_means = x_pre_use.mean(axis=0)
        x_pre_use = x_pre_use.fillna(donor_means)
        x_pre_np = x_pre_use.to_numpy(dtype=float)

        if x_pre_np.shape[1] == 0:
            continue

        w = _synthetic_weights(x_pre_np, y_pre_use)

        x_post_use = x_post.fillna(donor_means)
        y_post_use = y_post_full
        valid_post = y_post_use.notna()
        if valid_post.sum() < min_post_days:
            continue

        y_post_np = y_post_use.loc[valid_post].to_numpy(dtype=float)
        x_post_np = x_post_use.loc[valid_post].to_numpy(dtype=float)

        if x_post_np.shape[0] == 0:
            continue

        synth_post = x_post_np @ w
        firm_effect = float(np.mean(y_post_np - synth_post))
        effects.append(firm_effect)
        used_firms += 1

    if not effects:
        return {
            "available": False,
            "message": "Insufficient data to compute SCDiD fallback.",
        }

    effects_arr = np.array(effects, dtype=float)
    avg_effect = float(np.mean(effects_arr))
    se = float(np.std(effects_arr, ddof=1) / math.sqrt(len(effects_arr))) if len(effects_arr) > 1 else np.nan

    return {
        "available": True,
        "n_firms_used": int(used_firms),
        "avg_post_effect": avg_effect,
        "se_across_firms": se,
        "ci95_low": float(avg_effect - 1.96 * se) if np.isfinite(se) else np.nan,
        "ci95_high": float(avg_effect + 1.96 * se) if np.isfinite(se) else np.nan,
        "notes": "Firm-by-firm synthetic control fallback using donor firms to build post-period counterfactuals.",
    }


def write_outputs(
    output_dir: Path,
    panel: pd.DataFrame,
    fe_result: OLSResult,
    regressor_names: list[str],
    diagnostics: dict[str, Any],
    scdid_result: dict[str, Any] | None,
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
        "fallback_scdid": scdid_result,
        "coefficients": coef_rows,
    }

    with open(output_dir / "rq1_summary.json", "w", encoding="ascii") as f:
        json.dump(payload, f, indent=2)

    summary_lines = [
        "RQ1 Regression Pipeline Summary",
        "=" * 40,
        f"Rows in panel: {len(panel):,}",
        f"Firms: {panel['company_id'].nunique():,}",
        f"Date range: {panel['date'].min().date()} to {panel['date'].max().date()}",
        "",
        "Main FE result:",
        f"  beta(Post): {fe_result.beta[0]:.6f}",
        f"  se(HC1):    {fe_result.se[0]:.6f}",
        f"  p-value:    {fe_result.p_value[0]:.6f}",
        "",
        "Pre-period mean diagnostic:",
        f"  firms:      {diagnostics.get('n_firms')}",
        f"  range:      {diagnostics.get('range')}",
        f"  std:        {diagnostics.get('std')}",
        f"  means_equal (tol): {diagnostics.get('means_equal')}",
        "",
    ]

    if scdid_result:
        summary_lines.append("SCDiD fallback:")
        summary_lines.append(f"  {json.dumps(scdid_result)}")

    with open(output_dir / "rq1_summary.txt", "w", encoding="ascii") as f:
        f.write("\n".join(summary_lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQ1 FE regression pipeline for media bias shift around 2020 recession.")

    parser.add_argument("--schema", type=str, default=DEFAULT_SCHEMA, help="DB schema name")
    parser.add_argument("--table", type=str, default=DEFAULT_TABLE, help="DB table name")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.90,
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
        help="If pre-period firm mean range exceeds this value, trigger SCDiD fallback",
    )
    parser.add_argument(
        "--skip-scdid",
        action="store_true",
        help="Skip SCDiD fallback even if diagnostic triggers",
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

    scdid_result = None
    if diag.get("trigger_scdid") and not args.skip_scdid:
        print("Pre-period means differ across firms: running SCDiD fallback...")
        scdid_result = run_scdid_fallback(panel)

    metadata = {
        "schema": args.schema,
        "table": args.table,
        "confidence_threshold": args.confidence_threshold,
        "vix_source": f"{args.vix_schema}.{args.vix_table}" if not args.vix_csv else args.vix_csv,
        "pre_period": [PRE_START, PRE_END],
        "post_period": [POST_START, POST_END],
    }

    print("Writing outputs...")
    write_outputs(
        output_dir=Path(args.output_dir),
        panel=panel,
        fe_result=fe_result,
        regressor_names=regressor_names,
        diagnostics=diag,
        scdid_result=scdid_result,
        metadata=metadata,
    )

    print("Done.")
    print(f"Main coefficient beta(Post): {fe_result.beta[0]:.6f}")


if __name__ == "__main__":
    main()
