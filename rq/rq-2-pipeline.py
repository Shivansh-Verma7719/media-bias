#!/usr/bin/env python3
"""
RQ2 regression pipeline.

Research question:
Did the 2020 recession shock affect stock prices differently pre versus post, at the market level?

Model:
    return(i,t) = alpha(i) + beta * Post(t) + gamma * X(i,t) + epsilon(i,t)

Where return(i,t) is firm daily stock return from close prices.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
import yfinance as yf
from dotenv import load_dotenv


# Firm universe source (aligned with RQ1 article universe)
DEFAULT_ARTICLE_SCHEMA = "public"
DEFAULT_ARTICLE_TABLE = "articles_no_title_deduped"
DEFAULT_COMPANY_TABLE = "top_companies"

# Price source for firm-level returns
DEFAULT_PRICE_SCHEMA = "public"
DEFAULT_PRICE_TABLE = "stock_prices"

# VIX source
DEFAULT_VIX_SCHEMA = "finance"
DEFAULT_VIX_TABLE = "vix_daily"

# Market benchmark source (for S&P 500 return control)
DEFAULT_MARKET_SCHEMA = "public"
DEFAULT_MARKET_TABLE = "stock_prices"
DEFAULT_MARKET_TICKER = "^GSPC"

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


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        flat = []
        for col in out.columns.to_flat_index():
            left = str(col[0]).strip() if len(col) > 0 else ""
            right = str(col[1]).strip() if len(col) > 1 else ""
            flat.append(left or right)
        out.columns = flat
    out = out.loc[:, ~out.columns.duplicated(keep="first")]
    return out


def _pick_col(df: pd.DataFrame, *candidates: str) -> str | None:
    normalized = {
        str(c).lower().replace(" ", "").replace("_", ""): c
        for c in df.columns
    }
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in normalized:
            return str(normalized[key])
    return None


def fetch_firm_universe(
    db_url: str,
    article_schema: str,
    article_table: str,
    company_table: str,
) -> pd.DataFrame:
    article_schema = validate_identifier(article_schema, "article_schema")
    article_table = validate_identifier(article_table, "article_table")
    company_table = validate_identifier(company_table, "company_table")

    query = f"""
        SELECT DISTINCT
            a.company_id::text AS company_id,
            c.symbol AS symbol
        FROM {article_schema}.{article_table} a
        JOIN public.{company_table} c
          ON c.id = a.company_id
        WHERE a.published_at::date BETWEEN %s AND %s
          AND a.company_id IS NOT NULL
          AND c.symbol IS NOT NULL
    """

    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (PRE_START, POST_END))
            rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=["company_id", "symbol"])
    if df.empty:
        raise RuntimeError("No firms found in article universe for RQ2 window.")

    df["company_id"] = df["company_id"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    return df.drop_duplicates()


def fetch_price_panel(
    db_url: str,
    symbols: list[str],
    price_schema: str,
    price_table: str,
) -> pd.DataFrame:
    if not symbols:
        raise RuntimeError("No symbols supplied for price query.")

    price_schema = validate_identifier(price_schema, "price_schema")
    price_table = validate_identifier(price_table, "price_table")

    query = f"""
        SELECT
            ticker AS symbol,
            date::date AS date,
            close::double precision AS close
        FROM {price_schema}.{price_table}
        WHERE ticker = ANY(%s)
          AND date::date BETWEEN %s AND %s
          AND close IS NOT NULL
        ORDER BY ticker, date
    """

    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (symbols, PRE_START, POST_END))
            rows = cur.fetchall()

    px = pd.DataFrame(rows, columns=["symbol", "date", "close"])
    if px.empty:
        raise RuntimeError(f"No stock prices found in {price_schema}.{price_table} for selected symbols.")

    px["date"] = pd.to_datetime(px["date"])
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px = px.dropna(subset=["close"]).sort_values(["symbol", "date"])

    px["daily_return"] = px.groupby("symbol")["close"].pct_change()
    px = px.dropna(subset=["daily_return"])
    return px[["symbol", "date", "daily_return"]]


def load_sp500_return(
    db_url: str,
    market_csv: str | None,
    market_schema: str,
    market_table: str,
    market_ticker: str,
) -> pd.DataFrame:
    if market_csv:
        mdf = pd.read_csv(market_csv)
        date_col = _pick_col(mdf, "Date")
        close_col = _pick_col(mdf, "Close", "Adj Close", "AdjClose")
        if not date_col or not close_col:
            raise ValueError("Market CSV must include Date and Close (or Adj Close) columns.")
        out = mdf[[date_col, close_col]].copy()
        out.columns = ["date", "close"]
    else:
        market_schema = validate_identifier(market_schema, "market_schema")
        market_table = validate_identifier(market_table, "market_table")
        query = f"""
            SELECT date::date AS date, close::double precision AS close
            FROM {market_schema}.{market_table}
            WHERE ticker = %s
              AND date::date BETWEEN %s AND %s
              AND close IS NOT NULL
            ORDER BY date
        """
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (market_ticker, PRE_START, POST_END))
                rows = cur.fetchall()
        out = pd.DataFrame(rows, columns=["date", "close"])

        # Fallback to yfinance if market ticker not present in DB table.
        if out.empty:
            y = yf.download(market_ticker, start=PRE_START, end=POST_END, auto_adjust=False, progress=False, interval="1d")
            if y.empty:
                raise RuntimeError(
                    f"No S&P 500 benchmark data found in {market_schema}.{market_table} and yfinance fallback returned empty."
                )
            y = _normalize_columns(y.reset_index())
            date_col = _pick_col(y, "Date")
            close_col = _pick_col(y, "Close", "Adj Close", "AdjClose")
            if not date_col or not close_col:
                raise RuntimeError("Unexpected yfinance schema for market benchmark.")
            out = y[[date_col, close_col]].copy()
            out.columns = ["date", "close"]

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date")

    out["sp500_return"] = out["close"].pct_change()
    out = out.dropna(subset=["sp500_return"])
    return out[["date", "sp500_return"]]


def load_vix(db_url: str, vix_schema: str, vix_table: str) -> pd.DataFrame:
    vix_schema = validate_identifier(vix_schema, "vix_schema")
    vix_table = validate_identifier(vix_table, "vix_table")

    query = f"""
        SELECT trade_date::date AS date, close::double precision AS vix
        FROM {vix_schema}.{vix_table}
        WHERE trade_date::date BETWEEN %s AND %s
          AND close IS NOT NULL
        ORDER BY trade_date
    """

    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (PRE_START, POST_END))
            rows = cur.fetchall()

    out = pd.DataFrame(rows, columns=["date", "vix"])
    if out.empty:
        raise RuntimeError(f"No VIX data found in {vix_schema}.{vix_table}.")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["vix"] = pd.to_numeric(out["vix"], errors="coerce")
    out = out.dropna(subset=["date", "vix"])
    return out


def build_return_panel(
    universe: pd.DataFrame,
    price_returns: pd.DataFrame,
    sp500_return: pd.DataFrame,
    vix: pd.DataFrame,
) -> pd.DataFrame:
    panel = universe.merge(price_returns, on="symbol", how="inner")
    panel = panel.merge(sp500_return, on="date", how="left")
    panel = panel.merge(vix, on="date", how="left")

    panel["sp500_return"] = panel["sp500_return"].ffill().bfill()
    panel["vix"] = panel["vix"].ffill().bfill()

    panel["post"] = (panel["date"] >= pd.Timestamp(POST_START)).astype(int)

    panel = panel.dropna(subset=["daily_return", "post", "sp500_return", "vix"])
    panel = panel[(panel["date"] >= PRE_START) & (panel["date"] <= POST_END)]

    pre = panel[(panel["date"] >= PRE_START) & (panel["date"] <= PRE_END)]
    post = panel[(panel["date"] >= POST_START) & (panel["date"] <= POST_END)]
    if pre.empty or post.empty:
        raise RuntimeError("Insufficient pre/post observations after panel construction.")

    return panel


def _normal_2sided_p(t_stat: np.ndarray) -> np.ndarray:
    abs_t = np.abs(t_stat)
    return np.array([math.erfc(float(v) / math.sqrt(2.0)) for v in abs_t])


def run_entity_fe_ols(panel: pd.DataFrame) -> tuple[OLSResult, list[str]]:
    cols = ["post", "sp500_return", "vix"]
    y = panel["daily_return"].astype(float)
    x = panel[cols].astype(float)

    y_tilde = (y - y.groupby(panel["company_id"]).transform("mean")).to_numpy()
    x_tilde = np.column_stack(
        [
            (x[c] - x[c].groupby(panel["company_id"]).transform("mean")).to_numpy()
            for c in cols
        ]
    )

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

    meat = np.zeros((k, k), dtype=float)
    for i in range(n):
        xi = x_tilde[i : i + 1, :].T
        meat += (resid[i] ** 2) * (xi @ xi.T)

    vcov = (n / max(n - k, 1)) * (xtx_inv @ meat @ xtx_inv)
    se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))

    t_stat = beta / np.where(se == 0, np.nan, se)
    p_vals = _normal_2sided_p(np.nan_to_num(t_stat, nan=0.0))
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se

    sse = float(np.sum(resid**2))
    sst = float(np.sum((y_tilde - np.mean(y_tilde)) ** 2))
    r2 = 1.0 - (sse / sst if sst > 0 else np.nan)

    return (
        OLSResult(
            beta=beta,
            se=se,
            t_stat=t_stat,
            p_value=p_vals,
            ci_low=ci_low,
            ci_high=ci_high,
            r2=r2,
            n_obs=n,
            n_regressors=k,
        ),
        cols,
    )


def pre_period_mean_diagnostic(panel: pd.DataFrame, tolerance: float) -> dict[str, Any]:
    pre = panel[(panel["date"] >= PRE_START) & (panel["date"] <= PRE_END)]
    pre_means = pre.groupby("company_id")["daily_return"].mean()

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
    w, *_ = np.linalg.lstsq(x_pre, y_pre, rcond=None)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0:
        return np.ones(x_pre.shape[1]) / x_pre.shape[1]
    return w / s


def run_scdid_fallback(panel: pd.DataFrame, min_pre_days: int = 180, min_post_days: int = 180) -> dict[str, Any]:
    pivot = panel.pivot_table(index="date", columns="company_id", values="daily_return", aggfunc="mean")
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

        valid_pre = y_pre_full.notna() & (x_pre.notna().sum(axis=1) >= 5)
        if valid_pre.sum() < min_pre_days:
            continue

        x_pre_use = x_pre.loc[valid_pre].copy()
        y_pre_use = y_pre_full.loc[valid_pre].to_numpy(dtype=float)

        donor_means = x_pre_use.mean(axis=0)
        x_pre_np = x_pre_use.fillna(donor_means).to_numpy(dtype=float)
        if x_pre_np.shape[1] == 0:
            continue

        w = _synthetic_weights(x_pre_np, y_pre_use)

        valid_post = y_post_full.notna()
        if valid_post.sum() < min_post_days:
            continue

        y_post_np = y_post_full.loc[valid_post].to_numpy(dtype=float)
        x_post_np = x_post.fillna(donor_means).loc[valid_post].to_numpy(dtype=float)
        if x_post_np.shape[0] == 0:
            continue

        synth_post = x_post_np @ w
        effects.append(float(np.mean(y_post_np - synth_post)))
        used_firms += 1

    if not effects:
        return {"available": False, "message": "Insufficient data to compute SCDiD fallback."}

    arr = np.array(effects, dtype=float)
    avg = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else np.nan
    return {
        "available": True,
        "n_firms_used": int(used_firms),
        "avg_post_effect": avg,
        "se_across_firms": se,
        "ci95_low": float(avg - 1.96 * se) if np.isfinite(se) else np.nan,
        "ci95_high": float(avg + 1.96 * se) if np.isfinite(se) else np.nan,
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

    panel.to_csv(output_dir / "rq2_daily_panel.csv", index=False)

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

    pd.DataFrame(coef_rows).to_csv(output_dir / "rq2_fe_coefficients.csv", index=False)

    payload: dict[str, Any] = {
        "metadata": metadata,
        "model": {
            "formula": "daily_return_it = alpha_i + beta*post_t + gamma1*sp500_return_t + gamma2*vix_t + eps_it",
            "estimator": "Entity fixed-effects OLS (within), HC1 robust SE",
            "n_obs": fe_result.n_obs,
            "n_regressors": fe_result.n_regressors,
            "r2_within": fe_result.r2,
            "main_beta_post": float(fe_result.beta[0]),
            "main_beta_post_se": float(fe_result.se[0]),
            "main_beta_post_p": float(fe_result.p_value[0]),
        },
        "diagnostics": {"pre_period_means": diagnostics},
        "fallback_scdid": scdid_result,
        "coefficients": coef_rows,
    }

    with open(output_dir / "rq2_summary.json", "w", encoding="ascii") as f:
        import json

        json.dump(payload, f, indent=2)

    lines = [
        "RQ2 Regression Pipeline Summary",
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
        import json

        lines.append("SCDiD fallback:")
        lines.append(f"  {json.dumps(scdid_result)}")

    with open(output_dir / "rq2_summary.txt", "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RQ2 FE regression pipeline for post-2020 shift in firm daily returns.")

    p.add_argument("--article-schema", type=str, default=DEFAULT_ARTICLE_SCHEMA)
    p.add_argument("--article-table", type=str, default=DEFAULT_ARTICLE_TABLE)
    p.add_argument("--company-table", type=str, default=DEFAULT_COMPANY_TABLE)

    p.add_argument("--price-schema", type=str, default=DEFAULT_PRICE_SCHEMA)
    p.add_argument("--price-table", type=str, default=DEFAULT_PRICE_TABLE)

    p.add_argument("--vix-schema", type=str, default=DEFAULT_VIX_SCHEMA)
    p.add_argument("--vix-table", type=str, default=DEFAULT_VIX_TABLE)

    p.add_argument("--market-csv", type=str, default=None, help="Optional market benchmark CSV with Date/Close.")
    p.add_argument("--market-schema", type=str, default=DEFAULT_MARKET_SCHEMA)
    p.add_argument("--market-table", type=str, default=DEFAULT_MARKET_TABLE)
    p.add_argument("--market-ticker", type=str, default=DEFAULT_MARKET_TICKER)

    p.add_argument("--pre-mean-tolerance", type=float, default=0.005)
    p.add_argument("--skip-scdid", action="store_true")
    p.add_argument("--output-dir", type=str, default="results/rq2")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    db_url = resolve_db_url()

    print("Fetching firm universe (aligned with RQ1 article sample)...")
    universe = fetch_firm_universe(
        db_url,
        args.article_schema,
        args.article_table,
        args.company_table,
    )

    print("Fetching firm stock prices and computing daily returns...")
    price_returns = fetch_price_panel(
        db_url,
        symbols=sorted(universe["symbol"].unique().tolist()),
        price_schema=args.price_schema,
        price_table=args.price_table,
    )

    print("Loading S&P 500 market return control...")
    market_returns = load_sp500_return(
        db_url,
        market_csv=args.market_csv,
        market_schema=args.market_schema,
        market_table=args.market_table,
        market_ticker=args.market_ticker,
    )

    print("Loading VIX control...")
    vix = load_vix(db_url, args.vix_schema, args.vix_table)

    print("Building regression panel...")
    panel = build_return_panel(universe, price_returns, market_returns, vix)

    print("Running entity FE regression...")
    fe_result, regressor_names = run_entity_fe_ols(panel)

    print("Running pre-period diagnostic...")
    diag = pre_period_mean_diagnostic(panel, args.pre_mean_tolerance)

    scdid_result = None
    if diag.get("trigger_scdid") and not args.skip_scdid:
        print("Pre-period means differ across firms: running SCDiD fallback...")
        scdid_result = run_scdid_fallback(panel)

    metadata = {
        "article_source": f"{args.article_schema}.{args.article_table}",
        "company_table": args.company_table,
        "price_source": f"{args.price_schema}.{args.price_table}",
        "market_source": args.market_csv if args.market_csv else f"{args.market_schema}.{args.market_table}:{args.market_ticker}",
        "vix_source": f"{args.vix_schema}.{args.vix_table}",
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
