#!/usr/bin/env python3
"""Temporary pipeline: fetch VIX from yfinance and upsert into finance.vix_daily."""

from __future__ import annotations

import argparse
import os
from datetime import date

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import errors
from psycopg2.extras import execute_values
import yfinance as yf
from dotenv import load_dotenv


DEFAULT_SCHEMA = "finance"
DEFAULT_TABLE = "vix_daily"
DEFAULT_SYMBOL = "^VIX"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance output columns to a predictable single level."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        flat = []
        for col in out.columns.to_flat_index():
            left = str(col[0]).strip() if len(col) > 0 else ""
            right = str(col[1]).strip() if len(col) > 1 else ""
            flat.append(left or right)
        out.columns = flat

    # Keep first occurrence when duplicated after flattening.
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


def get_db_url() -> str:
    load_dotenv()
    db_url = os.getenv("POOLER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("Missing POOLER_DATABASE_URL/DATABASE_URL in environment")
    return db_url


def fetch_vix(start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(
        DEFAULT_SYMBOL,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        interval="1d",
    )

    if df.empty:
        raise RuntimeError("No VIX data returned by yfinance")

    df = df.reset_index()
    df = _normalize_columns(df)

    date_col = _pick_col(df, "Date")
    open_col = _pick_col(df, "Open")
    high_col = _pick_col(df, "High")
    low_col = _pick_col(df, "Low")
    close_col = _pick_col(df, "Close")
    adj_close_col = _pick_col(df, "Adj Close", "AdjClose")
    volume_col = _pick_col(df, "Volume")

    if date_col is None:
        raise RuntimeError("Unexpected yfinance schema: missing Date column")

    if close_col is None:
        raise RuntimeError("Unexpected yfinance schema: missing Close column")

    # Build canonical shape while tolerating missing optional columns.
    out = pd.DataFrame()
    out["trade_date"] = df[date_col]
    out["open"] = df[open_col] if open_col else np.nan
    out["high"] = df[high_col] if high_col else np.nan
    out["low"] = df[low_col] if low_col else np.nan
    out["close"] = df[close_col]
    out["adj_close"] = df[adj_close_col] if adj_close_col else df[close_col]
    out["volume"] = df[volume_col] if volume_col else np.nan

    df = out
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

    for c in ["open", "high", "low", "close", "adj_close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = df.dropna(subset=["trade_date", "close"]).copy()
    df["symbol"] = DEFAULT_SYMBOL
    df["source"] = "yfinance"
    return df


def upsert_vix(db_url: str, schema: str, table: str, df: pd.DataFrame) -> int:
    schema = schema.strip()
    table = table.strip()

    rows = [
        (
            r.trade_date,
            r.symbol,
            float(r.open) if pd.notna(r.open) else None,
            float(r.high) if pd.notna(r.high) else None,
            float(r.low) if pd.notna(r.low) else None,
            float(r.close) if pd.notna(r.close) else None,
            float(r.adj_close) if pd.notna(r.adj_close) else None,
            int(float(r.volume)) if pd.notna(r.volume) else None,
            r.source,
        )
        for r in df.itertuples(index=False)
    ]

    if not rows:
        return 0

    query = f"""
        INSERT INTO {schema}.{table}
        (
            trade_date, symbol, open, high, low, close, adj_close, volume, source
        )
        VALUES %s
        ON CONFLICT (trade_date)
        DO UPDATE SET
            symbol = EXCLUDED.symbol,
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            adj_close = EXCLUDED.adj_close,
            volume = EXCLUDED.volume,
            source = EXCLUDED.source,
            updated_at = NOW();
    """

    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, rows, page_size=500)
            inserted = cur.rowcount
        conn.commit()

    return inserted


def get_current_role(db_url: str) -> str:
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_user")
            row = cur.fetchone()
    return str(row[0]) if row else "unknown_role"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch VIX from yfinance and upsert into Postgres.")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=str(date.today()), help="End date YYYY-MM-DD (exclusive in yfinance)")
    parser.add_argument("--schema", type=str, default=DEFAULT_SCHEMA)
    parser.add_argument("--table", type=str, default=DEFAULT_TABLE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_url = get_db_url()

    print("Fetching VIX from yfinance...")
    vix = fetch_vix(args.start, args.end)
    print(f"Fetched {len(vix):,} rows ({vix['trade_date'].min()} to {vix['trade_date'].max()}).")

    print(f"Upserting into {args.schema}.{args.table}...")
    try:
        upsert_count = upsert_vix(db_url, args.schema, args.table, vix)
    except errors.InsufficientPrivilege as exc:
        role = get_current_role(db_url)
        fqtn = f"{args.schema}.{args.table}"
        print("Permission error while writing VIX data.")
        print(f"Current role: {role}")
        print("Run these SQL statements with an admin/owner role:")
        print()
        print(f"GRANT USAGE ON SCHEMA {args.schema} TO {role};")
        print(f"GRANT SELECT, INSERT, UPDATE ON TABLE {fqtn} TO {role};")
        print(f"ALTER TABLE {fqtn} OWNER TO {role};")
        print()
        print("Then rerun: python rq/temp_vix_pipeline.py")
        raise RuntimeError("Insufficient privileges to upsert VIX data.") from exc
    print(f"Upsert completed. Driver rowcount: {upsert_count}.")


if __name__ == "__main__":
    main()
