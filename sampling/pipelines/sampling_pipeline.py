import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from matplotlib.ticker import FuncFormatter
from psycopg2 import sql


# -----------------------------
# Global pipeline configuration
# -----------------------------
SCHEMA_NAME = "indian_cos"
ARTICLE_TABLE = "articles_fixed"
COMPANIES_TABLE = "companies"
TARGET_TABLE = "articles_stratified"

# Optional behavior toggles
TRUNCATE_TARGET_BEFORE_LOAD = False
DRY_RUN = False

# Outputs
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
PRE_MATRIX_FILENAME = "pre_stratification_2d_matrix.csv"
POST_MATRIX_FILENAME = "post_stratification_2d_matrix.csv"
PRE_CDF_FILENAME = "pre_stratification_cdf.png"
POST_CDF_FILENAME = "post_stratification_cdf.png"


def resolve_db_url() -> str:
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path)
    else:
        print(f"Warning: .env not found at {env_path}")

    db_url = os.getenv("POOLER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("Neither POOLER_DATABASE_URL nor DATABASE_URL found in .env")

    return db_url


def _filter_and_dedupe_cte(schema_name: str, article_table: str, companies_table: str) -> sql.SQL:
    return sql.SQL(
        """
        WITH filtered AS (
            SELECT
                a.id,
                a.company_id,
                c.symbol,
                c.name AS company_name,
                CAST(EXTRACT(YEAR FROM a.published_at) AS INTEGER) AS year,
                LOWER(TRIM(REGEXP_REPLACE(COALESCE(a.title, ''), '\\s+', ' ', 'g'))) AS title_norm,
                a.published_at
            FROM {schema}.{articles} a
            JOIN {schema}.{companies} c
                ON a.company_id = c.id
            WHERE a.published_at IS NOT NULL
              AND COALESCE(TRIM(a.title), '') <> ''
              AND (
                    LOWER(a.title) LIKE '%' || LOWER(c.name) || '%'
                    OR LOWER(a.title) LIKE '%' || LOWER(c.symbol) || '%'
              )
        ),
        deduped AS (
            SELECT
                id,
                company_id,
                symbol,
                company_name,
                year,
                ROW_NUMBER() OVER (
                    PARTITION BY company_id, title_norm
                    ORDER BY published_at DESC, id DESC
                ) AS rn
            FROM filtered
        )
        """
    ).format(
        schema=sql.Identifier(schema_name),
        articles=sql.Identifier(article_table),
        companies=sql.Identifier(companies_table),
    )


def ensure_target_table(cur, schema_name: str, source_table: str, target_table: str) -> None:
    cur.execute(
        sql.SQL("CREATE SCHEMA IF NOT EXISTS {schema};").format(
            schema=sql.Identifier(schema_name)
        )
    )

    cur.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {schema}.{target} (
                LIKE {schema}.{source} INCLUDING ALL
            );
            """
        ).format(
            schema=sql.Identifier(schema_name),
            target=sql.Identifier(target_table),
            source=sql.Identifier(source_table),
        )
    )


def table_count(cur, schema_name: str, table_name: str) -> int:
    cur.execute(
        sql.SQL("SELECT COUNT(*) FROM {schema}.{table}").format(
            schema=sql.Identifier(schema_name),
            table=sql.Identifier(table_name),
        )
    )
    return int(cur.fetchone()[0])


def candidate_count(cur, schema_name: str, source_table: str, companies_table: str) -> int:
    query = sql.SQL(
        """
        {cte}
        SELECT COUNT(*)
        FROM deduped
        WHERE rn = 1;
        """
    ).format(cte=_filter_and_dedupe_cte(schema_name, source_table, companies_table))

    cur.execute(query)
    return int(cur.fetchone()[0])


def insert_filtered_deduped_rows(
    cur,
    schema_name: str,
    source_table: str,
    companies_table: str,
    target_table: str,
) -> int:
    query = sql.SQL(
        """
        {cte}
        INSERT INTO {schema}.{target}
        SELECT a.*
        FROM {schema}.{source} a
        JOIN deduped d
            ON a.id = d.id
        WHERE d.rn = 1
          AND NOT EXISTS (
              SELECT 1
              FROM {schema}.{target} t
              WHERE t.id = a.id
          );
        """
    ).format(
        cte=_filter_and_dedupe_cte(schema_name, source_table, companies_table),
        schema=sql.Identifier(schema_name),
        source=sql.Identifier(source_table),
        target=sql.Identifier(target_table),
    )

    cur.execute(query)
    return cur.rowcount


def _query_df(conn, query: sql.SQL) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=columns)


def fetch_yearly_counts_all_articles_df(
    conn,
    schema_name: str,
    article_table: str,
    companies_table: str,
) -> pd.DataFrame:
    query = sql.SQL(
        """
        SELECT
            c.symbol,
            CAST(EXTRACT(YEAR FROM a.published_at) AS INTEGER) AS year,
            COUNT(*) AS article_count
        FROM {schema}.{articles} a
        JOIN {schema}.{companies} c
            ON a.company_id = c.id
        WHERE a.published_at IS NOT NULL
        GROUP BY c.symbol, CAST(EXTRACT(YEAR FROM a.published_at) AS INTEGER)
        ORDER BY c.symbol, year;
        """
    ).format(
        schema=sql.Identifier(schema_name),
        articles=sql.Identifier(article_table),
        companies=sql.Identifier(companies_table),
    )

    return _query_df(conn, query)


def fetch_yearly_counts_df(
    conn,
    schema_name: str,
    article_table: str,
    companies_table: str,
) -> pd.DataFrame:
    query = sql.SQL(
        """
        {cte}
        SELECT
            symbol,
            year,
            COUNT(*) AS article_count
        FROM deduped
        WHERE rn = 1
        GROUP BY symbol, year
        ORDER BY symbol, year;
        """
    ).format(cte=_filter_and_dedupe_cte(schema_name, article_table, companies_table))

    return _query_df(conn, query)


def fetch_company_counts_all_articles(
    conn,
    schema_name: str,
    article_table: str,
    companies_table: str,
) -> np.ndarray:
    query = sql.SQL(
        """
        SELECT
            c.symbol,
            COUNT(*) AS article_count
        FROM {schema}.{articles} a
        JOIN {schema}.{companies} c
            ON a.company_id = c.id
        GROUP BY c.symbol
        ORDER BY c.symbol;
        """
    ).format(
        schema=sql.Identifier(schema_name),
        articles=sql.Identifier(article_table),
        companies=sql.Identifier(companies_table),
    )

    df = _query_df(conn, query)
    if df.empty:
        return np.array([], dtype=float)

    return df["article_count"].astype(float).to_numpy()


def build_2d_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    matrix = (
        df.pivot(index="symbol", columns="year", values="article_count")
        .fillna(0)
        .astype(int)
    )

    year_cols = sorted([c for c in matrix.columns], key=int)
    matrix = matrix[year_cols]
    matrix["Total"] = matrix.sum(axis=1)
    matrix = matrix.sort_values(by="Total", ascending=False)

    total_row = matrix.sum(axis=0)
    total_row.name = "Total"
    matrix = pd.concat([matrix, total_row.to_frame().T])

    matrix.columns = [str(c) if c != "Total" else "Total" for c in matrix.columns]
    matrix.index.name = None
    return matrix


def fetch_company_counts(
    conn,
    schema_name: str,
    article_table: str,
    companies_table: str,
) -> np.ndarray:
    query = sql.SQL(
        """
        {cte}
        SELECT
            symbol,
            COUNT(*) AS article_count
        FROM deduped
        WHERE rn = 1
        GROUP BY symbol
        ORDER BY symbol;
        """
    ).format(cte=_filter_and_dedupe_cte(schema_name, article_table, companies_table))

    df = _query_df(conn, query)
    if df.empty:
        return np.array([], dtype=float)

    return df["article_count"].astype(float).to_numpy()


def plot_cdf_dual_scale(counts: np.ndarray, title: str, output_path: Path) -> None:
    if counts.size == 0:
        print(f"Skipping CDF plot ({output_path.name}): no data.")
        return

    counts = np.sort(counts)
    n = len(counts)
    cdf = np.arange(1, n + 1)

    p25 = np.percentile(counts, 25)
    p50 = np.percentile(counts, 50)
    p75 = np.percentile(counts, 75)
    p90 = np.percentile(counts, 90)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    colors = {"25th": "#1abc9c", "50th": "#f39c12", "75th": "#e74c3c", "90th": "#9b59b6"}
    line_color = "royalblue"

    for ax, is_log in [(ax1, False), (ax2, True)]:
        ax.plot(counts, cdf, linestyle="-", color=line_color, linewidth=2)
        ax.fill_between(counts, cdf, color=line_color, alpha=0.2)

        ax.set_ylabel("Cumulative Number of Companies", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.7, which="both")
        ax.set_ylim(0, n + 5)

        if is_log:
            min_pos = np.min(counts[counts > 0]) if np.any(counts > 0) else 1
            ax.set_xlim(left=min_pos * 0.8)
            ax.set_title("CDF: Log Scale", fontsize=14, fontweight="bold")
            ax.set_xlabel("Number of Articles per Company (log scale)", fontsize=12)
            ax.set_xscale("log")

            stats_text = (
                f"Total Companies: {n}\n"
                f"Total Articles: {int(np.sum(counts)):,}\n"
                f"Mean: {np.mean(counts):.1f}\n"
                f"Median: {np.median(counts):.1f}\n"
                f"Std Dev: {np.std(counts):.1f}\n"
                f"Min: {int(np.min(counts))}\n"
                f"Max: {int(np.max(counts)):,}\n"
                f"Zero-count: {np.sum(counts == 0)}"
            )
            props = dict(boxstyle="round", facecolor="white", alpha=0.8)
            ax.text(
                0.95,
                0.05,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=props,
            )
        else:
            ax.set_xlim(left=0)
            ax.set_title("CDF: Linear Scale", fontsize=14, fontweight="bold")
            ax.set_xlabel("Number of Articles per Company", fontsize=12)

            def millions_formatter(x, _pos):
                if x >= 1e6:
                    return f"{x * 1e-6:g}M"
                if x >= 1e3:
                    return f"{x * 1e-3:g}K"
                return f"{x:g}"

            ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))

        percentiles = [
            (25, p25, int(n * 0.25), colors["25th"]),
            (50, p50, int(n * 0.50), colors["50th"]),
            (75, p75, int(n * 0.75), colors["75th"]),
            (90, p90, int(n * 0.90), colors["90th"]),
        ]

        for perc, p_val, num_companies, color in percentiles:
            ax.axhline(y=num_companies, color=color, linestyle="--", alpha=0.6)
            if p_val > 0 or not is_log:
                ax.axvline(x=p_val, color=color, linestyle="--", alpha=0.6)

            text_x = p_val * (1.10 if is_log else 1.02)
            if not is_log and p_val == 0:
                text_x = np.max(counts) * 0.02

            ax.text(
                text_x,
                num_companies + (n * 0.01),
                f"{perc}th: {int(p_val)} articles ({num_companies} co.)",
                color=color,
                fontsize=9,
                alpha=0.9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved CDF plot: {output_path}")


def save_matrix(matrix: pd.DataFrame, output_path: Path, label: str) -> None:
    if matrix.empty:
        print(f"{label}: no data available.")
        return
    matrix.to_csv(output_path)
    print(f"Saved {label}: {output_path}")


def main() -> None:
    db_url = resolve_db_url()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Connecting to database...")
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            print("Step 1: Generating pre-stratification 2D matrix...")
            pre_df = fetch_yearly_counts_all_articles_df(
                conn,
                SCHEMA_NAME,
                ARTICLE_TABLE,
                COMPANIES_TABLE,
            )
            pre_matrix = build_2d_matrix(pre_df)
            save_matrix(pre_matrix, OUTPUT_DIR / PRE_MATRIX_FILENAME, "pre-stratification matrix")

            print("Step 2: Ensuring target table exists...")
            ensure_target_table(cur, SCHEMA_NAME, ARTICLE_TABLE, TARGET_TABLE)

            if TRUNCATE_TARGET_BEFORE_LOAD:
                cur.execute(
                    sql.SQL("TRUNCATE TABLE {schema}.{table}").format(
                        schema=sql.Identifier(SCHEMA_NAME),
                        table=sql.Identifier(TARGET_TABLE),
                    )
                )
                print(f"Truncated {SCHEMA_NAME}.{TARGET_TABLE}")

            print("Step 3: Running title filter + dedupe load...")
            candidates = candidate_count(cur, SCHEMA_NAME, ARTICLE_TABLE, COMPANIES_TABLE)
            print(f"Candidate rows after filter+dedupe: {candidates}")

            if not DRY_RUN:
                before_count = table_count(cur, SCHEMA_NAME, TARGET_TABLE)
                inserted = insert_filtered_deduped_rows(
                    cur,
                    SCHEMA_NAME,
                    ARTICLE_TABLE,
                    COMPANIES_TABLE,
                    TARGET_TABLE,
                )
                after_count = table_count(cur, SCHEMA_NAME, TARGET_TABLE)
                print(f"Rows in {SCHEMA_NAME}.{TARGET_TABLE} before insert: {before_count}")
                print(f"Rows reported inserted this run: {inserted}")
                print(f"Rows in {SCHEMA_NAME}.{TARGET_TABLE} after insert: {after_count}")
            else:
                print("Dry run enabled. Skipping insert into target table.")

            print("Step 4: Generating post-stratification 2D matrix...")
            post_df = fetch_yearly_counts_df(conn, SCHEMA_NAME, TARGET_TABLE, COMPANIES_TABLE)
            post_matrix = build_2d_matrix(post_df)
            save_matrix(post_matrix, OUTPUT_DIR / POST_MATRIX_FILENAME, "post-stratification matrix")

            print("Step 5: Generating pre/post CDF plots (linear + log)...")
            pre_counts = fetch_company_counts_all_articles(
                conn,
                SCHEMA_NAME,
                ARTICLE_TABLE,
                COMPANIES_TABLE,
            )
            post_counts = fetch_company_counts(conn, SCHEMA_NAME, TARGET_TABLE, COMPANIES_TABLE)

            plot_cdf_dual_scale(
                pre_counts,
                "CDF: Pre-Stratification Article Distribution",
                OUTPUT_DIR / PRE_CDF_FILENAME,
            )
            plot_cdf_dual_scale(
                post_counts,
                "CDF: Post-Stratification Article Distribution",
                OUTPUT_DIR / POST_CDF_FILENAME,
            )

    print("Pipeline completed.")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()