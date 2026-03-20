import os
from pathlib import Path

import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv


TABLE_NAME = "articles_no_title_deduped"
SOURCE_TABLE = "articles_stratified"
COMPANY_TABLE = "top_companies"
TRUNCATE_TARGET = False
DRY_RUN = False


def resolve_db_url() -> str:
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path)
    else:
        print(f"Warning: .env not found at {env_path}")

    db_url = os.getenv("POOLER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("Neither POOLER_DATABASE_URL nor DATABASE_URL found in .env")

    return db_url


def ensure_target_table(cur, source_table: str, target_table: str) -> None:
    # Clone source schema so all article columns and defaults are preserved.
    cur.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS public.{target} (
                LIKE public.{source} INCLUDING ALL
            );
            """
        ).format(target=sql.Identifier(target_table), source=sql.Identifier(source_table))
    )


def compute_candidate_count(cur, source_table: str, company_table: str) -> int:
    count_query = sql.SQL(
        """
        WITH filtered AS (
            SELECT
                a.id,
                a.company_id,
                c.symbol,
                c.name AS company_name,
                LOWER(TRIM(REGEXP_REPLACE(COALESCE(a.title, ''), '\\s+', ' ', 'g'))) AS title_norm,
                a.published_at
            FROM public.{source} a
            JOIN public.{company} c
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
                ROW_NUMBER() OVER (
                    PARTITION BY company_id, title_norm
                    ORDER BY published_at DESC, id DESC
                ) AS rn
            FROM filtered
        )
        SELECT COUNT(*)
        FROM deduped
        WHERE rn = 1;
        """
    ).format(source=sql.Identifier(source_table), company=sql.Identifier(company_table))

    cur.execute(count_query)
    return int(cur.fetchone()[0])


def insert_rows(cur, source_table: str, company_table: str, target_table: str) -> int:
    insert_query = sql.SQL(
        """
        WITH filtered AS (
            SELECT
                a.id,
                a.company_id,
                c.symbol,
                c.name AS company_name,
                LOWER(TRIM(REGEXP_REPLACE(COALESCE(a.title, ''), '\\s+', ' ', 'g'))) AS title_norm,
                a.published_at
            FROM public.{source} a
            JOIN public.{company} c
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
                ROW_NUMBER() OVER (
                    PARTITION BY company_id, title_norm
                    ORDER BY published_at DESC, id DESC
                ) AS rn
            FROM filtered
        )
        INSERT INTO public.{target}
        SELECT a.*
        FROM public.{source} a
        JOIN deduped d
            ON a.id = d.id
        WHERE d.rn = 1
        ON CONFLICT (id) DO NOTHING;
        """
    ).format(
        source=sql.Identifier(source_table),
        company=sql.Identifier(company_table),
        target=sql.Identifier(target_table),
    )

    cur.execute(insert_query)
    return cur.rowcount


def table_count(cur, table_name: str) -> int:
    cur.execute(
        sql.SQL("SELECT COUNT(*) FROM public.{table}").format(
            table=sql.Identifier(table_name)
        )
    )
    return int(cur.fetchone()[0])


def main() -> None:
    db_url = resolve_db_url()

    print("Connecting to database...")
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            ensure_target_table(cur, SOURCE_TABLE, TABLE_NAME)

            if TRUNCATE_TARGET:
                cur.execute(
                    sql.SQL("TRUNCATE TABLE public.{table}").format(
                        table=sql.Identifier(TABLE_NAME)
                    )
                )
                print(f"Truncated public.{TABLE_NAME}")

            candidate_count = compute_candidate_count(cur, SOURCE_TABLE, COMPANY_TABLE)
            print(f"Candidate rows after filter+dedupe: {candidate_count}")

            if DRY_RUN:
                print("Dry run complete. No rows inserted.")
                return

            before_count = table_count(cur, TABLE_NAME)
            inserted = insert_rows(cur, SOURCE_TABLE, COMPANY_TABLE, TABLE_NAME)
            after_count = table_count(cur, TABLE_NAME)

            print(f"Rows in public.{TABLE_NAME} before insert: {before_count}")
            print(f"Rows reported inserted this run: {inserted}")
            print(f"Rows in public.{TABLE_NAME} after insert: {after_count}")

    print("Done.")


if __name__ == "__main__":
    main()
