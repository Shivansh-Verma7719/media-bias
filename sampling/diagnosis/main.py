import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table


console = Console()


# -----------------------------
# Global diagnosis configuration
# -----------------------------
SCHEMA_NAME = "public"
ARTICLE_TABLE = "articles_no_title_deduped"
# Set to None to auto-detect top_companies/companies in the schema
COMPANIES_TABLE: str | None = None
SAMPLE_SIZE = 12
AUTO_PURGE = False


def resolve_db_url() -> str:
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    db_url = os.getenv("POOLER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("Neither POOLER_DATABASE_URL nor DATABASE_URL was found")
    return db_url


def pick_companies_table(cur, schema_name: str, preferred: str | None) -> str:
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["top_companies", "companies"])

    seen = set()
    for table_name in candidates:
        if table_name in seen:
            continue
        seen.add(table_name)

        cur.execute(
            "SELECT to_regclass(%s)",
            (f"{schema_name}.{table_name}",),
        )
        exists = cur.fetchone()[0]
        if exists:
            return table_name

    raise ValueError(
        f"Could not find companies table in schema '{schema_name}'. Tried: {', '.join(candidates)}"
    )


def total_rows(cur, schema_name: str, article_table: str) -> int:
    cur.execute(
        sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
            sql.Identifier(schema_name),
            sql.Identifier(article_table),
        )
    )
    return int(cur.fetchone()[0])


def fetch_title_failures(
    cur,
    schema_name: str,
    article_table: str,
    companies_table: str,
    sample_size: int,
) -> tuple[set[int], list[tuple[int, str, str, str]]]:
    query = sql.SQL(
        """
        SELECT
            a.id,
            COALESCE(a.title, '') AS title,
            COALESCE(c.name, '') AS company_name,
            COALESCE(c.symbol, '') AS symbol
        FROM {schema}.{articles} a
        LEFT JOIN {schema}.{companies} c
            ON c.id = a.company_id
        WHERE
            COALESCE(TRIM(a.title), '') = ''
            OR c.id IS NULL
            OR NOT (
                LOWER(a.title) LIKE '%%' || LOWER(c.name) || '%%'
                OR LOWER(a.title) LIKE '%%' || LOWER(c.symbol) || '%%'
            )
        ORDER BY a.published_at DESC NULLS LAST, a.id DESC
        """
    ).format(
        schema=sql.Identifier(schema_name),
        articles=sql.Identifier(article_table),
        companies=sql.Identifier(companies_table),
    )

    cur.execute(query)
    rows = cur.fetchall()
    bad_ids = {int(r[0]) for r in rows}
    return bad_ids, rows[:sample_size]


def fetch_duplicate_rows(
    cur,
    schema_name: str,
    article_table: str,
    companies_table: str,
    sample_size: int,
) -> tuple[set[int], list[tuple[int, str, str, str, int]]]:
    query = sql.SQL(
        """
        WITH valid AS (
            SELECT
                a.id,
                a.company_id,
                COALESCE(a.title, '') AS title,
                COALESCE(c.name, '') AS company_name,
                COALESCE(c.symbol, '') AS symbol,
                LOWER(TRIM(REGEXP_REPLACE(COALESCE(a.title, ''), '\\s+', ' ', 'g'))) AS title_norm,
                a.published_at
            FROM {schema}.{articles} a
            JOIN {schema}.{companies} c
                ON c.id = a.company_id
            WHERE
                COALESCE(TRIM(a.title), '') <> ''
                AND (
                    LOWER(a.title) LIKE '%%' || LOWER(c.name) || '%%'
                    OR LOWER(a.title) LIKE '%%' || LOWER(c.symbol) || '%%'
                )
        ), ranked AS (
            SELECT
                id,
                title,
                company_name,
                symbol,
                ROW_NUMBER() OVER (
                    PARTITION BY company_id, title_norm
                    ORDER BY published_at DESC NULLS LAST, id DESC
                ) AS rn
            FROM valid
        )
        SELECT id, title, company_name, symbol, rn
        FROM ranked
        WHERE rn > 1
        ORDER BY rn DESC, id DESC
        """
    ).format(
        schema=sql.Identifier(schema_name),
        articles=sql.Identifier(article_table),
        companies=sql.Identifier(companies_table),
    )

    cur.execute(query)
    rows = cur.fetchall()
    dup_ids = {int(r[0]) for r in rows}
    return dup_ids, rows[:sample_size]


def render_dashboard(
    schema_name: str,
    article_table: str,
    companies_table: str,
    total: int,
    title_fail_ids: set[int],
    duplicate_ids: set[int],
    title_samples: list[tuple[int, str, str, str]],
    duplicate_samples: list[tuple[int, str, str, str, int]],
) -> None:
    purge_ids = title_fail_ids | duplicate_ids

    overview = Table(box=box.SIMPLE_HEAVY)
    overview.add_column("Metric", style="bold cyan")
    overview.add_column("Value", justify="right", style="bold white")
    overview.add_row("Schema", schema_name)
    overview.add_row("Table", article_table)
    overview.add_row("Companies table", companies_table)
    overview.add_row("Total rows", f"{total:,}")
    overview.add_row("Title-check failures", f"{len(title_fail_ids):,}")
    overview.add_row("Duplicate rows", f"{len(duplicate_ids):,}")
    overview.add_row("Rows to purge (union)", f"{len(purge_ids):,}")
    overview.add_row("Rows remaining after purge", f"{max(total - len(purge_ids), 0):,}")

    title_tbl = Table(title="Title-Check Failures (sample)", box=box.ROUNDED)
    title_tbl.add_column("id", justify="right", style="yellow")
    title_tbl.add_column("symbol", style="magenta")
    title_tbl.add_column("company", style="cyan")
    title_tbl.add_column("title", style="white")

    if title_samples:
        for rid, title, cname, symbol in title_samples:
            title_tbl.add_row(str(rid), symbol or "-", cname or "-", (title or "")[:120])
    else:
        title_tbl.add_row("-", "-", "-", "No failures detected")

    dup_tbl = Table(title="Duplicate Rows (sample, rn > 1)", box=box.ROUNDED)
    dup_tbl.add_column("id", justify="right", style="yellow")
    dup_tbl.add_column("symbol", style="magenta")
    dup_tbl.add_column("company", style="cyan")
    dup_tbl.add_column("rn", justify="right", style="red")
    dup_tbl.add_column("title", style="white")

    if duplicate_samples:
        for rid, title, cname, symbol, rn in duplicate_samples:
            dup_tbl.add_row(str(rid), symbol or "-", cname or "-", str(rn), (title or "")[:110])
    else:
        dup_tbl.add_row("-", "-", "-", "-", "No duplicates detected")

    console.print(Panel.fit(overview, title="Diagnosis Dashboard", border_style="green"))
    console.print(Columns([title_tbl, dup_tbl], equal=True, expand=True))


def purge_rows(cur, schema_name: str, article_table: str, ids_to_delete: set[int]) -> int:
    if not ids_to_delete:
        return 0

    delete_query = sql.SQL(
        "DELETE FROM {}.{} WHERE id = ANY(%s)"
    ).format(
        sql.Identifier(schema_name),
        sql.Identifier(article_table),
    )

    cur.execute(delete_query, (sorted(ids_to_delete),))
    return cur.rowcount


def main() -> None:
    db_url = resolve_db_url()

    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            companies_table = pick_companies_table(cur, SCHEMA_NAME, COMPANIES_TABLE)
            total = total_rows(cur, SCHEMA_NAME, ARTICLE_TABLE)

            title_fail_ids, title_samples = fetch_title_failures(
                cur,
                SCHEMA_NAME,
                ARTICLE_TABLE,
                companies_table,
                SAMPLE_SIZE,
            )
            duplicate_ids, duplicate_samples = fetch_duplicate_rows(
                cur,
                SCHEMA_NAME,
                ARTICLE_TABLE,
                companies_table,
                SAMPLE_SIZE,
            )

            render_dashboard(
                SCHEMA_NAME,
                ARTICLE_TABLE,
                companies_table,
                total,
                title_fail_ids,
                duplicate_ids,
                title_samples,
                duplicate_samples,
            )

            purge_ids = title_fail_ids | duplicate_ids
            if not purge_ids:
                console.print("[bold green]No rows to purge.[/bold green]")
                return

            should_purge = AUTO_PURGE or Confirm.ask(
                f"Delete {len(purge_ids):,} rows from {SCHEMA_NAME}.{ARTICLE_TABLE}?",
                default=False,
            )

            if not should_purge:
                console.print("[yellow]Purge skipped.[/yellow]")
                return

            deleted = purge_rows(cur, SCHEMA_NAME, ARTICLE_TABLE, purge_ids)
            conn.commit()
            console.print(f"[bold green]Deleted rows:[/bold green] {deleted:,}")


if __name__ == "__main__":
    main()
