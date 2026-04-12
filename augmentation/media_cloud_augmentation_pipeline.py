import datetime as dt
import hashlib
import os
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql
from psycopg2.extras import Json, execute_values

try:
    from rich import box
    from rich.console import Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    import mediacloud.api
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependency: mediacloud. Install with `pip install mediacloud`."
    ) from exc


# -----------------------------
# Global pipeline configuration
# -----------------------------
SCHEMA_NAME = "public"
TARGET_TABLE = "articles_no_title_deduped"
COMPANIES_TABLE = "top_companies"
COMPANIES_METADATA_COLUMN = "metadata"

COMPANIES_ID_COLUMN = "id"
COMPANIES_NAME_COLUMN = "name"
COMPANIES_SYMBOL_COLUMN = "symbol"

COLLECTION_MODE = "INDIA_NATIONAL"  # Options: US_NATIONAL, INDIA_NATIONAL
COLLECTIONS = {
    "US_NATIONAL": 34412234,
    "INDIA_NATIONAL": 34412118,
}

START_YEAR = 2015
END_YEAR = dt.date.today().year

MEDIA_CLOUD_API_KEY_ENV = "MEDIA_CLOUD_API_KEY"
REQUESTS_PER_MINUTE = 100
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2

DRY_RUN = False
PRINT_EVERY_N_PAGES = 2
USE_RICH_MODE = "auto"  # Options: auto, on, off
SKIP_COMPLETED_COMPANIES = True
FORCE_RERUN = False


def render_live_dashboard(state: Dict) -> Group:
    if not HAS_RICH:
        raise RuntimeError("Rich is not installed")

    metrics_panel = Panel(
        (
            f"Status: {state['status']}\n"
            f"Current: {state['current_company']} / {state['current_year']}\n"
            f"Companies done: {state['companies_done']} / {state['companies_total']}\n"
            f"Cells done: {state['cells_done']} / {state['cells_total']}\n"
            f"Total fetched: {state['grand_fetched']:,}\n"
            f"Total inserted: {state['grand_inserted']:,}\n"
            f"Errors: {len(state['recent_errors'])}"
        ),
        title="Augmentation Live Metrics",
        box=box.ROUNDED,
    )

    cell_table = Table(title="Recent Company-Year Cells", box=box.SIMPLE_HEAVY)
    cell_table.add_column("Company", style="cyan")
    cell_table.add_column("Year", style="magenta")
    cell_table.add_column("Fetched", justify="right")
    cell_table.add_column("Existing", justify="right")
    cell_table.add_column("TitleOut", justify="right")
    cell_table.add_column("DupOut", justify="right")
    cell_table.add_column("Candidate", justify="right")
    cell_table.add_column("Inserted", justify="right", style="green")

    for row in list(state["recent_cells"]):
        cell_table.add_row(
            row["symbol"],
            str(row["year"]),
            str(row["fetched"]),
            str(row["existing"]),
            str(row["title_filtered_out"]),
            str(row["duplicate_filtered_out"]),
            str(row["candidate_new"]),
            str(row["inserted"]),
        )

    errors_panel = Panel(
        "\n".join(state["recent_errors"]) if state["recent_errors"] else "No recent errors",
        title="Recent Errors",
        box=box.ROUNDED,
    )

    return Group(metrics_panel, cell_table, errors_panel)


def should_use_rich() -> bool:
    mode = (os.getenv("AUGMENT_USE_RICH") or USE_RICH_MODE).strip().lower()

    if mode == "on":
        return HAS_RICH
    if mode == "off":
        return False

    # auto mode: use Rich only when running in an interactive terminal
    return HAS_RICH and sys.stdout.isatty()


def print_cell_stats(symbol: str, year: int, stats: Dict[str, int]) -> None:
    print(
        f"CELL {symbol} {year} | "
        f"fetched={stats['fetched']} existing={stats['existing']} "
        f"title_out={stats['title_filtered_out']} dup_out={stats['duplicate_filtered_out']} "
        f"candidate={stats['candidate_new']} inserted={stats['inserted']}"
    )


def print_company_summary(symbol: str, company_rollup: Dict[str, int]) -> None:
    print(
        f"COMPANY {symbol} SUMMARY | "
        f"cells={company_rollup['cells']} fetched={company_rollup['fetched']} "
        f"existing={company_rollup['existing']} title_out={company_rollup['title_filtered_out']} "
        f"dup_out={company_rollup['duplicate_filtered_out']} "
        f"candidate={company_rollup['candidate_new']} inserted={company_rollup['inserted']}"
    )


def build_run_key() -> str:
    return f"{SCHEMA_NAME}.{TARGET_TABLE}:{COLLECTION_MODE}:{START_YEAR}-{END_YEAR}"


def should_force_rerun() -> bool:
    raw = os.getenv("AUGMENT_FORCE_RERUN", str(FORCE_RERUN)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _as_metadata_dict(value) -> Dict:
    if isinstance(value, dict):
        return value
    return {}


def is_company_completed_for_run(metadata: Dict, run_key: str) -> bool:
    run_meta = metadata.get("augmentation", {}).get(run_key, {})
    return bool(run_meta.get("completed", False))


def upsert_company_metadata(
    conn,
    company: Dict,
    run_key: str,
    *,
    year: Optional[int] = None,
    year_status: Optional[str] = None,
    year_stats: Optional[Dict] = None,
    year_error: Optional[str] = None,
    complete_company: bool = False,
    company_rollup: Optional[Dict] = None,
) -> None:
    metadata = _as_metadata_dict(company.get("metadata", {}))
    metadata.setdefault("augmentation", {})
    run_meta = metadata["augmentation"].setdefault(
        run_key,
        {
            "schema": SCHEMA_NAME,
            "table": TARGET_TABLE,
            "collection_mode": COLLECTION_MODE,
            "start_year": START_YEAR,
            "end_year": END_YEAR,
            "status": "in_progress",
            "completed": False,
            "years": {},
            "totals": {},
            "updated_at": None,
        },
    )

    now_iso = dt.datetime.utcnow().isoformat() + "Z"
    run_meta["updated_at"] = now_iso

    if year is not None and year_status is not None:
        year_key = str(year)
        year_payload = {
            "status": year_status,
            "updated_at": now_iso,
        }
        if year_stats is not None:
            year_payload["stats"] = year_stats
        if year_error is not None:
            year_payload["error"] = year_error
        run_meta.setdefault("years", {})[year_key] = year_payload

    if company_rollup is not None:
        run_meta["totals"] = company_rollup

    if complete_company:
        run_meta["status"] = "completed"
        run_meta["completed"] = True

    with conn.cursor() as cur:
        query = sql.SQL(
            """
            UPDATE {schema}.{companies}
            SET {metadata_col} = %s
            WHERE {id_col} = %s
            """
        ).format(
            schema=sql.Identifier(SCHEMA_NAME),
            companies=sql.Identifier(COMPANIES_TABLE),
            metadata_col=sql.Identifier(COMPANIES_METADATA_COLUMN),
            id_col=sql.Identifier(COMPANIES_ID_COLUMN),
        )
        cur.execute(query, (Json(metadata), company["id"]))

    company["metadata"] = metadata


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


def resolve_media_cloud_key() -> str:
    key = os.getenv(MEDIA_CLOUD_API_KEY_ENV)
    if not key:
        raise ValueError(f"{MEDIA_CLOUD_API_KEY_ENV} is not set in .env")
    return key


def normalize_text(value: Optional[str]) -> str:
    text = (value or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def clean_company_name(name: str) -> str:
    cleaned = (
        name.replace(" Inc.", "")
        .replace(" Corp.", "")
        .replace(" Corporation", "")
        .replace(" Ltd.", "")
        .replace(" Limited", "")
        .replace(" LLC", "")
        .replace(",", "")
    )
    return cleaned.strip()


def make_article_hash(title: Optional[str], url: Optional[str]) -> str:
    payload = f"{normalize_text(title)}|{normalize_text(url)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def title_mentions_company(title: str, company_name: str, company_symbol: str) -> bool:
    title_norm = normalize_text(title)
    company_name_norm = normalize_text(company_name)
    symbol_norm = normalize_text(company_symbol)

    return (company_name_norm and company_name_norm in title_norm) or (
        symbol_norm and symbol_norm in title_norm
    )


def year_date_bounds(year: int) -> Tuple[dt.date, dt.date]:
    return dt.date(year, 1, 1), dt.date(year, 12, 31)


def extract_domain(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        return urlparse(url).netloc.replace("www.", "").strip().lower() or None
    except Exception:
        return None


class SimpleRateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.min_interval = 60.0 / max(1, requests_per_minute)
        self.last_ts = 0.0

    def wait(self) -> None:
        now = time.time()
        elapsed = now - self.last_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_ts = time.time()


def table_exists(conn, schema_name: str, table_name: str) -> bool:
    query = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
        )
    """
    with conn.cursor() as cur:
        cur.execute(query, (schema_name, table_name))
        return bool(cur.fetchone()[0])


def get_target_columns(conn, schema_name: str, table_name: str) -> Set[str]:
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
    """
    with conn.cursor() as cur:
        cur.execute(query, (schema_name, table_name))
        return {row[0] for row in cur.fetchall()}


def get_column_metadata(conn, schema_name: str, table_name: str, column_name: str) -> Optional[Dict[str, str]]:
    query = """
        SELECT is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
          AND column_name = %s
    """
    with conn.cursor() as cur:
        cur.execute(query, (schema_name, table_name, column_name))
        row = cur.fetchone()

    if not row:
        return None

    return {
        "is_nullable": row[0],
        "column_default": row[1],
    }


def requires_manual_id_insert(conn, schema_name: str, table_name: str, target_columns: Set[str]) -> bool:
    if "id" not in target_columns:
        return False

    meta = get_column_metadata(conn, schema_name, table_name, "id")
    if not meta:
        return False

    # If id has a default (typically sequence/identity) or allows nulls, omit it from insert.
    if meta["column_default"] is not None:
        return False
    if (meta["is_nullable"] or "").upper() == "YES":
        return False

    return True


def reserve_id_block(conn, schema_name: str, table_name: str, count: int) -> int:
    if count <= 0:
        return 1

    with conn.cursor() as cur:
        # Serialize id allocation within this table for safety.
        cur.execute(
            sql.SQL("LOCK TABLE {schema}.{table} IN SHARE ROW EXCLUSIVE MODE").format(
                schema=sql.Identifier(schema_name),
                table=sql.Identifier(table_name),
            )
        )
        cur.execute(
            sql.SQL("SELECT COALESCE(MAX(id), 0) + 1 FROM {schema}.{table}").format(
                schema=sql.Identifier(schema_name),
                table=sql.Identifier(table_name),
            )
        )
        start_id = int(cur.fetchone()[0])

    return start_id


def fetch_companies(conn, include_metadata: bool) -> List[Dict]:
    select_columns = [
        sql.Identifier(COMPANIES_ID_COLUMN),
        sql.Identifier(COMPANIES_NAME_COLUMN),
        sql.Identifier(COMPANIES_SYMBOL_COLUMN),
    ]
    if include_metadata:
        select_columns.append(sql.Identifier(COMPANIES_METADATA_COLUMN))

    query = sql.SQL(
        """
        SELECT {select_cols}
        FROM {schema}.{companies}
        WHERE COALESCE(TRIM({name_col}::text), '') <> ''
          AND COALESCE(TRIM({symbol_col}::text), '') <> ''
        ORDER BY {symbol_col}
        """
    ).format(
        select_cols=sql.SQL(", ").join(select_columns),
        name_col=sql.Identifier(COMPANIES_NAME_COLUMN),
        symbol_col=sql.Identifier(COMPANIES_SYMBOL_COLUMN),
        schema=sql.Identifier(SCHEMA_NAME),
        companies=sql.Identifier(COMPANIES_TABLE),
    )

    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    companies = []
    for row in rows:
        company = {"id": row[0], "name": row[1], "symbol": row[2]}
        company["metadata"] = _as_metadata_dict(row[3]) if include_metadata else {}
        companies.append(company)

    return companies


def fetch_existing_hashes(
    conn,
    company_id: int,
    year: int,
) -> Set[str]:
    start_date, end_date = year_date_bounds(year)

    query = sql.SQL(
        """
        SELECT title, url
        FROM {schema}.{table}
        WHERE company_id = %s
          AND published_at >= %s
          AND published_at < %s
        """
    ).format(
        schema=sql.Identifier(SCHEMA_NAME),
        table=sql.Identifier(TARGET_TABLE),
    )

    with conn.cursor() as cur:
        cur.execute(query, (company_id, start_date, end_date + dt.timedelta(days=1)))
        rows = cur.fetchall()

    hashes: Set[str] = set()
    for title, url in rows:
        hashes.add(make_article_hash(title, url))
    return hashes


def build_mc_query(company_name: str, company_symbol: str) -> str:
    cleaned_name = clean_company_name(company_name)
    # Query is broad enough for recall; strict title filter is enforced before insert.
    return f'("{cleaned_name}" OR "{company_symbol}") AND language:en'


def fetch_story_page(
    search_api,
    query: str,
    start_date: dt.date,
    end_date: dt.date,
    collection_id: int,
    pagination_token,
    rate_limiter: SimpleRateLimiter,
):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            rate_limiter.wait()
            return search_api.story_list(
                query,
                start_date=start_date,
                end_date=end_date,
                collection_ids=[collection_id],
                pagination_token=pagination_token,
            )
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            sleep_s = RETRY_BACKOFF_SECONDS * attempt
            print(
                f"  Retry {attempt}/{MAX_RETRIES - 1} after Media Cloud error: {exc} (sleep {sleep_s}s)"
            )
            time.sleep(sleep_s)


def iterate_stories_for_company_year(
    search_api,
    company_name: str,
    company_symbol: str,
    year: int,
    collection_id: int,
    rate_limiter: SimpleRateLimiter,
):
    start_date, end_date = year_date_bounds(year)
    query = build_mc_query(company_name, company_symbol)

    pagination_token = None
    page_idx = 0

    while True:
        page_idx += 1
        page, pagination_token = fetch_story_page(
            search_api,
            query,
            start_date,
            end_date,
            collection_id,
            pagination_token,
            rate_limiter,
        )

        if PRINT_EVERY_N_PAGES > 0 and page_idx % PRINT_EVERY_N_PAGES == 0:
            print(f"    fetched page {page_idx}, page_size={len(page)}")

        yield page

        if pagination_token is None:
            break


def resolve_media_outlet_ids(
    conn,
    domains_with_names: Dict[str, str],
    target_columns: Set[str],
) -> Dict[str, int]:
    if "media_outlet_id" not in target_columns:
        return {}

    if not domains_with_names:
        return {}

    if not table_exists(conn, SCHEMA_NAME, "media_outlets"):
        return {}

    domains = sorted(domains_with_names.keys())

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT domain, id FROM {schema}.media_outlets WHERE domain = ANY(%s)"
            ).format(schema=sql.Identifier(SCHEMA_NAME)),
            (domains,),
        )
        existing = {row[0]: row[1] for row in cur.fetchall()}

        missing = [d for d in domains if d not in existing]
        if missing:
            values = [(d, domains_with_names.get(d) or d) for d in missing]
            execute_values(
                cur,
                sql.SQL(
                    """
                    INSERT INTO {schema}.media_outlets (domain, name)
                    VALUES %s
                    ON CONFLICT (domain) DO NOTHING
                    """
                ).format(schema=sql.Identifier(SCHEMA_NAME)).as_string(conn),
                values,
            )

            cur.execute(
                sql.SQL(
                    "SELECT domain, id FROM {schema}.media_outlets WHERE domain = ANY(%s)"
                ).format(schema=sql.Identifier(SCHEMA_NAME)),
                (missing,),
            )
            for domain, outlet_id in cur.fetchall():
                existing[domain] = outlet_id

    return existing


def build_insert_rows(
    candidates: Sequence[Dict],
    company_id: int,
    outlet_id_map: Dict[str, int],
    target_columns: Set[str],
    include_manual_id: bool,
    id_start: Optional[int],
) -> Tuple[List[str], List[Tuple]]:
    preferred_order = [
        "id",
        "title",
        "content",
        "url",
        "source",
        "published_at",
        "media_outlet_id",
        "company_id",
        "social_data",
        "raw_content",
    ]

    insert_columns = [c for c in preferred_order if c in target_columns]
    if not include_manual_id:
        insert_columns = [c for c in insert_columns if c != "id"]

    rows: List[Tuple] = []
    for idx, art in enumerate(candidates):
        domain = extract_domain(art.get("url"))
        row_map = {
            "id": (id_start + idx) if (include_manual_id and id_start is not None) else None,
            "title": art.get("title"),
            "content": None,
            "url": art.get("url"),
            "source": art.get("media_name") or domain,
            "published_at": art.get("publish_date"),
            "media_outlet_id": outlet_id_map.get(domain),
            "company_id": company_id,
            "social_data": Json(
                {
                    "language": art.get("language"),
                    "media_cloud_id": art.get("id"),
                }
            ),
            "raw_content": None,
        }
        rows.append(tuple(row_map[col] for col in insert_columns))

    return insert_columns, rows


def insert_articles(
    conn,
    candidates: Sequence[Dict],
    company_id: int,
    target_columns: Set[str],
) -> int:
    if not candidates:
        return 0

    domains_with_names = {}
    for art in candidates:
        domain = extract_domain(art.get("url"))
        if domain:
            domains_with_names[domain] = art.get("media_name") or domain

    outlet_id_map = resolve_media_outlet_ids(conn, domains_with_names, target_columns)

    manual_id_mode = requires_manual_id_insert(conn, SCHEMA_NAME, TARGET_TABLE, target_columns)
    id_start = reserve_id_block(conn, SCHEMA_NAME, TARGET_TABLE, len(candidates)) if manual_id_mode else None

    insert_columns, rows = build_insert_rows(
        candidates,
        company_id,
        outlet_id_map,
        target_columns,
        include_manual_id=manual_id_mode,
        id_start=id_start,
    )

    if not insert_columns:
        raise ValueError(
            f"No insertable columns found in {SCHEMA_NAME}.{TARGET_TABLE}. "
            "Expected at least one of title/url/source/published_at/company_id."
        )

    insert_sql = sql.SQL("INSERT INTO {schema}.{table} ({cols}) VALUES %s").format(
        schema=sql.Identifier(SCHEMA_NAME),
        table=sql.Identifier(TARGET_TABLE),
        cols=sql.SQL(", ").join([sql.Identifier(c) for c in insert_columns]),
    )

    with conn.cursor() as cur:
        execute_values(cur, insert_sql.as_string(conn), rows, page_size=200)

    return len(rows)


def augment_company_year(
    conn,
    search_api,
    company: Dict,
    year: int,
    collection_id: int,
    rate_limiter: SimpleRateLimiter,
    target_columns: Set[str],
) -> Dict[str, int]:
    company_id = company["id"]
    company_name = company["name"]
    company_symbol = company["symbol"]

    existing_hashes = fetch_existing_hashes(conn, company_id, year)
    seen_hashes = set(existing_hashes)

    fetched = 0
    title_filtered_out = 0
    duplicate_filtered_out = 0
    to_insert: List[Dict] = []

    for page in iterate_stories_for_company_year(
        search_api,
        company_name,
        company_symbol,
        year,
        collection_id,
        rate_limiter,
    ):
        for story in page:
            fetched += 1
            title = (story.get("title") or "").strip()
            url = (story.get("url") or "").strip()
            publish_date = story.get("publish_date")

            if not title or not url:
                continue

            if not title_mentions_company(title, company_name, company_symbol):
                title_filtered_out += 1
                continue

            h = make_article_hash(title, url)
            if h in seen_hashes:
                duplicate_filtered_out += 1
                continue

            seen_hashes.add(h)
            to_insert.append(
                {
                    "id": story.get("id"),
                    "title": title,
                    "url": url,
                    "publish_date": publish_date,
                    "media_name": story.get("media_name"),
                    "language": story.get("language"),
                }
            )

    inserted = 0
    if not DRY_RUN and to_insert:
        inserted = insert_articles(conn, to_insert, company_id, target_columns)

    return {
        "fetched": fetched,
        "existing": len(existing_hashes),
        "title_filtered_out": title_filtered_out,
        "duplicate_filtered_out": duplicate_filtered_out,
        "candidate_new": len(to_insert),
        "inserted": inserted,
    }


def main() -> None:
    db_url = resolve_db_url()
    api_key = resolve_media_cloud_key()

    if COLLECTION_MODE not in COLLECTIONS:
        raise ValueError(
            f"Invalid COLLECTION_MODE={COLLECTION_MODE}. Use one of {list(COLLECTIONS.keys())}."
        )
    collection_id = COLLECTIONS[COLLECTION_MODE]

    print("Starting Media Cloud augmentation pipeline")
    print(f"Schema/Table: {SCHEMA_NAME}.{TARGET_TABLE}")
    print(f"Companies table: {SCHEMA_NAME}.{COMPANIES_TABLE}")
    print(f"Collection: {COLLECTION_MODE} ({collection_id})")
    print(f"Years: {START_YEAR}..{END_YEAR}")
    print(f"Rate limit: {REQUESTS_PER_MINUTE} requests/min")
    print(f"Dry run: {DRY_RUN}")
    print(f"Rich logging active: {should_use_rich()}")

    with psycopg2.connect(db_url) as conn:
        conn.autocommit = False

        if not table_exists(conn, SCHEMA_NAME, TARGET_TABLE):
            raise ValueError(f"Target table does not exist: {SCHEMA_NAME}.{TARGET_TABLE}")
        if not table_exists(conn, SCHEMA_NAME, COMPANIES_TABLE):
            raise ValueError(f"Companies table does not exist: {SCHEMA_NAME}.{COMPANIES_TABLE}")

        company_columns = get_target_columns(conn, SCHEMA_NAME, COMPANIES_TABLE)
        has_metadata_column = COMPANIES_METADATA_COLUMN in company_columns
        run_key = build_run_key()

        target_columns = get_target_columns(conn, SCHEMA_NAME, TARGET_TABLE)
        companies = fetch_companies(conn, include_metadata=has_metadata_column)

        if has_metadata_column and SKIP_COMPLETED_COMPANIES and not should_force_rerun():
            filtered_companies = []
            skipped = 0
            for company in companies:
                if is_company_completed_for_run(company.get("metadata", {}), run_key):
                    skipped += 1
                    continue
                filtered_companies.append(company)
            companies = filtered_companies
            if skipped > 0:
                print(f"Skipping {skipped} companies already marked completed for run {run_key}")

        if not companies:
            print("No companies found. Exiting.")
            return

        search_api = mediacloud.api.SearchApi(api_key)
        rate_limiter = SimpleRateLimiter(REQUESTS_PER_MINUTE)

        cells_total = len(companies) * (END_YEAR - START_YEAR + 1)
        state = {
            "status": "Running",
            "current_company": "-",
            "current_year": "-",
            "companies_done": 0,
            "companies_total": len(companies),
            "cells_done": 0,
            "cells_total": cells_total,
            "grand_fetched": 0,
            "grand_inserted": 0,
            "recent_cells": deque(maxlen=12),
            "recent_errors": deque(maxlen=8),
        }

        use_rich = should_use_rich()

        if use_rich:
            with Live(render_live_dashboard(state), refresh_per_second=4) as live:
                for idx, company in enumerate(companies, start=1):
                    company_rollup = {
                        "cells": 0,
                        "fetched": 0,
                        "existing": 0,
                        "title_filtered_out": 0,
                        "duplicate_filtered_out": 0,
                        "candidate_new": 0,
                        "inserted": 0,
                    }

                    state["current_company"] = company["symbol"]
                    state["current_year"] = START_YEAR
                    live.update(render_live_dashboard(state))

                    for year in range(START_YEAR, END_YEAR + 1):
                        state["current_year"] = year
                        live.update(render_live_dashboard(state))
                        try:
                            stats = augment_company_year(
                                conn,
                                search_api,
                                company,
                                year,
                                collection_id,
                                rate_limiter,
                                target_columns,
                            )
                            if has_metadata_column and not DRY_RUN:
                                upsert_company_metadata(
                                    conn,
                                    company,
                                    run_key,
                                    year=year,
                                    year_status="success",
                                    year_stats=stats,
                                )
                            if not DRY_RUN:
                                conn.commit()

                            state["cells_done"] += 1
                            state["grand_fetched"] += stats["fetched"]
                            state["grand_inserted"] += stats["inserted"]
                            company_rollup["cells"] += 1
                            company_rollup["fetched"] += stats["fetched"]
                            company_rollup["existing"] += stats["existing"]
                            company_rollup["title_filtered_out"] += stats["title_filtered_out"]
                            company_rollup["duplicate_filtered_out"] += stats["duplicate_filtered_out"]
                            company_rollup["candidate_new"] += stats["candidate_new"]
                            company_rollup["inserted"] += stats["inserted"]

                            state["recent_cells"].appendleft(
                                {
                                    "symbol": company["symbol"],
                                    "year": year,
                                    **stats,
                                }
                            )
                            live.console.print(
                                f"CELL {company['symbol']} {year} | "
                                f"fetched={stats['fetched']} existing={stats['existing']} "
                                f"title_out={stats['title_filtered_out']} dup_out={stats['duplicate_filtered_out']} "
                                f"candidate={stats['candidate_new']} inserted={stats['inserted']}"
                            )
                        except Exception as exc:
                            conn.rollback()
                            state["cells_done"] += 1
                            state["recent_errors"].appendleft(
                                f"{company['symbol']} {year}: {str(exc)[:180]}"
                            )
                            if has_metadata_column and not DRY_RUN:
                                try:
                                    upsert_company_metadata(
                                        conn,
                                        company,
                                        run_key,
                                        year=year,
                                        year_status="error",
                                        year_error=str(exc)[:500],
                                    )
                                    conn.commit()
                                except Exception:
                                    conn.rollback()
                            live.console.print(f"ERROR {company['symbol']} {year}: {exc}")

                        live.update(render_live_dashboard(state))

                    state["companies_done"] = idx
                    if has_metadata_column and not DRY_RUN:
                        try:
                            upsert_company_metadata(
                                conn,
                                company,
                                run_key,
                                complete_company=True,
                                company_rollup=company_rollup,
                            )
                            conn.commit()
                        except Exception as meta_exc:
                            conn.rollback()
                            state["recent_errors"].appendleft(
                                f"{company['symbol']} meta finalize: {str(meta_exc)[:180]}"
                            )
                    print_company_summary(company["symbol"], company_rollup)
                    live.update(render_live_dashboard(state))

                state["status"] = "Completed"
                live.update(render_live_dashboard(state))
        else:
            for idx, company in enumerate(companies, start=1):
                company_rollup = {
                    "cells": 0,
                    "fetched": 0,
                    "existing": 0,
                    "title_filtered_out": 0,
                    "duplicate_filtered_out": 0,
                    "candidate_new": 0,
                    "inserted": 0,
                }

                state["current_company"] = company["symbol"]
                print(f"\n[{idx}/{len(companies)}] Company {company['symbol']} - {company['name']}")

                for year in range(START_YEAR, END_YEAR + 1):
                    state["current_year"] = year
                    try:
                        stats = augment_company_year(
                            conn,
                            search_api,
                            company,
                            year,
                            collection_id,
                            rate_limiter,
                            target_columns,
                        )
                        if has_metadata_column and not DRY_RUN:
                            upsert_company_metadata(
                                conn,
                                company,
                                run_key,
                                year=year,
                                year_status="success",
                                year_stats=stats,
                            )
                        if not DRY_RUN:
                            conn.commit()

                        state["cells_done"] += 1
                        state["grand_fetched"] += stats["fetched"]
                        state["grand_inserted"] += stats["inserted"]
                        company_rollup["cells"] += 1
                        company_rollup["fetched"] += stats["fetched"]
                        company_rollup["existing"] += stats["existing"]
                        company_rollup["title_filtered_out"] += stats["title_filtered_out"]
                        company_rollup["duplicate_filtered_out"] += stats["duplicate_filtered_out"]
                        company_rollup["candidate_new"] += stats["candidate_new"]
                        company_rollup["inserted"] += stats["inserted"]

                        print_cell_stats(company["symbol"], year, stats)
                    except Exception as exc:
                        conn.rollback()
                        state["cells_done"] += 1
                        state["recent_errors"].appendleft(
                            f"{company['symbol']} {year}: {str(exc)[:180]}"
                        )
                        if has_metadata_column and not DRY_RUN:
                            try:
                                upsert_company_metadata(
                                    conn,
                                    company,
                                    run_key,
                                    year=year,
                                    year_status="error",
                                    year_error=str(exc)[:500],
                                )
                                conn.commit()
                            except Exception:
                                conn.rollback()
                        print(f"ERROR {company['symbol']} {year}: {exc}")

                    # Plain live stats snapshot for nohup-friendly logs.
                    print(
                        f"LIVE cells={state['cells_done']}/{state['cells_total']} "
                        f"companies={state['companies_done']}/{state['companies_total']} "
                        f"fetched_total={state['grand_fetched']} inserted_total={state['grand_inserted']}"
                    )

                state["companies_done"] = idx
                if has_metadata_column and not DRY_RUN:
                    try:
                        upsert_company_metadata(
                            conn,
                            company,
                            run_key,
                            complete_company=True,
                            company_rollup=company_rollup,
                        )
                        conn.commit()
                    except Exception as meta_exc:
                        conn.rollback()
                        state["recent_errors"].appendleft(
                            f"{company['symbol']} meta finalize: {str(meta_exc)[:180]}"
                        )
                print_company_summary(company["symbol"], company_rollup)

            state["status"] = "Completed"

        print("\nDone.")
        print(f"Total fetched from Media Cloud: {state['grand_fetched']}")
        print(f"Total inserted into {SCHEMA_NAME}.{TARGET_TABLE}: {state['grand_inserted']}")


if __name__ == "__main__":
    main()
