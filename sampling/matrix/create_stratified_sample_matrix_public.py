import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv


def _resolve_db_url() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sampling_dir = os.path.dirname(current_dir)
    project_dir = os.path.dirname(sampling_dir)
    env_path = os.path.join(project_dir, '.env')

    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        print(f"Warning: .env file not found at {env_path}")

    db_url = os.getenv('POOLER_DATABASE_URL') or os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError('Neither POOLER_DATABASE_URL nor DATABASE_URL found in .env')

    return db_url


def main() -> None:
    db_url = _resolve_db_url()

    print('Connecting to database...')
    conn = psycopg2.connect(db_url)

    diagnostics_query = """
    WITH sample_filtered AS (
        SELECT
            a.id,
            a.company_id,
            LOWER(TRIM(REGEXP_REPLACE(COALESCE(a.title, ''), '\\s+', ' ', 'g'))) AS title_norm,
            a.published_at
        FROM public.articles_sample a
        JOIN public.top_companies c ON a.company_id = c.id
        WHERE a.published_at IS NOT NULL
          AND COALESCE(TRIM(a.title), '') <> ''
          AND (
                LOWER(a.title) LIKE '%' || LOWER(c.name) || '%'
                OR LOWER(a.title) LIKE '%' || LOWER(c.symbol) || '%'
          )
    ),
    sample_deduped AS (
        SELECT
            id,
            ROW_NUMBER() OVER (
                PARTITION BY company_id, title_norm
                ORDER BY published_at DESC, id DESC
            ) AS rn
        FROM sample_filtered
    ),
    strat_filtered AS (
        SELECT
            a.id,
            a.company_id,
            LOWER(TRIM(REGEXP_REPLACE(COALESCE(a.title, ''), '\\s+', ' ', 'g'))) AS title_norm,
            a.published_at
        FROM public.articles_stratified a
        JOIN public.top_companies c ON a.company_id = c.id
        WHERE a.published_at IS NOT NULL
          AND COALESCE(TRIM(a.title), '') <> ''
          AND (
                LOWER(a.title) LIKE '%' || LOWER(c.name) || '%'
                OR LOWER(a.title) LIKE '%' || LOWER(c.symbol) || '%'
          )
    ),
    strat_deduped AS (
        SELECT
            id,
            ROW_NUMBER() OVER (
                PARTITION BY company_id, title_norm
                ORDER BY published_at DESC, id DESC
            ) AS rn
        FROM strat_filtered
    )
    SELECT
        (SELECT COUNT(*) FROM public.articles_sample) AS sample_total,
        (SELECT COUNT(*) FROM public.articles_stratified) AS stratified_total,
        (SELECT COUNT(*) FROM sample_deduped WHERE rn = 1) AS sample_filtered_deduped,
        (SELECT COUNT(*) FROM strat_deduped WHERE rn = 1) AS stratified_filtered_deduped;
    """

    diag_cur = conn.cursor()
    diag_cur.execute(diagnostics_query)
    sample_total, stratified_total, sample_filtered_deduped, stratified_filtered_deduped = diag_cur.fetchone()
    diag_cur.close()

    print('Diagnostics:')
    print(f'  articles_sample total rows: {sample_total}')
    print(f'  articles_stratified total rows: {stratified_total}')
    print(f'  sample after filter+dedupe: {sample_filtered_deduped}')
    print(f'  stratified after filter+dedupe: {stratified_filtered_deduped}')

    query = """
    WITH filtered AS (
        SELECT
            a.id,
            a.company_id,
            c.symbol,
            c.name AS company_name,
            CAST(EXTRACT(YEAR FROM a.published_at) AS INTEGER) AS year,
            LOWER(TRIM(REGEXP_REPLACE(COALESCE(a.title, ''), '\\s+', ' ', 'g'))) AS title_norm,
            a.published_at
        FROM public.articles_stratified a
        JOIN public.top_companies c
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
    ),
    yearly_counts AS (
        SELECT
            symbol,
            year,
            COUNT(*) AS article_count
        FROM deduped
        WHERE rn = 1
        GROUP BY symbol, year
    )
    SELECT symbol, year, article_count
    FROM yearly_counts
    ORDER BY symbol, year;
    """

    print('Fetching filtered and deduplicated yearly counts...')
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        print('No rows found after filtering.')
        return

    matrix = df.pivot(index='symbol', columns='year', values='article_count').fillna(0).astype(int)

    year_cols = sorted([c for c in matrix.columns], key=int)
    matrix = matrix[year_cols]

    matrix['Total'] = matrix.sum(axis=1)
    matrix = matrix.sort_values(by='Total', ascending=False)

    total_row = matrix.sum(axis=0)
    total_row.name = 'Total'
    matrix = pd.concat([matrix, total_row.to_frame().T])

    matrix.columns = [str(c) if c != 'Total' else 'Total' for c in matrix.columns]
    matrix.index.name = None

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stratified_sample_matrix_public_filtered.csv')
    matrix.to_csv(output_path)

    print(f'Successfully generated matrix at: {output_path}')
    print(f'Rows (including Total): {len(matrix)}')
    print('Preview:')
    print(matrix.head())


if __name__ == '__main__':
    main()
