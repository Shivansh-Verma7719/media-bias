create table public.top_companies (
  id integer not null,
  name text null,
  symbol text null,
  current_page integer null,
  is_processed boolean null,
  last_updated timestamp without time zone null,
  created_at timestamp without time zone null,
  article_count bigint null,
  sector text null,
  exposure numeric null,
  constraint top_companies_pkey primary key (id)
) TABLESPACE pg_default;