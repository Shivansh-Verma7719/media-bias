create table public.articles_no_title_deduped (
  id integer not null,
  title text null,
  content text null,
  url text null,
  source text null,
  published_at timestamp without time zone null,
  media_outlet_id integer null,
  company_id integer null,
  social_data jsonb null,
  created_at timestamp without time zone null,
  content_scraped boolean null,
  last_scraped_at timestamp without time zone null,
  scraping_retry_count integer null,
  scraping_error text null,
  raw_content text null,
  pos_score double precision null,
  neutral_score double precision null,
  neg_score double precision null,
  link_health text null,
  link_ok boolean null,
  metadata jsonb null,
  constraint articles_no_title_deduped_pkey primary key (id),
  constraint articles_no_title_deduped_company_id_fkey foreign KEY (company_id) references top_companies (id)
) TABLESPACE pg_default;

create index IF not exists articles_no_title_deduped_company_id_published_at_idx on public.articles_no_title_deduped using btree (company_id, published_at desc) TABLESPACE pg_default
where
  (pos_score is null);