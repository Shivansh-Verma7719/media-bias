create table public.stock_prices (
  id serial not null,
  ticker text not null,
  date timestamp without time zone not null,
  open numeric(20, 10) not null,
  high numeric(20, 10) not null,
  low numeric(20, 10) not null,
  close numeric(20, 10) not null,
  volume bigint not null,
  created_at timestamp without time zone null default CURRENT_TIMESTAMP,
  constraint stock_prices_pkey primary key (id),
  constraint stock_prices_ticker_date_key unique (ticker, date),
  constraint fk_stock_prices_ticker foreign KEY (ticker) references companies (symbol)
) TABLESPACE pg_default;