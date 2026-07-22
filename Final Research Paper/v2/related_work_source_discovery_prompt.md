# Source-discovery prompt — find Related Work papers

Use with a **search-capable** model (Claude / ChatGPT / Perplexity / Gemini with web, or
a deep-research tool). NotebookLM cannot do this — it only reads what you upload.

---

I am writing the Related Work section of a quantitative-finance / NLP research paper and
need you to **find 25–30 real, verifiable academic papers** I can cite. Search the web
and only return papers that actually exist; for each one give enough detail that I can
verify it (authors, exact title, venue, year, and a **DOI or working URL**). Do not
invent papers, and do not pad the list with sources you cannot confirm.

## What the paper is about

I study the **tone of news headlines about large S&P 500 firms** over 2015–2025 and how
it relates to stock returns around the **2020 (COVID) shock**. Unlike work that reads a
single market-wide sentiment index, I score each headline with a **target-dependent
(aspect-level) sentiment model** that measures the tone directed at the **specific firm**
the headline names, then build a daily firm-level stance signal. I test three things:

- **RQ1** — did firm-level media stance shift after 2020? (two-way fixed-effects panel)
- **RQ2** — did firm-level returns shift after 2020? (two-way fixed-effects panel with
  S&P 500 and VIX controls)
- **RQ3** — did the **stance ↔ return relationship** change? (weekly bivariate VAR with
  Granger causality tested in **both directions**, stationarity checks, and **empirically
  dated structural breaks** rather than an imposed 2020 date)

## Why this matters / the gap I'm positioning against

Most sentiment–market studies work at the **aggregate or single-outlet** level, so they
cannot attribute tone to the individual firm a headline is about. Firm-level attribution
is hard, and language-model sentiment is known to be sensitive to model and prompt
choice. And almost no one tests the **bidirectional lead–lag** relation between
firm-level stance and firm-level returns on **stationarity-verified series with estimated
break dates**. My sources need to support and motivate exactly these points.

## Find sources in three groups (8–12 each, **mostly 2022 or later**)

1. **Media sentiment and market outcomes** — news or social-media sentiment predicting
   market/returns; COVID-era news–market dynamics; aggregate/market-level sentiment
   indices; cross-channel (news vs. social) and cross-horizon studies.
2. **Firm-level / target-dependent sentiment** — aspect-based or entity-level sentiment;
   finance-domain language models (FinBERT-type); and the caution literature on how
   LM-measured sentiment shifts with model or prompt.
3. **Dynamic links and structural change** — media–return feedback / Granger causality /
   direction-testing; structural breaks and regime shifts around 2020; econometrics of
   inference under breaks.

## Output

For each paper, one line:
`Group # — Authors, "Exact title," Venue/Journal, Year. DOI or URL. — one clause on what
it did/found and which of my points it supports.`

A few foundational method papers (e.g. target-dependent stance methods) may be older than
2022 if essential; flag those. Prioritize peer-reviewed and well-cited work.
