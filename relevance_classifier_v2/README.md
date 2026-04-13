# Relevance Classifier v2

Binary classifier identifying whether a news article is **materially relevant**
to an investor holding stock in the named S&P 500 company.

---

## Definition of Relevance

An article is **RELEVANT** only if ALL three conditions hold:

1. The named company is the **PRIMARY subject** (not mentioned in passing)
2. The article covers a **material corporate event**:
   - Earnings, revenue, guidance, dividends, buybacks
   - M&A, major contracts, partnerships, divestitures
   - Regulatory/legal actions against the company as an institution
   - Executive appointments/departures (CEO/CFO/board level)
   - Major layoffs, restructuring, plant closures
   - Major product launches or discontinuations (not minor features)
   - Labor strikes or disputes with operational impact
   - Credit ratings, debt issuance, bankruptcy
3. It is **NOT** any of the following:
   - Individual employee/driver/customer incident or crime
   - Entertainment, lifestyle, or consumer content
   - Unconfirmed rumour or product leak
   - Consumer shopping guide, deal, or price comparison
   - Macro/political article where company is one example among many
   - Analyst firm making a market prediction (not corporate subject)
   - Social media trend with no documented business impact
   - Minor app feature or UI update

**When in doubt → IRRELEVANT.**

---

## Pipeline

### Step 0 — Fetch Filtered Articles from Supabase
```bash
python 00_fetch_filtered.py -o 00_filtered_clean.csv --per_company 500
```
- Requires DB credentials in `.env` (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`)
- Filters: company name must appear in article title (entity alignment fix)
- Excludes known-contaminated companies (Boeing)
- Deduplicates near-identical headlines (TF-IDF cosine ≥ 0.85)

### Step 1 — LLM Annotation
```bash
python 01_annotate.py \
    -i 00_filtered_clean.csv \
    -o 01_annotated.csv \
    -k <CEREBRAS_API_KEY>
```
- Uses `llama3.1-8b` via Cerebras API (60k tokens/min → 4.5s sleep between batches)
- 40 few-shot examples covering every failure mode
- Checkpoint/resume supported

### Step 1b — Audit Irrelevant Class (QA, run once, report in paper)
```bash
python 01_audit_irrelevant.py \
    -i 04_training_data_verified.csv \
    -o irrelevant_audit.csv \
    -k <CEREBRAS_API_KEY> \
    -n 300
```
- Samples 300 irrelevant articles stratified by company
- Reports false-negative rate — document this number in the paper
- Action threshold: FN > 5% requires cleaning irrelevant class

### Step 2 — Human Gold Set Annotation
```bash
# First annotator
python 02_annotate_gold.py -i unlabeled_pool.csv -o gold_annotator1.csv -a your_name

# Second annotator (same input, different output)
python 02_annotate_gold.py -i unlabeled_pool.csv -o gold_annotator2.csv -a second_name

# Compute inter-annotator agreement (κ > 0.70 required for paper)
python 02_annotate_gold.py --iaa gold_annotator1.csv gold_annotator2.csv
```
- Displays full rubric in terminal before each article
- Progress saved after every label (safe to interrupt with Ctrl+C)
- Target: 500 train + 200 test gold labels

### Step 3 — Build Training Set
```bash
python 04_build_training_set.py \
    --llm 01_annotated.csv \
    --gold 300_train.csv \
    --output 04_training_data.csv
```
- Applies post-hoc rule filters (consumer deals, individual crimes, lifestyle, wrong-entity)
- Drops uncertain LLM labels (hedging language in reason field)
- Per-company cap: 200 irrelevant / 67 relevant

### Step 4 — Second-Pass Verification of Relevant Articles
```bash
python 05_verify_relevant.py \
    -i 04_training_data.csv \
    -o 04_training_data_verified.csv \
    -k <CEREBRAS_API_KEY>
```
- Strict CONFIRM/REJECT binary check on all LLM-labeled relevant articles
- Gold manual rows are never modified
- Defaults to REJECT for any failed/missed batch

### Step 5 — Train FinBERT Gold-Only Baseline
```bash
# Run on GPU (VM)
python 03_train_finbert.py \
    -i 04_training_data_verified.csv \
    -s model_gold_only \
    --gold_only
```
- Trains only on `source == gold_manual` rows (~240 rows)
- Establishes a clean baseline F1 to report in the paper

### Step 6 — Semi-Supervised Expansion
```bash
python 06_semisup_expand.py \
    --model     model_gold_only \
    --gold      04_training_data_verified.csv \
    --unlabeled 00_filtered_clean.csv \
    --output    06_semisup_training.csv \
    --threshold 0.90

# Re-train with expanded set
python 03_train_finbert.py \
    -i 06_semisup_training.csv \
    -s model_semisup
```
- Runs gold model on unlabeled pool; keeps p ≥ 0.90 as pseudo-labels
- Standard self-training / pseudo-labeling approach (citable in paper)
- This is the model used for full inference

### Step 7 — Evaluate on Held-Out Test Set
```bash
python 04_evaluate.py --model_path model_semisup --test 300_test.csv
```
- Reports F1, precision, recall on gold test set
- Lists all false positives and false negatives for analysis

### Step 8 — Full Inference on 483k Articles
```bash
python 05_full_inference.py --model model_semisup --output full_predictions_sp500.csv
```

---

## Known Data Quality Issues (documented for paper)

| Company   | Issue | Fix Applied |
|-----------|-------|-------------|
| Boeing    | DB tagging corrupted — Indian politics/Bollywood articles tagged to Boeing | Excluded entirely from training |
| AT&T      | Media Cloud returns articles mentioning AT&T in body (CenturyLink/CFTC contamination) | Company-name-in-title filter in `00_fetch_filtered.py` |
| Visa Inc. | AmEx articles tagged to Visa when body mentions "visa card" | Company-name-in-title filter |

Root cause: Media Cloud full-text search returns articles where the company name
appears anywhere in the body. We store only titles. The entity filter requires
the company name to appear in the headline, ensuring title-level entity alignment.

---

## Methodology Statement (for paper)

> "We constructed a gold-standard dataset of N articles annotated by two raters
> (Cohen's κ = X) following a formal relevance rubric. Articles were sourced from
> Supabase using a company-name-in-title filter to ensure entity alignment at the
> headline level. A FinBERT model was fine-tuned on the gold set using stratified
> 5-fold cross-validation with class-balanced loss, achieving baseline F1 = Y.
> The training set was then expanded via self-training: the baseline model was
> applied to an unlabeled pool of M articles, and predictions with confidence
> ≥ 0.90 were retained as pseudo-labels. The final model achieved F1 = Z on a
> held-out test set of 200 gold annotations."

---

## File Inventory

| File | Description |
|------|-------------|
| `00_fetch_filtered.py` | Fetch from Supabase with entity + dedup filters |
| `00_filtered_clean.csv` | Filtered article pool (step 0 output) |
| `01_annotate.py` | LLM annotation with 40 few-shot examples |
| `01_annotated.csv` | LLM-labeled articles |
| `01_audit_irrelevant.py` | QA: measure false-negative rate in irrelevant class |
| `02_annotate_gold.py` | Interactive human annotation tool with rubric + IAA |
| `03_train_finbert.py` | FinBERT fine-tuning (gold-only baseline + full mode) |
| `04_build_training_set.py` | Combine gold + LLM with post-hoc filters |
| `04_training_data_verified.csv` | Current best training set (use as starting point) |
| `04_evaluate.py` | Evaluate trained model on held-out test set |
| `05_verify_relevant.py` | Second-pass CONFIRM/REJECT on LLM-relevant articles |
| `05_full_inference.py` | Run model on full 483k article corpus |
| `06_semisup_expand.py` | Semi-supervised expansion via pseudo-labeling |
| `300_train.csv` | Gold training split (240 rows, always included) |
| `300_test.csv` | Gold test split (60 rows, NEVER used during training) |
