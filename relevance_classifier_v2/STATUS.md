# Relevance Classifier v2 — Ground Truth Status

> **READ THIS FIRST before making any changes.**
> Last updated: 2026-04-12

---

## What We Are Building

Binary classifier: given a news article title + company name, predict whether the article is **materially relevant** to an investor holding that S&P 500 stock.

**Relevance definition (strict):**
- Company is the PRIMARY subject (not passing mention)
- Covers a MATERIAL corporate event (earnings, M&A, executive departure, major layoff, regulatory action, major product launch, labor strike, credit/bankruptcy)
- NOT: individual incidents, consumer content, product leaks, macro/political pieces where company is one example, analyst firms making predictions (vs being the subject)
- When in doubt → IRRELEVANT

---

## Current Data Inventory

| File | Rows | Relevant | Irrelevant | Notes |
|------|------|----------|------------|-------|
| `300_train.csv` | 240 | 37 | 203 | Gold manual (corrected 2026-04-12: fixed 16 mislabels) |
| `gold_a1.csv` | 310 | 138 | 172 | Claude-annotated gold (2026-04-12) |
| `combined_gold.csv` | 550 | 175 (31.8%) | 375 (68.2%) | = 300_train + gold_a1, Apple normalized |
| `300_test.csv` | 60 | 13 | 47 | **HELD OUT — never train on this** |
| `00_filtered_clean.csv` | 14,934 | unlabeled | unlabeled | Supabase fetch, entity-filtered, deduped |

**Companies with 0 relevant training examples:** Tesla, McDonald's, Bank of America, Morgan Stanley  
**True relevant rate in the wild:** ~10–15% (training set is 31.8% — miscalibrated)

---

## What Has Been Tried

### Attempt 1: Semi-supervised (FAILED — do not repeat)
- Trained gold-only model → ran inference on 00_filtered_clean.csv → kept p≥0.90 as pseudo-labels
- Result: 51.5% relevant in pseudo-labels (confirmation bias from miscalibrated model)
- model_semisup performed WORSE than gold-only on test set
- **Script: `06_semisup_expand.py` — DO NOT USE**

### Attempt 2: Gold-only FinBERT (current best)
- Trained on `combined_gold.csv` (550 rows), 5-fold CV, 4 epochs
- CV Mean F1: 0.6733 ± 0.0404
- Test set results (300_test.csv, 60 rows):
  - Overall accuracy: 76.5% on scored (9/60 uncertain)
  - Relevant precision: **0.444** — unacceptable
  - Relevant recall: 0.800
  - Relevant F1: 0.571
- Root cause: too little data, training class balance (32%) ≠ true rate (~10–15%)

---

## Root Cause of Low Precision

1. **Too few training examples** — 550 total, 175 relevant. Model doesn't generalize.
2. **Wrong class balance** — 32% relevant in training vs ~10–15% reality. Balanced loss over-compensates, model predicts relevant too aggressively.
3. **4 companies with 0 relevant examples** — model has never seen what "relevant" looks like for Tesla, McDonald's, Bank of America, Morgan Stanley.

---

## Correct Path Forward

### Step 1: LLM-annotate the full filtered pool (run on VM)
```bash
python 01_annotate.py \
    -i 00_filtered_clean.csv \
    -o 01_annotated_full.csv \
    -k <CEREBRAS_API_KEY>
```
- The current prompt in `01_annotate.py` is strict and good (40 examples, tie-breaking rule: "when uncertain → IRRELEVANT")
- Expected output: ~10–15% relevant (~1,500–2,200 relevant articles)
- Previous LLM annotation failed because the OLD prompt was weak (gave 50/50). This prompt is different.
- Checkpoint/resume supported — safe to interrupt

### Step 2: Second-pass verification of LLM relevant labels
```bash
python 05_verify_relevant.py \
    -i 01_annotated_full.csv \
    -o 01_annotated_verified.csv \
    -k <CEREBRAS_API_KEY>
```
- Runs strict CONFIRM/REJECT on every article the LLM labeled relevant
- Defaults to REJECT if batch fails — conservative

### Step 3: Build combined training set
```bash
python 04_build_training_set.py \
    --llm 01_annotated_verified.csv \
    --gold combined_gold.csv \
    --output training_final.csv \
    --company_cap 300
```
- Gold rows always included in full
- LLM rows filtered (post-hoc rules, uncertainty markers, company caps)
- Expected total: ~10,000–14,000 rows at ~10–15% relevant

### Step 4a: Train DeBERTa-v3-base (primary — stronger architecture)
```bash
python 03_train_finbert.py \
    -i combined_gold.csv \
    -s model_deberta \
    --base_model microsoft/deberta-v3-base \
    --gold_only --epochs 6
```

### Step 4b: Train FinBERT (for comparison)
```bash
python 03_train_finbert.py \
    -i combined_gold.csv \
    -s model_finbert \
    --base_model ProsusAI/finbert \
    --gold_only --epochs 6
```
- 6 epochs (was 4 — model still learning at epoch 4)
- Class weights capped at 1.5:1 (was fully balanced 2:1 — caused over-prediction)
- Script is now model-agnostic via --base_model argument

### Step 5: Evaluate
```bash
python 04_evaluate.py --model_path model_final --test 300_test.csv
```
- Uses asymmetric threshold: P(relevant) ≥ 0.75 to predict relevant (standard ML practice for imbalanced classes — not a rule hack)
- Target: relevant precision ≥ 0.90, recall ≥ 0.80, overall accuracy ≥ 90%

---

## Pipeline Scripts (what each does)

| Script | Status | Purpose |
|--------|--------|---------|
| `00_fetch_filtered.py` | ✅ Use | Fetch from Supabase with entity filter + dedup |
| `01_annotate.py` | ✅ Use | LLM annotation with strict prompt (Cerebras/Groq) |
| `04_build_training_set.py` | ✅ Use | Combine gold + LLM with filters and company caps |
| `03_train_finbert.py` | ✅ Use | FinBERT fine-tuning with k-fold CV |
| `04_evaluate.py` | ✅ Use | Evaluate on held-out test set |
| `05_verify_relevant.py` | ✅ Use | Second-pass LLM verification of relevant labels |
| `05_full_inference.py` | ✅ Use | Full inference on 700k+ article corpus |
| `02_annotate_gold.py` | ✅ Use | Interactive human annotation tool |
| `01_audit_irrelevant.py` | ✅ Use | QA tool for false-negative rate |
| `07_sample_for_annotation.py` | ✅ Use | Samples articles for manual annotation, weighted by underrepresented companies |
| `06_semisup_expand.py` | ❌ Abandoned | Semi-supervised failed — do not use |
| `02_prepare_training_data.py` | ❌ Redundant | Replaced by `04_build_training_set.py` |
| `write_gold.py` | ❌ One-time | Done — generated gold_a1.csv |
| `combine_gold.py` | ✅ Keep | Rebuilds combined_gold.csv from 300_train + gold_a1 |

---

## Files to Ignore / Not Train On

- `06_semisup_training.csv` — artifact of failed semi-supervised attempt
- `01_annotated.csv` — old LLM annotation with weak prompt (gave 50/50 split)
- `model_semisup/` — trained on failed semi-sup data, worse than gold-only

---

## Evaluation Target

| Metric | Current | Target |
|--------|---------|--------|
| Relevant precision | 0.444 | ≥ 0.90 |
| Relevant recall | 0.800 | ≥ 0.80 |
| Relevant F1 | 0.571 | ≥ 0.85 |
| Overall accuracy | 76.5% | ≥ 90% |

---

## Key Constraints

- **300_test.csv is sacred** — 60 rows, 13 relevant, 47 irrelevant. Never train on it.
- **No semi-supervised expansion** — self-training with a miscalibrated model propagates errors
- **Minimal hard-coded rules** — the model must learn; rules don't generalize and are indefensible in a paper
- **Asymmetric threshold is fine** — adjusting the decision boundary for imbalanced classes is standard ML practice, not a hack. Document it in the paper.
- **VM training** — CPU-only at cloud@10.2.94.119, use screen/tmux for long runs
