# Relevance Classifier v2 — Ground Truth Status

> **READ THIS FIRST before making any changes.**
> Last updated: 2026-04-13

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
| `manual_annotations.csv` | 750 | 227 (30.3%) | 523 (69.7%) | Single source of truth — all annotated gold |
| `combined_gold.csv` | 750 | 227 (30.3%) | 523 (69.7%) | Training-ready (adds source=gold_manual column) |
| `300_test.csv` | 60 | 13 | 47 | **HELD OUT — never train on this** |
| `00_filtered_clean.csv` | 14,934 | unlabeled | unlabeled | Supabase fetch, entity-filtered, deduped |

**To add more annotations:** run `07_sample_for_annotation.py` → annotate → add to `manual_annotations.csv` → run `combine_gold.py`

**Companies with 0 relevant training examples:** McDonald's, Bank of America, Morgan Stanley  
**True relevant rate in the wild:** ~10–15% (training set is 30.3% — still miscalibrated)

---

## Results History

| Attempt | Model | Train rows | CV Mean F1 | Precision | Recall | F1 | Accuracy | Uncertain |
|---------|-------|-----------|-----------|-----------|--------|----|----------|-----------|
| 1 | FinBERT (4ep, balanced) | 550 | 0.6733 | 0.444 | 0.800 | 0.571 | 76.5% | 9/60 |
| 2 | FinBERT (6ep, 1.5:1 cap) | 550 | 0.6692 | 0.500 | 0.125 | 0.200 | 81.8% | 16/60 |
| 3 | FinBERT (6ep, 1.5:1 cap) | 750 | 0.6706 | — | — | — | — | — |
| **4** | **DeBERTa-v3-base (6ep)** | **750** | **0.7197** | **1.000** | **0.583** | **0.737** | **91.2%** | **3/60** |

**Current best: DeBERTa-v3-base** (`model_deberta`)

---

## Current Status (2026-04-13)

DeBERTa-v3-base with 750 training rows achieves:
- ✅ Precision: **1.000** (target ≥ 0.90 — MET)
- ✅ Accuracy: **91.2%** (target ≥ 90% — MET)
- ❌ Recall: **0.583** (target ≥ 0.80 — GAP: 5 FN articles)
- ❌ Relevant F1: **0.737** (target ≥ 0.85)

**3 uncertain articles** (0.51–0.74): Airbnb 1,900 layoffs (p=0.737), Microsoft Windows preview (p=0.552), Uber app redesign (p=0.510)

**5 false negatives** (all score very low, 0.014–0.337):
- Netflix reconsidering risky spending (p=0.337)
- AT&T carrier target post Time Warner deal (p=0.307)
- Walmart Q3 Earnings takeaways (p=0.270)
- GM global sales trail VW (p=0.098)
- Walmart Rival Sale vs Amazon Prime Day (p=0.014)

**Root cause of remaining gap:** model hasn't seen enough relevant examples for Walmart, GM, AT&T, Netflix. More annotations for these companies will directly improve recall.

---

## Path Forward

### Option A: Annotate more (recommended)
Run `07_sample_for_annotation.py` targeting Walmart, GM, AT&T, Netflix — add to `manual_annotations.csv` — retrain DeBERTa.

### Option B: Lower threshold
The Airbnb layoffs article (p=0.737) would be caught at threshold 0.70. But the 5 FNs score too low (0.014–0.337) to be helped by threshold tuning alone.

**Do both in sequence:** annotate more → retrain → then tune threshold if needed.

---

## What Has Been Tried and Failed

### Semi-supervised expansion (FAILED — do not repeat)
- Trained gold-only model → ran inference on 00_filtered_clean.csv → kept p≥0.90 as pseudo-labels
- Result: 51.5% relevant in pseudo-labels (confirmation bias from miscalibrated model)
- **Script: `06_semisup_expand.py` — DO NOT USE**

### FinBERT with class weight cap 1.5:1 (WORSE)
- Attempt 2: capping weights to prevent over-prediction collapsed recall to 0.125
- DeBERTa is the better architecture — use it going forward

---

## Pipeline Scripts

| Script | Status | Purpose |
|--------|--------|---------|
| `00_fetch_filtered.py` | ✅ Use | Fetch from Supabase with entity filter + dedup |
| `03_train_finbert.py` | ✅ Use | Model-agnostic fine-tuning (FinBERT/DeBERTa/etc) with k-fold CV |
| `04_evaluate.py` | ✅ Use | Evaluate on held-out test set |
| `02_annotate_gold.py` | ✅ Use | Interactive human annotation tool |
| `07_sample_for_annotation.py` | ✅ Use | Samples articles weighted by underrepresented companies |
| `combine_gold.py` | ✅ Use | Rebuilds combined_gold.csv from manual_annotations.csv |
| `05_full_inference.py` | ✅ Use | Full inference on 700k+ article corpus |
| `01_audit_irrelevant.py` | ✅ Use | QA tool for false-negative rate |
| `06_semisup_expand.py` | ❌ Abandoned | Semi-supervised failed — do not use |
| `01_annotate.py` | ⚠️ Caution | LLM annotation — previously caused leakage issues |
| `05_verify_relevant.py` | ⚠️ Caution | Second-pass LLM verification |

---

## Key Constraints

- **300_test.csv is sacred** — 60 rows, 13 relevant, 47 irrelevant. Never train on it.
- **No semi-supervised expansion** — self-training with a miscalibrated model propagates errors
- **Minimal hard-coded rules** — the model must learn; rules don't generalize and are indefensible in a paper
- **Asymmetric threshold is fine** — P(relevant) ≥ 0.75 is standard ML practice for imbalanced classes. Document in paper.
- **DeBERTa is the primary model** — consistently outperforms FinBERT
- **VM training** — CPU-only at cloud@10.2.94.119, use nohup for long runs (~9 hours for DeBERTa)
- **manual_annotations.csv is the single source of truth** — all annotation additions go here first

---

## Evaluation Targets

| Metric | Current (DeBERTa) | Target |
|--------|-------------------|--------|
| Relevant precision | **1.000** ✅ | ≥ 0.90 |
| Relevant recall | 0.583 ❌ | ≥ 0.80 |
| Relevant F1 | 0.737 ❌ | ≥ 0.85 |
| Overall accuracy | **91.2%** ✅ | ≥ 90% |
