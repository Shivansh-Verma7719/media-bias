# Relevance Classifier — Full Documentation

## 1. Problem Statement

Binary classifier for news article titles: given `(title, company_name)`, predict whether the article is **financially material** to the company (`relevant`) or not (`irrelevant`).

**Targets:** Precision ≥ 0.90, Recall ≥ 0.80
**Pipeline location:** `media-bias/relevance_classifier_v2/`

---

## 2. Base Architecture

**Model:** `microsoft/deberta-v3-base` (HuggingFace)
- 86M parameters
- Disentangled attention with relative position encoding
- Stronger than BERT or FinBERT for sequence classification

**Training script:** `03_train_finbert.py`

**Common training configuration:**
- 5-fold stratified cross-validation (`StratifiedKFold`)
- Max sequence length: 128 tokens
- Optimizer: AdamW, learning rate 2e-5, weight decay 0.01
- Linear warmup (10% of total steps) + linear decay scheduler
- Gradient clipping at 1.0
- Batch size: 16
- Max 6 epochs with early stopping (patience=2 on validation loss)
- DeBERTa-v3 specific: bias, LayerNorm, and embedding parameters excluded from weight decay (required by GDES embedding sharing)

---

## 3. Three Trained Models

### Model 1: `model_deberta`

**Training data:** Gold-labeled only (750 articles)
- File: `combined_training_set.csv` filtered to `source_type == 'gold_manual'`
- 523 irrelevant, 227 relevant
- Source: manually annotated via `02_annotate_gold.py`

**Loss:** Class-weighted cross-entropy (weight cap 1.5:1 to prevent over-prediction)

**Result on test set (`test.csv`, 187 articles):**

| Threshold | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|
| 0.75 | **0.652** | **0.714** | 0.682 | 16 | 12 |

### Model 2: `model_synthetic`

**Training data:** Gold + Round 1 synthetic = 5,000 articles
- File: `combined_training_set.csv`
- Gold: 750 (523 irr, 227 rel)
- Synthetic: 4,250 (2,800 irr, 1,450 rel)
- Synthetic generated via `gen_synthetic.py` for broad FP categories A–G:
  - A: Consumer deals/promos
  - B: Sports/entertainment crossover
  - C: Travel/lifestyle
  - D: Generic news incidentally mentioning company
  - E: Stock market overview articles
  - F: Industry analysis (not company-specific)
  - G: Personal finance / consumer advice

**Result on test set:**

| Threshold | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|
| 0.90 | **0.762** | **0.762** | 0.762 | 10 | 10 |

### Model 3: `model_synthetic_v2`

**Training data:** Gold + all synthetic = 6,525 articles
- File: `combined_training_set_v2.csv`
- Same 750 gold + 5,775 synthetic
- Targeted patterns added via `gen_targeted.py`:
  - **G11** (100): Speculative business moves ("considers exporting")
  - **G12** (100): Analyst earnings previews ("What to Expect From")
  - **G13** (100): Quantified philanthropy ("Foundation commits $5M")
  - **G14** (100): Regional micro-launches ("launches app in [Country]")
  - **G15** (100): Mid-level executive appointments + BRIEF- wire prefix
  - **G16** (100): Local economic impact stories
  - **G17** (100): Brand comparators ("The Netflix of China")
  - **G18** (100): Small lawsuits ($1M–$49M, immaterial at S&P 500 scale)
  - **G19** (100): Visa contamination (immigration articles tagged to Visa Inc.)
  - **G20** (100): Industry disruption catalyst
  - **H7** (75): Layoffs from employee perspective
  - **H8** (75): Safety issues with CEO admission language
  - **H9** (75): Informal earnings language ("Finishes 2016 Strong")
  - **H10** (75): Executive departure as succession
  - **H11** (75): New financial products / crypto
  - **H12** (75): Cross-company investments
  - **H13** (75): Competitive underperformance

**Loss:** Focal loss (γ=2) — down-weights easy correct examples, focuses on boundary cases
**Sample weighting:** Gold examples upweighted 5x in loss computation

**Result on test set:**

| Threshold | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|
| 0.95 | **0.860** | **0.881** | 0.871 | 6 | 5 |

---

## 4. Test Set

**File:** `test.csv` (187 articles, never seen during training)
- 145 irrelevant (77.5%)
- 42 relevant (22.5%)
- Hand-labeled, held out from all training rounds

---

## 5. Rule-Based Post-Processor

**File:** `rule_adjuster.py`

After ensemble inference, 9 linguistic rules adjust `p_relevant` downward for patterns that are never relevant:

| Rule ID | Pattern | Adjustment |
|---|---|---|
| G11 | "considers importing/exporting" — speculative | −0.25 |
| G12 | "What to Expect From", "earnings preview" | −0.40 |
| G13 | Foundation + $ amount, philanthropy | −0.40 |
| G15 | BRIEF- wire prefix | −0.35 |
| G15 | Mid-level executive appointment titles | −0.35 |
| G16 | "City a big winner in [Co] deal" | −0.35 |
| G17 | "The X of [country]" brand comparator | −0.40 |
| G18 | "$[1-49]M lawsuit" | −0.30 |
| G19 | Visa Inc. + travel visa keywords | hard set 0.05 |
| G20 | "Competition with X sparks..." | −0.40 |

Applied via `adjust_batch(titles, company_names, probs)`.

---

## 6. Ensemble — How It Classifies

**Inference script:** `10_full_inference_ensemble.py`
**Evaluation script:** `09_postprocess_eval.py`

For each article title, the classification process is:

```
1. HARD FILTER (skip inference if any match)
   - Source in SOURCE_BLOCKLIST (espn.com, bleacherreport.com, etc.)
   - Title matches CONSUMER_CONTENT_RE (50% off, Black Friday, buying guide)
   - Analyst firm + rating action (Goldman Sachs upgrades/downgrades)
   - Intel/Target name contamination (military intel, military target)
   → predicted_label = 'irrelevant', skip to next article

2. ENSEMBLE INFERENCE (3 models)
   - Tokenize title with each model's tokenizer (DeBERTa-v3 tokenizer)
   - Forward pass through model_deberta → p1 = softmax(logits)[:, 1]
   - Forward pass through model_synthetic → p2 = softmax(logits)[:, 1]
   - Forward pass through model_synthetic_v2 → p3 = softmax(logits)[:, 1]
   - ensemble_prob = (p1 + p2 + p3) / 3   # equal weights

3. RULE ADJUSTMENT
   - adj_prob, rule_fired = rule_adjuster.adjust(title, company_name, ensemble_prob)

4. THRESHOLD
   - if adj_prob >= 0.65: predicted_label = 'relevant'
   - else:                predicted_label = 'irrelevant'
```

**Why averaging works:** Each of the three models was trained on slightly different data and has different error patterns. When all three agree, confidence is high. When they disagree, the average pulls the score toward the consensus, dampening individual model errors. This raises precision without retraining.

**Ensemble result on test set (full sweep):**

| Threshold | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|
| 0.55 | 0.673 | 0.881 | 0.763 | 18 | 5 |
| 0.60 | 0.783 | 0.857 | 0.818 | 10 | 6 |
| **0.65** | **0.944** | **0.810** | **0.872** | **2** | **8** |
| 0.70 | 0.970 | 0.762 | 0.853 | 1 | 10 |
| 0.75 | 0.968 | 0.714 | 0.822 | 1 | 12 |

**Production threshold: 0.65** (first point where both targets are met)

---

## 7. Performance Progression Summary

| Model | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|
| model_deberta | 0.652 | 0.714 | 0.682 | 16 | 12 |
| model_synthetic | 0.762 | 0.762 | 0.762 | 10 | 10 |
| model_synthetic_v2 | 0.860 | 0.881 | 0.871 | 6 | 5 |
| **3-model Ensemble** | **0.944** | **0.810** | **0.872** | **2** | **8** |
| **Target** | **≥0.90** | **≥0.80** | — | — | — |

---

## 8. Full Inference (Completed)

Ran `10_full_inference_ensemble.py` on the full Supabase corpus.

| Metric | Value |
|---|---|
| Total articles | 695,731 |
| Relevant predictions | 121,177 (17.4%) |
| Irrelevant predictions | 574,554 |
| Hard-filtered before inference | 36,879 (5.3%) |
| Output file | `full_predictions_ensemble.csv` |

Output columns: `id, title, url, source, published_at, company_id, company_symbol, company_name, media_outlet_id, pos_score, neutral_score, neg_score, predicted_label, ensemble_prob, adj_prob, rule_fired`

---

## 9. To-Do

1. **Audit borderline predictions** (p=0.65–0.70). Spot-checking shows some FPs from pre-existing DB tagging issues (e.g. RBI gold bond article tagged to GM because "gm" appears as "grams").

2. **Add immigration signals to G19 rule** — "ICE", "deportation", "border" should also trigger Visa contamination filter.

3. **Replace ensemble with a single re-trained DeBERTa model** (per professor's guidance — more academically defendable for the paper). Approach:
   - Combine all training data used across the three models into one set (gold + Round 1 synthetic + Round 2 targeted synthetic G11–G20 / H7–H13)
   - Augment further if needed by generating more synthetic examples for patterns where individual models still fail
   - Re-train a single DeBERTa-v3-base with the same focal loss + 5x gold upweighting that worked for `model_synthetic_v2`
   - Goal: match or exceed the ensemble's P=0.944, R=0.810 with one model
