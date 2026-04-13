"""
Stage 4: Build the final FinBERT training set from two sources:
  1. 300_train.csv  — 240 gold manually-labeled rows (always fully included)
  2. 01_annotated.csv — v2 LLM annotations (new prompt + 40 few-shot examples)

Applies:
  - Post-hoc rule filters to catch systematic LLM errors
  - Reason-field sanity checks (catch hedged/uncertain labels)
  - Per-company cap to reduce company dominance
  - Final class distribution report

Usage:
  python 04_build_training_set.py \
      --llm 01_annotated.csv \
      --gold 300_train.csv \
      --output 04_training_data.csv
"""
import re
import argparse
import pandas as pd
from collections import Counter

# ── Post-hoc rule filters (same logic as 05_full_inference.py) ───────────────
# Articles matching these patterns are forced to irrelevant regardless of LLM label

CONSUMER_DEAL_RE = re.compile(
    r'(\d+\s*%\s*off'
    r'|\$\s*\d+\s*off'
    r'|only\s+\$\s*\d+'
    r'|save\s+\$\s*\d+'
    r'|deals?\s+of\s+the\s+day'
    r'|best\s+.{0,40}\s+deals?'
    r'|buying\s+guide'
    r'|gift\s+(guide|ideas?|list)'
    r'|black\s+friday\s+.{0,40}(deal|sale|offer|sav|discount|bargain)'
    r'|cyber\s+monday'
    r'|prime\s+day'
    r'|hands[\s\-]on\s+(review|preview|with)'
    r'|unboxing'
    r'|review:\s'
    r'|vs\.?\s+.{0,30}:\s+which'
    r'|record\s+low\s+price'
    r'|lowest\s+ever\s+price'
    r')',
    re.IGNORECASE
)

INDIVIDUAL_CRIME_RE = re.compile(
    r'\b(driver\s+(charged|arrested|accused|convicted|jailed|sentenced)'
    r'|employee\s+(arrested|fired|charged|jailed|convicted)'
    r'|worker\s+(arrested|charged|jailed|convicted)'
    r'|man\s+(killed|arrested|charged|shot|stabbed)'
    r'|woman\s+(killed|arrested|charged|shot|stabbed)'
    r'|passenger\s+(killed|raped|assaulted|attacked)'
    r'|rider\s+(killed|raped|assaulted|attacked)'
    r'|shooting\s+at\s+(a\s+)?(walmart|starbucks|mcdonald)'
    r'|murder(ed|er)?\s+(uber|lyft|airbnb)'
    r'|(uber|lyft|airbnb)\s+(driver|employee)\s+(rape|assault|murder|kill|attack|charged|arrested))\b',
    re.IGNORECASE
)

LIFESTYLE_RE = re.compile(
    r'\b(zodiac|horoscope|ideal\s+drink|valentine.s\s+(menu|drink|gift)'
    r'|holiday\s+menu|seasonal\s+menu|new\s+flavor|new\s+menu\s+item'
    r'|refresher|latte\s+(recipe|secret|hack)'
    r'|halloween\s+(costume|treat|decoration)'
    r'|christmas\s+(gift|decoration|deal)'
    r'|gift\s+ideas?\s+for|what\s+to\s+buy|shopping\s+list)\b',
    re.IGNORECASE
)

# Analyst firm → if acting as analyst (not subject), force irrelevant
ANALYST_FIRMS = {
    'Goldman Sachs', 'Morgan Stanley', 'JPMorgan Chase', 'Bank of America',
    'Wells Fargo Securities', 'Citigroup', 'UBS', 'Barclays',
    'Stifel', 'Jefferies', 'Raymond James', 'Piper Sandler',
}

ANALYST_ACTION_RE = re.compile(
    r'\b(upgrades?|downgrades?'
    r'|raises?\s+(?:its\s+)?(?:price\s+)?target'
    r'|cuts?\s+(?:its\s+)?(?:price\s+)?target'
    r'|maintains?\s+(?:its\s+)?(?:buy|sell|neutral|hold|overweight|underweight)'
    r'|initiates?\s+(?:coverage|buy|sell|neutral|hold|overweight|underweight)'
    r'|reiterates?\s+(?:buy|sell|neutral|hold|overweight|underweight)'
    r'|boosts?\s+(?:price\s+)?target|lowers?\s+(?:price\s+)?target'
    r'|lifts?\s+(?:price\s+)?target|slashes?\s+(?:price\s+)?target'
    r'|trims?\s+(?:price\s+)?target|bumps?\s+(?:price\s+)?target)\b',
    re.IGNORECASE
)

# Wrong-entity patterns (company name matches a different entity)
WRONG_ENTITY = {
    'Visa Inc.': re.compile(
        r'\b(immigration|immigrant|work\s+visa|student\s+visa|travel\s+visa'
        r'|tourist\s+visa|H-?1B|green\s+card|deportat|border\s+patrol'
        r'|customs|asylum|refugee|visa\s+application|visa\s+requirement'
        r'|visa\s+renewal|visa\s+ban|entry\s+visa|exit\s+visa|transit\s+visa'
        r'|visa\s+free|visa\s+waiver)\b',
        re.IGNORECASE
    ),
    'Intel': re.compile(
        r'\b(military\s+intel(?:ligence)?|intelligence\s+agenc'
        r'|CIA\s+intel|NSA\s+intel|spy|espionage|gather(?:ing)?\s+intel'
        r'|tower\s+22|drone\s+attack|airstrike|counterterrorism'
        r'|street\s+intel|competitive\s+intel(?:ligence)?)\b',
        re.IGNORECASE
    ),
}

# Reason-field uncertainty markers — LLM hedging means low-confidence label
UNCERTAINTY_RE = re.compile(
    r'\b(unclear|ambiguous|borderline|could\s+go\s+either\s+way'
    r'|hard\s+to\s+say|difficult\s+to\s+determine|might\s+be'
    r'|possibly\s+relevant|tangentially|loosely\s+related'
    r'|not\s+entirely\s+clear|somewhat\s+relevant)\b',
    re.IGNORECASE
)


def apply_post_hoc_filters(row: pd.Series) -> str:
    """Return corrected label after applying rule-based post-hoc filters."""
    title = str(row.get('title', ''))
    label = str(row.get('label', '')).strip().lower()
    company = str(row.get('company_name', ''))
    reason = str(row.get('reason', ''))

    # Drop uncertain LLM labels
    if UNCERTAINTY_RE.search(reason):
        return 'DROP'

    # Force irrelevant for known false-positive patterns
    if label == 'relevant':
        if CONSUMER_DEAL_RE.search(title):
            return 'irrelevant'
        if INDIVIDUAL_CRIME_RE.search(title):
            return 'irrelevant'
        if LIFESTYLE_RE.search(title):
            return 'irrelevant'
        if company in ANALYST_FIRMS and ANALYST_ACTION_RE.search(title):
            return 'irrelevant'
        if company in WRONG_ENTITY and WRONG_ENTITY[company].search(title):
            return 'irrelevant'

    return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm",         "-l", type=str, required=True,
                        help="v2 LLM-annotated CSV (01_annotated.csv)")
    parser.add_argument("--gold",        "-g", type=str, required=True,
                        help="Gold manual training split (300_train.csv)")
    parser.add_argument("--output",      "-o", type=str, default="04_training_data.csv")
    parser.add_argument("--company_cap", "-c", type=int, default=200,
                        help="Max LLM articles per company (default 200)")
    args = parser.parse_args()

    # ── Load gold labels ──────────────────────────────────────────────────────
    gold = pd.read_csv(args.gold)
    gold = gold.dropna(subset=['title', 'label']).copy()
    gold['label'] = gold['label'].str.strip().str.lower()
    gold = gold[gold['label'].isin(['relevant', 'irrelevant'])]
    gold['source'] = 'gold_manual'
    gold_ids = set(gold['id'].astype(str))
    print(f"Gold manual rows: {len(gold)}")
    print(f"  {gold['label'].value_counts().to_dict()}")

    # ── Load v2 LLM annotations ───────────────────────────────────────────────
    llm = pd.read_csv(args.llm)
    llm = llm.dropna(subset=['title', 'label']).copy()
    llm['label'] = llm['label'].str.strip().str.lower()
    llm = llm[llm['label'].isin(['relevant', 'irrelevant'])]
    llm['id'] = llm['id'].astype(str)

    # Remove any IDs already in gold set
    llm = llm[~llm['id'].isin(gold_ids)]
    print(f"\nLLM annotations after dedup with gold: {len(llm)}")

    # Drop rows with no company tag — these are untagged/mismatched articles
    if 'company_name' in llm.columns:
        before = len(llm)
        llm = llm[llm['company_name'].notna() & (llm['company_name'].str.strip() != '')]
        print(f"Dropped {before - len(llm)} rows with missing company_name")

    # ── Apply post-hoc filters ────────────────────────────────────────────────
    llm['label'] = llm.apply(apply_post_hoc_filters, axis=1)
    dropped = (llm['label'] == 'DROP').sum()
    corrected = (llm['label'] != llm['label']).sum()  # already applied above
    llm = llm[llm['label'] != 'DROP']
    print(f"Post-hoc filter: dropped {dropped} uncertain labels")
    print(f"After filtering: {len(llm)}")
    print(f"  {llm['label'].value_counts().to_dict()}")

    # ── Company cap ───────────────────────────────────────────────────────────
    # Cap per company using explicit loop (avoids pandas groupby dropping columns)
    # Cap irrelevant at args.company_cap, relevant at args.company_cap // 3
    # This prevents any single company from dominating either class
    print(f"\nApplying per-company cap (irr={args.company_cap}, rel={args.company_cap//3})...")
    company_col = 'company_name' if 'company_name' in llm.columns else None
    if company_col:
        rel_cap = args.company_cap // 3
        irr_cap = args.company_cap
        capped_parts = []
        for company in llm[company_col].unique():
            grp = llm[llm[company_col] == company]
            rel_grp = grp[grp['label'] == 'relevant']
            irr_grp = grp[grp['label'] == 'irrelevant']
            capped_parts.append(rel_grp.sample(min(len(rel_grp), rel_cap), random_state=42))
            capped_parts.append(irr_grp.sample(min(len(irr_grp), irr_cap), random_state=42))
        llm = pd.concat(capped_parts, ignore_index=True)
        print(f"After company cap: {len(llm)}")
        print(f"  {llm['label'].value_counts().to_dict()}")
        print("\nPer-company breakdown:")
        for company in sorted(llm[company_col].unique()):
            grp = llm[llm[company_col] == company]
            rel_n = (grp['label'] == 'relevant').sum()
            irr_n = (grp['label'] == 'irrelevant').sum()
            print(f"  {company}: {rel_n} rel / {irr_n} irr")
    llm['source'] = 'llm_v2'

    # ── Combine ───────────────────────────────────────────────────────────────
    keep_cols = ['id', 'title', 'label', 'source']
    if company_col:
        keep_cols.append(company_col)

    combined = pd.concat([
        gold[keep_cols],
        llm[[c for c in keep_cols if c in llm.columns]]
    ], ignore_index=True)
    combined = combined.drop_duplicates(subset='id', keep='first')

    # ── Final report ──────────────────────────────────────────────────────────
    total = len(combined)
    n_rel = (combined['label'] == 'relevant').sum()
    n_irr = (combined['label'] == 'irrelevant').sum()
    rel_pct = 100 * n_rel / total

    print("\n" + "="*55)
    print("FINAL TRAINING SET")
    print("="*55)
    print(f"Total rows:    {total}")
    print(f"  Relevant:    {n_rel} ({rel_pct:.1f}%)")
    print(f"  Irrelevant:  {n_irr} ({100-rel_pct:.1f}%)")
    print(f"  Gold manual: {(combined['source']=='gold_manual').sum()}")
    print(f"  LLM v2:      {(combined['source']=='llm_v2').sum()}")

    if company_col in combined.columns:
        print(f"\nUnique companies: {combined[company_col].nunique()}")

    if rel_pct < 20:
        print("\nWARNING: relevant class < 20% — class weights in 03_train_finbert.py")
        print("will compensate, but consider collecting more relevant examples.")
    elif rel_pct > 45:
        print("\nWARNING: relevant class > 45% — may be overfit to relevant patterns.")

    combined.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
