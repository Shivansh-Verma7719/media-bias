"""
Combine 300_train.csv (existing hand-labeled gold) and gold_a1.csv (new gold)
into a single combined_gold.csv for FinBERT training.

Usage:
  python combine_gold.py
"""
import pandas as pd

OLD  = '300_train.csv'
NEW  = 'gold_a1.csv'
OUT  = 'combined_gold.csv'

old = pd.read_csv(OLD)
old['label']  = old['label'].str.strip().str.lower()
old['source'] = 'gold_manual'
old = old[['id', 'title', 'company_name', 'label', 'source']]

new = pd.read_csv(NEW)
new['label']  = new['label'].str.strip().str.lower()
new['source'] = 'gold_manual'
new = new[['id', 'title', 'company_name', 'label', 'source']]

combined = pd.concat([old, new], ignore_index=True)
combined  = combined.drop_duplicates(subset='id', keep='first')

# Normalize company name variants
combined['company_name'] = combined['company_name'].replace({'Apple': 'Apple Inc.'})
combined  = combined[combined['label'].isin(['relevant', 'irrelevant'])]

n     = len(combined)
n_rel = (combined['label'] == 'relevant').sum()
n_irr = (combined['label'] == 'irrelevant').sum()

print(f"300_train.csv : {len(old)} rows")
print(f"gold_a1.csv   : {len(new)} rows")
print(f"Duplicates    : {len(old) + len(new) - n}")
print(f"Combined      : {n} rows | relevant={n_rel} ({100*n_rel/n:.1f}%) | irrelevant={n_irr} ({100*n_irr/n:.1f}%)")
print(f"\nPer-company breakdown:")
print(combined.groupby('company_name')['label'].value_counts().unstack(fill_value=0).to_string())

combined.to_csv(OUT, index=False)
print(f"\nSaved to {OUT}")
print(f"\nNext steps (on VM):")
print(f"  python 03_train_finbert.py -i {OUT} -s model_gold_only --gold_only")
print(f"  python 06_semisup_expand.py --model model_gold_only --gold {OUT} --unlabeled 00_filtered_clean.csv --output 06_semisup_training.csv")
print(f"  python 03_train_finbert.py -i 06_semisup_training.csv -s model_semisup")
print(f"  python 04_evaluate.py --model model_semisup --test 300_test.csv")
