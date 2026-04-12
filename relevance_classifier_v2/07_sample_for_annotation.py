"""
Sample articles from 00_filtered_clean.csv for manual annotation.

Prioritises:
  1. Companies with 0 relevant examples in the current training set
  2. Companies with few relevant examples relative to their article count
  3. Shuffled so annotator sees variety

Output is a CSV ready to pass to 02_annotate_gold.py.

Usage:
  python 07_sample_for_annotation.py \
      --pool 00_filtered_clean.csv \
      --gold combined_gold.csv \
      --output annotation_batch.csv \
      --n 200
"""
import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool",   default="00_filtered_clean.csv",
                        help="Unlabeled article pool")
    parser.add_argument("--gold",   default="combined_gold.csv",
                        help="Current gold training set (to exclude already-labeled IDs)")
    parser.add_argument("--output", default="annotation_batch.csv")
    parser.add_argument("--n",      type=int, default=200,
                        help="Total articles to sample")
    args = parser.parse_args()

    pool = pd.read_csv(args.pool)
    gold = pd.read_csv(args.gold)

    # Exclude already-labeled IDs
    labeled_ids = set(gold['id'].astype(str))
    pool = pool[~pool['id'].astype(str).isin(labeled_ids)].copy()
    print(f"Unlabeled pool after excluding gold: {len(pool)} articles")

    # Count relevant examples per company in current training set
    rel_counts = gold[gold['label'] == 'relevant'].groupby('company_name').size()
    print("\nCurrent relevant examples per company:")
    for company in sorted(pool['company_name'].unique()):
        count = rel_counts.get(company, 0)
        print(f"  {company}: {count} relevant")

    # Weight companies inversely by their relevant count
    # Companies with 0 relevant get highest weight
    def company_weight(company):
        count = rel_counts.get(company, 0)
        return 1.0 / (count + 1)  # +1 to avoid division by zero

    pool['_weight'] = pool['company_name'].map(company_weight)

    # Stratified sample: take proportionally more from underrepresented companies
    # but ensure every company gets at least some representation
    companies = pool['company_name'].unique()
    per_company_min = max(2, args.n // (len(companies) * 3))  # at least 2 per company

    sampled_parts = []

    # First pass: minimum per company
    for company in companies:
        grp = pool[pool['company_name'] == company]
        n = min(len(grp), per_company_min)
        sampled_parts.append(grp.sample(n, random_state=42))

    sampled_min = pd.concat(sampled_parts)
    remaining_pool = pool[~pool['id'].isin(sampled_min['id'])]
    remaining_n = args.n - len(sampled_min)

    # Second pass: fill remainder weighted by underrepresentation
    if remaining_n > 0 and len(remaining_pool) > 0:
        weights = remaining_pool['_weight'] / remaining_pool['_weight'].sum()
        n_sample = min(remaining_n, len(remaining_pool))
        extra = remaining_pool.sample(n_sample, weights=weights, random_state=42)
        sampled_parts.append(extra)

    result = pd.concat(sampled_parts).drop_duplicates(subset='id')
    result = result.drop(columns=['_weight'], errors='ignore')
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    print(f"\nSampled {len(result)} articles for annotation")
    print("Per company:")
    print(result['company_name'].value_counts().to_string())

    result.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")
    print(f"\nNext: annotate with")
    print(f"  python 02_annotate_gold.py -i {args.output} -o annotated_batch.csv -a <your_name>")
    print(f"Then merge into combined_gold.csv with combine_gold.py")

if __name__ == "__main__":
    main()
