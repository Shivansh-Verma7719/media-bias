"""
Fetches articles from articles_stratified in batches and applies two modes:

Mode 1 (default): General fetch - titles containing company names, sampled
from multiple offsets across the DB for diversity.

Mode 2 (--tricky): Targeted fetch - titles containing company names AND
tricky consumer/deal/irrelevant patterns that BERT tends to misclassify.
Used to augment training data with hard negatives.
"""
import os
import re
import random
import argparse
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Patterns that strongly suggest consumer/deal/irrelevant articles
TRICKY_PATTERNS = re.compile(
    r'\b(sale|deal|discount|price|cheap|cheapest|lowest price|record low|'
    r'review|reviewed|hands.on|unboxing|best buy|buying guide|vs\.|versus|'
    r'warning|tip|tips|trick|tricks|how to|tutorial|'
    r'recall|lawsuit|death|dies|fired|arrest|crime|scandal|'
    r'rumor|leak|leaked|concept|render|spotted|'
    r'gift|holiday|black friday|cyber monday|prime day|'
    r'coupon|promo|promotion|limited.time|offer)\b',
    re.IGNORECASE
)

def load_company_names(company_file):
    with open(company_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def build_pattern(company_names):
    sorted_names = sorted(company_names, key=len, reverse=True)
    pattern_str = r'\b(?:' + '|'.join(map(re.escape, sorted_names)) + r')\b'
    return re.compile(pattern_str, re.IGNORECASE)

def main():
    parser = argparse.ArgumentParser(description="Fetch a random sample of titles from DB.")
    parser.add_argument("--output", "-o", type=str, default="00_filtered_sample.csv")
    parser.add_argument("--target", "-t", type=int, default=4000, help="Number of titles to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    sb = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

    total_db_rows = 483_514
    batch_size = 500
    n_batches = (args.target + batch_size - 1) // batch_size
    # Pick random start offsets, each batch fetches 500 consecutive rows
    offsets = sorted(random.sample(range(total_db_rows - batch_size), n_batches))

    print(f"Fetching {args.target} titles in {n_batches} batches of {batch_size}...")
    collected = []
    for offset in tqdm(offsets, desc="Fetching batches"):
        r = sb.table('articles_no_title_deduped') \
              .select('id,title,pos_score,neutral_score,neg_score,company_id') \
              .not_.is_('title', 'null') \
              .range(offset, offset + batch_size - 1) \
              .execute()
        collected.extend(r.data)

    random.shuffle(collected)
    collected = collected[:args.target]

    df = pd.DataFrame(collected)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} titles to {args.output}")
    print("\nSample titles:")
    for t in df['title'].head(10).tolist():
        print(f"  {t}")

if __name__ == "__main__":
    main()
