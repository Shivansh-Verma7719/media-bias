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
    parser = argparse.ArgumentParser(description="Fetch and pre-filter titles from DB.")
    parser.add_argument("--output", "-o", type=str, default="00_filtered_3k.csv")
    parser.add_argument("--companies", "-c", type=str, default="company_names.txt")
    parser.add_argument("--target", "-t", type=int, default=3000, help="Target number of filtered titles")
    parser.add_argument("--batch_size", "-b", type=int, default=10000, help="Rows to fetch per DB call")
    parser.add_argument("--num_segments", "-n", type=int, default=20, help="Number of DB segments to sample from")
    parser.add_argument("--tricky", action="store_true", help="Fetch tricky irrelevant articles for hard negative mining")
    parser.add_argument("--exclude", "-e", type=str, default=None, help="CSV file of already seen IDs to exclude")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sb = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
    company_names = load_company_names(args.companies)
    company_pattern = build_pattern(company_names)
    print(f"Loaded {len(company_names)} company names.")
    if args.tricky:
        print("Mode: TRICKY — targeting consumer/deal/irrelevant articles.")
    else:
        print("Mode: GENERAL — random diverse sample.")

    # Load existing IDs to exclude
    existing_ids = set()
    if args.exclude:
        for f in args.exclude.split(','):
            f = f.strip()
            if os.path.exists(f):
                df_ex = pd.read_csv(f, usecols=['id'])
                existing_ids.update(df_ex['id'].astype(str).tolist())
    print(f"Excluding {len(existing_ids)} already seen IDs.")

    # Spread offsets across DB for diversity
    random.seed(args.seed)
    total_db_rows = 3_800_000
    segment_size = total_db_rows // args.num_segments
    offsets = [i * segment_size + random.randint(0, segment_size // 2) for i in range(args.num_segments)]

    collected = []
    seen_ids = set(existing_ids)

    with tqdm(total=args.target, desc="Collecting titles") as pbar:
        for offset in offsets:
            if len(collected) >= args.target:
                break

            r = sb.table('articles_stratified') \
                  .select('id,title,pos_score,neutral_score,neg_score,company_id') \
                  .not_.is_('pos_score', 'null') \
                  .not_.is_('title', 'null') \
                  .range(offset, offset + args.batch_size - 1) \
                  .execute()

            for row in r.data:
                if str(row['id']) in seen_ids:
                    continue
                title = str(row.get('title', ''))

                has_company = company_pattern.search(title)
                if not has_company:
                    continue

                if args.tricky:
                    # Only collect if it also matches a tricky pattern
                    if not TRICKY_PATTERNS.search(title):
                        continue

                collected.append(row)
                seen_ids.add(str(row['id']))
                pbar.update(1)
                if len(collected) >= args.target:
                    break

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
