"""
Fetches articles from articles_stratified in batches, applies company name
pre-filter on titles, and saves until target_count filtered titles are collected.
"""
import os
import re
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from supabase import create_client
import argparse

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

def load_company_names(company_file):
    with open(company_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def build_pattern(company_names):
    sorted_names = sorted(company_names, key=len, reverse=True)
    pattern_str = r'\b(?:' + '|'.join(map(re.escape, sorted_names)) + r')\b'
    return re.compile(pattern_str, re.IGNORECASE)

def main():
    parser = argparse.ArgumentParser(description="Fetch and pre-filter titles from DB until target count reached.")
    parser.add_argument("--output", "-o", type=str, default="00_filtered_3k.csv")
    parser.add_argument("--companies", "-c", type=str, default="company_names.txt")
    parser.add_argument("--target", "-t", type=int, default=3000, help="Target number of filtered titles")
    parser.add_argument("--batch_size", "-b", type=int, default=5000, help="Rows to fetch per DB call")
    parser.add_argument("--max_batches", "-m", type=int, default=20, help="Max DB batches to try")
    args = parser.parse_args()

    sb = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
    company_names = load_company_names(args.companies)
    pattern = build_pattern(company_names)
    print(f"Loaded {len(company_names)} company names.")

    # Fetch from multiple offsets spread across the full table for diversity
    # From testing: ~15% of rows pass the title filter, so fetch ~20k per segment
    import random
    random.seed(42)

    total_db_rows = 3_800_000
    num_segments = 15
    segment_size = total_db_rows // num_segments
    offsets = [i * segment_size + random.randint(0, segment_size // 2) for i in range(num_segments)]

    collected = []
    seen_ids = set()

    with tqdm(total=args.target, desc="Collecting filtered titles") as pbar:
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
                if pattern.search(title):
                    collected.append(row)
                    seen_ids.add(str(row['id']))
                    pbar.update(1)
                    if len(collected) >= args.target:
                        break

    random.shuffle(collected)
    collected = collected[:args.target]

    df = pd.DataFrame(collected)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} filtered titles to {args.output}")
    print("\nTop 10 companies by article count:")
    print(df['company_id'].value_counts().head(10))

if __name__ == "__main__":
    main()
