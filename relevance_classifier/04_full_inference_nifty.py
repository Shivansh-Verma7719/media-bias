"""
Full inference for NIFTY pipeline.
Fetches from indian_cos.articles_stratified via psycopg2 (schema not exposed via REST),
runs DistilBERT relevance inference, saves results incrementally.
Resumes from checkpoint if interrupted.
"""
import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import psycopg2
import psycopg2.extras
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

DB_BATCH_SIZE       = 5000
INFERENCE_BATCH_SIZE = 128
MAX_LEN             = 128
CONFIDENCE_THRESHOLD = 0.75
TABLE               = 'indian_cos.articles_stratified'

# Post-filter: force consumer/deal titles to irrelevant regardless of BERT
POST_FILTER = re.compile(
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

# Analyst firms in the NIFTY universe that publish research on other stocks.
# Articles where they act as analyst (not corporate subject) → irrelevant for them.
ANALYST_FIRMS = {
    'HDFC Bank', 'ICICI Bank', 'Kotak Mahindra Bank', 'Axis Bank',
    'State Bank of India', 'Motilal Oswal', 'Emkay Global',
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


def apply_filters(title: str, label: str, company_name: str) -> str:
    """Apply all post-inference filters. Returns the final label."""
    t = str(title)

    # 1. Consumer/deal content (global)
    if POST_FILTER.search(t):
        return 'irrelevant'

    # 2. Analyst-firm filter
    if company_name in ANALYST_FIRMS:
        firm_re = re.compile(re.escape(company_name), re.IGNORECASE)
        if firm_re.search(t) and ANALYST_ACTION_RE.search(t):
            return 'irrelevant'

    return label


def get_conn():
    tenant_id = os.getenv('POOLER_TENANT_ID', 'your-tenant-id')
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=int(os.getenv('DB_PORT', 5432)),
        dbname=os.getenv('DB_NAME', 'postgres'),
        user=f"postgres.{tenant_id}",
        password=os.getenv('DB_PASSWORD'),
        connect_timeout=15,
    )


class TitleDataset(Dataset):
    def __init__(self, titles, tokenizer):
        self.titles = titles
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            str(self.titles[idx]),
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


def run_inference(model, tokenizer, titles, device):
    dataset = TitleDataset(titles, tokenizer)
    loader = DataLoader(dataset, batch_size=INFERENCE_BATCH_SIZE, shuffle=False,
                        num_workers=2 if str(device) != 'cpu' else 0)
    model.eval()
    preds, confs = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            preds.extend(torch.argmax(probs, dim=1).cpu().tolist())
            confs.extend(torch.max(probs, dim=1).values.cpu().tolist())
    return preds, confs


def main():
    parser = argparse.ArgumentParser(description="NIFTY full inference via psycopg2")
    parser.add_argument("--model_path", "-m", type=str, default="relevance_classifier/best_model_nifty")
    parser.add_argument("--output", "-o", type=str, default="relevance_classifier/full_predictions_nifty.csv")
    parser.add_argument("--db_batch", type=int, default=DB_BATCH_SIZE)
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model path {args.model_path} not found. Run 02_train_bert.py first.")
        sys.exit(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_path)
    model = DistilBertForSequenceClassification.from_pretrained(args.model_path)
    model = model.to(device)
    label_map = {1: 'relevant', 0: 'irrelevant'}

    # Connect to DB
    print(f"Connecting to {os.getenv('DB_HOST')} → {TABLE}")
    conn = get_conn()
    cur = conn.cursor()

    # Load company map
    print("Loading company map...")
    cur.execute("SELECT id, name, symbol FROM indian_cos.companies")
    company_map = {row[0]: {'name': row[1], 'symbol': row[2]} for row in cur.fetchall()}
    print(f"  {len(company_map)} companies loaded.")

    # Check for existing output to resume
    processed_ids = set()
    if os.path.exists(args.output):
        existing = pd.read_csv(args.output, usecols=['id'])
        processed_ids = set(existing['id'].astype(str).tolist())
        print(f"Resuming: {len(processed_ids)} articles already processed.")
    else:
        pd.DataFrame(columns=[
            'id', 'title', 'url', 'source', 'published_at',
            'company_id', 'company_symbol', 'company_name',
            'media_outlet_id', 'pos_score', 'neutral_score', 'neg_score',
            'predicted_label', 'confidence_score'
        ]).to_csv(args.output, index=False)

    # Count total
    cur.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE title IS NOT NULL")
    total_rows = cur.fetchone()[0]
    print(f"  Total articles: {total_rows:,}")

    # Keyset pagination by id
    last_id = 0
    total_processed = len(processed_ids)
    total_relevant = 0

    with tqdm(total=total_rows, initial=total_processed, desc="Inference", unit="articles") as pbar:
        while True:
            cur.execute(f"""
                SELECT id, title, url, source, published_at,
                       company_id, media_outlet_id,
                       pos_score, neutral_score, neg_score
                FROM {TABLE}
                WHERE title IS NOT NULL AND id > %s
                ORDER BY id
                LIMIT %s
            """, (last_id, args.db_batch))

            rows = cur.fetchall()
            if not rows:
                break

            last_id = rows[-1][0]

            # Skip already processed
            rows = [r for r in rows if str(r[0]) not in processed_ids]

            if rows:
                df = pd.DataFrame(rows, columns=[
                    'id', 'title', 'url', 'source', 'published_at',
                    'company_id', 'media_outlet_id',
                    'pos_score', 'neutral_score', 'neg_score'
                ])
                titles = df['title'].fillna('').tolist()

                preds, confs = run_inference(model, tokenizer, titles, device)

                # Resolve company names first — needed by apply_filters
                df['company_symbol'] = df['company_id'].apply(
                    lambda x: company_map.get(x, {}).get('symbol', '') if x else '')
                df['company_name'] = df['company_id'].apply(
                    lambda x: company_map.get(x, {}).get('name', '') if x else '')

                # Apply confidence threshold
                labels = [
                    label_map[p] if c >= CONFIDENCE_THRESHOLD else 'uncertain'
                    for p, c in zip(preds, confs)
                ]
                # Apply all post-inference filters
                df['predicted_label'] = [
                    apply_filters(t, l, c)
                    for t, l, c in zip(df['title'], labels, df['company_name'])
                ]
                df['confidence_score'] = confs

                out_cols = ['id', 'title', 'url', 'source', 'published_at',
                            'company_id', 'company_symbol', 'company_name',
                            'media_outlet_id', 'pos_score', 'neutral_score', 'neg_score',
                            'predicted_label', 'confidence_score']
                df[out_cols].to_csv(args.output, mode='a', header=False, index=False)

                n_relevant = sum(1 for p, c in zip(preds, confs) if p == 1 and c >= CONFIDENCE_THRESHOLD)
                total_relevant += n_relevant
                total_processed += len(rows)
                processed_ids.update(df['id'].astype(str).tolist())
                pbar.update(len(rows))
                pbar.set_postfix({
                    'relevant': total_relevant,
                    'relevant%': f'{100*total_relevant/max(total_processed,1):.1f}%',
                    'last_id': last_id
                })

    cur.close()
    conn.close()

    print(f"\nDone. Total processed: {total_processed:,}")
    print(f"Relevant: {total_relevant:,} ({100*total_relevant/max(total_processed,1):.1f}%)")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
