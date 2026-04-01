"""
Full pipeline inference script.
Fetches all articles from articles_stratified in batches, runs DistilBERT
relevance inference on each batch, and saves results incrementally to CSV.
Resumes from checkpoint if interrupted.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

DB_BATCH_SIZE = 5000      # Rows fetched from DB per request
INFERENCE_BATCH_SIZE = 128  # Titles per BERT forward pass
MAX_LEN = 128
CONFIDENCE_THRESHOLD = 0.75  # Below this → 'uncertain', not hard-classified

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
    parser = argparse.ArgumentParser(description="Full inference: fetch from DB, run BERT, save CSV")
    parser.add_argument("--model_path", "-m", type=str, default="relevance_classifier/best_model")
    parser.add_argument("--output", "-o", type=str, default="relevance_classifier/full_predictions.csv")
    parser.add_argument("--db_batch", type=int, default=DB_BATCH_SIZE)
    args = parser.parse_args()

    # Setup DB
    from supabase import create_client
    sb = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

    # Load company map
    print("Loading company map...")
    co_resp = sb.table('companies').select('id,symbol,name').execute()
    company_map = {str(row['id']): {'symbol': row['symbol'], 'name': row['name']} for row in co_resp.data}
    print(f"  {len(company_map)} companies loaded.")

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

    # Check for existing output to resume
    processed_ids = set()
    if os.path.exists(args.output):
        existing = pd.read_csv(args.output, usecols=['id'])
        processed_ids = set(existing['id'].astype(str).tolist())
        print(f"Resuming: {len(processed_ids)} articles already processed.")
    else:
        # Write header
        pd.DataFrame(columns=[
            'id', 'title', 'url', 'source', 'published_at',
            'company_id', 'company_symbol', 'company_name',
            'media_outlet_id', 'pos_score', 'neutral_score', 'neg_score',
            'predicted_label', 'confidence_score'
        ]).to_csv(args.output, index=False)

    # Count total for progress bar
    print("Counting total rows...")
    count_resp = sb.table('articles_no_title_deduped').select('id', count='exact').limit(1).execute()
    total_rows = count_resp.count
    print(f"  Total articles: {total_rows:,}")

    # Use keyset pagination on id to avoid gaps with .range()
    last_id = 0
    total_processed = len(processed_ids)
    total_relevant = 0

    with tqdm(total=total_rows, initial=total_processed, desc="Inference", unit="articles") as pbar:
        while True:
            # Fetch batch from DB ordered by id for stable pagination
            resp = sb.table('articles_no_title_deduped') \
                      .select('id,title,url,source,published_at,company_id,media_outlet_id,pos_score,neutral_score,neg_score') \
                      .not_.is_('title', 'null') \
                      .gt('id', last_id) \
                      .order('id') \
                      .limit(args.db_batch) \
                      .execute()

            rows = resp.data
            if not rows:
                break

            last_id = rows[-1]['id']

            # Skip already processed
            rows = [r for r in rows if str(r['id']) not in processed_ids]

            if rows:
                df = pd.DataFrame(rows)
                titles = df['title'].fillna('').tolist()

                # Run BERT inference
                preds, confs = run_inference(model, tokenizer, titles, device)

                df['predicted_label'] = [
                    label_map[p] if c >= CONFIDENCE_THRESHOLD else 'uncertain'
                    for p, c in zip(preds, confs)
                ]
                df['confidence_score'] = confs
                df['company_symbol'] = df['company_id'].apply(
                    lambda x: company_map.get(str(x), {}).get('symbol', '') if x else '')
                df['company_name'] = df['company_id'].apply(
                    lambda x: company_map.get(str(x), {}).get('name', '') if x else '')

                # Keep only output columns
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

            offset += args.db_batch

    print(f"\nDone. Total processed: {total_processed:,}")
    print(f"Relevant: {total_relevant:,} ({100*total_relevant/max(total_processed,1):.1f}%)")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
