"""
Stage 5: Full inference on articles_no_title_deduped (483k rows).
Uses FinBERT model trained in Stage 3. Resumes from checkpoint.
Applies post-inference filters for known false-positive patterns.

Usage:
  python 05_full_inference.py --model_path best_model --output full_predictions_v2.csv
"""
import os, re, sys, argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

DB_BATCH_SIZE       = 5000
INFERENCE_BATCH_SIZE = 128
MAX_LEN             = 128
CONFIDENCE_THRESHOLD = 0.65  # lowered from 0.75 — reduces uncertain articles

# ── Post-inference filters ────────────────────────────────────────────────────
# Consumer / deal content — model still misclassifies these as relevant
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

SOURCE_BLOCKLIST = {
    'axs.com', 'bleacherreport.com', 'cbssports.com', 'sbnation.com',
    'sportingnews.com', 'foxsports.com', 'espn.com', 'fansided.com',
    'nbcsports.com', 'hollywoodlife.com', 'deadspin.com', 'usmagazine.com',
    'radaronline.com', 'pitchfork.com', 'monstersandcritics.com',
}

ANALYST_FIRMS = {
    'Goldman Sachs', 'Morgan Stanley', 'JPMorgan Chase', 'Bank of America',
    'Wells Fargo', 'Citigroup', 'UBS', 'Barclays',
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

COMPANY_EXCLUSIONS = {
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
        r'|street\s+intel|competitive\s+intel(?:ligence)?'
        r'|tower\s+22|drone\s+attack|airstrike|counterterrorism)\b',
        re.IGNORECASE
    ),
    'Target': re.compile(
        r'\b(military\s+target|bombing\s+target|airstrike\s+target'
        r'|missile\s+target|shooting\s+target|target\s+practice'
        r'|archery\s+target|sniper\s+target)\b',
        re.IGNORECASE
    ),
}


def apply_filters(title: str, label: str, source: str, company_name: str) -> str:
    t = str(title)
    if POST_FILTER.search(t):
        return 'irrelevant'
    if str(source).lower() in SOURCE_BLOCKLIST:
        return 'irrelevant'
    if company_name in ANALYST_FIRMS:
        firm_re = re.compile(re.escape(company_name), re.IGNORECASE)
        if firm_re.search(t) and ANALYST_ACTION_RE.search(t):
            return 'irrelevant'
    if company_name in COMPANY_EXCLUSIONS and COMPANY_EXCLUSIONS[company_name].search(t):
        return 'irrelevant'
    return label


class TitleDataset(Dataset):
    def __init__(self, titles, tokenizer):
        self.titles = titles
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        enc = self.tokenizer.encode_plus(
            str(self.titles[idx]),
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
        }


def run_inference(model, tokenizer, titles, device):
    ds = TitleDataset(titles, tokenizer)
    dl = DataLoader(ds, batch_size=INFERENCE_BATCH_SIZE, shuffle=False,
                    num_workers=2 if str(device) != 'cpu' else 0)
    model.eval()
    preds, confs = [], []
    with torch.no_grad():
        for batch in dl:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            probs = torch.nn.functional.softmax(out.logits, dim=-1)
            preds.extend(torch.argmax(probs, dim=1).cpu().tolist())
            confs.extend(torch.max(probs, dim=1).values.cpu().tolist())
    return preds, confs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, default="best_model")
    parser.add_argument("--output",     "-o", type=str, default="full_predictions_v2.csv")
    parser.add_argument("--db_batch",         type=int, default=DB_BATCH_SIZE)
    args = parser.parse_args()

    from supabase import create_client
    sb = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

    print("Loading company map...")
    co_resp = sb.table('companies').select('id,symbol,name').execute()
    company_map = {str(r['id']): {'symbol': r['symbol'], 'name': r['name']} for r in co_resp.data}
    print(f"  {len(company_map)} companies loaded.")

    print(f"Loading model from {args.model_path}...")
    if not os.path.exists(args.model_path):
        print(f"ERROR: {args.model_path} not found. Run 03_train_finbert.py first.")
        sys.exit(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  device={device}")
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = BertForSequenceClassification.from_pretrained(args.model_path).to(device)
    label_map = {1: 'relevant', 0: 'irrelevant'}

    processed_ids = set()
    if os.path.exists(args.output):
        existing = pd.read_csv(args.output, usecols=['id'])
        processed_ids = set(existing['id'].astype(str).tolist())
        print(f"Resuming: {len(processed_ids):,} already processed.")
    else:
        pd.DataFrame(columns=[
            'id', 'title', 'url', 'source', 'published_at',
            'company_id', 'company_symbol', 'company_name',
            'media_outlet_id', 'pos_score', 'neutral_score', 'neg_score',
            'predicted_label', 'confidence_score'
        ]).to_csv(args.output, index=False)

    print("Counting total rows...")
    count_resp = sb.table('articles_no_title_deduped').select('id', count='exact').limit(1).execute()
    total_rows = count_resp.count
    print(f"  Total articles: {total_rows:,}")

    last_id = 0
    total_processed = len(processed_ids)
    total_relevant  = 0

    with tqdm(total=total_rows, initial=total_processed, desc="Inference", unit="articles") as pbar:
        while True:
            resp = (sb.table('articles_no_title_deduped')
                      .select('id,title,url,source,published_at,company_id,media_outlet_id,'
                              'pos_score,neutral_score,neg_score')
                      .not_.is_('title', 'null')
                      .gt('id', last_id)
                      .order('id')
                      .limit(args.db_batch)
                      .execute())

            rows = resp.data
            if not rows:
                break
            last_id = rows[-1]['id']
            rows = [r for r in rows if str(r['id']) not in processed_ids]

            if rows:
                df = pd.DataFrame(rows)
                titles = df['title'].fillna('').tolist()
                preds, confs = run_inference(model, tokenizer, titles, device)

                df['company_symbol'] = df['company_id'].apply(
                    lambda x: company_map.get(str(x), {}).get('symbol', '') if x else '')
                df['company_name'] = df['company_id'].apply(
                    lambda x: company_map.get(str(x), {}).get('name', '') if x else '')

                labels = [
                    label_map[p] if c >= CONFIDENCE_THRESHOLD else 'uncertain'
                    for p, c in zip(preds, confs)
                ]
                df['predicted_label'] = [
                    apply_filters(t, l, s, c)
                    for t, l, s, c in zip(df['title'], labels, df['source'], df['company_name'])
                ]
                df['confidence_score'] = confs

                out_cols = [
                    'id', 'title', 'url', 'source', 'published_at',
                    'company_id', 'company_symbol', 'company_name',
                    'media_outlet_id', 'pos_score', 'neutral_score', 'neg_score',
                    'predicted_label', 'confidence_score'
                ]
                df[out_cols].to_csv(args.output, mode='a', header=False, index=False)

                n_rel = sum(1 for p, c in zip(preds, confs) if p == 1 and c >= CONFIDENCE_THRESHOLD)
                total_relevant  += n_rel
                total_processed += len(rows)
                processed_ids.update(df['id'].astype(str).tolist())
                pbar.update(len(rows))
                pbar.set_postfix({
                    'relevant':   total_relevant,
                    'relevant%':  f'{100*total_relevant/max(total_processed,1):.1f}%',
                    'last_id':    last_id,
                })

    print(f"\nDone. Processed: {total_processed:,}")
    print(f"Relevant: {total_relevant:,} ({100*total_relevant/max(total_processed,1):.1f}%)")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
