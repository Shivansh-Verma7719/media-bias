"""
Stage 10: Full inference using 3-model ensemble + rule-based post-processor.

Ensemble: model_deberta + model_synthetic + model_synthetic_v2 (equal weights).
Threshold: p_adj >= 0.65 → relevant, else irrelevant.

Resumes from checkpoint — safe to interrupt and restart.

Usage:
  python 10_full_inference_ensemble.py \
      -m model_deberta model_synthetic model_synthetic_v2 \
      -o full_predictions_ensemble.csv

  # Custom weights (deberta:1, synthetic:1, synthetic_v2:2)
  python 10_full_inference_ensemble.py \
      -m model_deberta model_synthetic model_synthetic_v2 \
      -w 1 1 2 \
      -o full_predictions_ensemble.csv
"""
import os
import re
import sys
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from dotenv import load_dotenv

from rule_adjuster import adjust_batch

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

DB_BATCH_SIZE        = 5000
INFERENCE_BATCH_SIZE = 256
MAX_LEN              = 128
ENSEMBLE_THRESHOLD   = 0.65

# Hard blocklist: sports/entertainment sources never contain relevant financial news
SOURCE_BLOCKLIST = {
    'axs.com', 'bleacherreport.com', 'cbssports.com', 'sbnation.com',
    'sportingnews.com', 'foxsports.com', 'espn.com', 'fansided.com',
    'nbcsports.com', 'hollywoodlife.com', 'deadspin.com', 'usmagazine.com',
    'radaronline.com', 'pitchfork.com', 'monstersandcritics.com',
}

# Hard content filter: consumer deal / product review content
CONSUMER_CONTENT_RE = re.compile(
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
    re.IGNORECASE,
)

# Hard filter: analyst rating actions by named firms
# These are not the company's own news; we exclude them from relevant.
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
    re.IGNORECASE,
)

# Hard company-name contamination filters not covered by rule_adjuster
COMPANY_EXCLUSIONS = {
    'Intel': re.compile(
        r'\b(military\s+intel(?:ligence)?|intelligence\s+agenc'
        r'|CIA\s+intel|NSA\s+intel|spy|espionage|gather(?:ing)?\s+intel'
        r'|street\s+intel|competitive\s+intel(?:ligence)?'
        r'|tower\s+22|drone\s+attack|airstrike|counterterrorism)\b',
        re.IGNORECASE,
    ),
    'Target': re.compile(
        r'\b(military\s+target|bombing\s+target|airstrike\s+target'
        r'|missile\s+target|shooting\s+target|target\s+practice'
        r'|archery\s+target|sniper\s+target)\b',
        re.IGNORECASE,
    ),
}


def hard_filter(title: str, source: str, company_name: str) -> bool:
    """Return True if this article should be forced irrelevant before inference."""
    if str(source).lower() in SOURCE_BLOCKLIST:
        return True
    if CONSUMER_CONTENT_RE.search(str(title)):
        return True
    if company_name in ANALYST_FIRMS:
        co_re = re.compile(re.escape(company_name), re.IGNORECASE)
        if co_re.search(str(title)) and ANALYST_ACTION_RE.search(str(title)):
            return True
    if company_name in COMPANY_EXCLUSIONS and COMPANY_EXCLUSIONS[company_name].search(str(title)):
        return True
    return False


class TitleDataset(Dataset):
    def __init__(self, titles, tokenizer):
        self.titles = titles
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.titles[idx]),
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def get_probs(model, tokenizer, titles: list, device) -> np.ndarray:
    ds = TitleDataset(titles, tokenizer)
    dl = DataLoader(
        ds,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=2 if str(device) != 'cpu' else 0,
    )
    probs = []
    with torch.no_grad():
        for batch in dl:
            inputs = {k: v.to(device) for k, v in batch.items()}
            out = model(**inputs)
            p = torch.softmax(out.logits, dim=1)[:, 1]
            probs.extend(p.cpu().tolist())
    return np.array(probs)


def load_models(model_paths: list[str], weights: list[float], device):
    models, tokenizers = [], []
    for path in model_paths:
        print(f"  Loading {path}...")
        tokenizers.append(AutoTokenizer.from_pretrained(path))
        m = AutoModelForSequenceClassification.from_pretrained(path).to(device)
        m.eval()
        m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
        models.append(m)
    return models, tokenizers, np.array(weights) / sum(weights)


def ensemble_probs(models, tokenizers, weights, titles: list, device) -> np.ndarray:
    combined = np.zeros(len(titles))
    for model, tokenizer, w in zip(models, tokenizers, weights):
        combined += get_probs(model, tokenizer, titles, device) * w
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models',   '-m', nargs='+', required=True,
                        help='Model paths in order (e.g. model_deberta model_synthetic model_synthetic_v2)')
    parser.add_argument('--weights',  '-w', nargs='+', type=float, default=None,
                        help='Per-model weights (default: equal)')
    parser.add_argument('--output',   '-o', type=str, default='full_predictions_ensemble.csv')
    parser.add_argument('--db_batch', type=int, default=DB_BATCH_SIZE)
    parser.add_argument('--threshold', '-t', type=float, default=ENSEMBLE_THRESHOLD)
    args = parser.parse_args()

    weights = args.weights if args.weights else [1.0] * len(args.models)
    if len(weights) != len(args.models):
        print('ERROR: number of weights must match number of models')
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  |  Threshold: {args.threshold}')
    print(f'Models: {args.models}')

    print('\nLoading models...')
    models, tokenizers, weights_norm = load_models(args.models, weights, device)

    from supabase import create_client
    sb = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

    print('\nLoading company map...')
    co_resp = sb.table('companies').select('id,symbol,name').execute()
    company_map = {
        str(r['id']): {'symbol': r['symbol'], 'name': r['name']}
        for r in co_resp.data
    }
    print(f'  {len(company_map)} companies loaded.')

    # Resume support
    processed_ids: set[str] = set()
    if os.path.exists(args.output):
        existing = pd.read_csv(args.output, usecols=['id'])
        processed_ids = set(existing['id'].astype(str).tolist())
        print(f'Resuming: {len(processed_ids):,} already processed.')
    else:
        pd.DataFrame(columns=[
            'id', 'title', 'url', 'source', 'published_at',
            'company_id', 'company_symbol', 'company_name',
            'media_outlet_id', 'pos_score', 'neutral_score', 'neg_score',
            'predicted_label', 'ensemble_prob', 'adj_prob', 'rule_fired',
        ]).to_csv(args.output, index=False)

    count_resp = (sb.table('articles_no_title_deduped')
                    .select('id', count='exact')
                    .limit(1)
                    .execute())
    total_rows = count_resp.count
    print(f'Total articles in DB: {total_rows:,}')

    last_id = 0
    total_processed = len(processed_ids)
    total_relevant  = 0
    total_hard_filtered = 0

    out_cols = [
        'id', 'title', 'url', 'source', 'published_at',
        'company_id', 'company_symbol', 'company_name',
        'media_outlet_id', 'pos_score', 'neutral_score', 'neg_score',
        'predicted_label', 'ensemble_prob', 'adj_prob', 'rule_fired',
    ]

    with tqdm(total=total_rows, initial=total_processed,
              desc='Inference', unit='articles') as pbar:
        while True:
            resp = (sb.table('articles_no_title_deduped')
                      .select('id,title,url,source,published_at,company_id,'
                              'media_outlet_id,pos_score,neutral_score,neg_score')
                      .not_.is_('title', 'null')
                      .gt('id', last_id)
                      .order('id')
                      .limit(args.db_batch)
                      .execute())

            rows = resp.data
            if not rows:
                break
            last_id = rows[-1]['id']

            new_rows = [r for r in rows if str(r['id']) not in processed_ids]
            if not new_rows:
                continue

            df = pd.DataFrame(new_rows)
            df['company_symbol'] = df['company_id'].apply(
                lambda x: company_map.get(str(x), {}).get('symbol', '') if x else '')
            df['company_name'] = df['company_id'].apply(
                lambda x: company_map.get(str(x), {}).get('name', '') if x else '')

            titles       = df['title'].fillna('').tolist()
            sources      = df['source'].fillna('').tolist()
            company_names = df['company_name'].tolist()

            # Hard filter: mark as irrelevant before running models
            hard_mask = np.array([
                hard_filter(t, s, c)
                for t, s, c in zip(titles, sources, company_names)
            ])
            inference_idx = np.where(~hard_mask)[0]

            df['ensemble_prob'] = np.nan
            df['adj_prob']      = np.nan
            df['rule_fired']    = ''
            df['predicted_label'] = 'irrelevant'

            total_hard_filtered += int(hard_mask.sum())

            if len(inference_idx) > 0:
                inf_titles   = [titles[i]        for i in inference_idx]
                inf_companies = [company_names[i] for i in inference_idx]

                ens_p = ensemble_probs(models, tokenizers, weights_norm,
                                       inf_titles, device)
                adj_p, rules = adjust_batch(inf_titles, inf_companies, ens_p.tolist())
                adj_p = np.array(adj_p)

                for out_i, (orig_i, ep, ap, rule) in enumerate(
                    zip(inference_idx, ens_p, adj_p, rules)
                ):
                    df.at[orig_i, 'ensemble_prob'] = round(float(ep), 4)
                    df.at[orig_i, 'adj_prob']      = round(float(ap), 4)
                    df.at[orig_i, 'rule_fired']    = rule
                    df.at[orig_i, 'predicted_label'] = (
                        'relevant' if ap >= args.threshold else 'irrelevant'
                    )

            n_rel = (df['predicted_label'] == 'relevant').sum()
            total_relevant  += int(n_rel)
            total_processed += len(new_rows)
            processed_ids.update(df['id'].astype(str).tolist())

            df[out_cols].to_csv(args.output, mode='a', header=False, index=False)

            pbar.update(len(new_rows))
            pbar.set_postfix({
                'relevant':   total_relevant,
                'rel%':       f'{100 * total_relevant / max(total_processed, 1):.1f}%',
                'hard_filt':  total_hard_filtered,
                'last_id':    last_id,
            })

    print(f'\nDone.')
    print(f'  Processed:     {total_processed:,}')
    print(f'  Hard-filtered: {total_hard_filtered:,}')
    print(f'  Relevant:      {total_relevant:,}  '
          f'({100 * total_relevant / max(total_processed, 1):.1f}%)')
    print(f'  Output:        {args.output}')


if __name__ == '__main__':
    main()
