"""
Stage 4: Evaluate the trained model against the held-out 60-row test set.
Prints precision, recall, F1, and shows all errors for analysis.

Usage:
  python 04_evaluate.py --model_path best_model --test 300_test.csv
"""
import argparse, os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

MAX_LEN = 128
# Asymmetric thresholds: higher bar to call something relevant.
# Standard practice for imbalanced binary classification — adjusts decision
# boundary to favour precision on the minority class.
RELEVANT_THRESHOLD   = 0.75   # P(relevant) must exceed this to predict relevant
IRRELEVANT_THRESHOLD = 0.50   # P(irrelevant) must exceed this to predict irrelevant


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


def run_inference(model, tokenizer, titles, device):
    ds = TitleDataset(titles, tokenizer)
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    model.eval()
    rel_probs = []
    with torch.no_grad():
        for batch in dl:
            inputs = {k: v.to(device) for k, v in batch.items()}
            out = model(**inputs)
            probs = torch.nn.functional.softmax(out.logits, dim=-1)
            rel_probs.extend(probs[:, 1].cpu().tolist())
    return rel_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, default="best_model")
    parser.add_argument("--test",       "-t", type=str, default="300_test.csv")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.model_path}  |  device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)

    df = pd.read_csv(args.test)
    df = df.dropna(subset=['title', 'label']).copy()
    df['label'] = df['label'].str.strip().str.lower()
    df = df[df['label'].isin(['relevant', 'irrelevant'])]

    rel_probs = run_inference(model, tokenizer, df['title'].tolist(), device)
    df['p_relevant'] = rel_probs
    df['predicted']  = [
        'relevant'   if p >= RELEVANT_THRESHOLD   else
        'irrelevant' if p <= (1 - IRRELEVANT_THRESHOLD) else
        'uncertain'
        for p in rel_probs
    ]

    uncertain = df[df['predicted'] == 'uncertain']
    scored    = df[df['predicted'] != 'uncertain']

    print(f"\nTotal: {len(df)}  |  Scored: {len(scored)}  |  Uncertain: {len(uncertain)}")
    print(f"Thresholds: relevant≥{RELEVANT_THRESHOLD}  irrelevant≥{IRRELEVANT_THRESHOLD}")

    y_true = scored['label'].tolist()
    y_pred = scored['predicted'].tolist()

    print("\n" + "="*55)
    print("Classification Report")
    print("="*55)
    print(classification_report(y_true, y_pred, target_names=['irrelevant', 'relevant'], digits=3))

    cm = confusion_matrix(y_true, y_pred, labels=['irrelevant', 'relevant'])
    print("Confusion Matrix (rows=true, cols=pred):")
    print(f"               pred_irr  pred_rel")
    print(f"  true_irr       {cm[0][0]:>5}     {cm[0][1]:>5}")
    print(f"  true_rel       {cm[1][0]:>5}     {cm[1][1]:>5}")

    fp = scored[(scored['label'] == 'irrelevant') & (scored['predicted'] == 'relevant')]
    fn = scored[(scored['label'] == 'relevant')   & (scored['predicted'] == 'irrelevant')]

    print(f"\n── False Positives ({len(fp)}) ──────────────────────────────")
    for _, r in fp.sort_values('p_relevant', ascending=False).iterrows():
        print(f"  p_rel={r['p_relevant']:.3f} [{r.get('company_name','')}] {r['title'][:80]}")

    print(f"\n── False Negatives ({len(fn)}) ──────────────────────────────")
    for _, r in fn.sort_values('p_relevant', ascending=False).iterrows():
        print(f"  p_rel={r['p_relevant']:.3f} [{r.get('company_name','')}] {r['title'][:80]}")

    if len(uncertain) > 0:
        print(f"\n── Uncertain (skipped) ({len(uncertain)}) ─────────────────────────")
        for _, r in uncertain.sort_values('p_relevant', ascending=False).iterrows():
            print(f"  p_rel={r['p_relevant']:.3f} [{r.get('company_name','')}] {r['title'][:80]}")


if __name__ == "__main__":
    main()
