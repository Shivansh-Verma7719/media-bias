import argparse, torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MAX_LEN = 128

class TitleDataset(Dataset):
    def __init__(self, titles, tokenizer):
        self.titles = titles
        self.tokenizer = tokenizer
    def __len__(self): return len(self.titles)
    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.titles[idx]), max_length=MAX_LEN,
                             padding='max_length', truncation=True, return_tensors='pt')
        return {k: v.squeeze(0) for k, v in enc.items()}

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", "-m", default="model_deberta")
parser.add_argument("--test", "-t", default="test.csv")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading {args.model_path} on {device}")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
model.eval()

df = pd.read_csv(args.test).dropna(subset=['title', 'label']).copy()
df['label'] = df['label'].str.strip().str.lower()
df = df[df['label'].isin(['relevant', 'irrelevant'])]

ds = TitleDataset(df['title'].tolist(), tokenizer)
dl = DataLoader(ds, batch_size=64, shuffle=False)
probs = []
with torch.no_grad():
    for batch in dl:
        inputs = {k: v.to(device) for k, v in batch.items()}
        out = model(**inputs)
        p = torch.nn.functional.softmax(out.logits, dim=-1)
        probs.extend(p[:, 1].cpu().tolist())

df['p_rel'] = probs
true = df['label'].map({'relevant': 1, 'irrelevant': 0}).tolist()

print(f"\n{'Threshold':>10}  {'Precision':>9}  {'Recall':>7}  {'F1':>7}  {'FP':>4}  {'FN':>4}")
print("-" * 55)
for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]:
    preds = [1 if p >= t else 0 for p in probs]
    tp = sum(a == 1 and b == 1 for a, b in zip(true, preds))
    fp = sum(a == 0 and b == 1 for a, b in zip(true, preds))
    fn = sum(a == 1 and b == 0 for a, b in zip(true, preds))
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec  = tp / (tp + fn) if tp + fn > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
    print(f"{t:>10.2f}  {prec:>9.3f}  {rec:>7.3f}  {f1:>7.3f}  {fp:>4}  {fn:>4}")

print("\n── TP p_rel distribution (all relevant articles) ──")
tp_df = df[df['label'] == 'relevant']
for lo, hi in [(0.0,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.75),(0.75,0.8),(0.8,0.85),(0.85,0.9),(0.9,1.01)]:
    n = int(((tp_df['p_rel'] >= lo) & (tp_df['p_rel'] < hi)).sum())
    print(f"  [{lo:.2f}-{hi:.2f})  {n:>3}  {'█' * n}")
