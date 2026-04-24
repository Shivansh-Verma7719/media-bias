"""
Ensemble evaluation: average p_rel from multiple trained models, sweep thresholds.

Usage:
  python 08_ensemble_eval.py -t test.csv -m model_deberta model_synthetic
  python 08_ensemble_eval.py -t test.csv -m model_deberta model_synthetic --weights 1 2
"""
import argparse
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_score, recall_score, f1_score


class TitleDataset(Dataset):
    def __init__(self, titles, tokenizer, max_len=128):
        self.titles = titles
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.titles[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def get_probs(model_path, titles, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    ds = TitleDataset(titles, tokenizer)
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    probs = []
    with torch.no_grad():
        for batch in dl:
            inputs = {k: v.to(device) for k, v in batch.items()}
            out = model(**inputs)
            p = torch.softmax(out.logits, dim=1)[:, 1]
            probs.extend(p.cpu().tolist())
    return np.array(probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",    "-t", required=True)
    parser.add_argument("--models",  "-m", nargs="+", required=True)
    parser.add_argument("--weights", "-w", nargs="+", type=float, default=None,
                        help="Per-model weights for averaging (default: equal)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(args.test).dropna(subset=["title", "label"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["relevant", "irrelevant"])].reset_index(drop=True)
    true_labels = (df["label"] == "relevant").astype(int).to_numpy()
    titles = df["title"].tolist()

    weights = args.weights if args.weights else [1.0] * len(args.models)
    assert len(weights) == len(args.models), "Number of weights must match number of models"
    weights = np.array(weights) / sum(weights)

    all_probs = []
    for model_path, w in zip(args.models, weights):
        print(f"Loading {model_path} (weight={w:.2f})...")
        p = get_probs(model_path, titles, device)
        all_probs.append(p * w)
        solo_preds = (p >= 0.70).astype(int)
        sp = precision_score(true_labels, solo_preds, zero_division=0)
        sr = recall_score(true_labels, solo_preds, zero_division=0)
        print(f"  Solo @ 0.70: P={sp:.3f} R={sr:.3f}")

    ensemble_probs = np.sum(all_probs, axis=0)

    print(f"\nEnsemble of: {args.models}")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'FP':>5} {'FN':>5} {'N_pred':>8}")
    for t in np.arange(0.40, 0.91, 0.05):
        preds = (ensemble_probs >= t).astype(int)
        n_pred = preds.sum()
        if n_pred == 0:
            continue
        p = precision_score(true_labels, preds, zero_division=0)
        r = recall_score(true_labels, preds, zero_division=0)
        f = f1_score(true_labels, preds, zero_division=0)
        fp = int(((preds == 1) & (true_labels == 0)).sum())
        fn = int(((preds == 0) & (true_labels == 1)).sum())
        flag = " <-- P>=0.90" if p >= 0.90 else ""
        print(f"  {t:>8.2f} {p:>10.3f} {r:>8.3f} {f:>8.3f} {fp:>5} {fn:>5} {n_pred:>8}{flag}")

    print("\n── FP analysis at threshold 0.70 ──")
    preds_070 = (ensemble_probs >= 0.70).astype(int)
    df["p_ensemble"] = ensemble_probs
    fps = df[(df["label"] == "irrelevant") & (preds_070 == 1)].sort_values("p_ensemble", ascending=False)
    for _, r in fps.iterrows():
        print(f"  p={r['p_ensemble']:.3f} [{r.get('company_name', '')}] {r['title'][:90]}")


if __name__ == "__main__":
    main()
