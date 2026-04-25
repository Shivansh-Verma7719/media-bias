"""
Post-processing evaluation: ensemble + rule-based adjustment + threshold sweep.

Usage:
  # Single model + rules
  python 09_postprocess_eval.py -t test.csv -m model_synthetic_v2

  # 3-model ensemble + rules
  python 09_postprocess_eval.py -t test.csv \
      -m model_deberta model_synthetic model_synthetic_v2

  # Ensemble without rules (to compare)
  python 09_postprocess_eval.py -t test.csv \
      -m model_deberta model_synthetic model_synthetic_v2 --no_rules
"""
import argparse
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_score, recall_score, f1_score

from rule_adjuster import adjust_batch


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


def sweep(true_labels, probs, label=""):
    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'FP':>5} {'FN':>5} {'N_pred':>8}")
    print("-"*60)
    best = {"f1": 0}
    for t in np.arange(0.40, 0.96, 0.05):
        preds = (probs >= t).astype(int)
        n = preds.sum()
        if n == 0:
            continue
        p = precision_score(true_labels, preds, zero_division=0)
        r = recall_score(true_labels, preds, zero_division=0)
        f = f1_score(true_labels, preds, zero_division=0)
        fp = int(((preds == 1) & (true_labels == 0)).sum())
        fn = int(((preds == 0) & (true_labels == 1)).sum())
        flag = " <-- P>=0.90" if p >= 0.90 else ""
        print(f"  {t:>8.2f} {p:>10.3f} {r:>8.3f} {f:>8.3f} {fp:>5} {fn:>5} {n:>8}{flag}")
        if f > best["f1"]:
            best = {"threshold": round(float(t), 2), "f1": f, "precision": p, "recall": r, "fp": fp, "fn": fn}
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",     "-t", required=True)
    parser.add_argument("--models",   "-m", nargs="+", required=True)
    parser.add_argument("--weights",  "-w", nargs="+", type=float, default=None)
    parser.add_argument("--no_rules", action="store_true",
                        help="Skip rule-based adjustment (compare ensemble alone)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Models: {args.models}")

    df = pd.read_csv(args.test).dropna(subset=["title", "label"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["relevant", "irrelevant"])].reset_index(drop=True)
    true_labels = (df["label"] == "relevant").astype(int).to_numpy()
    titles = df["title"].tolist()
    companies = df["company_name"].tolist() if "company_name" in df.columns else [""] * len(df)

    weights = args.weights if args.weights else [1.0] * len(args.models)
    weights = np.array(weights) / sum(weights)

    all_probs = []
    for model_path, w in zip(args.models, weights):
        print(f"\nLoading {model_path} (weight={w:.2f})...")
        p = get_probs(model_path, titles, device)
        all_probs.append(p * w)
        solo_best = sweep(true_labels, p, label=f"Solo: {model_path}")
        print(f"  Best solo: P={solo_best['precision']:.3f} R={solo_best['recall']:.3f} "
              f"F1={solo_best['f1']:.3f} @ t={solo_best['threshold']}")

    ensemble_probs = np.sum(all_probs, axis=0)
    ensemble_best = sweep(true_labels, ensemble_probs,
                          label=f"Ensemble ({len(args.models)} models, no rules)")
    print(f"\n  Best ensemble: P={ensemble_best['precision']:.3f} R={ensemble_best['recall']:.3f} "
          f"F1={ensemble_best['f1']:.3f} @ t={ensemble_best['threshold']}")

    if not args.no_rules:
        adjusted_probs, fired_rules = adjust_batch(titles, companies, ensemble_probs.tolist())
        adjusted_probs = np.array(adjusted_probs)

        rule_counts = {}
        for r in fired_rules:
            if r:
                rule_counts[r] = rule_counts.get(r, 0) + 1
        if rule_counts:
            print(f"\n  Rules fired ({sum(rule_counts.values())} total):")
            for rule, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
                print(f"    {rule}: {count}")

        adj_best = sweep(true_labels, adjusted_probs,
                         label="Ensemble + rule post-processor")
        print(f"\n  Best with rules: P={adj_best['precision']:.3f} R={adj_best['recall']:.3f} "
              f"F1={adj_best['f1']:.3f} @ t={adj_best['threshold']}")

        # Show remaining FPs at best threshold
        t_best = adj_best["threshold"]
        preds = (adjusted_probs >= t_best).astype(int)
        df["p_adj"] = adjusted_probs
        df["rule"] = fired_rules
        fps = df[(df["label"] == "irrelevant") & (preds == 1)].sort_values("p_adj", ascending=False)
        fns = df[(df["label"] == "relevant") & (preds == 0)].sort_values("p_adj", ascending=False)

        print(f"\n── Remaining FPs at t={t_best} ({len(fps)}) ──")
        for _, r in fps.iterrows():
            print(f"  p={r['p_adj']:.3f} rule={r['rule'] or '-':30s} "
                  f"[{r.get('company_name', '')}] {r['title'][:70]}")

        print(f"\n── Remaining FNs at t={t_best} ({len(fns)}) ──")
        for _, r in fns.iterrows():
            print(f"  p={r['p_adj']:.3f} [{r.get('company_name', '')}] {r['title'][:70]}")

        # Summary comparison
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"  Ensemble alone:   P={ensemble_best['precision']:.3f}  R={ensemble_best['recall']:.3f}  "
              f"F1={ensemble_best['f1']:.3f}  FP={ensemble_best['fp']}  FN={ensemble_best['fn']}")
        print(f"  + Rule adjuster:  P={adj_best['precision']:.3f}  R={adj_best['recall']:.3f}  "
              f"F1={adj_best['f1']:.3f}  FP={adj_best['fp']}  FN={adj_best['fn']}")
        print(f"  Target:           P>=0.90  R>=0.80")


if __name__ == "__main__":
    main()
