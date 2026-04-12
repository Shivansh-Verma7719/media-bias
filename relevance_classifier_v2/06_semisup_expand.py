"""
Stage 6: Semi-supervised expansion via self-training / pseudo-labeling.

1. Load the gold-only trained model (from 03_train_finbert.py --gold_only)
2. Run inference on unlabeled articles from 00_fetch_filtered.py output
3. Keep predictions with confidence >= threshold as pseudo-labels
4. Combine gold + pseudo-labels into a new training CSV
5. Print distribution — you then re-run 03_train_finbert.py on this CSV

This approach is standard in NLP (cite: "self-training" / "pseudo-labeling").
The confidence threshold is the key hyperparameter: higher = cleaner but fewer.
Start at 0.90; if pseudo-labels are too few, try 0.85.

Usage:
  python 06_semisup_expand.py \
      --model  model_gold_only \
      --gold   04_training_data_verified.csv \
      --unlabeled 00_filtered_clean.csv \
      --output 06_semisup_training.csv \
      --threshold 0.90
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm


class TitleDataset(Dataset):
    def __init__(self, titles, tokenizer, max_len=128):
        self.titles    = titles
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        enc = self.tokenizer.encode_plus(
            str(self.titles[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
        }


def run_inference(model, tokenizer, titles, batch_size, device, max_len=128):
    """Returns (pred_labels, confidences) arrays."""
    ds = TitleDataset(titles, tokenizer, max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(dl, desc="Inference", unit="batch"):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    probs_arr = np.vstack(all_probs)  # shape (N, 2): [p_irrelevant, p_relevant]
    pred_labels  = np.argmax(probs_arr, axis=1)         # 0=irrelevant, 1=relevant
    confidences  = np.max(probs_arr, axis=1)            # max class prob
    return pred_labels, confidences, probs_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True, help="Path to gold-only trained model dir")
    parser.add_argument("--gold",       required=True, help="Gold training CSV (source=gold_manual rows)")
    parser.add_argument("--unlabeled",  required=True, help="Unlabeled articles CSV (00_filtered_clean.csv)")
    parser.add_argument("--output",     default="06_semisup_training.csv")
    parser.add_argument("--threshold",  type=float, default=0.90,
                        help="Min confidence for a pseudo-label to be kept (default 0.90)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_pseudo_per_company", type=int, default=300,
                        help="Cap pseudo-labels per company to prevent imbalance")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from {args.model}...")
    tokenizer = BertTokenizer.from_pretrained(args.model)
    model     = BertForSequenceClassification.from_pretrained(args.model)
    model     = model.to(device)

    # ── Load gold data ────────────────────────────────────────────────────────
    gold = pd.read_csv(args.gold)
    if "source" in gold.columns:
        gold = gold[gold["source"] == "gold_manual"].copy()
    gold_ids = set(gold["id"].astype(str))
    print(f"Gold rows: {len(gold)} ({gold['label'].value_counts().to_dict()})")

    # ── Load unlabeled pool ───────────────────────────────────────────────────
    unlabeled = pd.read_csv(args.unlabeled)
    unlabeled["id"] = unlabeled["id"].astype(str)
    # Remove articles already in gold
    unlabeled = unlabeled[~unlabeled["id"].isin(gold_ids)].reset_index(drop=True)
    print(f"Unlabeled pool: {len(unlabeled)} articles (after removing gold IDs)")

    # ── Inference ─────────────────────────────────────────────────────────────
    titles = unlabeled["title"].tolist()
    preds, confs, probs = run_inference(model, tokenizer, titles, args.batch_size, device)

    unlabeled["_pred"]  = preds
    unlabeled["_conf"]  = confs
    unlabeled["_p_rel"] = probs[:, 1]

    # ── Filter by confidence threshold ────────────────────────────────────────
    high_conf = unlabeled[unlabeled["_conf"] >= args.threshold].copy()
    high_conf["label"]  = high_conf["_pred"].map({1: "relevant", 0: "irrelevant"})
    high_conf["source"] = "pseudo_label"

    print(f"\nPseudo-labels at threshold {args.threshold}:")
    print(f"  Total high-conf:  {len(high_conf)}")
    print(f"  Relevant:         {(high_conf['label']=='relevant').sum()}")
    print(f"  Irrelevant:       {(high_conf['label']=='irrelevant').sum()}")

    # ── Company cap on pseudo-labels ─────────────────────────────────────────
    if "company_name" in high_conf.columns:
        capped = []
        for company, grp in high_conf.groupby("company_name"):
            rel = grp[grp["label"] == "relevant"]
            irr = grp[grp["label"] == "irrelevant"]
            rel_cap = args.max_pseudo_per_company // 3
            irr_cap = args.max_pseudo_per_company
            capped.append(rel.sample(min(len(rel), rel_cap), random_state=42))
            capped.append(irr.sample(min(len(irr), irr_cap), random_state=42))
        high_conf = pd.concat(capped, ignore_index=True)
        print(f"\nAfter company cap ({args.max_pseudo_per_company} irr / "
              f"{args.max_pseudo_per_company//3} rel per company):")
        print(f"  Total: {len(high_conf)}")
        print(f"  {high_conf['label'].value_counts().to_dict()}")

    # ── Combine ───────────────────────────────────────────────────────────────
    keep_cols = ["id", "title", "label", "source"]
    if "company_name" in high_conf.columns:
        keep_cols.append("company_name")

    combined = pd.concat([
        gold[[c for c in keep_cols if c in gold.columns]],
        high_conf[[c for c in keep_cols if c in high_conf.columns]],
    ], ignore_index=True)
    combined = combined.drop_duplicates(subset="id", keep="first")

    n_total = len(combined)
    n_rel   = (combined["label"] == "relevant").sum()
    n_irr   = (combined["label"] == "irrelevant").sum()
    n_gold  = (combined["source"] == "gold_manual").sum()
    n_pslab = (combined["source"] == "pseudo_label").sum()

    print(f"\n{'='*55}")
    print(f"SEMI-SUPERVISED TRAINING SET")
    print(f"{'='*55}")
    print(f"Total:          {n_total}")
    print(f"  Relevant:     {n_rel}  ({100*n_rel/n_total:.1f}%)")
    print(f"  Irrelevant:   {n_irr}  ({100*n_irr/n_total:.1f}%)")
    print(f"  Gold manual:  {n_gold}")
    print(f"  Pseudo-label: {n_pslab}  (threshold={args.threshold})")

    combined.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")
    print(f"\nNext: re-train with the expanded set:")
    print(f"  python 03_train_finbert.py -i {args.output} -s model_semisup")


if __name__ == "__main__":
    main()
