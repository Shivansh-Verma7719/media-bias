"""
Stage 3: Fine-tune FinBERT (ProsusAI/finbert) on the combined training set.
Uses stratified k-fold CV and saves the best checkpoint.

Two training modes:
  --gold_only   Train on gold-labeled rows only (source == 'gold_manual').
                Use this to establish a clean baseline F1 before adding
                LLM pseudo-labels. Report this number in the paper.
  (default)     Train on all rows (gold + pseudo-labels / LLM).

Usage:
  # Gold-only baseline (for paper)
  python 03_train_finbert.py -i 04_training_data_verified.csv -s model_gold_only --gold_only

  # Full training set
  python 03_train_finbert.py -i 04_training_data_verified.csv -s model_full
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import pandas as pd
import os

PRE_TRAINED_MODEL = 'ProsusAI/finbert'


class TitleDataset(Dataset):
    def __init__(self, titles, labels, tokenizer, max_len):
        self.titles = titles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        enc = self.tokenizer.encode_plus(
            str(self.titles[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_epoch(model, loader, optimizer, scheduler, device, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="  Train", leave=False):
        optimizer.zero_grad()
        ids   = batch['input_ids'].to(device)
        mask  = batch['attention_mask'].to(device)
        lbls  = batch['labels'].to(device)
        out   = model(input_ids=ids, attention_mask=mask)
        loss  = criterion(out.logits, lbls)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    return total_loss / len(loader)


def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_labels, losses = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbls = batch['labels'].to(device)
            out  = model(input_ids=ids, attention_mask=mask, labels=lbls)
            losses.append(out.loss.item())
            preds = torch.argmax(out.logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(lbls.cpu().tolist())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1  = f1_score(all_labels, all_preds, pos_label=1, average='binary')
    return acc, np.mean(losses), f1, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",           "-i", type=str, required=True)
    parser.add_argument("--model_save_path", "-s", type=str, default="best_model")
    parser.add_argument("--batch_size",      "-b", type=int, default=16)
    parser.add_argument("--epochs",          "-e", type=int, default=4)
    parser.add_argument("--kfolds",          "-k", type=int, default=5)
    parser.add_argument("--max_len",         "-m", type=int, default=128)
    parser.add_argument("--gold_only",             action="store_true",
                        help="Train on gold_manual rows only (baseline mode)")
    args = parser.parse_args()

    df = pd.read_csv(args.input).dropna(subset=['title', 'label'])
    label_map = {'relevant': 1, 'irrelevant': 0}
    df['target'] = df['label'].str.lower().map(label_map)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)

    if args.gold_only:
        if 'source' not in df.columns:
            raise ValueError("--gold_only requires a 'source' column in the CSV")
        df = df[df['source'] == 'gold_manual'].copy()
        print(f"GOLD-ONLY BASELINE MODE: {len(df)} gold samples")
    else:
        print(f"FULL TRAINING MODE: {len(df)} samples")

    print(df['target'].value_counts())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)
    titles  = df['title'].to_numpy()
    targets = df['target'].to_numpy()

    skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=42)
    fold_f1s = []
    best_f1  = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(titles, targets)):
        print(f"\n══════ Fold {fold+1}/{args.kfolds} ══════")

        train_titles, val_titles = titles[train_idx], titles[val_idx]
        train_targets, val_targets = targets[train_idx], targets[val_idx]

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_targets),
            y=train_targets
        )
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float).to(device)
        )

        train_ds = TitleDataset(train_titles, train_targets, tokenizer, args.max_len)
        val_ds   = TitleDataset(val_titles,   val_targets,   tokenizer, args.max_len)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

        model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL, num_labels=2, ignore_mismatched_sizes=True)
        model = model.to(device)

        optimizer = AdamW(model.parameters(), lr=2e-5)
        total_steps = len(train_dl) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )

        fold_best_f1 = 0.0
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_dl, optimizer, scheduler, device, criterion)
            val_acc, val_loss, val_f1, preds, reals = eval_epoch(model, val_dl, device)
            print(f"  Epoch {epoch+1}/{args.epochs} | "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"acc={val_acc:.4f} f1={val_f1:.4f}")

            if val_f1 > fold_best_f1:
                fold_best_f1 = val_f1
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    print(f"  ★ New best F1={val_f1:.4f} — saving to {args.model_save_path}")
                    model.save_pretrained(args.model_save_path)
                    tokenizer.save_pretrained(args.model_save_path)
                    # Save classification report for best fold
                    report = classification_report(
                        reals, preds,
                        target_names=['irrelevant', 'relevant'],
                        digits=3
                    )
                    with open(os.path.join(args.model_save_path, 'best_fold_report.txt'), 'w') as f:
                        f.write(f"Fold {fold+1}, Epoch {epoch+1}\n\n{report}")

        fold_f1s.append(fold_best_f1)
        print(f"  Fold {fold+1} best F1: {fold_best_f1:.4f}")

    print("\n" + "="*50)
    print("Cross-Validation F1 Results")
    print("="*50)
    for i, f in enumerate(fold_f1s):
        print(f"  Fold {i+1}: {f:.4f}")
    print(f"  Mean F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print(f"  Best global F1: {best_f1:.4f}")
    print(f"  Model saved to: {args.model_save_path}")


if __name__ == "__main__":
    main()
