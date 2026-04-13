"""
Stage 3: Fine-tune a transformer classifier on the combined training set.
Uses stratified k-fold CV and saves the best checkpoint.

Model-agnostic: pass any HuggingFace model via --base_model.
Recommended models to compare:
  ProsusAI/finbert          (financial domain, 110M params — original)
  microsoft/deberta-v3-base (stronger architecture, 86M params — try this)
  roberta-base              (general, 125M params)

Usage:
  python 03_train_finbert.py -i combined_gold.csv -s model_deberta --gold_only
  python 03_train_finbert.py -i combined_gold.csv -s model_finbert --gold_only --base_model ProsusAI/finbert
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import pandas as pd
import os

DEFAULT_MODEL = 'microsoft/deberta-v3-base'


class TitleDataset(Dataset):
    def __init__(self, titles, labels, tokenizer, max_len):
        self.titles = titles
        self.labels = labels
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
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_epoch(model, loader, optimizer, scheduler, device, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="  Train", leave=False):
        optimizer.zero_grad()
        labels = batch.pop('labels').to(device)
        inputs = {k: v.to(device) for k, v in batch.items()}
        out = model(**inputs)
        loss = criterion(out.logits, labels)
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
            labels = batch.pop('labels').to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            out = model(**inputs, labels=labels)
            losses.append(out.loss.item())
            preds = torch.argmax(out.logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1  = f1_score(all_labels, all_preds, pos_label=1, average='binary')
    return acc, np.mean(losses), f1, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",           "-i", type=str, required=True)
    parser.add_argument("--model_save_path", "-s", type=str, default="best_model")
    parser.add_argument("--base_model",      "-M", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace model name (default: microsoft/deberta-v3-base)")
    parser.add_argument("--batch_size",      "-b", type=int, default=16)
    parser.add_argument("--epochs",          "-e", type=int, default=6)
    parser.add_argument("--kfolds",          "-k", type=int, default=5)
    parser.add_argument("--max_len",         "-m", type=int, default=128)
    args = parser.parse_args()

    print(f"Base model: {args.base_model}")

    df = pd.read_csv(args.input).dropna(subset=['title', 'label'])
    label_map = {'relevant': 1, 'irrelevant': 0}
    df['target'] = df['label'].str.lower().map(label_map)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)
    print(f"Training on {len(df)} samples")

    print(df['target'].value_counts())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
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
        # Cap weight ratio at 1.5 — fully balanced (~2:1) causes the model to
        # over-predict relevant when the true relevant rate is ~10-15%.
        if len(class_weights) == 2 and class_weights[1] / class_weights[0] > 1.5:
            class_weights[1] = class_weights[0] * 1.5
        print(f"  Class weights: irr={class_weights[0]:.3f} rel={class_weights[1]:.3f}")
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float).to(device)
        )

        train_ds = TitleDataset(train_titles, train_targets, tokenizer, args.max_len)
        val_ds   = TitleDataset(val_titles,   val_targets,   tokenizer, args.max_len)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
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
                    report = classification_report(
                        reals, preds,
                        target_names=['irrelevant', 'relevant'],
                        digits=3
                    )
                    with open(os.path.join(args.model_save_path, 'best_fold_report.txt'), 'w') as f:
                        f.write(f"Base model: {args.base_model}\n"
                                f"Fold {fold+1}, Epoch {epoch+1}\n\n{report}")

        fold_f1s.append(fold_best_f1)
        print(f"  Fold {fold+1} best F1: {fold_best_f1:.4f}")

    print("\n" + "="*50)
    print(f"Base model: {args.base_model}")
    print("Cross-Validation F1 Results")
    print("="*50)
    for i, f in enumerate(fold_f1s):
        print(f"  Fold {i+1}: {f:.4f}")
    print(f"  Mean F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print(f"  Best global F1: {best_f1:.4f}")
    print(f"  Model saved to: {args.model_save_path}")


if __name__ == "__main__":
    main()
