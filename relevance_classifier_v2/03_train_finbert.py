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

# Gold-labeled rows get 5x weight in loss — they are ground truth, synthetic adds coverage
GOLD_WEIGHT = 5.0


class FocalLoss(torch.nn.Module):
    """Focal loss: down-weights easy correct examples, focuses on hard boundary cases.
    gamma=0 reduces to standard cross-entropy. gamma=2 is the standard focal setting."""
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = torch.nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma) * ce


class TitleDataset(Dataset):
    def __init__(self, titles, labels, tokenizer, max_len, weights=None):
        self.titles = titles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.weights = weights if weights is not None else np.ones(len(titles), dtype=np.float32)

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
        item['sample_weight'] = torch.tensor(self.weights[idx], dtype=torch.float)
        return item


def train_epoch(model, loader, optimizer, scheduler, device, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="  Train", leave=False):
        optimizer.zero_grad()
        labels = batch.pop('labels').to(device)
        sample_weights = batch.pop('sample_weight').to(device)
        inputs = {k: v.to(device) for k, v in batch.items()}
        out = model(**inputs)
        loss = (criterion(out.logits, labels) * sample_weights).mean()
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    return total_loss / len(loader)


def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs, losses = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop('labels').to(device)
            batch.pop('sample_weight', None)
            inputs = {k: v.to(device) for k, v in batch.items()}
            out = model(**inputs, labels=labels)
            losses.append(out.loss.item())
            probs = torch.softmax(out.logits, dim=1)[:, 1]
            preds = torch.argmax(out.logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1  = f1_score(all_labels, all_preds, pos_label=1, average='binary')
    return acc, np.mean(losses), f1, all_preds, all_labels, all_probs


def threshold_sweep(probs, labels, model_path):
    """Sweep p_rel thresholds, print precision/recall table, save optimal threshold."""
    from sklearn.metrics import precision_score, recall_score
    probs = np.array(probs)
    labels = np.array(labels)
    print("\n  Threshold sweep (predict relevant if p_rel >= threshold):")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'N_pred':>8}")
    best = {'threshold': 0.5, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    for t in np.arange(0.40, 0.85, 0.05):
        preds = (probs >= t).astype(int)
        n_pred = preds.sum()
        if n_pred == 0:
            continue
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        flag = " <-- P>=0.90" if p >= 0.90 else ""
        print(f"  {t:>10.2f} {p:>10.3f} {r:>8.3f} {f:>8.3f} {n_pred:>8}{flag}")
        if f > best['f1']:
            best = {'threshold': round(float(t), 2), 'f1': f, 'precision': p, 'recall': r}
    import json
    threshold_file = os.path.join(model_path, 'optimal_threshold.json')
    with open(threshold_file, 'w') as f:
        json.dump(best, f, indent=2)
    print(f"\n  Best threshold saved: {best} -> {threshold_file}")


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
    parser.add_argument("--test_file",       "-t", type=str, default=None,
                        help="Optional: run threshold sweep on this CSV (title,label) after training")
    parser.add_argument("--focal_gamma",     "-g", type=float, default=2.0,
                        help="Focal loss gamma (0=standard CE, 2=standard focal). Default: 2.0")
    args = parser.parse_args()

    print(f"Base model: {args.base_model}")

    df = pd.read_csv(args.input).dropna(subset=['title', 'label'])
    label_map = {'relevant': 1, 'irrelevant': 0}
    df['target'] = df['label'].str.lower().map(label_map)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)

    # Gold upweighting: verified annotations get GOLD_WEIGHT, synthetic get 1.0
    if 'source_type' in df.columns:
        df['sample_weight'] = df['source_type'].apply(
            lambda s: GOLD_WEIGHT if str(s).strip() == 'gold_manual' else 1.0
        )
        n_gold = (df['source_type'] == 'gold_manual').sum()
        print(f"Training on {len(df)} samples ({n_gold} gold x{GOLD_WEIGHT} weight, {len(df)-n_gold} synthetic x1.0)")
    else:
        df['sample_weight'] = 1.0
        print(f"Training on {len(df)} samples (no source_type column — uniform weights)")

    print(df['target'].value_counts())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    titles  = df['title'].to_numpy()
    targets = df['target'].to_numpy()
    weights = df['sample_weight'].to_numpy(dtype=np.float32)

    skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=42)
    fold_f1s = []
    best_f1  = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(titles, targets)):
        print(f"\n══════ Fold {fold+1}/{args.kfolds} ══════")

        train_titles, val_titles = titles[train_idx], titles[val_idx]
        train_targets, val_targets = targets[train_idx], targets[val_idx]
        train_weights, val_weights = weights[train_idx], weights[val_idx]

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
        cw_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        if args.focal_gamma > 0:
            criterion = FocalLoss(weight=cw_tensor, gamma=args.focal_gamma)
        else:
            criterion = torch.nn.CrossEntropyLoss(weight=cw_tensor, reduction='none')

        train_ds = TitleDataset(train_titles, train_targets, tokenizer, args.max_len, train_weights)
        val_ds   = TitleDataset(val_titles,   val_targets,   tokenizer, args.max_len, val_weights)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        model = model.to(device)

        # DeBERTa-v3: exclude bias, LayerNorm, and embeddings from weight decay
        # (GDES embedding sharing breaks down if embeddings are decayed)
        no_decay = {'bias', 'LayerNorm.weight', 'LayerNorm.bias'}
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay) and 'embeddings.' not in n],
                'weight_decay': 0.01,
            },
            {
                'params': [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay) or 'embeddings.' in n],
                'weight_decay': 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
        total_steps = len(train_dl) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )

        fold_best_f1 = 0.0
        best_val_loss = float('inf')
        patience_counter = 0
        EARLY_STOP_PATIENCE = 2

        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_dl, optimizer, scheduler, device, criterion)
            val_acc, val_loss, val_f1, preds, reals, val_probs = eval_epoch(model, val_dl, device)
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

            # Early stopping on val_loss (more stable than F1 for small val sets)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1} (val_loss didn't improve for {EARLY_STOP_PATIENCE} epochs)")
                    break

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

    if args.test_file:
        print(f"\n{'='*50}")
        print(f"Threshold sweep on: {args.test_file}")
        print("="*50)
        test_df = pd.read_csv(args.test_file).dropna(subset=['title', 'label'])
        test_df['target'] = test_df['label'].str.lower().map({'relevant': 1, 'irrelevant': 0})
        test_df = test_df.dropna(subset=['target'])
        test_df['target'] = test_df['target'].astype(int)
        best_model = AutoModelForSequenceClassification.from_pretrained(args.model_save_path).to(device)
        test_ds = TitleDataset(test_df['title'].to_numpy(), test_df['target'].to_numpy(),
                               tokenizer, args.max_len)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        _, _, _, _, test_labels, test_probs = eval_epoch(best_model, test_dl, device)
        threshold_sweep(test_probs, test_labels, args.model_save_path)


if __name__ == "__main__":
    main()
