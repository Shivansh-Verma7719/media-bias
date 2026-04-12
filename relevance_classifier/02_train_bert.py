import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import os
import argparse

class TitleDataset(Dataset):
    def __init__(self, titles, labels, tokenizer, max_len):
        self.titles = titles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.titles)
        
    def __getitem__(self, item):
        title = str(self.titles[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            title,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, dataloader, optimizer, scheduler, device, criterion):
    model = model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use class-weighted loss to handle relevant/irrelevant imbalance
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)

def eval_model(model, dataloader, device):
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    predictions = []
    real_values = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            predictions.extend(preds.cpu().tolist())
            real_values.extend(labels.cpu().tolist())
            
    accuracy = correct_predictions.double() / len(dataloader.dataset)
    return accuracy.item(), np.mean(losses), predictions, real_values

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Train BERT Classifier on Annotated Data")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input CSV with annotated titles (from Stage 2)")
    parser.add_argument("--batch_size", "-b", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", "-e", type=int, default=3, help="Training epochs per fold")
    parser.add_argument("--kfolds", "-k", type=int, default=5, help="Number of splits for CV")
    parser.add_argument("--max_len", "-m", type=int, default=128, help="Max sequence length")
    parser.add_argument("--model_save_path", "-s", type=str, default="best_model", help="Path to save best checkpoint")
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    
    # Preprocessing
    df = df.dropna(subset=['title', 'label'])
    print(f"Loaded {len(df)} annotated samples.")
    
    # Map 'relevant' -> 1, 'irrelevant' -> 0 (case insensitive)
    label_map = {'relevant': 1, 'irrelevant': 0}
    df['target'] = df['label'].astype(str).str.lower().map(label_map)
    
    # Check for unmapped labels
    if df['target'].isna().any():
        unknown_labels = df[df['target'].isna()]['label'].unique()
        print(f"Warning: Dropping rows with unknown labels: {unknown_labels}")
        df = df.dropna(subset=['target'])
        
    df['target'] = df['target'].astype(int)
    
    print("\nClass distribution:")
    print(df['target'].value_counts())
    
    # Setup hardware and model configs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    
    # Cross Validation
    skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=42)
    
    fold_accuracies = []
    best_acc = 0.0
    
    # Data columns to arrays for SKF
    titles = df['title'].to_numpy()
    targets = df['target'].to_numpy()
    
    print(f"\nStarting {args.kfolds}-Fold Cross Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(titles, targets)):
        print(f"\n======== Fold {fold + 1} / {args.kfolds} ========")
        
        train_titles, val_titles = titles[train_idx], titles[val_idx]
        train_targets, val_targets = targets[train_idx], targets[val_idx]
        
        # Class-weighted loss to handle relevant/irrelevant imbalance
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_targets),
            y=train_targets
        )
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float).to(device)
        )

        train_dataset = TitleDataset(train_titles, train_targets, tokenizer, args.max_len)
        val_dataset = TitleDataset(val_titles, val_targets, tokenizer, args.max_len)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Init model for this fold
        model = DistilBertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=2)
        model = model.to(device)

        optimizer = AdamW(model.parameters(), lr=2e-5)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),  # 10% warmup
            num_training_steps=total_steps
        )
        
        fold_best_acc = 0
        
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, criterion)
            val_acc, val_loss, preds, reals = eval_model(model, val_loader, device)
            
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > fold_best_acc:
                fold_best_acc = val_acc
                
                # Check if this is the global best
                if val_acc > best_acc:
                    best_acc = val_acc
                    print(f"New best global model ({val_acc:.4f})! Saving to {args.model_save_path}...")
                    model.save_pretrained(args.model_save_path)
                    tokenizer.save_pretrained(args.model_save_path)
                    
        fold_accuracies.append(fold_best_acc)
        print(f"Best accuracy for Fold {fold + 1}: {fold_best_acc:.4f}")
        
    print("\n" + "="*40)
    print("Cross Validation Results")
    print("="*40)
    for i, acc in enumerate(fold_accuracies):
        print(f"Fold {i+1}: {acc:.4f}")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
    print(f"Best tracking model saved with val accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
