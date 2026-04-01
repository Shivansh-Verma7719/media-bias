import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm
import argparse
import os

class InferenceDataset(Dataset):
    def __init__(self, titles, tokenizer, max_len):
        self.titles = titles
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.titles)
        
    def __getitem__(self, item):
        title = str(self.titles[item])
        
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
            'attention_mask': encoding['attention_mask'].flatten()
        }

def get_predictions(model, data_loader, device):
    model = model.eval()
    
    predictions = []
    prediction_probs = []
    
    with torch.no_grad():
        for d in tqdm(data_loader, desc="Running Inference"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Application of softmax to logits to get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            preds = torch.argmax(probs, dim=1)
            
            # Confidence score of the chosen class
            confidences = torch.max(probs, dim=1).values
            
            predictions.extend(preds.cpu().tolist())
            prediction_probs.extend(confidences.cpu().tolist())
            
    return predictions, prediction_probs

def main():
    parser = argparse.ArgumentParser(description="Stage 4: Run Relevance Inference on large dataset")
    parser.add_argument("--input", "-i", type=str, required=True, help="Large input CSV of pre-filtered titles")
    parser.add_argument("--output", "-o", type=str, default="04_final_predictions.csv", help="Output CSV with annotations")
    parser.add_argument("--model_path", "-m", type=str, default="best_model", help="Path to saved BERT model")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Prediction batch size")
    parser.add_argument("--max_len", "-l", type=int, default=128, help="Max sequence length")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Dataset size: {len(df)} rows")
    
    # Needs to handle missing titles
    if 'title' not in df.columns:
        raise ValueError("Provided CSV does not have a 'title' column.")
        
    original_size = len(df)
    df = df.dropna(subset=['title']).copy()
    print(f"Dropped {original_size - len(df)} rows without titles.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load Fine-Tuned Model
    print(f"Loading trained DistilBERT from {args.model_path}...")
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path {args.model_path} doesn't exist. Have you run Stage 3?")
        
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_path)
    model = DistilBertForSequenceClassification.from_pretrained(args.model_path)
    model = model.to(device)
    
    # Prepare Dataloader
    titles = df['title'].to_numpy()
    dataset = InferenceDataset(titles, tokenizer, args.max_len)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2 if str(device) == 'cuda' else 0, shuffle=False)
    
    print("\nStarting batched inference. This may take a while depending on hardware...")
    preds, probs = get_predictions(model, dataloader, device)
    
    # Assign back to dataframe
    df['predicted_target'] = preds
    df['confidence_score'] = probs
    
    # Map back to human readable labels, with confidence threshold
    CONFIDENCE_THRESHOLD = 0.75  # Below this → 'uncertain', not hard-classified
    label_map = {1: 'relevant', 0: 'irrelevant'}
    df['predicted_label'] = [
        label_map[p] if c >= CONFIDENCE_THRESHOLD else 'uncertain'
        for p, c in zip(preds, probs)
    ]
    
    # Show stats
    print("\nInference Complete.")
    print("Label distribution:")
    print(df['predicted_label'].value_counts(normalize=True))
    
    # Save output
    print(f"\nSaving to {args.output}...")
    df.to_csv(args.output, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
