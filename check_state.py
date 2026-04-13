import pandas as pd, json, os

print("=== CURRENT STATE OF ALL DATA FILES ===")
print()

files = {
    "00_filtered_sp500_sample.csv": "INPUT: 4000 articles to annotate",
    "01_annotated.csv": "LLM LABELS: annotated articles",
    "04_training_data_verified.csv": "FINAL TRAINING SET (last saved)",
    "300_train.csv": "GOLD TRAIN SET (hand-labeled)",
    "300_test.csv": "GOLD TEST SET (hand-labeled)",
}

for fname, desc in files.items():
    path = f"relevance_classifier_v2/{fname}"
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="utf-8")
        print(f"{fname}")
        print(f"  Purpose : {desc}")
        print(f"  Rows    : {len(df)}")
        if "label" in df.columns:
            print(f"  Labels  : {df['label'].value_counts().to_dict()}")
        if "company_name" in df.columns:
            print(f"  Companies: {df['company_name'].nunique()} unique")
        print()
    else:
        print(f"{fname}: NOT FOUND")
        print()

ck = "relevance_classifier_v2/01_annotated.csv.checkpoint.json"
if os.path.exists(ck):
    with open(ck) as f:
        ckdata = json.load(f)
    done = ckdata.get("done_ids", [])
    print(f"CHECKPOINT: {len(done)} article IDs annotated so far")
    print(f"  Remaining from 4000: {4000 - len(done)}")
else:
    print("No checkpoint file found")
