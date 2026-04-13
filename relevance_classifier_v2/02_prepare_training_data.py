"""
Stage 2: Merge the re-annotated 4k LLM labels with the clean 240-row
manually-labeled training split. Output: 02_training_data.csv

Usage:
  python 02_prepare_training_data.py \
      --annotated 01_annotated.csv \
      --clean_train 300_train.csv \
      --output 02_training_data.csv
"""
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotated",   "-a", type=str, required=True,
                        help="LLM-annotated CSV (output of 01_annotate.py)")
    parser.add_argument("--clean_train", "-c", type=str, required=True,
                        help="Clean manually-labeled training split (300_train.csv)")
    parser.add_argument("--output",      "-o", type=str, default="02_training_data.csv")
    args = parser.parse_args()

    # ── Load LLM annotations ─────────────────────────────────────────────────
    llm = pd.read_csv(args.annotated)
    llm = llm.dropna(subset=['title', 'label']).copy()
    llm['label'] = llm['label'].str.strip().str.lower()
    llm = llm[llm['label'].isin(['relevant', 'irrelevant'])]
    llm['source'] = 'llm'

    # ── Load clean manual training labels ────────────────────────────────────
    clean = pd.read_csv(args.clean_train)
    clean = clean.dropna(subset=['title', 'label']).copy()
    clean['label'] = clean['label'].str.strip().str.lower()
    clean = clean[clean['label'].isin(['relevant', 'irrelevant'])]
    clean['source'] = 'manual'

    # ── Remove from LLM set any IDs already in clean set (dedup) ────────────
    clean_ids = set(clean['id'].astype(str))
    llm = llm[~llm['id'].astype(str).isin(clean_ids)]
    print(f"LLM annotations after dedup:   {len(llm)}")
    print(f"Clean manual training rows:     {len(clean)}")

    # ── Combine (manual rows first so they take priority in any edge cases) ──
    combined = pd.concat([clean[['id', 'title', 'label', 'source']],
                          llm[['id', 'title', 'label', 'source']]],
                         ignore_index=True)
    combined = combined.drop_duplicates(subset='id', keep='first')

    print(f"\nCombined training set: {len(combined)} rows")
    print(combined['label'].value_counts())
    print(combined['source'].value_counts())

    combined.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
