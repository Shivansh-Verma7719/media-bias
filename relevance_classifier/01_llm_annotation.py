"""
Stage 1: Annotate titles as relevant/irrelevant using Groq API (zero-shot).
Batches titles into a single prompt call for efficiency.
"""
import os
import json
import time
import pandas as pd
from tqdm import tqdm
import argparse
from groq import Groq

SYSTEM_PROMPT = """You are a financial news relevance classifier.

Your task: classify each news article title as RELEVANT or IRRELEVANT.

RELEVANT = the article is about a company's business performance, financial results,
growth, products, strategy, mergers, acquisitions, stock price, earnings, revenue,
market share, leadership changes that affect the business, partnerships, or any event
that directly impacts the company as a business.

IRRELEVANT = the company is mentioned in the title but the article is NOT about the
company's business performance. Examples: an employee crime, a celebrity endorsed by
the brand, a sports sponsorship mention, a general industry article where the company
is briefly mentioned, political opinions, social issues.

Respond ONLY with a JSON array. Each element must have:
- "id": the id from input
- "label": exactly "relevant" or "irrelevant"
- "reason": one short sentence explaining why

Example output format:
[
  {"id": 1, "label": "relevant", "reason": "Reports Q3 earnings beat"},
  {"id": 2, "label": "irrelevant", "reason": "About an employee lawsuit unrelated to business performance"}
]
"""

def classify_batch(client, model_name, batch):
    items = [{"id": row["id"], "title": row["title"]} for row in batch]
    prompt = f"Classify these titles:\n{json.dumps(items, indent=2)}"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
    )
    text = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    return json.loads(text)

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Annotate titles using Groq API (zero-shot)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input CSV of pre-filtered titles")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output CSV with annotations")
    parser.add_argument("--api_key", "-k", type=str, default=None, help="Groq API key (or set GROQ_API_KEY env var)")
    parser.add_argument("--batch_size", "-b", type=int, default=50, help="Titles per API call")
    parser.add_argument("--model", "-m", type=str, default="llama-3.3-70b-versatile", help="Groq model to use")
    parser.add_argument("--title_col", type=str, default="title", help="Name of the title column")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Provide Groq API key via --api_key or GROQ_API_KEY env var")

    client = Groq(api_key=api_key)

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    df = df.dropna(subset=[args.title_col]).copy()
    df['id'] = df['id'].astype(str)
    print(f"Annotating {len(df)} titles with batch_size={args.batch_size}, model={args.model}...")

    rows = df[['id', args.title_col]].rename(columns={args.title_col: 'title'}).to_dict('records')
    failed_batches = []
    total_batches = (len(rows) + args.batch_size - 1) // args.batch_size
    start_time = time.time()

    # Load checkpoint if output file already exists (resume support)
    checkpoint_path = args.output + ".checkpoint.json"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            results = json.load(f)
        done_ids = {str(r['id']) for r in results}
        rows = [r for r in rows if str(r['id']) not in done_ids]
        print(f"Resuming from checkpoint: {len(results)} already annotated, {len(rows)} remaining.")
    else:
        results = []

    for i in tqdm(range(0, len(rows), args.batch_size), desc="Annotating", total=(len(rows) + args.batch_size - 1) // args.batch_size, unit="batch"):
        batch = rows[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        retries = 3
        for attempt in range(retries):
            try:
                parsed = classify_batch(client, args.model, batch)
                results.extend(parsed)
                # Save checkpoint after every batch
                with open(checkpoint_path, 'w') as f:
                    json.dump(results, f)
                break
            except Exception as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt * 5
                    tqdm.write(f"Batch {batch_num} failed (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    tqdm.write(f"Batch {batch_num} failed after {retries} attempts: {e}. Skipping.")
                    failed_batches.append(batch_num)

        # Print status every 5 batches
        if batch_num % 5 == 0 or batch_num == total_batches:
            elapsed = time.time() - start_time
            rate = batch_num / elapsed if elapsed > 0 else 0
            eta = (total_batches - batch_num) / rate if rate > 0 else 0
            relevant = sum(1 for r in results if r.get('label') == 'relevant')
            irrelevant = sum(1 for r in results if r.get('label') == 'irrelevant')
            tqdm.write(
                f"[{batch_num}/{total_batches}] "
                f"Annotated: {len(results)} titles | "
                f"Relevant: {relevant} | Irrelevant: {irrelevant} | "
                f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s"
            )

        time.sleep(0.5)

    results_df = pd.DataFrame(results)
    results_df['id'] = results_df['id'].astype(str)
    final_df = df.merge(results_df[['id', 'label', 'reason']], on='id', how='left')

    # Clean up checkpoint on successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"\nAnnotated {results_df['label'].notna().sum()} / {len(df)} titles.")
    print("Label distribution:")
    print(final_df['label'].value_counts())

    if failed_batches:
        print(f"\nWarning: {len(failed_batches)} batches failed and were skipped: {failed_batches}")

    final_df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()
