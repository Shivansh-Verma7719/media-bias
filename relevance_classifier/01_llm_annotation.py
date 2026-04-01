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

SYSTEM_PROMPT = """You are a highly precise financial news relevance classifier for a media bias research project.

YOUR CORE QUESTION: "Would a portfolio manager holding this stock need to read this article?"
  YES, clearly → RELEVANT
  NO, or MAYBE → IRRELEVANT

The article must be PRIMARILY ABOUT the named company. A passing mention does not qualify.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RELEVANT — include if any of these apply:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Financials & Markets
   - Earnings, revenue, profit, guidance, forecasts, dividends, buybacks
   - Stock price movement WITH a stated business reason (not pure macro)
   - Analyst upgrades/downgrades/initiations ON THIS COMPANY (see analyst rule below)
   - Credit ratings, bond issuance, debt restructuring, bankruptcy filings

2. Corporate Actions
   - Mergers, acquisitions, divestitures, spin-offs, joint ventures
   - Major partnerships or contracts (signed, announced, or terminated)
   - Restructuring, layoffs, plant closures, major hiring drives
   - CEO/CFO/board-level appointments or departures that affect strategy

3. Products & Strategy
   - Major product launches, updates, or discontinuations
   - Significant product recalls or safety alerts
   - AI/tech pivots, R&D breakthroughs, patent wins or losses
   - Supply chain disruptions, factory issues, logistics crises

4. Regulatory & Legal (corporate-level only — see individual rule below)
   - Government antitrust investigations or rulings against the company
   - SEC/DOJ/FTC probes or settlements involving the company as an institution
   - FDA approvals, rejections, or warning letters
   - Major government fines, sanctions, or tariffs directly targeting the company
   - Class-action lawsuits (many plaintiffs against the company)

5. Brand & Reputation (material events only)
   - Officially signed celebrity/athlete endorsement DEALS or contract terminations
   - Documented consumer boycotts showing measurable sales impact or corporate response
   - Corporate-level data breaches or cybersecurity incidents
   - Severe executive scandals that trigger board action or regulatory scrutiny

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IRRELEVANT — exclude if any of these apply:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Individual/Local Incidents
   - A single employee's lawsuit, arrest, or misconduct (not class-action, not executive)
   - A single store robbery, accident, or local incident
   - One customer's complaint or viral social media story with no corporate response

2. Consumer & Lifestyle Content
   - Shopping guides, deal roundups, gift lists ("Best Amazon deals this week")
   - Product reviews or comparisons written for consumers, not investors
   - Lifestyle articles that name-drop the brand ("Celebrities who love Starbucks")

3. Passing Mentions & Context
   - Macro/industry articles where the company appears as one of several examples
   - Articles primarily about a competitor that briefly compare this company
   - Economic or political analysis that references the company incidentally

4. Non-Material Entertainment
   - A sponsored sports team's game result (unless sponsorship deal itself is news)
   - A celebrity casually spotted wearing or using the product (not a signed deal)
   - Social media trends or memes involving the brand with no business impact

5. Analyst Firm vs. Analyst Target Rule (IMPORTANT)
   - "Goldman Sachs upgrades Tesla" → RELEVANT for Tesla, IRRELEVANT for Goldman Sachs
   - Classify based on whose business is being analyzed, not who is doing the analyzing
   - If the article is tagged to the analyst's firm (e.g., Goldman), mark IRRELEVANT

6. Company-as-Context vs. Company-as-Subject
   - IRRELEVANT: "How inflation is hitting retailers like Walmart, Target, and Costco"
   - RELEVANT: "Walmart cuts full-year guidance citing inflation pressure"
   - The test: is the company the subject of the headline, or just supporting evidence?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIE-BREAKING RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When genuinely uncertain, choose IRRELEVANT.
A clean dataset with some missed relevant articles is better than a noisy dataset with false positives polluting the analysis.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Respond ONLY with a JSON array. Each element must have:
- "id": the id from input
- "label": exactly "relevant" or "irrelevant"
- "reason": one short sentence stating what the article is about and why it qualifies or doesn't

EXAMPLES (study these carefully — they cover boundary cases):
[
  {"id": 1, "label": "relevant", "reason": "Tesla Q3 earnings beat estimates — direct financial result"},
  {"id": 2, "label": "irrelevant", "reason": "Shopping guide listing Amazon deals — affiliate content, not business news"},
  {"id": 3, "label": "relevant", "reason": "Nike signs LeBron James to new endorsement deal — official contract announcement"},
  {"id": 4, "label": "irrelevant", "reason": "LeBron James spotted wearing Nike shoes — casual mention, no signed deal"},
  {"id": 5, "label": "relevant", "reason": "Goldman Sachs upgrades Tesla price target — analyst action on Tesla's stock"},
  {"id": 6, "label": "irrelevant", "reason": "Goldman Sachs upgrades Tesla — article tagged to Goldman, not the company being rated"},
  {"id": 7, "label": "relevant", "reason": "Apple faces DOJ antitrust lawsuit — corporate-level regulatory action"},
  {"id": 8, "label": "irrelevant", "reason": "Apple store employee arrested for theft — individual incident, non-material"},
  {"id": 9, "label": "irrelevant", "reason": "Inflation hits retailers like Walmart, Target, Costco — company is one example in macro piece"},
  {"id": 10, "label": "relevant", "reason": "Walmart cuts guidance citing inflation — company is the subject with material financial impact"}
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
