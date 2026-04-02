"""
Stage 1 (Few-Shot variant): Annotate titles using Groq API.

Key difference from 01_llm_annotation.py:
- Articles are grouped by company before batching
- For each company, 5 relevant + 5 irrelevant examples are pulled
  from the 300 manually labeled rows and injected into the prompt
- Falls back to the generic hardcoded examples if a company has
  fewer than 3 manual examples on either side
"""
import os
import json
import time
import random
import pandas as pd
from tqdm import tqdm
import argparse
from groq import Groq

# ── Hardcoded fallback examples (used when manual set has no coverage) ──────
FALLBACK_EXAMPLES = [
    {"title": "Tesla Q3 earnings beat estimates", "label": "relevant", "reason": "Direct financial result"},
    {"title": "Amazon deals this week", "label": "irrelevant", "reason": "Shopping guide, affiliate content"},
    {"title": "Nike signs LeBron James to new endorsement deal", "label": "relevant", "reason": "Official signed contract"},
    {"title": "LeBron James spotted wearing Nike shoes", "label": "irrelevant", "reason": "Casual mention, no signed deal"},
    {"title": "Goldman Sachs upgrades Tesla price target", "label": "relevant", "reason": "Analyst action on Tesla stock"},
    {"title": "Apple faces DOJ antitrust lawsuit", "label": "relevant", "reason": "Corporate-level regulatory action"},
    {"title": "Apple store employee arrested for theft", "label": "irrelevant", "reason": "Individual incident, non-material"},
    {"title": "Inflation hits retailers like Walmart, Target, Costco", "label": "irrelevant", "reason": "Company is one example in macro piece"},
    {"title": "Walmart cuts guidance citing inflation", "label": "relevant", "reason": "Company is subject with material financial impact"},
    {"title": "Best streaming shows on Netflix this week", "label": "irrelevant", "reason": "Consumer entertainment guide, not business news"},
]

SYSTEM_PROMPT_TEMPLATE = """You are a highly precise financial news relevance classifier for a media bias research project.

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

4. Regulatory & Legal (corporate-level only)
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
   - A single employee's lawsuit, arrest, or misconduct
   - A single store robbery, accident, or local incident
   - One customer's complaint with no corporate response

2. Consumer & Lifestyle Content
   - Shopping guides, deal roundups, gift lists
   - Product reviews or comparisons written for consumers
   - Lifestyle articles that name-drop the brand

3. Passing Mentions & Context
   - Macro/industry articles where the company appears as one of several examples
   - Articles primarily about a competitor that briefly compare this company
   - Economic or political analysis that references the company incidentally

4. Non-Material Entertainment
   - A sponsored sports team's game result
   - A celebrity casually spotted using the product
   - Social media trends or memes with no business impact

5. Analyst Firm vs. Analyst Target Rule
   - "Goldman Sachs upgrades Tesla" → RELEVANT for Tesla, IRRELEVANT for Goldman Sachs
   - Classify based on whose business is being analyzed, not who is doing the analyzing

6. Company-as-Context vs. Company-as-Subject
   - The test: is the company the subject of the headline, or just supporting evidence?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIE-BREAKING RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When genuinely uncertain, choose IRRELEVANT.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES FOR {company_name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
These are real human-labeled examples specifically for {company_name}. Use them to
understand what counts as relevant vs irrelevant for this specific company:

{examples_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Respond ONLY with a JSON array. Each element must have:
- "id": the id from input
- "label": exactly "relevant" or "irrelevant"
- "reason": one short sentence stating what the article is about and why it qualifies or doesn't
"""


def load_manual_examples(manual_csv: str) -> dict:
    """
    Load the 300 manually labeled rows and group by company.
    Returns dict: { company_name -> { 'relevant': [...], 'irrelevant': [...] } }
    """
    df = pd.read_csv(manual_csv, usecols=['id', 'title', 'company_name', 'label'])
    df['label'] = df['label'].str.strip().str.lower()

    # Normalise label variants: 'relevant (?)' -> 'relevant', etc.
    df['label'] = df['label'].apply(
        lambda x: 'relevant' if 'relevant' in x and 'irrelevant' not in x else 'irrelevant'
    )

    examples: dict = {}
    for company, group in df.groupby('company_name'):
        rel = group[group['label'] == 'relevant'][['id', 'title', 'label']].to_dict('records')
        irr = group[group['label'] == 'irrelevant'][['id', 'title', 'label']].to_dict('records')
        examples[company] = {'relevant': rel, 'irrelevant': irr}

    return examples


def get_few_shot_examples(company_name: str, manual_examples: dict, n: int = 5) -> list:
    """
    Return up to n relevant + n irrelevant examples for the given company.
    Falls back to generic examples if the company has fewer than 3 on either side.
    """
    company_data = manual_examples.get(company_name, {})
    rel = company_data.get('relevant', [])
    irr = company_data.get('irrelevant', [])

    # Need at least 3 on each side to use company-specific examples
    if len(rel) >= 3 and len(irr) >= 3:
        sampled_rel = random.sample(rel, min(n, len(rel)))
        sampled_irr = random.sample(irr, min(n, len(irr)))
        examples = [
            {"title": e['title'], "label": "relevant", "reason": "Human-labeled relevant example"}
            for e in sampled_rel
        ] + [
            {"title": e['title'], "label": "irrelevant", "reason": "Human-labeled irrelevant example"}
            for e in sampled_irr
        ]
        random.shuffle(examples)
        return examples
    else:
        # Fall back to generic examples
        return FALLBACK_EXAMPLES


def build_examples_block(examples: list) -> str:
    lines = []
    for e in examples:
        lines.append(f'  Title: "{e["title"]}"')
        lines.append(f'  Label: {e["label"]}')
        lines.append(f'  Reason: {e["reason"]}')
        lines.append("")
    return "\n".join(lines)


def build_system_prompt(company_name: str, examples: list) -> str:
    examples_block = build_examples_block(examples)
    return SYSTEM_PROMPT_TEMPLATE.format(
        company_name=company_name,
        examples_block=examples_block,
    )


def classify_batch(client, model_name: str, batch: list, system_prompt: str) -> list:
    items = [{"id": row["id"], "title": row["title"]} for row in batch]
    prompt = f"Classify these {batch[0].get('company_name', '')} titles:\n{json.dumps(items, indent=2)}"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
    )
    text = response.choices[0].message.content.strip()

    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description="Stage 1 (Few-Shot): Annotate titles using company-specific examples")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input CSV with id, title, company_name columns")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output CSV with annotations")
    parser.add_argument("--manual", type=str, required=True, help="Path to 300 manually labeled CSV")
    parser.add_argument("--api_key", "-k", type=str, default=None)
    parser.add_argument("--batch_size", "-b", type=int, default=50)
    parser.add_argument("--model", "-m", type=str, default="llama-3.3-70b-versatile")
    parser.add_argument("--n_examples", type=int, default=5, help="Number of examples per label side (max 5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    api_key = args.api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Provide Groq API key via --api_key or GROQ_API_KEY env var")

    client = Groq(api_key=api_key)

    # Load input
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    df = df.dropna(subset=['title']).copy()
    df['id'] = df['id'].astype(str)

    if 'company_name' not in df.columns:
        raise ValueError("Input CSV must have a 'company_name' column. Run 00_fetch_and_filter.py with company info.")

    # Load manual examples
    print(f"Loading manual examples from {args.manual}...")
    manual_examples = load_manual_examples(args.manual)
    covered = [c for c, v in manual_examples.items() if len(v['relevant']) >= 3 and len(v['irrelevant']) >= 3]
    print(f"  Companies with enough manual examples for few-shot: {len(covered)}")
    print(f"  Companies falling back to generic examples: {df['company_name'].nunique() - len(covered)}")

    # Load checkpoint
    checkpoint_path = args.output + ".checkpoint.json"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            results = json.load(f)
        done_ids = {str(r['id']) for r in results}
        print(f"Resuming from checkpoint: {len(results)} already annotated.")
    else:
        results = []
        done_ids = set()

    # ── KEY CHANGE: group by company, inject company-specific examples ───────
    # Group rows by company so each batch is single-company
    df_remaining = df[~df['id'].isin(done_ids)].copy()
    print(f"\nAnnotating {len(df_remaining)} remaining titles across {df_remaining['company_name'].nunique()} companies...")

    failed_batches = []
    start_time = time.time()
    total_batches = sum(
        (len(group) + args.batch_size - 1) // args.batch_size
        for _, group in df_remaining.groupby('company_name')
    )

    with tqdm(total=len(df_remaining), desc="Annotating", unit="titles") as pbar:
        for company_name, company_df in df_remaining.groupby('company_name'):

            # Get company-specific few-shot examples
            examples = get_few_shot_examples(company_name, manual_examples, n=args.n_examples)
            system_prompt = build_system_prompt(company_name, examples)

            rows = company_df[['id', 'title', 'company_name']].to_dict('records')

            for i in range(0, len(rows), args.batch_size):
                batch = rows[i:i + args.batch_size]
                batch_num = i // args.batch_size + 1
                retries = 3

                for attempt in range(retries):
                    try:
                        parsed = classify_batch(client, args.model, batch, system_prompt)
                        results.extend(parsed)
                        with open(checkpoint_path, 'w') as f:
                            json.dump(results, f)
                        break
                    except Exception as e:
                        if attempt < retries - 1:
                            wait = 2 ** attempt * 5
                            tqdm.write(f"{company_name} batch {batch_num} failed (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                            time.sleep(wait)
                        else:
                            tqdm.write(f"{company_name} batch {batch_num} failed after {retries} attempts. Skipping.")
                            failed_batches.append(f"{company_name}:{batch_num}")

                pbar.update(len(batch))
                time.sleep(0.5)

    # Merge back with original df
    results_df = pd.DataFrame(results)
    results_df['id'] = results_df['id'].astype(str)
    final_df = df.merge(results_df[['id', 'label', 'reason']], on='id', how='left')

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"\nAnnotated {results_df['label'].notna().sum()} / {len(df)} titles.")
    print("Label distribution:")
    print(final_df['label'].value_counts())

    if failed_batches:
        print(f"\nWarning: {len(failed_batches)} batches failed: {failed_batches}")

    final_df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
