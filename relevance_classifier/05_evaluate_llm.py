"""
Evaluation script: measures LLM annotation accuracy against human labels.

Steps:
1. Strip human labels from the 300-row CSV → input for LLM
2. Run LLM annotation (same prompt as 01_llm_annotation_fewshot.py)
3. Compare LLM labels vs human labels → accuracy report
"""
import os
import json
import time
import random
import argparse
import pandas as pd
from tqdm import tqdm
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# ── Reuse the same system prompt from the few-shot script ───────────────────
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
   - Analyst upgrades/downgrades/initiations ON THIS COMPANY
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
   - Class-action lawsuits (must explicitly say "class-action" or involve a massive group; single-user lawsuits are IRRELEVANT)

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
   - One customer's complaint or lawsuit (e.g., "Unhappy subscriber sues Netflix" -> IRRELEVANT)

2. Consumer, Blog & Lifestyle Content
   - Shopping guides, deal roundups, holiday sales, "Black Friday", "% off", "$ off"
   - Product reviews, beta testing, "how-to" articles, software tips (e.g. "Microsoft begs you to stop using IE" -> IRRELEVANT)
   - Lifestyle journalism or listicles that name-drop the brand

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
{examples_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Respond ONLY with a JSON array. Each element must have:
- "id": the id from input
- "label": exactly "relevant" or "irrelevant"
- "reason": one short sentence
"""

FALLBACK_EXAMPLES = [
    {"title": "Tesla Q3 earnings beat estimates", "label": "relevant"},
    {"title": "Amazon deals this week", "label": "irrelevant"},
    {"title": "Apple faces DOJ antitrust lawsuit", "label": "relevant"},
    {"title": "Apple store employee arrested for theft", "label": "irrelevant"},
    {"title": "Walmart cuts guidance citing inflation", "label": "relevant"},
    {"title": "Inflation hits retailers like Walmart, Target, Costco", "label": "irrelevant"},
]


def normalise_label(label: str) -> str:
    label = str(label).strip().lower()
    if 'irrelevant' in label:
        return 'irrelevant'
    if 'relevant' in label:
        return 'relevant'
    return 'unknown'


def load_few_shot_examples(df: pd.DataFrame, company: str, n: int = 5, exclude_ids: set = None) -> list:
    """
    For a given company, pull n relevant + n irrelevant examples from the human-labeled set,
    excluding the rows currently being evaluated (no data leakage).
    """
    company_df = df[df['company_name'] == company].copy()
    if exclude_ids:
        company_df = company_df[~company_df['id'].astype(str).isin(exclude_ids)]

    rel = company_df[company_df['human_label'] == 'relevant']
    irr = company_df[company_df['human_label'] == 'irrelevant']

    if len(rel) >= 2 and len(irr) >= 2:
        sampled_rel = rel.sample(min(n, len(rel)))
        sampled_irr = irr.sample(min(n, len(irr)))
        examples = (
            [{"title": r['title'], "label": "relevant"} for _, r in sampled_rel.iterrows()] +
            [{"title": r['title'], "label": "irrelevant"} for _, r in sampled_irr.iterrows()]
        )
        random.shuffle(examples)
        return examples
    return FALLBACK_EXAMPLES


def build_system_prompt(company_name: str, examples: list) -> str:
    lines = []
    for e in examples:
        lines.append(f'  Title: "{e["title"]}"')
        lines.append(f'  Label: {e["label"]}')
        lines.append("")
    examples_block = "\n".join(lines)
    return SYSTEM_PROMPT_TEMPLATE.format(
        company_name=company_name,
        examples_block=examples_block,
    )


def classify_batch(client, model_name: str, batch: list, system_prompt: str) -> list:
    items = [{"id": row["id"], "title": row["title"]} for row in batch]
    prompt = f"Classify these titles:\n{json.dumps(items, indent=2)}"

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
    return json.loads(text.strip())


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM annotation accuracy vs human labels")
    parser.add_argument("--manual", type=str,
                        default="relevance_classifier/300 random articles - 300-random-articles.csv",
                        help="Path to manually labeled CSV")
    parser.add_argument("--output", "-o", type=str,
                        default="relevance_classifier/05_llm_eval_results.csv",
                        help="Output CSV with LLM predictions vs human labels")
    parser.add_argument("--api_key", "-k", type=str, default=None)
    parser.add_argument("--model", "-m", type=str, default="llama-3.3-70b-versatile")
    parser.add_argument("--batch_size", "-b", type=int, default=20)
    parser.add_argument("--n_examples", type=int, default=5)
    parser.add_argument("--zero_shot", action="store_true", help="Disable few-shot examples — pure zero-shot classification")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    api_key = args.api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Provide Groq API key via --api_key or GROQ_API_KEY env var")

    client = Groq(api_key=api_key)

    # ── Step 1: Load and clean the 300-row human-labeled CSV ────────────────
    print(f"Loading manual labels from {args.manual}...")
    raw = pd.read_csv(args.manual, usecols=['id', 'title', 'company_name', 'label'])
    raw = raw.dropna(subset=['title', 'label', 'company_name']).copy()
    raw['id'] = raw['id'].astype(str)
    raw['human_label'] = raw['label'].apply(normalise_label)
    raw = raw[raw['human_label'].isin(['relevant', 'irrelevant'])].copy()
    print(f"  {len(raw)} rows with valid human labels.")
    print(f"  Human label distribution:\n{raw['human_label'].value_counts().to_string()}")

    # ── Step 2: LLM annotation grouped by company (few-shot) ────────────────
    checkpoint_path = args.output + ".checkpoint.json"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            results = json.load(f)
        done_ids = {str(r['id']) for r in results}
        print(f"Resuming from checkpoint: {len(results)} already annotated.")
    else:
        results = []
        done_ids = set()

    remaining = raw[~raw['id'].isin(done_ids)]
    print(f"\nRunning LLM annotation on {len(remaining)} titles...")

    mode = "zero-shot" if args.zero_shot else "few-shot"
    print(f"Mode: {mode}")

    with tqdm(total=len(remaining), desc="Annotating", unit="titles") as pbar:
        for company, group in remaining.groupby('company_name'):
            batch_ids = set(group['id'].tolist())

            if args.zero_shot:
                examples = []
                examples_block = "No examples provided — use the rules above to classify."
                system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                    company_name=company,
                    examples_block=examples_block,
                )
            else:
                examples = load_few_shot_examples(raw, company, n=args.n_examples, exclude_ids=batch_ids)
                system_prompt = build_system_prompt(company, examples)

            rows = group[['id', 'title']].to_dict('records')

            for i in range(0, len(rows), args.batch_size):
                batch = rows[i:i + args.batch_size]
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
                            tqdm.write(f"{company} failed (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                            time.sleep(wait)
                        else:
                            tqdm.write(f"{company} failed after {retries} attempts. Skipping batch.")

                pbar.update(len(batch))
                time.sleep(0.5)

    # ── Step 3: Merge and compare ────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df['id'] = results_df['id'].astype(str)
    results_df['llm_label'] = results_df['label'].apply(normalise_label)

    merged = raw.merge(results_df[['id', 'llm_label', 'reason']], on='id', how='left')
    merged = merged.dropna(subset=['llm_label'])

    # Overall accuracy
    merged['correct'] = merged['human_label'] == merged['llm_label']
    overall_acc = merged['correct'].mean()

    # Per-class accuracy
    rel_acc  = merged[merged['human_label'] == 'relevant']['correct'].mean()
    irr_acc  = merged[merged['human_label'] == 'irrelevant']['correct'].mean()

    # Per-company accuracy
    per_company = merged.groupby('company_name')['correct'].agg(['mean', 'count']).rename(
        columns={'mean': 'accuracy', 'count': 'n'}
    ).sort_values('accuracy')

    print(f"\n{'='*50}")
    print(f"LLM ANNOTATION ACCURACY vs HUMAN LABELS")
    print(f"{'='*50}")
    print(f"Overall accuracy:     {overall_acc:.1%}  ({merged['correct'].sum()}/{len(merged)})")
    print(f"Relevant accuracy:    {rel_acc:.1%}  (correct on relevant titles)")
    print(f"Irrelevant accuracy:  {irr_acc:.1%}  (correct on irrelevant titles)")
    print(f"\nPer-company accuracy:")
    print(per_company.to_string())

    # Confusion matrix
    tp = ((merged['human_label'] == 'relevant')   & (merged['llm_label'] == 'relevant')).sum()
    tn = ((merged['human_label'] == 'irrelevant') & (merged['llm_label'] == 'irrelevant')).sum()
    fp = ((merged['human_label'] == 'irrelevant') & (merged['llm_label'] == 'relevant')).sum()
    fn = ((merged['human_label'] == 'relevant')   & (merged['llm_label'] == 'irrelevant')).sum()

    print(f"\nConfusion matrix:")
    print(f"  True Positives  (relevant → relevant):     {tp}")
    print(f"  True Negatives  (irrelevant → irrelevant): {tn}")
    print(f"  False Positives (irrelevant → relevant):   {fp}  ← LLM overcalls relevant")
    print(f"  False Negatives (relevant → irrelevant):   {fn}  ← LLM undercalls relevant")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"\n  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1 Score:  {f1:.1%}")

    # Save full comparison
    merged[['id', 'title', 'company_name', 'human_label', 'llm_label', 'reason', 'correct']].to_csv(
        args.output, index=False
    )
    print(f"\nFull comparison saved to {args.output}")

    # Show disagreements
    disagreements = merged[~merged['correct']][['title', 'company_name', 'human_label', 'llm_label', 'reason']]
    print(f"\nDisagreements ({len(disagreements)} total):")
    print(disagreements.to_string(max_colwidth=80))

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


if __name__ == "__main__":
    main()
