"""
Stage 5: Second-pass verification of LLM-labeled relevant articles.

Takes every article the LLM labeled as 'relevant' and runs a strict
binary CONFIRM/REJECT check using a much tighter prompt.
Articles that fail the check are flipped to 'irrelevant'.

Usage:
  python 05_verify_relevant.py \
      -i 04_training_data.csv \
      -o 04_training_data_verified.csv \
      -k <CEREBRAS_API_KEY>
"""
import os, json, time, re, argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

PROVIDER_URL = "https://api.cerebras.ai/v1"
DEFAULT_MODEL = "llama3.1-8b"

SYSTEM_PROMPT = """You are a strict financial news verifier for a media bias research project.

Your job: verify whether each article is GENUINELY relevant to an investor holding stock in the named company.

CONFIRM only if ALL three conditions are true:
  1. The named company is the PRIMARY subject of the article (not just mentioned in passing)
  2. The article covers a MATERIAL corporate event:
       - Earnings, revenue, guidance, dividends, buybacks
       - M&A, major contracts, partnerships, divestitures
       - Regulatory/legal actions AGAINST the company as an institution
       - Executive appointments or departures (CEO/CFO/board level)
       - Major layoffs, restructuring, plant closures
       - Major product launches or discontinuations (not minor features)
       - Labor strikes or disputes with operational impact
       - Credit ratings, debt issuance, bankruptcy
  3. It is NOT any of the following:
       - An individual employee/driver/customer crime or incident
       - Entertainment or lifestyle content (show reviews, menu items, zodiac)
       - A product rumour or leak with no official announcement
       - A consumer shopping guide, deal, or price comparison
       - A macro/political article where the company is one example among many
       - An analyst firm making a market prediction (vs. being the subject)
       - A social media trend or boycott with no documented business impact
       - A minor app feature or UI update

REJECT if there is ANY doubt. When uncertain, always REJECT.

INPUT FORMAT: JSON array of objects with "id", "company", "title"
OUTPUT FORMAT: JSON array of objects with "id" and "verdict" ("CONFIRM" or "REJECT") only.
No explanation needed. No other text. Just the JSON array.
"""


def extract_verdicts(text: str) -> list:
    text = text.strip()
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array in response: {text[:200]!r}")
    raw = text[start:end+1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: regex extract id + verdict
        results = []
        for m in re.finditer(
            r'"id"\s*:\s*"?([^",}\s]+)"?.*?"verdict"\s*:\s*"(CONFIRM|REJECT)"',
            raw, re.DOTALL | re.IGNORECASE
        ):
            results.append({"id": m.group(1).strip(), "verdict": m.group(2).upper()})
        if results:
            return results
        raise ValueError(f"Could not parse verdicts: {raw[:300]!r}")


def verify_batch(client, model, batch):
    items = [{"id": str(r["id"]), "company": r["company_name"], "title": r["title"]}
             for r in batch]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Verify these articles:\n{json.dumps(items, indent=2)}"}
        ],
        temperature=0.0,
    )
    return extract_verdicts(response.choices[0].message.content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      "-i", type=str, required=True)
    parser.add_argument("--output",     "-o", type=str, required=True)
    parser.add_argument("--api_key",    "-k", type=str, default=None)
    parser.add_argument("--model",      "-m", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch_size", "-b", type=int, default=20)
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise ValueError("Provide API key via --api_key or CEREBRAS_API_KEY env var")

    client = OpenAI(api_key=api_key, base_url=PROVIDER_URL)

    df = pd.read_csv(args.input, encoding='utf-8')
    print(f"Loaded {len(df)} training rows")
    print(f"Labels before: {df['label'].value_counts().to_dict()}")

    # Only verify LLM-labeled relevant articles (gold stays untouched)
    to_verify = df[(df['source'] == 'llm_v2') & (df['label'] == 'relevant')].copy()
    print(f"\nLLM relevant articles to verify: {len(to_verify)}")
    print(f"Per company: {to_verify['company_name'].value_counts().to_dict()}")

    rows = to_verify.to_dict('records')
    checkpoint_path = args.output + ".verify_checkpoint.json"

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            verdicts = json.load(f)
        done_ids = {str(v['id']) for v in verdicts}
        rows = [r for r in rows if str(r['id']) not in done_ids]
        print(f"Resuming: {len(verdicts)} done, {len(rows)} remaining")
    else:
        verdicts = []

    total_batches = (len(rows) + args.batch_size - 1) // args.batch_size
    failed = []

    for i in tqdm(range(0, len(rows), args.batch_size), total=total_batches,
                  desc="Verifying", unit="batch"):
        batch = rows[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        for attempt in range(4):
            try:
                result = verify_batch(client, args.model, batch)
                verdicts.extend(result)
                with open(checkpoint_path, 'w') as f:
                    json.dump(verdicts, f)
                break
            except Exception as e:
                if attempt < 3:
                    wait = 30 * (2 ** attempt)
                    tqdm.write(f"Batch {batch_num} failed (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    tqdm.write(f"Batch {batch_num} failed after 4 attempts. Skipping.")
                    failed.append(batch_num)
        time.sleep(4.5)

    # Apply verdicts
    verdict_map = {str(v['id']): v['verdict'] for v in verdicts}
    confirmed = rejected = 0

    def apply_verdict(row):
        nonlocal confirmed, rejected
        if row['source'] == 'llm_v2' and row['label'] == 'relevant':
            verdict = verdict_map.get(str(row['id']), 'REJECT')  # default REJECT if missed
            if verdict == 'CONFIRM':
                confirmed += 1
                return 'relevant'
            else:
                rejected += 1
                return 'irrelevant'
        return row['label']

    df['label'] = df.apply(apply_verdict, axis=1)

    print(f"\nVerification results:")
    print(f"  CONFIRMED relevant: {confirmed}")
    print(f"  REJECTED (flipped to irrelevant): {rejected}")
    print(f"\nLabels after verification:")
    print(f"  {df['label'].value_counts().to_dict()}")

    if failed:
        print(f"\nWARNING: {len(failed)} batches failed and were defaulted to REJECT")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
