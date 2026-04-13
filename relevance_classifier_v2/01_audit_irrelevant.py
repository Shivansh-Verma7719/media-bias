"""
Irrelevant Audit: Measure the false-negative rate in the irrelevant class.

Takes a random sample of irrelevant-labeled articles and asks the LLM
whether any of them are actually relevant. Reports the FN rate and saves
any confirmed FNs so they can be flipped before training.

This is a one-time QA step. Run it once and report the FN rate in the paper.
If FN rate > 5%, the irrelevant class needs the same treatment as relevant.

Usage:
  python 01_audit_irrelevant.py \
      -i 04_training_data_verified.csv \
      -o irrelevant_audit.csv \
      -k <CEREBRAS_API_KEY> \
      -n 300
"""
import os, json, time, re, argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

PROVIDER_URL  = "https://api.cerebras.ai/v1"
DEFAULT_MODEL = "llama3.1-8b"

SYSTEM_PROMPT = """You are a strict financial news auditor for a media bias research project.

You will receive articles labeled as IRRELEVANT to an investor holding stock in the named company.
Your job: identify any that are ACTUALLY RELEVANT and have been incorrectly labeled.

An article is RELEVANT only if ALL three conditions are true:
  1. The named company is the PRIMARY subject (not just mentioned in passing)
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
       - Individual employee/driver/customer crime
       - Entertainment or lifestyle content
       - Product rumour or leak with no official announcement
       - Consumer shopping guide, deal, or price comparison
       - Macro/political article where company is one example among many
       - Analyst firm making a market prediction
       - Social media trend or boycott with no documented business impact
       - Minor app feature or UI update

OUTPUT FORMAT: JSON array — one object per input article.
Each object: {"id": "<id>", "verdict": "RELEVANT" or "IRRELEVANT"}
No explanation. No other text. Just the JSON array.
"""


def extract_verdicts(text: str) -> list:
    text = text.strip()
    start, end = text.find('['), text.rfind(']')
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array in response: {text[:200]!r}")
    raw = text[start:end+1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        results = []
        for m in re.finditer(
            r'"id"\s*:\s*"?([^",}\s]+)"?.*?"verdict"\s*:\s*"(RELEVANT|IRRELEVANT)"',
            raw, re.DOTALL | re.IGNORECASE
        ):
            results.append({"id": m.group(1).strip(), "verdict": m.group(2).upper()})
        if results:
            return results
        raise ValueError(f"Could not parse verdicts: {raw[:300]!r}")


def audit_batch(client, model, batch):
    items = [{"id": str(r["id"]), "company": r["company_name"], "title": r["title"]}
             for r in batch]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Audit these irrelevant articles:\n{json.dumps(items, indent=2)}"}
        ],
        temperature=0.0,
    )
    return extract_verdicts(response.choices[0].message.content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      "-i", required=True)
    parser.add_argument("--output",     "-o", default="irrelevant_audit.csv")
    parser.add_argument("--api_key",    "-k", default=None)
    parser.add_argument("--model",      "-m", default=DEFAULT_MODEL)
    parser.add_argument("--n_sample",   "-n", type=int, default=300,
                        help="Number of irrelevant articles to audit (default 300)")
    parser.add_argument("--batch_size", "-b", type=int, default=20)
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise ValueError("Provide API key via --api_key or CEREBRAS_API_KEY env var")

    client = OpenAI(api_key=api_key, base_url=PROVIDER_URL)

    df = pd.read_csv(args.input, encoding="utf-8")
    irrelevant = df[df["label"] == "irrelevant"].copy()

    # Sample — stratified by company so every company is represented
    if "company_name" in irrelevant.columns:
        n_companies = irrelevant["company_name"].nunique()
        per_company = max(1, args.n_sample // n_companies)
        sample = (
            irrelevant.groupby("company_name", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_company), random_state=args.seed))
        )
        # Top up to n_sample if we're under
        if len(sample) < args.n_sample:
            remaining = irrelevant[~irrelevant.index.isin(sample.index)]
            topup = remaining.sample(min(len(remaining), args.n_sample - len(sample)),
                                     random_state=args.seed)
            sample = pd.concat([sample, topup])
    else:
        sample = irrelevant.sample(min(len(irrelevant), args.n_sample), random_state=args.seed)

    sample = sample.reset_index(drop=True)
    print(f"Auditing {len(sample)} irrelevant articles (from {len(irrelevant)} total)")
    print(f"Per company: {sample['company_name'].value_counts().to_dict()}\n")

    rows = sample.to_dict("records")
    checkpoint_path = args.output + ".audit_checkpoint.json"

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            verdicts = json.load(f)
        done_ids = {str(v["id"]) for v in verdicts}
        rows = [r for r in rows if str(r["id"]) not in done_ids]
        print(f"Resuming: {len(verdicts)} done, {len(rows)} remaining")
    else:
        verdicts = []

    total_batches = (len(rows) + args.batch_size - 1) // args.batch_size
    failed = []

    for i in tqdm(range(0, len(rows), args.batch_size), total=total_batches,
                  desc="Auditing", unit="batch"):
        batch = rows[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        for attempt in range(4):
            try:
                result = audit_batch(client, args.model, batch)
                verdicts.extend(result)
                with open(checkpoint_path, "w") as f:
                    json.dump(verdicts, f)
                break
            except Exception as e:
                if attempt < 3:
                    wait = 30 * (2 ** attempt)
                    tqdm.write(f"Batch {batch_num} attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    tqdm.write(f"Batch {batch_num} failed after 4 attempts. Skipping.")
                    failed.append(batch_num)
        time.sleep(4.5)

    # Analyse results
    verdict_map = {str(v["id"]): v["verdict"] for v in verdicts}
    sample["audit_verdict"] = sample["id"].astype(str).map(verdict_map).fillna("IRRELEVANT")

    n_audited  = len(sample)
    n_flipped  = (sample["audit_verdict"] == "RELEVANT").sum()
    fn_rate    = 100 * n_flipped / n_audited if n_audited else 0

    print(f"\n{'='*50}")
    print(f"IRRELEVANT AUDIT RESULTS")
    print(f"{'='*50}")
    print(f"Articles audited:       {n_audited}")
    print(f"Confirmed irrelevant:   {n_audited - n_flipped}")
    print(f"Found RELEVANT (FN):    {n_flipped}")
    print(f"False-negative rate:    {fn_rate:.1f}%")

    if failed:
        print(f"\nWARNING: {len(failed)} batches skipped (defaulted to IRRELEVANT)")

    if fn_rate > 5:
        print("\nACTION REQUIRED: FN rate > 5% — apply the same LLM verification")
        print("to the entire irrelevant class before training.")
    elif fn_rate > 2:
        print("\nWARNING: FN rate 2-5% — document in paper, consider cleaning.")
    else:
        print("\nOK: FN rate < 2% — irrelevant class is clean enough for training.")

    if n_flipped > 0:
        fn_articles = sample[sample["audit_verdict"] == "RELEVANT"]
        print(f"\nFalse negatives by company:")
        print(fn_articles["company_name"].value_counts().to_string())
        print("\nSample FN titles:")
        for _, row in fn_articles.head(20).iterrows():
            title = str(row["title"]).encode("ascii", errors="replace").decode()
            print(f"  [{row['company_name']}] {title}")

    sample.to_csv(args.output, index=False)
    print(f"\nAudit results saved to {args.output}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


if __name__ == "__main__":
    main()
