"""
Stage 1: Annotate titles using Cerebras API with few-shot examples drawn from
the manually-labeled 300-article gold set.

Usage:
  python 01_annotate.py -i 00_filtered_nifty_sample.csv -o 01_annotated.csv
"""
import os, json, time, argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# Provider base URLs (all are OpenAI-compatible)
PROVIDER_URLS = {
    "groq":      "https://api.groq.com/openai/v1",
    "gemini":    "https://generativelanguage.googleapis.com/v1beta/openai/",
    "together":  "https://api.together.xyz/v1",
    "openrouter":"https://openrouter.ai/api/v1",
    "cerebras":  "https://api.cerebras.ai/v1",
    "openai":    None,  # default OpenAI
}

# ── System prompt ────────────────────────────────────────────────────────────
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
   - Major product launches, updates, or discontinuations (not minor feature tweaks)
   - Significant product recalls or safety alerts
   - AI/tech pivots, R&D breakthroughs, patent wins or losses
   - Supply chain disruptions, factory issues, logistics crises

4. Regulatory & Legal (corporate-level only)
   - Government antitrust investigations or rulings against the company
   - SEC/DOJ/FTC/FCA probes or settlements involving the company as an institution
   - FDA approvals, rejections, or warning letters
   - Major government fines, sanctions, or tariffs directly targeting the company
   - Class-action lawsuits (many plaintiffs against the company)
   - Regulatory body enforcement actions against the company

5. Brand & Reputation (material events only)
   - Officially signed celebrity/athlete endorsement DEALS or contract terminations
   - Documented consumer boycotts showing measurable sales impact or corporate response
   - Corporate-level data breaches or cybersecurity incidents
   - Severe executive scandals that trigger board action or regulatory scrutiny

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IRRELEVANT — exclude if any of these apply:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Individual/Local Incidents
   - A single employee's lawsuit, arrest, assault, or misconduct (not class-action, not executive)
   - A single store robbery, accident, shooting, or local incident
   - One customer's complaint or viral social media story with no corporate response

2. Consumer & Lifestyle Content
   - Shopping guides, deal roundups, gift lists ("Best Walmart deals this week")
   - Product reviews or comparisons written for consumers, not investors
   - Lifestyle articles that name-drop the brand
   - Minor incremental product feature updates (new emoji, UI tweak, app redesign)
   - Product rumours or leaks with no official announcement

3. Passing Mentions & Context
   - Macro/industry articles where the company appears as one of several examples
   - Articles primarily about a competitor or another company that briefly mention this one
   - Economic or political analysis that references the company incidentally

4. Non-Material Entertainment & Culture
   - A sponsored sports team's game result (unless the sponsorship deal itself is the news)
   - Content releases (new Netflix show, new album on Spotify) unless tied to earnings
   - Social media trends or memes involving the brand with no business impact
   - Workplace culture or office perks articles with no material financial impact

5. Analyst Firm vs. Analyst Target Rule (CRITICAL)
   - "Goldman Sachs upgrades Tesla" → RELEVANT for Tesla, IRRELEVANT for Goldman Sachs
   - Classify based on whose business is being analyzed, not who is doing the analyzing
   - Goldman Sachs/Morgan Stanley/JPMorgan making market predictions → IRRELEVANT for that firm
   - Goldman Sachs signing a deal, making an investment, facing a lawsuit → RELEVANT for Goldman Sachs

6. Company-as-Context vs. Company-as-Subject
   - IRRELEVANT: "How inflation is hitting retailers like Walmart, Target, and Costco"
   - RELEVANT: "Walmart cuts full-year guidance citing inflation pressure"
   - IRRELEVANT: "Apple takes a step into Microsoft's territory with Accenture deal" (Apple is subject)
   - The test: is this company the subject of the headline, or just supporting evidence?

7. Ambiguous Company Names
   - "Intel" in a military/intelligence context → IRRELEVANT (wrong Intel)
   - "Visa" in an immigration context → IRRELEVANT (wrong Visa)
   - "Target" in a military context → IRRELEVANT (wrong Target)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIE-BREAKING RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When genuinely uncertain, choose IRRELEVANT.
A clean dataset with some missed relevant articles is better than a noisy dataset.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Respond ONLY with a JSON array. Each element:
- "id": the id from input
- "label": exactly "relevant" or "irrelevant"
- "reason": one short sentence — what the article is about and why it qualifies or doesn't

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LABELED EXAMPLES (drawn from real financial news data — study every case):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RELEVANT — firm is the direct subject with material impact:
[
  {"id": "e1",  "label": "relevant",   "reason": "Regulator Blasts Wells Fargo for Deceptive Auto Insurance Program — enforcement action directly against Wells Fargo as an institution"},
  {"id": "e2",  "label": "relevant",   "reason": "Morning Agenda: The Wells Fargo Clawback — executive pay clawback, material corporate governance event for Wells Fargo"},
  {"id": "e3",  "label": "relevant",   "reason": "GM Strike after a week: Sales hold up while parts shortages loom — active labour strike impacting GM's operations and sales"},
  {"id": "e4",  "label": "relevant",   "reason": "Teamsters refusing to transport GM vehicles in solidarity with UAW — supply chain disruption directly hitting GM"},
  {"id": "e5",  "label": "relevant",   "reason": "Autoworker strike cost General Motors $1.1 billion — quantified financial loss from labour action on GM"},
  {"id": "e6",  "label": "relevant",   "reason": "Stifel Maintains Buy Rating for FedEx — analyst action rating FedEx's stock; relevant for FedEx, not Stifel"},
  {"id": "e7",  "label": "relevant",   "reason": "Intel Buys Mobileye for $15.3bn — major acquisition by Intel, direct corporate action with clear financial impact"},
  {"id": "e8",  "label": "relevant",   "reason": "Cramer: I have faith others can take charge at Intel after Brian Krzanich's ouster — CEO departure at Intel, strategy-affecting leadership change"},
  {"id": "e9",  "label": "relevant",   "reason": "Microsoft beats Amazon for Pentagon's $10 bln cloud computing contract — landmark contract win for Microsoft"},
  {"id": "e10", "label": "relevant",   "reason": "Walmart holiday quarter sales rise, ecommerce jumped 43 percent — quarterly financial results showing material growth for Walmart"},
  {"id": "e11", "label": "relevant",   "reason": "Netflix Q2 Earnings Preview: Can the Streaming Giant's Run Continue? — imminent earnings event directly about Netflix's financial performance"},
  {"id": "e12", "label": "relevant",   "reason": "California labour commission rules Uber drivers are employees — regulatory ruling with major operational and cost implications for Uber"},
  {"id": "e13", "label": "relevant",   "reason": "NYC becomes latest city to cut back on ties with Wells Fargo — institutional clients exiting, material reputational and revenue impact"},
  {"id": "e14", "label": "relevant",   "reason": "Goldman Sachs to invest $184 million in Brazil storage company — Goldman acting as principal investor; this is Goldman's own corporate action"},
  {"id": "e15", "label": "relevant",   "reason": "Airbnb employees say feel betrayed as 1,900 layoffs rip apart company — major restructuring event with direct financial and operational impact on Airbnb"},
  {"id": "e16", "label": "relevant",   "reason": "Starbucks union workers in Buffalo walk out over unsafe conditions — labour action directly affecting Starbucks store operations"},
  {"id": "e17", "label": "relevant",   "reason": "LG Chem shares slide amid electric vehicle battery-fire probe with GM — safety probe directly involving GM's EV supply chain"},
  {"id": "e18", "label": "relevant",   "reason": "Binance Card Holders as Mastercard and Visa Reassess Ties — Visa making a corporate-level decision to cut a major partner relationship"},
  {"id": "e19", "label": "relevant",   "reason": "AT&T touts Time Warner merger at U.S. Senate hearing — AT&T defending its own merger before regulators; direct corporate regulatory event"},
  {"id": "e20", "label": "relevant",   "reason": "LG Chem shares slide amid electric vehicle battery-fire probe with GM — safety probe directly involving GM's EV supply chain, material liability risk"}
]

IRRELEVANT — individual incident, consumer content, company-as-context, or wrong company:
[
  {"id": "e21", "label": "irrelevant", "reason": "Uber driver charged with raping unconscious 17-year-old — individual employee crime, not a corporate-level event or policy failure"},
  {"id": "e22", "label": "irrelevant", "reason": "Prosecutors: South Carolina man killed Uber rider — individual criminal act by a third party, not a systemic corporate news event"},
  {"id": "e23", "label": "irrelevant", "reason": "Microsoft brings Known Folder Migration feature to OneDrive consumer users — minor incremental product feature, no material financial or strategic impact"},
  {"id": "e24", "label": "irrelevant", "reason": "Microsoft is allegedly creating a Slack competitor — unconfirmed rumour with no official announcement or material corporate action"},
  {"id": "e25", "label": "irrelevant", "reason": "Microsoft's Last Windows Update For 600 Million Users Act Now — consumer security advisory article, not a corporate action or financial event"},
  {"id": "e26", "label": "irrelevant", "reason": "New Windows 10 preview leaks as Microsoft struggles to deliver — product preview leak, not a material corporate announcement"},
  {"id": "e27", "label": "irrelevant", "reason": "Walmart Deals for Days sale is live — TVs, AirPods, Instant Pot — promotional consumer deal roundup, not investor-relevant business news"},
  {"id": "e28", "label": "irrelevant", "reason": "Here's everything I'd buy at Walmart's Black Friday sale — consumer shopping guide, not business news"},
  {"id": "e29", "label": "irrelevant", "reason": "Intel officials warned well before Tower 22 attack of drone risks — military intelligence context; this is not Intel the semiconductor company"},
  {"id": "e30", "label": "irrelevant", "reason": "Apple takes another step into Microsoft's core territory with Accenture deal — Apple is the subject here; Microsoft appears only as a competitive reference"},
  {"id": "e31", "label": "irrelevant", "reason": "Shooting at Walmart in El Paso — local crime incident at a single store, not a corporate-level event"},
  {"id": "e32", "label": "irrelevant", "reason": "Starbucks Just Revealed The Ideal Drink For Each Zodiac Sign — lifestyle/entertainment content with zero investor relevance"},
  {"id": "e33", "label": "irrelevant", "reason": "Galaxy Z Flip 5G now up for preorder — consumer device availability article; AT&T is just the carrier, not the subject"},
  {"id": "e34", "label": "irrelevant", "reason": "No massages? Why Uber's workplace is different than other tech companies — workplace culture fluff, no material financial impact"},
  {"id": "e35", "label": "irrelevant", "reason": "Uber app redesign adds real-time language translation — minor app feature update, not a material business development"},
  {"id": "e36", "label": "irrelevant", "reason": "Facebook denies giving Spotify, Netflix other tech giants wider data access — Facebook is the primary subject; Netflix is one of several named companies"},
  {"id": "e37", "label": "irrelevant", "reason": "Goldman Sachs believes Trump's tariffs leave U.S. in event-driven bear market — Goldman acting as macro analyst making market predictions, not news about Goldman's own business"},
  {"id": "e38", "label": "irrelevant", "reason": "Why Bernie Sanders should talk more about Goldman Sachs — political opinion piece, not corporate or financial news about Goldman Sachs"},
  {"id": "e39", "label": "irrelevant", "reason": "Get Microsoft Office 2019 on sale for just $30 — promotional deal for consumers, not investor-relevant business news"},
  {"id": "e40", "label": "irrelevant", "reason": "Netflix Just Released a Surprise Sequel to Cloverfield — content release announcement; not tied to earnings, contracts, or financial results"}
]
"""


def extract_json_array(text: str) -> list:
    text = text.strip()
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON array found in response: {text[:200]!r}")
    raw = text[start:end+1]

    # First try clean parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fallback: extract individual objects with regex, only keep id+label
    # Handles malformed JSON caused by special chars in titles/reasons
    import re
    objects = []
    for m in re.finditer(
        r'\{\s*"id"\s*:\s*"?([^",}]+)"?\s*,.*?"label"\s*:\s*"(relevant|irrelevant)"',
        raw, re.DOTALL | re.IGNORECASE
    ):
        objects.append({"id": m.group(1).strip(), "label": m.group(2).strip(), "reason": ""})
    if objects:
        return objects

    raise ValueError(f"Could not parse JSON from response: {raw[:300]!r}")


def classify_batch(client, model_name, batch):
    items = [{"id": row["id"], "title": row["title"]} for row in batch]
    prompt = f"Classify these titles:\n{json.dumps(items, indent=2)}"
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.0,
        max_tokens=4096,
    )
    text = response.choices[0].message.content
    if not text or not text.strip():
        raise ValueError("Model returned empty response")
    return extract_json_array(text)



def main():
    parser = argparse.ArgumentParser(description="Stage 1 v2: Annotate titles with few-shot prompt")
    parser.add_argument("--input",      "-i", type=str, required=True)
    parser.add_argument("--output",     "-o", type=str, required=True)
    parser.add_argument("--api_key",    "-k", type=str, default=None)
    parser.add_argument("--batch_size", "-b", type=int, default=15)
    parser.add_argument("--model",      "-m", type=str, default="llama3.1-8b")
    parser.add_argument("--provider",   "-p", type=str, default="cerebras",
                        choices=list(PROVIDER_URLS.keys()),
                        help="API provider (default: groq)")
    parser.add_argument("--title_col",        type=str, default="title")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("CEREBRAS_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Provide API key via --api_key / GROQ_API_KEY / CEREBRAS_API_KEY env var")

    base_url = PROVIDER_URLS.get(args.provider)
    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"Provider: {args.provider} | Base URL: {base_url or 'OpenAI default'}")

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    df = df.dropna(subset=[args.title_col]).copy()
    df['id'] = df['id'].astype(str)
    print(f"Annotating {len(df)} titles | batch={args.batch_size} | model={args.model}")

    rows = df[['id', args.title_col]].rename(columns={args.title_col: 'title'}).to_dict('records')

    checkpoint_path = args.output + ".checkpoint.json"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            results = json.load(f)
        done_ids = {str(r['id']) for r in results}
        rows = [r for r in rows if str(r['id']) not in done_ids]
        print(f"Resuming: {len(results)} done, {len(rows)} remaining.")
    else:
        results = []

    total_batches = (len(rows) + args.batch_size - 1) // args.batch_size
    start_time = time.time()
    failed_batches = []

    for i in tqdm(range(0, len(rows), args.batch_size), total=total_batches,
                  desc="Annotating", unit="batch"):
        batch = rows[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        for attempt in range(4):
            try:
                parsed = classify_batch(client, args.model, batch)
                results.extend(parsed)
                with open(checkpoint_path, 'w') as f:
                    json.dump(results, f)
                break
            except Exception as e:
                if attempt < 3:
                    # Parse retry-after from 429 messages (e.g. "Please try again in 12m57s")
                    import re
                    wait = 30 * (2 ** attempt)
                    m = re.search(r'try again in (\d+)m([\d.]+)s', str(e))
                    if m:
                        wait = int(m.group(1)) * 60 + float(m.group(2)) + 5
                    tqdm.write(f"Batch {batch_num} failed (attempt {attempt+1}): {e}. Retrying in {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    tqdm.write(f"Batch {batch_num} failed after 4 attempts: {e}. Skipping.")
                    failed_batches.append(batch_num)

        if batch_num % 5 == 0 or batch_num == total_batches:
            elapsed = time.time() - start_time
            rate = batch_num / elapsed if elapsed > 0 else 0
            eta = (total_batches - batch_num) / rate if rate > 0 else 0
            rel = sum(1 for r in results if r.get('label') == 'relevant')
            irr = sum(1 for r in results if r.get('label') == 'irrelevant')
            tqdm.write(f"[{batch_num}/{total_batches}] {len(results)} annotated | "
                       f"relevant={rel} irrelevant={irr} | ETA={eta:.0f}s")

        time.sleep(4.5)  # llama3.1-8b: 60k tokens/min, ~4.5k tokens/req → ~13 req/min max

    results_df = pd.DataFrame(results)
    results_df['id'] = results_df['id'].astype(str)
    final_df = df.merge(results_df[['id', 'label', 'reason']], on='id', how='left')

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"\nAnnotated {results_df['label'].notna().sum()} / {len(df)} titles")
    print(final_df['label'].value_counts())
    if failed_batches:
        print(f"WARNING: {len(failed_batches)} batches skipped: {failed_batches}")

    final_df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
