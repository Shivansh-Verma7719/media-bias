"""
Targeted synthetic data generation based on error analysis of model_synthetic.

Generates examples for the 10 FP patterns and 7 FN patterns identified
from the 04_evaluate.py output on test.csv.

Output: targeted_synthetic.csv (irrelevant + relevant)
        combined_training_set_v2.csv (original 5k + targeted)
"""
import random
import io
import sys
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
random.seed(42)

SP500 = [
    "Apple", "Microsoft", "Amazon", "Alphabet", "Meta", "Tesla", "Nvidia",
    "Berkshire Hathaway", "ExxonMobil", "JPMorgan Chase", "Johnson & Johnson",
    "Visa", "Mastercard", "Procter & Gamble", "UnitedHealth", "Home Depot",
    "Chevron", "AbbVie", "Pfizer", "Merck", "Coca-Cola", "PepsiCo",
    "Costco", "Walmart", "McDonald's", "Starbucks", "Nike", "Netflix",
    "Salesforce", "Adobe", "Intel", "Qualcomm", "Broadcom", "Cisco",
    "Goldman Sachs", "Morgan Stanley", "Bank of America", "Wells Fargo",
    "Citigroup", "BlackRock", "American Express", "Airbnb", "Uber", "Lyft",
    "Ford", "General Motors", "Boeing", "Lockheed Martin", "Raytheon",
    "3M", "General Electric", "Honeywell", "Caterpillar", "Deere & Company",
    "UPS", "FedEx", "Delta Air Lines", "United Airlines",
    "AT&T", "Verizon", "T-Mobile", "Comcast", "Disney", "Warner Bros",
    "CVS Health", "Walgreens", "HCA Healthcare", "Eli Lilly",
    "ConocoPhillips", "Halliburton", "Schlumberger", "Waste Management",
    "Republic Services", "Dollar General", "Target", "Lowe's",
    "Kraft Heinz", "Mondelez", "Colgate-Palmolive", "Kimberly-Clark",
    "Aflac", "Travelers", "Progressive", "Allstate", "MetLife",
    "Charles Schwab", "Fidelity", "Vanguard",
]

COUNTRIES = ["Brazil", "India", "Germany", "France", "Japan", "Australia",
             "Canada", "Mexico", "South Korea", "Singapore", "UK", "Italy",
             "Spain", "Netherlands", "Sweden", "Poland", "Indonesia", "Vietnam"]

CITIES = ["Fort Collins", "Phoenix", "Austin", "Nashville", "Denver",
          "Charlotte", "Columbus", "Indianapolis", "Portland", "Salt Lake City",
          "Raleigh", "Atlanta", "Pittsburgh", "Cincinnati", "Tampa"]

STATES = ["Colorado", "Texas", "Tennessee", "Arizona", "North Carolina",
          "Ohio", "Utah", "Georgia", "Pennsylvania", "Indiana", "Florida"]

CAUSES = ["disaster response", "climate action", "veterans support",
          "education access", "workforce development", "food security",
          "housing affordability", "mental health", "STEM education",
          "community resilience", "hunger relief", "clean water access"]

SOCIAL_ISSUES = [
    "racial discrimination in promotion decisions",
    "gender pay gap allegations",
    "hostile work environment claims",
    "age discrimination among warehouse workers",
    "disability accommodation failures",
    "pregnancy discrimination in hiring",
    "sexual harassment in sales division",
    "religious discrimination in scheduling",
    "national origin discrimination claims",
    "retaliation against whistleblowers",
]

PRODUCTS = ["mobile app", "digital wallet", "loyalty rewards program",
            "subscription service", "streaming platform", "cloud dashboard",
            "enterprise software", "consumer app", "fintech feature",
            "health monitoring tool"]

FEATURES = ["two-factor authentication", "dark mode", "push notifications",
            "in-app tipping", "QR code payments", "voice search",
            "biometric login", "personalized feed", "offline mode",
            "real-time tracking"]

REGIONS = ["Southeast Asia", "Latin America", "Eastern Europe",
           "Sub-Saharan Africa", "the Middle East", "Central America",
           "the Nordics", "Southeast Europe", "West Africa", "South Asia"]

MID_ROLES = [
    "head of investor relations", "head of corporate communications",
    "head of digital marketing", "regional vice president for Europe",
    "chief of staff to the CEO", "head of sustainability",
    "VP of human resources", "director of government affairs",
    "head of supply chain analytics", "chief diversity officer",
    "head of global partnerships", "VP of product marketing",
]

COMPETING_INDUSTRIES = [
    "taxi", "hotel", "cable", "newspaper", "retail banking",
    "travel agency", "video rental", "landline phone",
    "brick-and-mortar retail", "traditional advertising",
]

SAFETY_ISSUES = [
    "braking problems", "battery overheating issues", "steering defect",
    "software glitch causing crashes", "airbag malfunction",
    "seatbelt failure", "fire risk in charging systems",
    "structural integrity issues", "door latch defect",
    "emissions control failure",
]

INFORMAL_METRICS = [
    ("revenue", ["beats expectations", "tops estimates", "exceeds forecasts"]),
    ("earnings", ["comes in ahead of Wall Street", "beats the Street", "surpasses consensus"]),
    ("profit", ["finishes strong", "caps year on high note", "closes out on high note"]),
]

COMPETITORS = ["Samsung", "Google", "Apple", "Microsoft", "Amazon",
               "Alibaba", "Huawei", "ByteDance", "Baidu", "Tencent",
               "Toyota", "Volkswagen", "BMW", "Mercedes", "Hyundai",
               "Walmart", "Target", "Costco", "Carrefour", "Aldi"]


def pick(lst):
    return random.choice(lst)


def small_amount():
    return random.choice([1, 2, 3, 4, 5, 7, 8, 10, 12, 15, 20, 25, 30, 40, 45])


def big_amount():
    v = random.choice([500, 750, 1000, 1200, 1500, 1800, 2000, 2500, 3000,
                       3400, 4000, 4500, 5000, 6000, 7500, 8000, 10000])
    return v


def amount_str(v):
    if v >= 1000:
        return f"${v//1000}.{(v % 1000)//100}B" if v % 1000 else f"${v//1000}B"
    return f"${v}M"


def quarter():
    return pick(["Q1", "Q2", "Q3", "Q4"])


def year():
    return pick(list(range(2015, 2026)))


def person():
    first = pick(["James", "Sarah", "Michael", "Emily", "David", "Jennifer",
                  "Robert", "Lisa", "William", "Karen", "Richard", "Susan",
                  "John", "Patricia", "Charles", "Linda", "Mark", "Barbara"])
    last = pick(["Johnson", "Smith", "Williams", "Brown", "Jones", "Miller",
                 "Davis", "Wilson", "Moore", "Taylor", "Anderson", "Thomas",
                 "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia"])
    return f"{first} {last}"


# ── IRRELEVANT TEMPLATES ──────────────────────────────────────────────────────

def gen_g11_speculative(n=100):
    """Company 'considers' or 'explores' — no decision made, speculative."""
    templates = [
        "{co} considers importing {product} to {country}",
        "{co} weighing options to expand into {country} market, sources say",
        "{co} exploring potential sale of {unit} unit, people familiar say",
        "{co} mulls {country} market entry as growth strategy",
        "Report: {co} considering move into {product} business",
        "{co} may exit {country} amid regulatory uncertainty, sources say",
        "{co} studying options for restructuring {unit} division",
        "{co} in early talks about potential {country} partnership — no deal imminent",
        "Sources: {co} weighing {action} but no decision expected soon",
        "{co} CEO says company is 'open to' exploring {product} opportunities",
    ]
    actions = ["a spinoff", "an IPO of its subsidiary", "a joint venture",
               "a strategic review", "a minority stake sale", "an asset sale"]
    units = ["media", "insurance", "consumer", "industrial", "logistics",
             "retail", "tech", "healthcare", "energy", "financial services"]
    products = ["LNG", "solar panels", "electric vehicles", "cloud services",
                "streaming content", "financial products", "insurance products"]
    rows = []
    for _ in range(n):
        co = pick(SP500)
        t = pick(templates).format(
            co=co,
            product=pick(products),
            country=pick(COUNTRIES),
            unit=pick(units),
            action=pick(actions),
        )
        rows.append({"title": t, "company_name": co, "label": "irrelevant"})
    return rows


def gen_g12_analyst_preview(n=100):
    """Analyst preview/expectation articles — look like earnings but aren't."""
    templates = [
        "What To Expect From {co}'s {year} After {adj} Fiscal {q}",
        "{co} Earnings Preview: What Analysts Are Watching This Quarter",
        "Here's What Wall Street Expects From {co} When It Reports {q}",
        "{co} {q} Earnings: Analyst Estimates and Key Metrics to Watch",
        "What to Watch When {co} Reports Earnings {day}",
        "{co} Earnings: {number} Things Investors Should Monitor",
        "Ahead of {co}'s {q} Report: Consensus Estimates and Risks",
        "{co}'s Upcoming Earnings: Will the Stock React?",
        "Breaking Down {co} Pre-Earnings: Bull vs Bear Case",
        "Analysts Divided on {co} Heading Into {q} Report",
    ]
    adjs = ["Solid", "Strong", "Mixed", "Disappointing", "Surprising",
            "Volatile", "Steady", "Modest", "Robust", "Challenging"]
    days = ["Friday", "Monday", "Tuesday", "after the bell",
            "before the open", "next week", "after market close"]
    rows = []
    for _ in range(n):
        co = pick(SP500)
        t = pick(templates).format(
            co=co, year=year(), adj=pick(adjs),
            q=quarter(), day=pick(days),
            number=pick([3, 4, 5, 6, 7]),
        )
        rows.append({"title": t, "company_name": co, "label": "irrelevant"})
    return rows


def gen_g13_quantified_philanthropy(n=100):
    """CSR/philanthropy with dollar amounts — looks material but isn't."""
    templates = [
        "The {co} Foundation Increases {cause} Commitment To ${amount} Million",
        "{co} pledges ${amount} million to {cause} initiative",
        "{co} Foundation awards ${amount}K grant to local {cause} nonprofit",
        "{co} commits ${amount} million to {city} {cause} fund",
        "{co} announces ${amount}M donation to support {cause}",
        "{co} Foundation launches ${amount}M initiative for {cause}",
        "{co} gives ${amount}M to {university} for {cause} scholarship",
        "{co} doubles {cause} funding to ${amount} million for {year}",
        "{co} Foundation reports ${amount}M in {cause} grants awarded this year",
        "{co}'s charitable arm commits to ${amount}M {cause} pledge",
    ]
    universities = ["MIT", "Stanford", "Harvard", "Howard University",
                    "Morehouse College", "UCLA", "Michigan", "Texas A&M"]
    rows = []
    for _ in range(n):
        co = pick(SP500)
        amt = random.choice([1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 50])
        t = pick(templates).format(
            co=co, cause=pick(CAUSES), amount=amt,
            city=pick(CITIES), year=year(),
            university=pick(universities),
        )
        rows.append({"title": t, "company_name": co, "label": "irrelevant"})
    return rows


def gen_g14_regional_micro_launch(n=100):
    """Product/service launch in specific region — geographic qualifier inflates."""
    templates = [
        "{co} launches new app and {feature} in {country}",
        "{co} expands {product} to {country} with {number} pilot users",
        "{co} rolls out {feature} in {country} ahead of broader rollout",
        "{co} introduces {product} in {region} market",
        "{co} tests {feature} in {city} starting this month",
        "{co} brings {product} to {country} users for the first time",
        "{co} soft-launches {feature} in {country} following local regulatory approval",
        "{co} extends {product} trial to {country} and {country2}",
        "{co} debuts {product} in {region} in bid to gain local users",
        "{co} adds {feature} to its {country} platform",
    ]
    rows = []
    for _ in range(n):
        co = pick(SP500)
        countries = random.sample(COUNTRIES, 2)
        t = pick(templates).format(
            co=co,
            feature=pick(FEATURES),
            product=pick(PRODUCTS),
            country=countries[0],
            country2=countries[1],
            region=pick(REGIONS),
            city=pick(CITIES),
            number=pick([500, 1000, 2000, 5000, 10000, 50000]),
        )
        rows.append({"title": t, "company_name": co, "label": "irrelevant"})
    return rows


def gen_g15_midlevel_appointment(n=100):
    """Non-C-suite executive appointment — not CEO/CFO/board level."""
    templates = [
        "BRIEF-{co} names {person} as {role}",
        "{co} appoints {person} to newly created {role} position",
        "{co} promotes {person} to {role}",
        "{co} hires {person} as {role} from {rival}",
        "{co} names {person} to lead {role} function",
        "{co} taps {person} for {role} role amid organizational changes",
        "{co} fills {role} role with internal promotion of {person}",
        "{co} announces {person} as new {role}",
        "{person} joins {co} as {role}",
        "{co} recruits {person} from {rival} as {role}",
    ]
    rivals = ["Goldman Sachs", "McKinsey", "Deloitte", "PwC", "Accenture",
              "JPMorgan", "Morgan Stanley", "Google", "Amazon", "Microsoft"]
    rows = []
    for _ in range(n):
        co = pick(SP500)
        t = pick(templates).format(
            co=co,
            person=person(),
            role=pick(MID_ROLES),
            rival=pick(rivals),
        )
        rows.append({"title": t, "company_name": co, "label": "irrelevant"})
    return rows


def gen_g16_local_economic(n=100):
    """Article about local/regional economic benefit from a company deal."""
    templates = [
        "{city} a big winner in {co}'s new purchase agreement with {partner}",
        "{co}'s new facility brings {number} jobs to {city}, officials say",
        "{state} gains economic boost as {co} expands operations",
        "How {co}'s decision to build in {city} benefits local workers",
        "{city} officials celebrate as {co} selects site for new {facility}",
        "{co} expansion to create {number} jobs in {city}, governor says",
        "{state} lands {co} investment worth ${amount}M in local economy",
        "{city} touts economic windfall from {co}'s new {facility}",
        "Local suppliers cheer {co}'s new production deal as {city} gains",
        "{co}'s new {facility} in {city} to employ {number} by {year}",
    ]
    partners = ["Apple", "Google", "Amazon", "Microsoft", "Boeing",
                "Ford", "General Motors", "Walmart", "Target", "Costco"]
    facilities = ["distribution center", "data center", "manufacturing plant",
                  "research campus", "logistics hub", "fulfillment center",
                  "corporate office", "training center"]
    rows = []
    for _ in range(n):
        co = pick(SP500)
        t = pick(templates).format(
            co=co,
            city=pick(CITIES),
            state=pick(STATES),
            partner=pick(partners),
            number=pick([50, 100, 200, 300, 500, 750, 1000, 1500, 2000]),
            amount=pick([50, 75, 100, 150, 200, 250]),
            facility=pick(facilities),
            year=year(),
        )
        rows.append({"title": t, "company_name": co, "label": "irrelevant"})
    return rows


def gen_g17_company_as_comparator(n=100):
    """Company name used as brand/archetype for another company, not primary subject."""
    templates = [
        "The {co} of {country} just {action} to conquer the US market",
        "Meet {country}'s answer to {co} — and why it's growing fast",
        "Is {startup} becoming the next {co}?",
        "Why analysts are calling {startup} '{country}'s {co}'",
        "{co2} is emerging as the {co} of {industry}",
        "{startup} aims to be the {co} of {industry} in {country}",
        "The startup that wants to do to {industry} what {co} did to retail",
        "{country} startup billed as the '{co} of Asia' raises ${amount}M",
        "Could {startup} be the {country} version of {co}?",
        "{co2} vs {co}: who wins the {country} market?",
    ]
    startups = ["iQiyi", "Grab", "Gojek", "Lazada", "Flipkart", "MercadoLibre",
                "OLX", "Coupang", "Sea Limited", "Rappi", "Jumia", "Tokopedia",
                "Bukalapak", "Careem", "Zomato", "Swiggy"]
    industries = ["food delivery", "e-commerce", "ride-hailing", "streaming",
                  "financial services", "healthcare", "logistics", "education"]
    cos2 = ["Alibaba", "Baidu", "Tencent", "ByteDance", "Xiaomi",
            "Samsung", "Hyundai", "SoftBank", "Rakuten", "Line"]
    actions = ["acquired a TV maker", "went public", "raised $1B",
               "launched in 10 new cities", "partnered with a US giant",
               "surpassed 100 million users", "filed for an IPO"]
    rows = []
    for _ in range(n):
        co = pick(SP500)
        t = pick(templates).format(
            co=co,
            country=pick(COUNTRIES),
            startup=pick(startups),
            industry=pick(industries),
            co2=pick(cos2),
            action=pick(actions),
            amount=pick([50, 100, 200, 300, 500]),
        )
        rows.append({"title": t, "company_name": co, "label": "irrelevant"})
    return rows


def gen_g18_small_lawsuit(n=100):
    """Small-dollar lawsuit ($1M-$50M) — immaterial at S&P 500 scale."""
    templates = [
        "{co} faces ${amount}M lawsuit over {issue}",
        "{co} sued by {plaintiff} over {issue}, ${amount}M claimed",
        "${amount}M lawsuit alleges {co} {issue}",
        "{co} hit with ${amount}M class action over {issue}",
        "Lawsuit filed against {co} over {issue}, seeks ${amount}M",
        "{co} {issue} lawsuit alleges violation of consumer protection law",
        "Former {co} customer files ${amount}M suit over {issue}",
        "{co} faces legal action over {issue} in {state}",
        "Group of {co} users file ${amount}M lawsuit alleging {issue}",
        "${amount}M suit: {co} accused of {issue} in workplace",
    ]
    plaintiffs = ["a group of customers", "a former employee",
                  "a consumer advocacy group", "a class of workers",
                  "a coalition of users", "a nonprofit organization",
                  "a group of shareholders"]
    rows = []
    for _ in range(n):
        co = pick(SP500)
        t = pick(templates).format(
            co=co,
            amount=small_amount(),
            issue=pick(SOCIAL_ISSUES),
            plaintiff=pick(plaintiffs),
            state=pick(STATES),
        )
        rows.append({"title": t, "company_name": co, "label": "irrelevant"})
    return rows


def gen_g19_visa_contamination(n=100):
    """Travel/immigration visa articles incorrectly tagged to Visa Inc."""
    templates = [
        "{country} wants concrete date for EU visa liberalization",
        "US delays visa program for {country} travelers amid security review",
        "Student visa rules tighten as universities face scrutiny",
        "{country} pushes for faster US visa processing after backlog grows",
        "New visa requirements affect {country} visitors to the US",
        "EU visa-free travel for {country} under review after border incidents",
        "Visa waiver program expanded to include {country}, officials say",
        "Work visa approvals fall as {country} professionals face longer waits",
        "{country} threatens to restrict US visa access in retaliation",
        "Visa backlog hits {country} workers applying for H-1B transfers",
        "Student visa subpoenas alarm {country} universities",
        "{country} nationals face new biometric visa requirements",
        "Tourist visa applications from {country} surge ahead of summer",
        "US tightens visa rules for {country} nationals over security concerns",
        "Visa processing delays strand {country} travelers at US airports",
    ]
    rows = []
    for _ in range(n):
        t = pick(templates).format(country=pick(COUNTRIES))
        rows.append({"title": t, "company_name": "Visa", "label": "irrelevant"})
    return rows


def gen_g20_industry_disruption_catalyst(n=100):
    """Company mentioned as cause of disruption; article is about the disrupted industry."""
    templates = [
        "Competition with {co} sparks audit of {industry} lenders",
        "{industry} players pivot as {co} disrupts market with new {product}",
        "How {co} is forcing the {industry} industry to reinvent itself",
        "{industry} giants struggle as {co} {action}",
        "{industry} braces for impact as {co} prepares to enter market",
        "{co} threat prompts {industry} industry to lobby for protection",
        "Investors dump {industry} stocks on fears of {co} disruption",
        "{industry} firms face existential crisis as {co} {action}",
        "The {industry} industry's answer to the {co} challenge",
        "{co}'s growth is reshaping how the {industry} industry operates",
    ]
    actions = ["cuts prices aggressively", "launches competing product",
               "wins major government contract", "partners with key supplier",
               "enters the market", "raises fresh capital",
               "announces new feature", "expands to new cities"]
    rows = []
    for _ in range(n):
        co = pick(SP500)
        t = pick(templates).format(
            co=co,
            industry=pick(COMPETING_INDUSTRIES),
            product=pick(PRODUCTS),
            action=pick(actions),
        )
        rows.append({"title": t, "company_name": co, "label": "irrelevant"})
    return rows


# ── RELEVANT TEMPLATES ────────────────────────────────────────────────────────

def gen_h7_layoffs_employee_pov(n=75):
    """Layoffs framed from employee perspective — model missed these."""
    templates = [
        "{co} employees say they feel betrayed as {number} layoffs {action}",
        "Workers at {co} express anger as company eliminates {number} roles",
        "{co} staff react with shock as {number} colleagues receive pink slips",
        "Insiders at {co} describe chaos as {number} jobs cut without warning",
        "{co} workers say morale has collapsed following {number}-person reduction",
        "'{co} has changed': employees describe fallout from {number} job cuts",
        "{co} employees told to clear desks as {number} roles axed in restructuring",
        "{number} {co} workers lose jobs in latest round of cost-cutting",
        "{co} employees blindsided as company announces {number} more layoffs",
        "{co} workforce reduced by {number} as restructuring accelerates",
    ]
    actions = ["rip apart teams", "hit without warning", "take effect immediately",
               "sweep through divisions", "gut entire departments", "shock the office"]
    rows = []
    for _ in range(n):
        co = pick(SP500)
        t = pick(templates).format(
            co=co,
            number=pick([500, 750, 900, 1000, 1200, 1500, 1900, 2000, 2500, 3000, 5000]),
            action=pick(actions),
        )
        rows.append({"title": t, "company_name": co, "label": "relevant"})
    return rows


def gen_h8_safety_admission(n=75):
    """CEO/exec admits safety problem — model missed these."""
    templates = [
        "{co} CEO admits {product} has {issue}",
        "{co} acknowledges {product} defect after {number} customer complaints",
        "{co} founder concedes {product} fell short of safety standards",
        "{co} exec admits {issue} with {product} model after pressure mounts",
        "{co} says it was 'too slow' to address {product} {issue}",
        "{co} acknowledges {issue} in {product}, promises fix within {weeks} weeks",
        "{co} CEO concedes {product} problems are 'more serious than we thought'",
        "{co} admits knowing about {product} {issue} before public reports",
        "{co} leadership acknowledges {product} recall was delayed due to {issue}",
        "Under pressure, {co} admits {product} has {issue} affecting {number} units",
    ]
    rows = []
    for _ in range(n):
        co = pick(SP500)
        products = ["Model 3", "flagship phone", "new SUV", "electric vehicle",
                    "blood glucose monitor", "insulin pump", "cardiac device",
                    "commercial aircraft", "consumer product", "medical device"]
        t = pick(templates).format(
            co=co,
            product=pick(products),
            issue=pick(SAFETY_ISSUES),
            number=pick([500, 1000, 2000, 5000, 10000, 50000, 100000]),
            weeks=pick([4, 6, 8, 12, 16]),
        )
        rows.append({"title": t, "company_name": co, "label": "relevant"})
    return rows


def gen_h9_informal_earnings(n=75):
    """Earnings/financial results with informal or indirect language."""
    templates = [
        "{co} finishes {year} strong, has high hopes for {next_year}",
        "{co} caps {year} with record {metric} as {driver} accelerates",
        "{co}'s {q}: better than feared, outlook encouraging",
        "{co} powers through {year} on {driver}, eyes {next_year} recovery",
        "{co} closes out {year} on high note despite {challenge}",
        "{co} wraps up {year} with a bang as {metric} {action}",
        "{co}'s year-end results: {adj} quarter, positive {next_year} view",
        "In a surprise, {co} {q} {action} — stock rallies after hours",
        "{co} {metric} tops estimates as management lifts {next_year} outlook",
        "{co} delivers solid close to {year} as {metric} {action}",
    ]
    metrics = ["revenue", "earnings", "profit", "margins", "cash flow",
               "free cash flow", "operating income", "gross profit"]
    drivers = ["cloud growth", "subscription momentum", "AI spending",
               "consumer demand", "pricing power", "cost discipline",
               "international expansion", "new product cycle"]
    challenges = ["macro headwinds", "supply chain pressure", "rising costs",
                  "currency drag", "slower consumer spending"]
    adjs = ["strong", "solid", "better-than-expected", "impressive", "resilient"]
    actions_list = ["surges", "beats consensus", "comes in ahead", "tops estimates",
                    "exceeds Wall Street forecasts", "beats the Street"]
    rows = []
    for _ in range(n):
        co = pick(SP500)
        y = year()
        t = pick(templates).format(
            co=co, year=y, next_year=y + 1,
            metric=pick(metrics), driver=pick(drivers),
            challenge=pick(challenges), adj=pick(adjs),
            action=pick(actions_list), q=quarter(),
        )
        rows.append({"title": t, "company_name": co, "label": "relevant"})
    return rows


def gen_h10_succession_departure(n=75):
    """Executive departure framed as insider/succession story."""
    templates = [
        "{person} once seen as possible {co} CEO successor now out at company",
        "Person widely expected to lead {co} departs in surprise move",
        "{co} loses senior executive seen as heir apparent to {role}",
        "The executive many thought would run {co} has left the company",
        "{co} heir-apparent exits as board searches for new leadership",
        "{person}, {co}'s likely next CEO, resigns following board dispute",
        "Surprise exit: {co}'s presumed future CEO departs after {months} months",
        "{co} loses {role} seen as succession frontrunner to rival firm",
        "Inside {co}'s leadership crisis: the departure of its next CEO",
        "{person} leaves {co}, ending speculation about CEO succession",
    ]
    roles = ["CFO", "COO", "president", "division head", "chief strategy officer",
             "head of core business", "co-president"]
    rows = []
    for _ in range(n):
        co = pick(SP500)
        t = pick(templates).format(
            co=co,
            person=person(),
            role=pick(roles),
            months=pick([6, 9, 12, 18, 24]),
        )
        rows.append({"title": t, "company_name": co, "label": "relevant"})
    return rows


def gen_h11_new_financial_product(n=75):
    """New financial product/crypto/digital currency launch."""
    templates = [
        "{co} launches cryptocurrency token to streamline payments",
        "{co} coin: {co}'s new digital currency explained",
        "{co} unveils blockchain-based payment system for enterprise clients",
        "{co} enters crypto space with new digital token backed by deposits",
        "{co} announces stablecoin to accelerate cross-border payments",
        "{co} introduces digital bond platform on blockchain",
        "{co} coin: inside the bank's end-run around traditional payments",
        "{co} launches tokenized fund on blockchain for institutional investors",
        "{co} debuts digital currency for same-day settlement between banks",
        "{co} files patent for blockchain-based payment rail",
    ]
    rows = []
    for _ in range(n):
        co = pick(["JPMorgan Chase", "Goldman Sachs", "Morgan Stanley",
                   "Bank of America", "Citigroup", "Wells Fargo",
                   "Mastercard", "Visa", "American Express",
                   "BlackRock", "Fidelity", "Charles Schwab",
                   "PayPal", "Square", "Robinhood"])
        t = pick(templates).format(co=co)
        rows.append({"title": t, "company_name": co, "label": "relevant"})
    return rows


def gen_h12_cross_company_investment(n=75):
    """Company A's stake/investment in Company B, framed from B's perspective."""
    templates = [
        "{co2} shares rise as investors learn of {co}'s stake",
        "{co2} surges after {co} discloses {pct}% ownership position",
        "{co} quietly builds ${amount}M stake in {co2}, filing shows",
        "Filing reveals {co} has acquired {pct}% of {co2}",
        "{co2} stock jumps as {co} emerges as major shareholder",
        "{co}'s investment in {co2} signals strategic interest in {industry}",
        "{co} takes {pct}% stake in {co2} worth ${amount}M",
        "{co2} CEO confirms {co} has made strategic investment in company",
        "{co} 13F filing reveals new {pct}% position in {co2}",
        "{co2} gains after {co} discloses significant new equity stake",
    ]
    industry_map = ["fintech", "healthcare AI", "autonomous vehicles",
                    "cloud computing", "cybersecurity", "logistics tech",
                    "clean energy", "biotech", "semiconductor design"]
    cos_list = SP500.copy()
    rows = []
    for _ in range(n):
        cos_pair = random.sample(cos_list, 2)
        co, co2 = cos_pair[0], cos_pair[1]
        t = pick(templates).format(
            co=co, co2=co2,
            pct=pick([3, 5, 7, 8, 10, 12, 15, 20]),
            amount=pick([100, 200, 300, 500, 750, 1000, 1500, 2000]),
            industry=pick(industry_map),
        )
        rows.append({"title": t, "company_name": co, "label": "relevant"})
    return rows


def gen_h13_competitive_underperformance(n=75):
    """Company underperforms vs competitor — relevant (material business metric)."""
    templates = [
        "{co} global sales trail {competitor} for {period}",
        "{co} cedes market share to {competitor} in {segment} for first time",
        "{co} falls behind {competitor} in {metric} — analysts cut targets",
        "{co} loses {segment} lead to {competitor} amid pricing pressure",
        "{co} market share slips as {competitor} gains ground in {segment}",
        "{co}'s {metric} lags {competitor} by widest margin in {years} years",
        "{co} now second to {competitor} in {segment}, data shows",
        "{competitor} overtakes {co} in {metric} as gap widens",
        "{co} unit sales miss as {competitor} captures {pct}% of market",
        "{co} revenue growth trails {competitor} for {quarters} straight quarters",
    ]
    segments = ["electric vehicles", "cloud services", "smartphone sales",
                "streaming subscriptions", "digital advertising", "payments",
                "semiconductor revenue", "enterprise software", "consumer electronics"]
    metrics = ["revenue", "unit sales", "market share", "subscriber growth",
               "profit margins", "delivery volumes", "active users"]
    rows = []
    for _ in range(n):
        co = pick(SP500)
        competitor = pick(COMPETITORS)
        t = pick(templates).format(
            co=co, competitor=competitor,
            segment=pick(segments), metric=pick(metrics),
            period=pick(["the third straight quarter", "fiscal 2024",
                         "the first time in five years", "Q3"]),
            years=pick([3, 5, 7, 10]),
            pct=pick([30, 35, 40, 45, 50, 55]),
            quarters=pick([2, 3, 4, 5, 6]),
        )
        rows.append({"title": t, "company_name": co, "label": "relevant"})
    return rows


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating targeted synthetic data based on FP/FN error analysis...")

    irr_rows = []
    irr_rows += gen_g11_speculative(100)
    irr_rows += gen_g12_analyst_preview(100)
    irr_rows += gen_g13_quantified_philanthropy(100)
    irr_rows += gen_g14_regional_micro_launch(100)
    irr_rows += gen_g15_midlevel_appointment(100)
    irr_rows += gen_g16_local_economic(100)
    irr_rows += gen_g17_company_as_comparator(100)
    irr_rows += gen_g18_small_lawsuit(100)
    irr_rows += gen_g19_visa_contamination(100)
    irr_rows += gen_g20_industry_disruption_catalyst(100)
    print(f"  Irrelevant (G11-G20): {len(irr_rows)}")

    rel_rows = []
    rel_rows += gen_h7_layoffs_employee_pov(75)
    rel_rows += gen_h8_safety_admission(75)
    rel_rows += gen_h9_informal_earnings(75)
    rel_rows += gen_h10_succession_departure(75)
    rel_rows += gen_h11_new_financial_product(75)
    rel_rows += gen_h12_cross_company_investment(75)
    rel_rows += gen_h13_competitive_underperformance(75)
    print(f"  Relevant (H7-H13): {len(rel_rows)}")

    targeted = pd.DataFrame(irr_rows + rel_rows)
    targeted["source_type"] = "synthetic"
    targeted = targeted.sample(frac=1, random_state=42).reset_index(drop=True)
    targeted.to_csv("targeted_synthetic.csv", index=False)
    print(f"\ntargeted_synthetic.csv: {len(targeted)} rows "
          f"({(targeted.label=='irrelevant').sum()} irr / {(targeted.label=='relevant').sum()} rel)")

    # Leakage check against test.csv
    try:
        test = pd.read_csv("test.csv")
        test_titles = set(test["title"].str.strip().str.lower())
        before = len(targeted)
        targeted = targeted[~targeted["title"].str.strip().str.lower().isin(test_titles)]
        print(f"Leakage check: {before} -> {len(targeted)} ({before - len(targeted)} removed)")
    except FileNotFoundError:
        print("test.csv not found — skipping leakage check")

    # Combine with existing training set
    try:
        existing = pd.read_csv("combined_training_set.csv")
        combined = pd.concat([existing, targeted], ignore_index=True)
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        combined.to_csv("combined_training_set_v2.csv", index=False)
        print(f"\ncombined_training_set_v2.csv: {len(combined)} rows")
        print(combined.groupby(["source_type", "label"]).size().to_string())
        irr_total = (combined.label == "irrelevant").sum()
        rel_total = (combined.label == "relevant").sum()
        print(f"\nOverall ratio (irr:rel): {irr_total/rel_total:.2f}:1")
    except FileNotFoundError:
        print("combined_training_set.csv not found — targeted_synthetic.csv only")


if __name__ == "__main__":
    main()
