"""
Synthetic training data generator for the relevance classifier.
Target: 4,250 synthetic rows + 750 gold = 5,000 combined.

Irrelevant (2,800 synthetic):
  Cat A  Discrimination / harassment lawsuits          300
  Cat B  Analyst commentary / upgrades-downgrades      300
  Cat C  Executive social / political commentary        300
  Cat D  Minor product / app / feature news            300
  Cat E  Individual employee / customer incidents       300
  Cat F  Social activism / PR / sponsorship            300
  Cat G  Hard borderline cases (8 sub-types)         1,000
         G1 Small fines < $50M
         G2 Unconfirmed rumours
         G3 Awards / recognition
         G4 Conference speeches / appearances
         G5 Industry / market-share reports (3rd party)
         G6 Consumer deal guides / shopping lists
         G7 Stock price movement without corporate catalyst
         G8 Company name in non-business context
         G9 Former executive joins another company
         G10 Analyst note on rival deal that mentions company in passing

Relevant (1,450 synthetic):
  Standard material events (sector-matched)          1,000
  Hard borderline relevant (unusual phrasing)          450
         H1 Unusual earnings phrasing
         H2 Product safety / recalls with material cost
         H3 Multi-year strategic plans
         H4 Regulatory actions phrased passively
         H5 M&A with informal / indirect phrasing
         H6 Guidance changes / profit warnings
"""
import csv, random
import pandas as pd
random.seed(42)

# ── Company universe ──────────────────────────────────────────────────────
SP500 = [
    "Apple","Microsoft","Amazon","Alphabet","Meta","Nvidia","Netflix","Salesforce","Oracle","IBM","Accenture","ServiceNow",
    "JPMorgan Chase","Goldman Sachs","Morgan Stanley","Bank of America","Wells Fargo","Citigroup","American Express","Capital One","Visa","Mastercard","BlackRock","Charles Schwab","Berkshire Hathaway",
    "Johnson & Johnson","Pfizer","Eli Lilly","AbbVie","Merck","CVS Health","UnitedHealth",
    "ExxonMobil","Chevron","ConocoPhillips","Halliburton",
    "AT&T","Verizon","Comcast","T-Mobile",
    "Walt Disney","Warner Bros. Discovery","Paramount Global",
    "Walmart","Target","Costco","Home Depot","Lowe's","Dollar General",
    "McDonald's","Starbucks","Yum Brands","Chipotle","Coca-Cola","PepsiCo","Mondelez","Kraft Heinz","Tyson Foods",
    "Nike","Under Armour","Ralph Lauren",
    "FedEx","UPS","CSX","Norfolk Southern",
    "General Motors","Ford","Stellantis","Tesla",
    "Boeing","Lockheed Martin","Raytheon","Northrop Grumman",
    "General Electric","Honeywell","3M","Emerson Electric","Caterpillar","Deere & Company",
    "Procter & Gamble","Colgate-Palmolive","Kimberly-Clark",
    "Uber","Airbnb","Booking Holdings",
    "Progressive","Allstate","Travelers","Aflac",
    "Waste Management","Republic Services","DuPont","Dow",
]

ANALYST_FIRMS = ["JPMorgan","Goldman Sachs","Morgan Stanley","Wells Fargo","Bank of America","Citi","Barclays","UBS","Deutsche Bank","RBC Capital Markets","Piper Sandler","Oppenheimer","Baird","Evercore ISI","Jefferies","Stifel","Mizuho","TD Cowen","Bernstein","Needham"]
ROLES = ["warehouse worker","sales representative","financial advisor","branch teller","store manager","delivery driver","software engineer","customer service representative","marketing manager","factory worker","retail associate","human resources manager","district manager","logistics coordinator"]
STATES = ["California","New York","Texas","Illinois","Florida","Washington","Georgia","Ohio","Pennsylvania","Massachusetts","Michigan","New Jersey"]
CITIES = ["Atlanta","Chicago","Houston","Los Angeles","Miami","New York","Phoenix","Seattle","Dallas","Detroit","Boston","Denver","Philadelphia","Minneapolis","Charlotte"]

# ═══════════════════════════════════════════════════════════════════════════
# IRRELEVANT GENERATORS
# ═══════════════════════════════════════════════════════════════════════════

# ── A: Discrimination / harassment (300) ─────────────────────────────────
disc_types = ["age discrimination","gender discrimination","racial discrimination","pregnancy discrimination","disability discrimination","pay discrimination","sexual harassment","racial harassment","religious discrimination","national origin discrimination"]
plaintiffs  = ["former employee","group of former employees","former manager","group of warehouse workers","former executive","group of loan officers","former sales representative","group of retail workers","former engineer","group of delivery drivers","former financial advisor"]
areas       = ["hiring practices","promotion decisions","pay practices","termination policies","scheduling practices","performance reviews","workplace culture"]
cat_a_t     = [
    "{co} faces {disc} lawsuit filed by {plaintiff}",
    "{co} sued by {plaintiff} alleging {disc} in {area}",
    "Former {co} {role} files {disc} complaint with EEOC",
    "{co} faces class-action lawsuit over alleged {disc} among {role}s",
    "{co} settles {disc} complaint filed by {plaintiff} for undisclosed amount",
    "EEOC opens investigation into {co} over alleged {disc} in {area}",
    "{co} faces new {disc} lawsuit from {plaintiff} in {state} court",
    "Group of former {co} {role}s alleges {disc} in federal class-action",
    "{co} denies wrongdoing in {disc} complaint filed by {plaintiff}",
]

def gen_a(n):
    rows, seen = [], set()
    attempts = 0
    while len(rows) < n and attempts < n * 25:
        attempts += 1
        co = random.choice(SP500)
        t  = random.choice(cat_a_t)
        title = t.format(co=co, disc=random.choice(disc_types), plaintiff=random.choice(plaintiffs),
                         area=random.choice(areas), role=random.choice(ROLES), state=random.choice(STATES))
        if title not in seen:
            seen.add(title); rows.append((title, co, "irrelevant"))
    return rows

# ── B: Analyst commentary (300) ──────────────────────────────────────────
ratings = ["overweight","outperform","buy","neutral","equal weight","hold","underperform","sell"]
up_reasons = ["cites strong earnings visibility","sees multiple expansion ahead","calls valuation compelling after selloff","sees margin recovery underway","flags improving demand trends","cites market share gains","says near-term headwinds are priced in","sees accelerating revenue growth","calls recent dip a buying opportunity","points to improving competitive position","raises estimates on guidance beat"]
dn_reasons = ["cites slowing demand environment","flags margin pressure ahead","says valuation stretched after recent rally","lowers estimates on weak guidance","flags rising competitive pressure","cites macro headwinds","says near-term catalysts are limited","flags execution risks","lowers price target on earnings miss","sees limited upside from current levels","warns of elevated inventory","flags deteriorating pricing power"]
actions_up = ["upgrades","raises","lifts","boosts"]
actions_dn = ["downgrades","cuts","lowers","trims"]

def gen_b(n):
    rows, seen = [], set()
    attempts = 0
    while len(rows) < n and attempts < n * 25:
        attempts += 1
        co   = random.choice(SP500)
        firm = random.choice(ANALYST_FIRMS)
        if random.random() > 0.5:
            action, reason = random.choice(actions_up), random.choice(up_reasons)
        else:
            action, reason = random.choice(actions_dn), random.choice(dn_reasons)
        title = f"{firm} {action} {co} to {random.choice(ratings)}, {reason}"
        if title not in seen:
            seen.add(title); rows.append((title, co, "irrelevant"))
    return rows

# ── C: Executive social / political commentary (300) ─────────────────────
exec_titles  = ["CEO","CFO","chairman","president","chief executive","co-founder","chief operating officer"]
macro_topics = ["inflation","interest rates","the labor market","immigration policy","AI regulation","climate policy","trade tariffs","geopolitical tensions","consumer spending","the housing market","deficit spending","energy policy","cryptocurrency regulation","supply chain resilience","tax policy","healthcare costs","banking regulation","data privacy law","the federal budget","global trade"]
generic_stmts = ["is the biggest challenge facing businesses today","requires coordinated global action to address effectively","will have long-lasting effects on the broader economy","is creating uncertainty for long-term investment decisions","is something all companies need to take seriously","is moving in the right direction but more work remains","presents both risks and opportunities for the private sector","demands a thoughtful policy response from legislators","will shape business strategy for years to come","is top of mind for executives across every industry","cannot be solved by the private sector alone","is more complex than most policymakers appreciate","is a concern shared by businesses of all sizes","requires urgent attention from elected officials"]

def gen_c(n):
    rows, seen = [], set()
    attempts = 0
    while len(rows) < n and attempts < n * 25:
        attempts += 1
        co    = random.choice(SP500)
        title = f"{co} {random.choice(exec_titles)} says {random.choice(macro_topics)} {random.choice(generic_stmts)}"
        if title not in seen:
            seen.add(title); rows.append((title, co, "irrelevant"))
    return rows

# ── D: Minor product / app / feature (300) ───────────────────────────────
minor_products = ["mobile app","customer app","loyalty app","enterprise dashboard","employee portal","website","checkout flow","digital wallet","online store","streaming platform","cloud console","developer portal","banking app","retail website","customer portal"]
minor_features = ["dark mode","fingerprint login","spending insights tab","push notification settings","improved search filters","new color themes","accessibility improvements","in-app chat support","simplified navigation menu","new onboarding tutorial","enhanced map view","a new rewards tracker","an updated help center","improved load times","a redesigned home screen","a new password manager","location-based alerts","a new tips section","a progress tracker","two-factor authentication prompts"]
minor_actions  = ["adds","launches","rolls out","introduces","updates","refreshes","releases","enables"]

def gen_d(n):
    rows, seen = [], set()
    attempts = 0
    while len(rows) < n and attempts < n * 25:
        attempts += 1
        co    = random.choice(SP500)
        title = f"{co} {random.choice(minor_actions)} {random.choice(minor_features)} to its {random.choice(minor_products)}"
        if title not in seen:
            seen.add(title); rows.append((title, co, "irrelevant"))
    return rows

# ── E: Individual employee / customer incidents (300) ─────────────────────
incident_types = [
    "arrested for alleged theft from delivery route",
    "charged with embezzling from customer accounts",
    "faces personal lawsuit unrelated to company activities",
    "charged with assault following workplace altercation",
    "arrested in connection with unrelated drug charges",
    "suspended following investigation into personal conduct",
    "charged with running fraud scheme at local branch",
    "faces restraining order filed by former coworker",
    "arrested for allegedly cashing fraudulent checks",
    "cited for personal safety violation",
    "charged with harassment following complaint by coworker",
    "arrested after incident caught on security camera",
    "faces criminal charges unrelated to employment",
    "sued by customer over personal conduct during shift",
]

def gen_e(n):
    rows, seen = [], set()
    attempts = 0
    while len(rows) < n and attempts < n * 25:
        attempts += 1
        co    = random.choice(SP500)
        title = f"{co} {random.choice(ROLES)} {random.choice(incident_types)} in {random.choice(CITIES)}"
        if title not in seen:
            seen.add(title); rows.append((title, co, "irrelevant"))
    return rows

# ── F: Social activism / PR / sponsorship (300) ───────────────────────────
act_t = [
    "{co} employees stage walkout over {policy}",
    "{co} workers circulate petition calling for {demand}",
    "{co} drops sponsorship of {event} following social media backlash",
    "{co} faces consumer boycott over {controversy}",
    "{co} ends partnership with {partner} after {reason}",
    "{co} employees protest {policy} at headquarters",
    "{co} faces criticism from advocacy groups over {controversy}",
    "{co} pulls advertising from {platform} following {reason}",
    "{co} workers in {city} rally to demand {demand}",
    "{co} faces social media campaign targeting {controversy}",
    "{co} employees sign open letter opposing {policy}",
]
policies      = ["return-to-office mandate","mandatory overtime policy","termination of diversity programs","performance tracking policy","new attendance tracking system","changes to remote work flexibility","healthcare benefit reductions","employee monitoring rollout","mandatory drug testing policy"]
demands       = ["better pay and benefits","remote work flexibility","reversal of return-to-office policy","improved workplace safety standards","greater transparency in pay practices","end to mandatory arbitration","paid family leave expansion"]
events        = ["major sporting event","annual awards show","college football sponsorship","arts festival","professional golf tournament","esports tournament","marathon sponsorship","music festival"]
partners      = ["celebrity spokesperson","social media influencer","sports personality","brand ambassador","podcast host"]
reasons_act   = ["controversial comments online","personal scandal","content moderation controversy","employee pressure","community backlash","offensive social media post"]
controversies = ["overseas manufacturing practices","ingredient sourcing policies","executive compensation disclosures","environmental claims on packaging","data privacy practices","price increase practices","marketing to minors allegations"]
platforms     = ["social media platform","streaming service","news outlet","podcast network","video sharing platform"]

def gen_f(n):
    rows, seen = [], set()
    attempts = 0
    while len(rows) < n and attempts < n * 25:
        attempts += 1
        co    = random.choice(SP500)
        t     = random.choice(act_t)
        title = t.format(co=co, policy=random.choice(policies), demand=random.choice(demands),
                         event=random.choice(events), partner=random.choice(partners),
                         reason=random.choice(reasons_act), controversy=random.choice(controversies),
                         platform=random.choice(platforms), city=random.choice(CITIES))
        if title not in seen:
            seen.add(title); rows.append((title, co, "irrelevant"))
    return rows

# ── G: Hard borderline irrelevant (1,000 total across 10 sub-types) ────────

# G1: Small fines < $50M — looks regulatory but isn't material at S&P 500 scale
small_fines   = ["$2 million","$5 million","$8 million","$10 million","$12 million","$15 million","$18 million","$20 million","$25 million","$30 million","$35 million","$40 million","$45 million"]
minor_viols   = ["a disclosure timing issue","a minor data handling violation","an advertising standards complaint","a customer privacy complaint","a minor labor board finding","a product labeling issue","an environmental reporting error","a minor fair lending violation","an outdated record-keeping practice","a minor consumer notification requirement"]
small_reg     = ["SEC","FTC","CFPB","state attorneys general","EEOC","EPA","local labor board"]
g1_t = [
    "{co} fined {fine} by {reg} over {viol}",
    "{co} pays {fine} to settle {viol} complaint with {reg}",
    "{co} agrees to {fine} penalty over {viol}",
    "{co} reaches {fine} agreement with {reg} to resolve {viol}",
    "{co} to pay {fine} in connection with {viol} investigation",
]

# G2: Unconfirmed rumours / "sources say"
rumour_actions = ["exploring a merger","considering selling its media division","in early discussions about a joint venture","weighing a potential spinoff","considering a headquarters relocation","evaluating strategic alternatives","exploring a buyout of a rival","mulling a share buyback","looking at entering the {market2} market","considering going private"]
markets2       = ["healthcare","logistics","financial services","streaming","electric vehicle","defense","real estate","retail banking"]
g2_t = [
    "Report: {co} {action} — company declines to comment",
    "Sources: {co} {action} — no deal imminent, sources say",
    "{co} said to be {action} in early-stage discussions",
    "Rumour: Could {co} be {action}? Analysts are watching",
    "Bloomberg: {co} {action}, though talks are preliminary",
]

# G3: Awards / rankings / recognition
award_types = ["Fortune's most admired companies list","LinkedIn's top companies to work for","Forbes' list of best employers","Time's 100 most influential companies","Glassdoor's top-rated employers","Barron's most sustainable companies","Fast Company's most innovative companies"]
award_years  = ["for the third consecutive year","for the fifth year running","for the first time","for the second straight year"]
g3_t = [
    "{co} named to {award} {year}",
    "{co} earns top score on {award}",
    "{co} CEO named to {award}",
    "{co} ranks among {award} {year}",
    "{co} recognized on {award} for workplace culture",
]

# G4: Conference appearances / speaking slots
conferences  = ["the World Economic Forum in Davos","the Allen & Company Sun Valley conference","the Goldman Sachs Communacopia conference","the JPMorgan Healthcare Conference","a Senate Commerce Committee hearing","the Milken Institute Global Conference","the Aspen Ideas Festival","the MIT Sloan Management Conference"]
conf_topics  = ["the future of AI","climate change and business","the global supply chain","workforce transformation","the digital economy","financial regulation","energy transition","innovation and growth"]
g4_t = [
    "{co} CEO to deliver keynote at {conf}",
    "{co} executive to speak at {conf} on {topic}",
    "{co} to present at {conf} next month",
    "{co} CEO joins panel discussion at {conf} on {topic}",
    "{co} CFO scheduled to appear at {conf} investor day",
]

# G5: Third-party industry / market-share reports
markets_g5  = ["cloud infrastructure","digital advertising","e-commerce","streaming services","electric vehicles","ride-sharing","smartphone sales","enterprise software","online payments","search advertising","semiconductor manufacturing","pharmaceutical distribution"]
pcts_g5     = ["18%","22%","27%","31%","34%","38%","41%","44%","52%","58%","63%"]
firms_g5    = ["Gartner","IDC","Forrester","Nielsen","eMarketer","S&P Global Market Intelligence","Bernstein Research","Morgan Stanley Research","Goldman Sachs Research"]
g5_t = [
    "Report: {co} holds {pct} market share in {market} per {firm}",
    "{firm} estimates {co} controls {pct} of the {market} market",
    "{firm} ranks {co} as the leading provider in {market}",
    "New {firm} data shows {co} gaining share in {market}",
    "{co} cited as top player in {market} by {firm} industry survey",
]

# G6: Consumer deal guides / shopping content (non-news)
products_g6  = ["laptops","smartphones","headphones","televisions","streaming plans","credit cards","airline miles cards","sneakers","running shoes","home appliances","food delivery","software subscriptions"]
seasons_g6   = ["Black Friday","Cyber Monday","Prime Day","the holiday season","back-to-school season","Valentine's Day","Mother's Day"]
g6_t = [
    "Best {co} deals this {season}: our top picks",
    "Is {co}'s latest {product} worth buying? A consumer review",
    "{co} vs competitors: which {product} is right for you",
    "{season}: the best {co} {product} deals you can get right now",
    "Save big on {co} {product} this {season} — here is how",
    "Consumer guide: {co} {product} ranked and reviewed",
]

# G7: Stock price movements with no corporate catalyst
price_moves  = ["falls to 52-week low","rises to 52-week high","slips","edges higher","extends its losing streak","pares earlier losses","underperforms the sector","outperforms the broader market","gives back recent gains","pulls back from record highs"]
market_moves = ["amid broader market selloff","as tech sector sees profit-taking","amid risk-off sentiment","as investors rotate into defensives","as the S&P 500 drops","following weaker-than-expected jobs data","amid rising interest rate fears","on no company-specific news","as investors take profits across sectors","following broader macro concerns"]
g7_t = [
    "{co} stock {move} {cause}",
    "{co} shares {move} {cause}",
    "{co} shares trade lower {cause} — no new company news",
    "Why is {co} stock {move} today? Here is what we know",
]

# G8: Company name in non-business context (stadium, geography, visa the word)
non_biz_contexts = [
    "{co} Field hosts record crowd for playoff game",
    "Concert at {co} Arena draws thousands despite rain",
    "Protests outside {co} headquarters over unrelated political issue",
    "Tourist spots near {co} campus rank among best in the area",
    "Local school renamed after {co} founder despite community debate",
    "New documentary examines working conditions at {co} suppliers overseas",
    "Comedian references {co} in viral stand-up set, clip goes viral",
    "City council debates naming rights deal for stadium near {co} offices",
]

# G9: Former executive joins another company (company not the actor)
exec_roles   = ["former CEO","former CFO","former chief technology officer","former chief operating officer","former president","former general counsel","former chief marketing officer","former chief revenue officer"]
new_cos      = ["a rival startup","a private equity firm","a venture capital fund","a fintech startup","a healthcare company","a government advisory role","a non-profit board","a hedge fund","a consulting firm"]
g9_t = [
    "{co} {role} joins {new_co} as its new chief executive",
    "Former {co} {role} named to lead {new_co}",
    "{co} {role} departs to become partner at {new_co}",
    "Former {co} {role} appointed to advisory board of {new_co}",
    "{co} {role} leaves to join {new_co} in senior role",
]

# G10: Analyst note on rival deal that merely mentions this company in passing
rival_cos   = ["Amazon","Apple","Microsoft","Google","Meta","JPMorgan Chase","Goldman Sachs","Bank of America","Walmart","Target","Costco","UPS","FedEx","Nike","McDonald's","Starbucks"]
deal_types_g10 = ["acquisition","merger","partnership","joint venture","divestiture","restructuring"]
mention_types = ["as a potential competitor in the space","as a company that could benefit from the deal","as a likely target in a broader sector consolidation","as a potential bidder that chose not to participate","as a company that may need to respond competitively","among the companies most affected by the deal"]
g10_t = [
    "{rival} {deal} seen as competitive threat to {co}, analyst says",
    "{rival}'s {deal} deal raises questions about {co}'s position, per {firm}",
    "How {rival}'s {deal} could reshape the competitive landscape for {co}",
    "{firm} flags {co} {mention} in note on {rival} {deal}",
    "Analyst says {rival} {deal} positions the company ahead of {co}",
]

def gen_g(n):
    """Generate hard borderline irrelevant examples across 10 sub-types."""
    per_sub = n // 10
    extra   = n - per_sub * 10
    counts  = [per_sub] * 10
    for i in range(extra):
        counts[i] += 1

    rows, seen = [], set()

    def _add(templates, co, **kw):
        nonlocal rows, seen
        t     = random.choice(templates)
        title = t.format(co=co, **kw)
        if title not in seen:
            seen.add(title)
            rows.append((title, co, "irrelevant"))
            return True
        return False

    def _fill(gen_fn, target_n):
        added   = 0
        attempts = 0
        while added < target_n and attempts < target_n * 30:
            attempts += 1
            if gen_fn():
                added += 1

    # G1
    def _g1():
        co = random.choice(SP500)
        return _add(g1_t, co, fine=random.choice(small_fines),
                    reg=random.choice(small_reg), viol=random.choice(minor_viols))
    _fill(_g1, counts[0])

    # G2
    def _g2():
        co     = random.choice(SP500)
        action = random.choice(rumour_actions).format(market2=random.choice(markets2))
        return _add(g2_t, co, action=action)
    _fill(_g2, counts[1])

    # G3
    def _g3():
        co = random.choice(SP500)
        return _add(g3_t, co, award=random.choice(award_types), year=random.choice(award_years))
    _fill(_g3, counts[2])

    # G4
    def _g4():
        co = random.choice(SP500)
        return _add(g4_t, co, conf=random.choice(conferences), topic=random.choice(conf_topics))
    _fill(_g4, counts[3])

    # G5
    def _g5():
        co = random.choice(SP500)
        return _add(g5_t, co, pct=random.choice(pcts_g5),
                    market=random.choice(markets_g5), firm=random.choice(firms_g5))
    _fill(_g5, counts[4])

    # G6
    def _g6():
        co = random.choice(SP500)
        return _add(g6_t, co, product=random.choice(products_g6), season=random.choice(seasons_g6))
    _fill(_g6, counts[5])

    # G7
    def _g7():
        co = random.choice(SP500)
        return _add(g7_t, co, move=random.choice(price_moves), cause=random.choice(market_moves))
    _fill(_g7, counts[6])

    # G8: company name in non-business context — use fixed templates, sub {co}
    def _g8():
        co    = random.choice(SP500)
        title = random.choice(non_biz_contexts).format(co=co)
        if title not in seen:
            seen.add(title); rows.append((title, co, "irrelevant")); return True
        return False
    _fill(_g8, counts[7])

    # G9
    def _g9():
        co = random.choice(SP500)
        return _add(g9_t, co, role=random.choice(exec_roles), new_co=random.choice(new_cos))
    _fill(_g9, counts[8])

    # G10
    def _g10():
        co     = random.choice(SP500)
        rival  = random.choice([r for r in rival_cos if r != co])
        return _add(g10_t, co, rival=rival,
                    deal=random.choice(deal_types_g10),
                    firm=random.choice(ANALYST_FIRMS),
                    mention=random.choice(mention_types))
    _fill(_g10, counts[9])

    return rows


# ═══════════════════════════════════════════════════════════════════════════
# RELEVANT GENERATORS  — sector-aware profiles
# ═══════════════════════════════════════════════════════════════════════════

PROFILES = {
    "tech": {
        "cos": {"Apple","Microsoft","Amazon","Alphabet","Meta","Nvidia","Netflix","Salesforce","Oracle","IBM","Accenture","ServiceNow"},
        "drivers":   ["strong cloud revenue growth","AI-driven demand acceleration","strong advertising revenue","record data center orders","strong subscription growth","robust enterprise software renewals"],
        "headwinds": ["slowing cloud growth","weaker advertising market","slower enterprise spending","elevated chip inventory","increased competition from cloud rivals","softer consumer device demand"],
        "targets":   ["AI startup","cloud infrastructure company","cybersecurity startup","data analytics firm","developer tools company","enterprise software company"],
        "units":     ["consumer hardware division","streaming unit","enterprise software arm","advertising business","cloud division"],
        "ab": ["$1.8","$2.5","$3.4","$5.1","$8.3","$14.0","$18.5","$22.0","$50.0","$90.0"],
        "am": ["$180","$310","$420","$650","$750","$900"],
        "contracts": ["cloud computing services","AI infrastructure","cybersecurity services","IT modernization","enterprise software licensing"],
        "margins":   ["52.3%","61.2%","67.4%","72.9%","76.4%","44.7%"],
        "govt": True,
        "extra": [
            "{co} cloud revenue grows {pct} year-over-year, {bm}",
            "{co} data center revenue surges {pct} on record AI chip demand",
            "{co} reports {pct} jump in advertising revenue as digital ad market recovers",
        ],
    },
    "finance": {
        "cos": {"JPMorgan Chase","Goldman Sachs","Morgan Stanley","Bank of America","Wells Fargo","Citigroup","American Express","Capital One","Visa","Mastercard","BlackRock","Charles Schwab","Berkshire Hathaway"},
        "drivers":   ["strong trading revenue","record investment banking fees","strong card spending volumes","wealth management growth","higher net interest income","record assets under management"],
        "headwinds": ["rising loan loss provisions","weaker trading revenue","slower deal activity","elevated credit card delinquencies","commercial real estate exposure","net interest margin compression"],
        "targets":   ["asset management firm","fintech startup","regional bank","brokerage firm","payments company","insurance company"],
        "units":     ["consumer banking arm","retail brokerage unit","insurance subsidiary","international operations","wealth management division"],
        "ab": ["$1.2","$1.8","$2.5","$3.4","$5.1","$8.3","$14.0","$18.5"],
        "am": ["$145","$215","$250","$380","$500","$750"],
        "contracts": ["financial advisory services","asset management services","corporate banking services"],
        "margins":   ["28.4%","31.2%","34.7%","37.1%","39.8%","42.3%"],
        "govt": False,
        "extra": [
            "{co} raises loan loss provisions to {amt_b} billion citing rising consumer delinquencies",
            "{co} investment banking revenue surges {pct} as M&A market recovers",
            "{co} wealth management division surpasses {amt_b} trillion in client assets",
            "{co} net interest income falls {pct} as rate environment pressures margins",
        ],
    },
    "pharma": {
        "cos": {"Johnson & Johnson","Pfizer","Eli Lilly","AbbVie","Merck","CVS Health","UnitedHealth"},
        "drivers":   ["strong prescription volume growth","blockbuster drug demand","record insurance enrollment","GLP-1 drug demand exceeding expectations","new drug approvals driving revenue"],
        "headwinds": ["patent cliff exposure","biosimilar competition","drug pricing pressure","weaker-than-expected prescription volumes","higher medical costs"],
        "targets":   ["biotech startup","specialty pharma company","clinical-stage drug developer","medical device company","pharmacy benefit manager"],
        "units":     ["consumer health division","medical devices unit","specialty pharma arm","biosimilars division","insurance unit"],
        "ab": ["$1.2","$1.8","$2.5","$3.4","$5.1","$8.3","$14.0","$22.0"],
        "am": ["$65","$110","$250","$380","$650"],
        "contracts": ["drug supply agreement","pharmacy benefit management","healthcare administration"],
        "margins":   ["18.4%","21.2%","24.7%","27.1%","29.8%","32.3%","35.6%"],
        "govt": False,
        "extra": [
            "{co} receives FDA approval for new {target} treatment, shares rise",
            "{co} reports positive Phase 3 trial results for key pipeline drug",
            "{co} raises full-year earnings guidance following stronger-than-expected drug sales",
        ],
    },
    "energy": {
        "cos": {"ExxonMobil","Chevron","ConocoPhillips","Halliburton"},
        "drivers":   ["higher oil prices","strong refining margins","record upstream production","strong natural gas demand","improved drilling efficiency"],
        "headwinds": ["lower oil prices","weak refining margins","declining production volumes","higher drilling costs","reduced capital expenditure from customers"],
        "targets":   ["upstream oil producer","natural gas company","shale oil operator","energy services company","offshore drilling company"],
        "units":     ["downstream refining unit","petrochemicals division","midstream operations","international upstream assets"],
        "ab": ["$1.2","$1.8","$2.5","$3.4","$5.1","$8.3","$11.2","$14.0"],
        "am": ["$110","$180","$250","$380","$500"],
        "contracts": ["offshore drilling services","energy infrastructure","oilfield services"],
        "margins":   ["18.4%","22.7%","27.1%","31.8%","35.3%","38.6%"],
        "govt": False,
        "extra": [
            "{co} boosts quarterly dividend {pct} as higher oil prices fuel record free cash flow",
            "{co} raises full-year production guidance after stronger-than-expected output",
        ],
    },
    "telecom": {
        "cos": {"AT&T","Verizon","Comcast","T-Mobile"},
        "drivers":   ["strong wireless postpaid net adds","record broadband subscriber growth","strong ARPU growth","reduced churn from improved network quality","strong bundled service adoption"],
        "headwinds": ["slowing postpaid phone net adds","elevated subscriber churn","rising network infrastructure costs","cord-cutting accelerating","pricing pressure from competitors"],
        "targets":   ["regional broadband provider","wireless spectrum holder","cable operator","fiber network company"],
        "units":     ["wireline business","satellite unit","enterprise division","media assets"],
        "ab": ["$1.2","$1.8","$2.5","$3.4","$5.1","$8.3"],
        "am": ["$110","$180","$250","$380","$500"],
        "contracts": ["network infrastructure services","enterprise connectivity","spectrum licensing"],
        "margins":   ["28.4%","31.2%","34.7%","37.1%","39.8%"],
        "govt": False,
        "extra": [
            "{co} adds record number of postpaid phone subscribers in {q}, {bm}",
            "{co} broadband net adds accelerate in {q} on fiber expansion",
        ],
    },
    "media": {
        "cos": {"Walt Disney","Warner Bros. Discovery","Paramount Global","Netflix"},
        "drivers":   ["strong streaming subscriber growth","theme park record attendance","strong box office performance","improved advertising revenue","successful content slate driving subscriptions"],
        "headwinds": ["streaming subscriber losses","declining linear TV ratings","rising content costs","theme park attendance softness","weaker advertising market"],
        "targets":   ["streaming platform","content studio","sports rights holder","production company"],
        "units":     ["linear television network","streaming unit","theme park division","film studio"],
        "ab": ["$1.2","$1.8","$2.1","$3.0","$5.1","$8.3"],
        "am": ["$110","$180","$250","$380","$500"],
        "contracts": ["content licensing deal","sports broadcasting rights","streaming distribution agreement"],
        "margins":   ["18.4%","21.2%","24.7%","27.1%","29.8%"],
        "govt": False,
        "extra": [
            "{co} streaming service adds record subscribers in {q}, {bm}",
            "{co} raises streaming subscription price for US subscribers",
        ],
    },
    "retail": {
        "cos": {"Walmart","Target","Costco","Home Depot","Lowe's","Dollar General"},
        "drivers":   ["strong comparable store sales growth","record membership fee revenue","strong online sales penetration","improving inventory management","strong seasonal demand"],
        "headwinds": ["declining comparable store sales","elevated inventory","weakening consumer spending","rising shrink losses","wage cost inflation"],
        "targets":   ["e-commerce platform","specialty retailer","delivery service","supply chain company"],
        "units":     ["international division","e-commerce unit","pharmacy chain","financial services arm"],
        "ab": ["$1.2","$1.8","$2.5","$3.4","$5.1","$8.3"],
        "am": ["$110","$180","$250","$380","$500"],
        "contracts": ["supply chain logistics","retail services","distribution services"],
        "margins":   ["24.4%","27.2%","29.7%","31.1%","33.8%","36.2%"],
        "govt": False,
        "extra": [
            "{co} comparable store sales {ec} {pct} in {q}, {bm}",
            "{co} raises full-year comparable sales guidance on stronger consumer demand",
        ],
    },
    "food": {
        "cos": {"McDonald's","Starbucks","Yum Brands","Chipotle","Coca-Cola","PepsiCo","Mondelez","Kraft Heinz","Tyson Foods"},
        "drivers":   ["strong same-store sales growth","menu price increases driving revenue","strong international comparable sales","record digital order mix","strong organic volume growth"],
        "headwinds": ["same-store sales decline","traffic softness amid consumer trade-down","elevated commodity costs","weaker-than-expected menu pricing","slowing international sales"],
        "targets":   ["restaurant chain","food brand","beverage company","plant-based food startup"],
        "units":     ["international division","company-owned restaurant portfolio","beverage unit","snack foods division"],
        "ab": ["$0.8","$1.2","$1.8","$2.1","$3.0","$5.1"],
        "am": ["$65","$110","$180","$250","$380"],
        "contracts": ["food distribution agreement","franchise system services","supply agreement"],
        "margins":   ["18.4%","21.2%","24.7%","27.1%","29.8%","32.3%"],
        "govt": False,
        "extra": [
            "{co} global same-store sales rise {pct} in {q}, {bm}",
            "{co} same-store sales decline {pct} as consumers trade down amid inflation",
            "{co} raises menu prices {pct} to offset rising ingredient costs",
        ],
    },
    "auto": {
        "cos": {"General Motors","Ford","Stellantis","Tesla"},
        "drivers":   ["strong vehicle delivery growth","record truck sales","improving EV production efficiency","better-than-expected vehicle pricing","strong fleet sales"],
        "headwinds": ["elevated warranty costs","slowing EV demand","production disruptions","weaker-than-expected deliveries","rising recall costs"],
        "targets":   ["EV battery maker","autonomous driving startup","auto software company","charging network operator"],
        "units":     ["EV unit","international operations","financial services arm","commercial vehicle division"],
        "ab": ["$0.8","$1.2","$1.8","$2.1","$3.0","$5.1"],
        "am": ["$110","$180","$250","$380","$500"],
        "contracts": ["fleet supply agreement","battery supply deal","autonomous vehicle development contract"],
        "margins":   ["8.4%","11.2%","14.7%","17.1%","19.8%","22.3%"],
        "govt": False,
        "extra": [
            "{co} delivers record number of vehicles in {q}, {bm}",
            "{co} recalls vehicles over potential safety defect, faces repair costs",
            "{co} raises EV production target by {pct} following strong order intake",
        ],
    },
    "defense": {
        "cos": {"Boeing","Lockheed Martin","Raytheon","Northrop Grumman"},
        "drivers":   ["record defense contract backlog","strong government defense spending","NATO ally orders driving revenue","strong missile systems demand","record hypersonic weapons orders"],
        "headwinds": ["commercial aircraft delivery delays","production cost overruns","supply chain disruptions","program schedule slippages","weaker commercial aviation demand"],
        "targets":   ["defense electronics company","satellite systems maker","missile defense startup","unmanned systems company"],
        "units":     ["commercial aviation division","space systems unit","defense electronics arm","rotary and mission systems division"],
        "ab": ["$1.8","$2.5","$3.4","$5.1","$8.3","$11.2","$14.0"],
        "am": ["$180","$250","$380","$500","$650"],
        "contracts": ["fighter jet maintenance","satellite communications","missile defense systems","unmanned aerial vehicle development","defense modernization"],
        "margins":   ["11.4%","13.2%","15.7%","17.1%","19.8%","21.3%"],
        "govt": True,
        "extra": [
            "{co} wins {amt_b} billion Pentagon contract for {contract_desc}",
            "{co} contract backlog reaches record {amt_b} billion on strong defense orders",
        ],
    },
    "industrial": {
        "cos": {"General Electric","Honeywell","3M","Emerson Electric","Caterpillar","Deere & Company"},
        "drivers":   ["record equipment order backlog","strong aerospace aftermarket demand","strong construction equipment demand","improving industrial production","record agricultural equipment sales"],
        "headwinds": ["declining equipment orders","slowing construction activity","weaker-than-expected agricultural demand","higher raw material costs","supply chain disruptions"],
        "targets":   ["industrial software company","automation startup","specialty materials maker","precision tools company"],
        "units":     ["power division","industrial automation arm","healthcare unit","safety products division"],
        "ab": ["$0.8","$1.2","$1.8","$2.5","$3.4","$5.1","$8.3"],
        "am": ["$110","$180","$250","$380","$500"],
        "contracts": ["industrial automation services","aerospace maintenance contract","infrastructure equipment supply"],
        "margins":   ["18.4%","21.2%","24.7%","27.1%","29.8%","32.3%"],
        "govt": True,
        "extra": ["{co} order backlog reaches record {amt_b} billion on strong industrial demand"],
    },
    "transport": {
        "cos": {"FedEx","UPS","CSX","Norfolk Southern"},
        "drivers":   ["strong package volume growth","record freight revenue per shipment","improved on-time delivery performance","strong e-commerce-driven volume","pricing power in tight capacity market"],
        "headwinds": ["declining package volume","lower revenue per shipment","slowing freight demand","higher fuel costs","labor cost inflation"],
        "targets":   ["last-mile delivery company","freight brokerage","logistics startup","warehouse operator"],
        "units":     ["air freight division","ground network","international logistics arm","supply chain services unit"],
        "ab": ["$0.8","$1.2","$1.8","$2.5","$3.4","$5.1"],
        "am": ["$110","$180","$250","$380","$500"],
        "contracts": ["logistics services","last-mile delivery contract","freight services agreement"],
        "margins":   ["14.4%","17.2%","19.7%","21.1%","23.8%","26.3%"],
        "govt": False,
        "extra": [
            "{co} package volume falls {pct} as freight market downturn deepens",
            "{co} raises per-shipment surcharge citing fuel and labor cost pressures",
        ],
    },
    "consumer_goods": {
        "cos": {"Procter & Gamble","Colgate-Palmolive","Kimberly-Clark","Nike","Under Armour","Ralph Lauren"},
        "drivers":   ["strong organic sales growth","pricing power offsetting volume declines","strong international emerging market sales","easing commodity cost tailwinds","innovation driving market share gains"],
        "headwinds": ["volume declines from price elasticity","private label competition intensifying","elevated commodity costs","weaker emerging market volumes","consumer trading down"],
        "targets":   ["personal care brand","specialty skincare company","household products brand","apparel brand"],
        "units":     ["feminine care brands","baby care division","grooming unit","oral care business","footwear division"],
        "ab": ["$0.8","$1.2","$1.8","$2.5","$3.4","$5.1","$8.3"],
        "am": ["$65","$110","$180","$250","$380","$450"],
        "contracts": ["distribution agreement","supply chain services","licensing agreement"],
        "margins":   ["38.4%","41.2%","44.7%","47.1%","49.8%","52.3%"],
        "govt": False,
        "extra": [
            "{co} organic sales growth of {pct} in {q} {bm} on pricing and volume",
            "{co} gross margin recovers to {margin_pct} as commodity cost headwinds ease",
        ],
    },
    "insurance": {
        "cos": {"Progressive","Allstate","Travelers","Aflac"},
        "drivers":   ["improving combined ratio","strong premium growth","better-than-expected catastrophe losses","strong personal auto pricing","record policy count growth"],
        "headwinds": ["elevated catastrophe losses","rising claims severity","deteriorating combined ratio","higher reinsurance costs","weaker-than-expected premium retention"],
        "targets":   ["specialty insurer","reinsurance company","insurtech startup","life insurance company"],
        "units":     ["personal lines division","commercial insurance arm","life and health unit","reinsurance operations"],
        "ab": ["$0.5","$0.8","$1.2","$1.8","$2.5","$3.4"],
        "am": ["$65","$110","$180","$250","$380"],
        "contracts": ["reinsurance agreement","insurance services","risk management contract"],
        "margins":   ["8.4%","11.2%","14.7%","17.1%","19.8%","22.3%"],
        "govt": False,
        "extra": [
            "{co} combined ratio improves to {margin_pct} as claims costs moderate",
            "{co} catastrophe losses reach {amt_b} billion from severe weather events",
        ],
    },
    "travel": {
        "cos": {"Uber","Airbnb","Booking Holdings"},
        "drivers":   ["record gross bookings","strong travel demand recovery","improving take rate","record nights booked","strong international travel volumes"],
        "headwinds": ["slowing gross booking growth","lower-than-expected take rate","weakening travel demand","rising driver and host supply costs","regulatory headwinds in key markets"],
        "targets":   ["travel technology company","ride-sharing operator","short-term rental platform","travel data company"],
        "units":     ["freight business","experiences division","international operations","financial services arm"],
        "ab": ["$0.8","$1.2","$1.8","$2.5","$3.4","$5.1"],
        "am": ["$110","$180","$250","$380","$500"],
        "contracts": ["delivery services agreement","platform licensing","data services agreement"],
        "margins":   ["14.4%","17.2%","19.7%","21.1%","23.8%"],
        "govt": False,
        "extra": [
            "{co} gross bookings reach record {amt_b} billion in {q}, {bm}",
            "{co} nights booked rise {pct} as international travel demand surges",
        ],
    },
    "other": {
        "cos": set(),
        "drivers":   ["strong volume growth","record revenue","pricing power","improved operational efficiency"],
        "headwinds": ["volume declines","weaker-than-expected pricing","higher operating costs","slowing demand"],
        "targets":   ["industry peer","specialty company","technology startup","service company"],
        "units":     ["international division","specialty unit","core business","non-core assets"],
        "ab": ["$1.2","$1.8","$2.5","$3.4","$5.1","$8.3"],
        "am": ["$65","$110","$180","$250","$380"],
        "contracts": ["services agreement","supply contract","logistics deal"],
        "margins":   ["18.4%","21.2%","24.7%","27.1%","29.8%","32.3%"],
        "govt": False,
        "extra": [],
    },
}

CO_PROFILE = {}
for sector, p in PROFILES.items():
    for co in p["cos"]:
        CO_PROFILE[co] = p

# Shared fill values
QUARTERS      = ["Q1","Q2","Q3","Q4"]
EPS_VALS      = ["$0.43","$0.87","$1.22","$1.87","$2.14","$2.56","$3.10","$3.78","$4.22","$4.91","$5.37","$6.14"]
DEAL_TYPES    = ["all-cash deal","stock-and-cash transaction","all-stock deal"]
BUYERS        = ["private equity firm","strategic acquirer","consortium of investors"]
JOBS          = ["800","1,200","1,600","2,000","2,500","3,000","4,000","5,500","7,000","10,000","12,000","18,000"]
DIVISIONS     = ["corporate functions","investment banking","retail operations","technology division","global operations","consumer unit","R&D division"]
REASONS_LAY   = ["slowing demand","cost reduction program","business reorganization","strategic review","automation initiative"]
EXEC_ACTIONS  = ["steps down after 8 years","announces planned retirement","departs following board review","resigns amid strategic disagreements","is replaced following activist pressure","steps down after record tenure"]
REPLACEMENTS  = ["names internal successor","launches external search","appoints industry veteran","promotes division head to top role"]
DEPARTURES    = ["retirement","resignation","departure"]
SUCCESSORS    = ["internal finance executive","former divisional CFO","Wall Street veteran","industry outsider"]
EXEC_DESCS    = ["industry veteran","longtime board member","divisional president","former rival executive"]
TRIGGERS      = ["board review","activist pressure","CEO transition","strategic shift"]
PCTS          = ["4%","6%","7%","8%","9%","11%","12%","14%","15%","17%","18%","21%","24%","27%","31%","38%","42%","47%"]
DIV_AMTS      = ["$0.45","$0.62","$0.78","$0.85","$0.94","$1.10","$1.25","$1.38"]
REGULATORS    = ["DOJ","SEC","FTC","CFPB","EPA","EU regulators","state attorneys general","CFTC"]
ALLEGATIONS   = ["antitrust violations","misleading investors","data privacy violations","price-fixing conspiracy","mortgage servicing failures","consumer protection violations","market manipulation","environmental violations"]
CREDIT_ACTS   = ["upgraded","downgraded","placed on negative watch","affirmed with stable outlook"]
AGENCIES      = ["S&P","Moody's","Fitch","DBRS"]
CREDIT_REASONS= ["strong capital position","improved cash flow generation","elevated debt levels","deteriorating free cash flow","diversified revenue mix","rising leverage","progress on cost reduction"]
DEBT_PURPOSES = ["refinance maturing debt","fund strategic acquisitions","strengthen balance sheet","finance share buyback program","fund capital expenditures"]
LABOR_ACTIONS = ["authorize strike","unionize","ratify new labor agreement","reject company contract offer"]
LOCATIONS     = ["its largest facility","flagship warehouse","main production plant","key distribution center"]
DEAL_DESCS    = ["new four-year labor agreement","tentative contract deal","landmark new agreement"]
RIVALS        = ["competitor","smaller challenger","domestic rival","overseas challenger"]
EC            = ["expands","contracts","widens","narrows"]
BEAT_MISSES   = ["beats analyst estimates","tops expectations","surpasses consensus","misses analyst estimates","falls short of expectations","comes in below consensus"]

BASE_REL_TEMPLATES = [
    "{co} reports {q} revenue of {amt_b} billion, {bm} on {driver}",
    "{co} posts {q} net income of {amt_b} billion, {bm}",
    "{co} {q} earnings of {eps} per share {bm}",
    "{co} raises full-year guidance after {q} beat on {driver}",
    "{co} cuts full-year guidance citing {headwind}",
    "{co} {q} gross margin {ec} to {margin_pct} on {driver}",
    "{co} to acquire {target} for {amt_b} billion in {deal_type}",
    "{co} acquires {target} for {amt_m} million to bolster {capability}",
    "{co} agrees to sell {unit} for {amt_b} billion to {buyer}",
    "{co} divests {unit} in {amt_b} billion deal as part of strategic review",
    "{co} announces {jobs} job cuts in {division} amid {reason_layoff}",
    "{co} to eliminate {jobs} positions as part of {amt_b} billion cost reduction plan",
    "{co} CEO {exec_action}, board {replacement_action}",
    "{co} CFO announces {departure}, company names {successor} as replacement",
    "{co} names {exec_desc} as new chief executive following {trigger}",
    "{co} announces {amt_b} billion share buyback program",
    "{co} raises quarterly dividend {pct} to {div} per share",
    "{co} board approves new {amt_b} billion share repurchase authorization",
    "{co} agrees to {amt_m} million settlement with {regulator} over {allegation}",
    "{co} faces {amt_b} billion fine from {regulator} over {allegation}",
    "{co} credit rating {credit_action} by {agency} citing {reason_credit}",
    "{co} issues {amt_b} billion in senior notes to {debt_purpose}",
    "{co} workers vote to {labor_action} at {location}",
    "{co} reaches {deal_desc} with union representing {jobs} workers",
    "{co} announces {jobs} corporate layoffs, shares fall on restructuring news",
    "{co} {q} operating income surges {pct} on {driver}",
    "{co} activist investor discloses {amt_b} billion stake pushes for margin improvement",
    "{co} reports {pct} drop in quarterly revenue, cuts annual guidance",
    "{co} acquires {target} in {amt_b} billion deal expanding into {capability}",
    "{co} loses {unit} contract to rival in blow to revenue outlook",
]

# Hard borderline RELEVANT templates (unusual phrasing the model hasn't seen)
HARD_REL_TEMPLATES = [
    # H1: Unusual earnings phrasing
    "{co} tops the Street on both revenue and earnings in {q}",
    "{co} posts better-than-expected {q} results as {driver}",
    "{co} {q} results: beats on top and bottom line, raises outlook",
    "{co} blows past {q} estimates on {driver}",
    "{co} delivers {q} beat, guides higher for the year",
    "{co} {q} miss: revenue and earnings both fall short as {headwind}",
    "{co} turns in a {q} quarter investors had feared, stock drops",
    "{co} {q}: revenue in line, margins disappoint, guidance cut",
    # H2: Product safety / recalls with material cost
    "{co} recalls {jobs} thousand units over safety defect, faces {amt_m} million repair bill",
    "{co} issues voluntary recall of {target} product over potential safety concern",
    "FDA orders {co} to halt production at facility over contamination findings",
    "{co} faces class-action over {target} product defect affecting millions of users",
    "{co} sets aside {amt_b} billion to cover costs of {target} product recall",
    # H3: Multi-year strategic plans announced
    "{co} unveils three-year plan to cut costs by {amt_b} billion",
    "{co} outlines five-year strategy to double revenue in {capability} segment",
    "{co} lays out roadmap to {pct} operating margin by {year}",
    "{co} announces plan to separate into two independent publicly traded companies",
    "{co} to spin off its {unit} as a standalone publicly traded company",
    # H4: Regulatory phrased passively / indirectly
    "{co} ordered by {regulator} to pay {amt_m} million over {allegation}",
    "Judge orders {co} to pay {amt_m} million in damages over {allegation}",
    "{co} loses {allegation} case, ordered to pay {amt_m} million in penalties",
    "{co} hit with {amt_m} million penalty by {regulator} over {allegation}",
    "{co} discloses {regulator} subpoena related to {allegation}",
    "{co} confirms receipt of {regulator} civil investigative demand over {allegation}",
    # H5: M&A informal / indirect phrasing
    "{co} in deal to take over {target} for {amt_b} billion",
    "{co} and {target} confirm merger talks, deal could be worth {amt_b} billion",
    "{co} snaps up {target} for {amt_m} million in bolt-on acquisition",
    "{co} to merge with {target} in {amt_b} billion combination",
    "Sources: {co} close to finalising acquisition of {target} for {amt_b} billion",
    "{co} confirms it received and rejected {amt_b} billion takeover bid",
    # H6: Guidance changes / profit warnings
    "{co} warns {q} results will miss estimates due to {headwind}",
    "{co} issues profit warning, slashes guidance citing {headwind}",
    "{co} suspends quarterly dividend amid cash flow concerns",
    "{co} says full-year revenue will fall short of prior guidance by {pct}",
    "{co} lowers {q} outlook, sends shares sharply lower in after-hours trading",
    "{co} sees {q} revenue of {amt_b} billion, below prior guidance and analyst consensus",
]

HARD_REL_YEARS = ["2025","2026","2027","2028"]

def get_profile(co):
    return CO_PROFILE.get(co, PROFILES["other"])

def fill_rel(tmpl, co, p):
    return (tmpl
        .replace("{co}", co)
        .replace("{q}", random.choice(QUARTERS))
        .replace("{amt_b}", random.choice(p["ab"]))
        .replace("{amt_m}", random.choice(p["am"]))
        .replace("{bm}", random.choice(BEAT_MISSES))
        .replace("{driver}", random.choice(p["drivers"]))
        .replace("{headwind}", random.choice(p["headwinds"]))
        .replace("{eps}", random.choice(EPS_VALS))
        .replace("{target}", random.choice(p["targets"]))
        .replace("{capability}", random.choice(p["targets"]))
        .replace("{deal_type}", random.choice(DEAL_TYPES))
        .replace("{unit}", random.choice(p["units"]))
        .replace("{buyer}", random.choice(BUYERS))
        .replace("{jobs}", random.choice(JOBS))
        .replace("{division}", random.choice(DIVISIONS))
        .replace("{reason_layoff}", random.choice(REASONS_LAY))
        .replace("{exec_action}", random.choice(EXEC_ACTIONS))
        .replace("{replacement_action}", random.choice(REPLACEMENTS))
        .replace("{departure}", random.choice(DEPARTURES))
        .replace("{successor}", random.choice(SUCCESSORS))
        .replace("{exec_desc}", random.choice(EXEC_DESCS))
        .replace("{trigger}", random.choice(TRIGGERS))
        .replace("{pct}", random.choice(PCTS))
        .replace("{div}", random.choice(DIV_AMTS))
        .replace("{regulator}", random.choice(REGULATORS))
        .replace("{allegation}", random.choice(ALLEGATIONS))
        .replace("{credit_action}", random.choice(CREDIT_ACTS))
        .replace("{agency}", random.choice(AGENCIES))
        .replace("{reason_credit}", random.choice(CREDIT_REASONS))
        .replace("{debt_purpose}", random.choice(DEBT_PURPOSES))
        .replace("{labor_action}", random.choice(LABOR_ACTIONS))
        .replace("{location}", random.choice(LOCATIONS))
        .replace("{deal_desc}", random.choice(DEAL_DESCS))
        .replace("{contract_desc}", random.choice(p["contracts"]))
        .replace("{rival}", random.choice(RIVALS))
        .replace("{margin_pct}", random.choice(p["margins"]))
        .replace("{ec}", random.choice(EC))
        .replace("{year}", random.choice(HARD_REL_YEARS))
    )

def gen_rel(n_standard, n_hard):
    rows, seen = [], set()

    # Standard templates
    attempts = 0
    while len(rows) < n_standard and attempts < n_standard * 40:
        attempts += 1
        co   = random.choice(SP500)
        p    = get_profile(co)
        tmpl = random.choice(BASE_REL_TEMPLATES + p["extra"])
        if "Pentagon" in tmpl and not p["govt"]:
            continue
        title = fill_rel(tmpl, co, p)
        if title not in seen:
            seen.add(title); rows.append((title, co, "relevant"))

    # Hard borderline
    attempts = 0
    while len(rows) < n_standard + n_hard and attempts < n_hard * 40:
        attempts += 1
        co   = random.choice(SP500)
        p    = get_profile(co)
        tmpl = random.choice(HARD_REL_TEMPLATES)
        title = fill_rel(tmpl, co, p)
        if title not in seen:
            seen.add(title); rows.append((title, co, "relevant"))

    return rows


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
print("Generating irrelevant categories A–F...")
irr_rows  = gen_a(300) + gen_b(300) + gen_c(300) + gen_d(300) + gen_e(300) + gen_f(300)
print(f"  A–F total: {len(irr_rows)}")

print("Generating irrelevant category G (hard borderline)...")
irr_rows += gen_g(1000)
print(f"  A–G total: {len(irr_rows)}")

print("Generating relevant rows (1,000 standard + 450 hard borderline)...")
rel_rows  = gen_rel(1000, 450)
print(f"  Relevant total: {len(rel_rows)}")

all_rows  = irr_rows + rel_rows
random.shuffle(all_rows)

with open("synthetic_data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["title","company_name","label"])
    writer.writerows(all_rows)

print(f"\nsynthetic_data.csv: {len(all_rows)} rows  ({sum(1 for r in all_rows if r[2]=='irrelevant')} irr / {sum(1 for r in all_rows if r[2]=='relevant')} rel)")

# Build combined
synthetic = pd.read_csv("synthetic_data.csv")
test      = pd.read_csv("test.csv", on_bad_lines="skip")
manual    = pd.read_csv("manual_annotations.csv")

test_titles     = set(test["title"].str.strip().str.lower())
n_before        = len(synthetic)
synthetic_clean = synthetic[~synthetic["title"].str.strip().str.lower().isin(test_titles)].copy()
print(f"Leakage check: {n_before} -> {len(synthetic_clean)} ({n_before - len(synthetic_clean)} removed)")

manual_out              = manual[["title","company_name","label"]].copy()
manual_out["source_type"] = "gold_manual"
synthetic_clean["source_type"] = "synthetic"

combined = pd.concat([manual_out, synthetic_clean[["title","company_name","label","source_type"]]], ignore_index=True)
combined.to_csv("combined_training_set.csv", index=False)

print(f"\ncombined_training_set.csv: {len(combined)} rows")
print(combined.groupby(["source_type","label"]).size().to_string())
print(f"\nOverall ratio (irr:rel): {combined[combined.label=='irrelevant'].shape[0] / combined[combined.label=='relevant'].shape[0]:.2f}:1")
