"""Write gold_a1.csv from manually decided labels for 310 articles."""
import pandas as pd

LABELS = {
    # AT&T
    2965080: 'irrelevant',  # Time Warner/Dodgers carriage — consumer content
    3495052: 'relevant',    # AT&T divests Xandr to Microsoft — divestiture
    3262424: 'irrelevant',  # Verizon won't copy AT&T fake 5G — Verizon is subject
    2926302: 'relevant',    # DirecTV acquisition closes after FCC approval — M&A
    3501067: 'irrelevant',  # Stock movers roundup — multiple companies
    3093081: 'relevant',    # Time Warner deal timing uncertain — M&A update
    3313528: 'relevant',    # AT&T CEO calls out CBS carriage fight — CEO/contract dispute
    3477632: 'relevant',    # GM and AT&T 5G in vehicles — major partnership
    3057938: 'irrelevant',  # DirecTV Now free Roku box promo — consumer deal
    3271677: 'irrelevant',  # Cody Gribble at AT&T Pebble Beach — golf tournament
    # Airbnb
    468951:  'irrelevant',  # Hocus Pocus cottage — entertainment/lifestyle
    357737:  'irrelevant',  # How to prevent ragers — consumer advice
    363655:  'irrelevant',  # Cities wrangling Airbnb rules — regulatory landscape article
    424603:  'relevant',    # Airbnb cloud contract extension — major contract
    211729:  'relevant',    # Airbnb regulatory challenge in Japan — material market regulatory risk
    477820:  'relevant',    # Airbnb reports spending/stay trends — revenue/guidance
    297710:  'irrelevant',  # Airbnb in your neighborhood? — consumer/lifestyle
    295216:  'relevant',    # Airbnb adds hotel listings, loyalty program — major product launch
    492006:  'irrelevant',  # Airbnb verifying properties — minor operational policy
    430838:  'irrelevant',  # Jeff Buckley home as Airbnb — entertainment/lifestyle
    # Amazon
    3416936: 'relevant',    # Amazon halts construction over Seattle tax — major operational decision
    2782343: 'relevant',    # Amazon unveils new smart-home products — major product launch
    2655551: 'irrelevant',  # Xbox appears on Amazon — Xbox product, Amazon is retailer
    11100451:'irrelevant',  # Five things to know today — news roundup
    11142160:'relevant',    # Amazon launches video streaming in Japan — major service launch
    2619201: 'irrelevant',  # Amazon Giveaway sweepstakes feature — minor feature
    3096969: 'irrelevant',  # Whirlpool/Amazon-Sears — Whirlpool is subject
    3490156: 'relevant',    # FDA chief on Amazon healthcare move — major business expansion
    11498784:'relevant',    # Amazon/Whole Foods warehouses needed — post-M&A strategy
    3147691: 'irrelevant',  # Whole Foods acquisition impact on consumer brands — macro/industry
    # Apple Inc.
    10919001:'irrelevant',  # Amazon/Apple/Facebook driving market down — market roundup
    10522655:'irrelevant',  # Alphabet eclipses Apple — Alphabet is subject
    9626426: 'relevant',    # Apple CEO on FBI iPhone court order — major legal/regulatory
    10152529:'irrelevant',  # ESPN Apple TV app four-game view — minor app feature
    9937679: 'relevant',    # Apple invests $1B in Softbank Vision Fund — major investment
    10837312:'relevant',    # Apple revenue rises on iPhone shipments — earnings
    10606281:'irrelevant',  # Apple chip supplier virus delays — supplier is subject
    11413418:'irrelevant',  # Apple Music Muzak versions — entertainment/consumer content
    9563878: 'irrelevant',  # Enlisting Apple/Google to fight terrorists — policy/macro
    9368407: 'irrelevant',  # Apple music service rumour on exclusives — unconfirmed rumour
    # Berkshire Hathaway
    11677509:'irrelevant',  # Buffett watcher on bitcoin bubble — market prediction
    11766275:'irrelevant',  # Brett Veatch top broker — individual employee
    11706222:'relevant',    # Berkshire lifts cap on buybacks — buybacks
    11784132:'irrelevant',  # Munger says Buffett not "deeply evil" — vague statement
    11745264:'relevant',    # Berkshire not getting love for Apple stake — portfolio analysis (holding co)
    11733284:'irrelevant',  # Buffett annual lunch auction — charity event
    11712408:'irrelevant',  # Fire Warren Buffett? — investor opinion
    11781023:'relevant',    # Buffett sold REIT, betting on housing with new investments — major investments
    11700745:'relevant',    # Berkshire Energy agreement to acquire Oncor — M&A
    11638953:'irrelevant',  # Precision Castparts charity criticism — subsidiary/charity
    # Broadcom
    12011608:'relevant',    # UK scrutinizes Broadcom-VMware deal — M&A regulatory
    12005505:'irrelevant',  # Broadcom unusual options activity — trading activity
    11945558:'relevant',    # Broadcom profit and revenue miss — earnings
    12008392:'relevant',    # Broadcom announces convertible preferred stock conversion — equity/debt
    11948559:'relevant',    # Broadcom: Maslowski employment terminated — executive departure
    12016859:'relevant',    # Broadcom and Google Cloud VMware license portability — major partnership
    12011574:'irrelevant',  # Broadcom unusual options activity — trading activity
    12016874:'irrelevant',  # Broadcom, Dollar General earnings roundup — multiple companies
    12011135:'relevant',    # Broadcom stock drops on Apple own chip report — major customer risk
    12010766:'irrelevant',  # After-hours Broadcom stock movement — generic stock article
    # ExxonMobil
    7453086: 'relevant',    # Shell/ExxonMobil exit California — divestiture
    1098045: 'relevant',    # ExxonMobil wins anti-SLAPP motion — legal action
    7447972: 'irrelevant',  # ExxonMobil New Year tweet during Australia fires — social media
    1083163: 'relevant',    # Exxon Mobil beats earnings — earnings
    1088698: 'irrelevant',  # Bank of America on ExxonMobil — analyst prediction
    1083730: 'relevant',    # Exxon Mobil oil discovery off Guyana — major discovery
    7420351: 'irrelevant',  # Chevron vs ExxonMobil comparison — Chevron is subject
    7380669: 'irrelevant',  # Nordic American Tanker contract with Exxon subsidiary — NAT is subject
    1083380: 'relevant',    # ExxonMobil CEO meets Russian energy officials — executive/geopolitical
    7385732: 'relevant',    # ExxonMobil dealt with Iran under Tillerson — legal/regulatory revelation
    # FedEx
    11690934:'irrelevant',  # Walgreens leverages FedEx deal — Walgreens is subject
    11708636:'irrelevant',  # Trump sent FedEx courier a check — individual/consumer incident
    11684912:'irrelevant',  # FedEx hiring 500 for holidays — minor seasonal hiring
    11785333:'irrelevant',  # Wyndham Championship FedEx Cup — sports tournament
    11796748:'irrelevant',  # NYT/FedEx data squabble — media commentary
    11886716:'relevant',    # Cash dividend from FedEx — dividend
    11652591:'relevant',    # FedEx indictment on illegal drug shipments — legal/regulatory
    11661893:'irrelevant',  # Jordan Spieth wins FedEx Cup — sports event
    11773809:'irrelevant',  # FedEx/Home Depot stock movers — stock roundup
    11868583:'irrelevant',  # FedEx mass shooting — individual crime
    # Ford Motor Company
    2020059: 'irrelevant',  # Lee Iacocca obituary — historical executive
    2024706: 'relevant',    # Ford U.S. sales decline 12.5% amid coronavirus — revenue
    2036854: 'relevant',    # Ford halts F-150 Lightning deliveries for quality checks — major product issue
    2032110: 'irrelevant',  # Will Ford stock reach $20? — analyst prediction
    2017456: 'irrelevant',  # Ford GT orders open — consumer product
    2022532: 'irrelevant',  # Ford special editions, Trump call — minor product/political
    2026052: 'irrelevant',  # Trump blasts Michigan AG at Ford plant — political
    2010914: 'relevant',    # Ford Motor Q2 profit beats estimates — earnings
    2034626: 'relevant',    # Ford to cut 3K jobs amid EV transition — major layoffs
    2005377: 'relevant',    # Ford CEO outlines plans to compete with ride-hailing — major strategy
    # General Motors
    5077682: 'relevant',    # GM to test self-driving cars on public roads — major product/tech
    5261075: 'relevant',    # Federal judge rejects GM fuel pump defect argument — legal action
    5086127: 'irrelevant',  # Credit Suisse upgrades GM — analyst rating
    5264819: 'relevant',    # GM to build Cruise Origin, electric trucks in Detroit — major product/facility
    5047866: 'relevant',    # GM U.S. sales fall 3.8% — revenue
    5328176: 'relevant',    # GM to lay off 1,000 software employees — major layoffs
    5014137: 'irrelevant',  # GM design chief Ed Welburn retires — not CEO/CFO/board level
    4968152: 'relevant',    # GM lays off 3,300 amid EV issues — major layoffs
    4996735: 'relevant',    # GM Chevy/Buick recall 6.5M vehicles — major product recall
    5354004: 'relevant',    # Mary Barra committed to GM EV push — CEO strategy statement
    # Home Depot
    7386554: 'irrelevant',  # First National Realty acquires Home Depot center — FNRP is subject
    7389043: 'irrelevant',  # Texas woman ends up in Mexico going to Home Depot — consumer incident
    7386294: 'irrelevant',  # Bernie Marcus built legacy — opinion/lifestyle
    7385943: 'irrelevant',  # Stock movers Carmax/Circle/Home Depot — roundup
    7383588: 'irrelevant',  # Home Depot hiring 80,000 for spring — seasonal hiring
    7386107: 'relevant',    # Home Depot gross margins decline year-over-year — earnings
    7383581: 'relevant',    # Home Depot announces Q4 EPS $1.52 — earnings
    7383096: 'relevant',    # Home Depot Q2 income rises 9.3% — earnings
    7388999: 'irrelevant',  # Home Depot shoppers pulling back + Buffett on housing — mixed/macro
    7385831: 'irrelevant',  # Escaped calf behind Connecticut Home Depot — consumer incident
    # Intel
    7447062: 'irrelevant',  # NASA legend startup wants to outdo Intel — startup article
    7657876: 'irrelevant',  # Adam Schiff threatens intel community — US intelligence, not Intel corp
    7458356: 'relevant',    # Fiat Chrysler joins BMW-Intel driverless car platform — major partnership
    7422263: 'irrelevant',  # Vault 7 tools, US intel rebuild — US intelligence agencies
    7878892: 'relevant',    # Mobileye IPO proceeds go to Intel — M&A/spinoff proceeds
    7925815: 'relevant',    # EU hits Intel with $400M antitrust fine — regulatory/legal
    7343525: 'relevant',    # Intel to cut 2.5% of Folsom workforce — layoffs
    7869874: 'relevant',    # Intel unveils new funding model for chip project — major financial
    7528728: 'irrelevant',  # Stocks biggest premarket movers roundup — stock roundup
    7869342: 'irrelevant',  # McCarthy subpoenas intel agency chiefs — US intelligence agencies
    # JPMorgan Chase
    6136114: 'relevant',    # Three charged in cyberattacks against JPMorgan — legal/security
    6303698: 'relevant',    # JPMorgan to pay $920M for illegal trading — legal settlement
    6264691: 'irrelevant',  # Jamie Dimon on tariffs and recession — macro commentary
    6306175: 'relevant',    # JPMorgan to kick off Q3 earnings season — earnings
    6193985: 'relevant',    # JPMorgan pays $136M credit card settlement — legal settlement
    6133074: 'irrelevant',  # JPMorgan head: vote for politician, not CEO — political opinion
    642315:  'relevant',    # JPMorgan responding to fair banking access requests — regulatory compliance
    6270798: 'irrelevant',  # JPMorgan planning 5-day office return — HR policy, not material event
    6353507: 'irrelevant',  # Jamie Dimon quotes on complacency — CEO opinion quotes
    6317154: 'relevant',    # JPMorgan publishes 2020 annual report — major corporate disclosure
    # Johnson & Johnson
    8833025: 'relevant',    # J&J new tests find no asbestos in recalled powder — product safety/recall
    8771402: 'relevant',    # J&J loses talcum powder cancer lawsuit — legal action
    8993590: 'relevant',    # J&J to buy Abiomed for $16.6B — M&A
    8859800: 'relevant',    # J&J CFO on Q2 earnings, raising guidance — earnings/guidance
    8897035: 'relevant',    # J&J vaccine batch ruined at Baltimore factory — major operational issue
    8854453: 'relevant',    # J&J discontinues talc baby powder — major product discontinuation
    8990707: 'relevant',    # J&J $5B buyback program, profit targets — buyback/guidance
    8812495: 'relevant',    # US justices reject J&J unit anti-psychotic drug — legal action
    8779869: 'relevant',    # J&J hit with $247M verdict in hip implant trial — legal verdict
    8909092: 'relevant',    # Fauci expects J&J vaccine to get back on track — vaccine regulatory status
    # Mastercard
    8706374: 'relevant',    # Western Union/Mastercard partnership — major partnership
    8637752: 'relevant',    # MasterCard profit rises 12.7% — earnings
    8705943: 'relevant',    # DACO suspects Mastercard of discriminating consumers — regulatory investigation
    8619006: 'relevant',    # Mastercard CFO on steady consumer growth — earnings/guidance
    8682161: 'irrelevant',  # Card fraud declining says Mastercard — industry report, not corporate event
    8723214: 'irrelevant',  # Mastercard whale trades — options trading activity
    8643093: 'irrelevant',  # EMV chip adoption 88% per Mastercard — industry statistics
    8701935: 'irrelevant',  # Wirex launches Mastercard card — Wirex is subject
    8728734: 'relevant',    # Mastercard hikes dividend 16%, $9B buyback — dividend/buyback
    8715163: 'relevant',    # Mastercard acquires CipherTrace for crypto fraud — M&A
    # Meta Platforms
    4927711: 'irrelevant',  # Meta head of communications leaves — not CEO/CFO/board level
    4896633: 'relevant',    # Instagram reaches 3 billion monthly users — major business milestone
    4936767: 'irrelevant',  # Facebook users lose followers mysteriously — minor consumer/tech issue
    4963727: 'relevant',    # Meta shuts down thousands of fake election-related accounts — major platform action
    4913966: 'relevant',    # Facebook says Apple privacy push will cost $10B — material guidance impact
    4925358: 'irrelevant',  # Turkish woman arrested for criticizing Instagram ban — individual incident
    4936654: 'irrelevant',  # Facebook janitors protest layoffs — contracted workers, limited operational impact
    4894072: 'irrelevant',  # MPA demands Meta stop calling Instagram teen content PG-13 — advocacy demand, not regulatory
    4916930: 'irrelevant',  # Kanye West suspended from Instagram 24 hours — individual/social media content
    4910808: 'relevant',    # WhatsApp says spyware firm targeted users in 24 countries — major legal/security action
    # Microsoft
    4053077: 'irrelevant',  # Virtualization industry analysis — Microsoft one of several companies
    4440969: 'relevant',    # Reports in Microsoft lawsuit find significant gender gap — legal action
    4395926: 'irrelevant',  # What data Windows 10 sends Microsoft — consumer/privacy content
    4399217: 'irrelevant',  # GitHub billionaires owning Microsoft stock — individual stock ownership detail
    4462445: 'irrelevant',  # Paul Allen dead at 65 — historical co-founder obituary
    4538964: 'irrelevant',  # IBM makes Watson available on Amazon/Microsoft/Google — IBM is subject
    4787564: 'relevant',    # Microsoft warned about coronavirus impact — guidance/earnings impact
    4205861: 'relevant',    # Microsoft to deliver Visual Studio for Mac — major product launch
    3981033: 'irrelevant',  # Microsoft Copilot accesses Gmail — minor feature update
    4052462: 'irrelevant',  # Why Microsoft can't ditch phones — opinion/analysis
    # Netflix
    3727606: 'irrelevant',  # Netflix streams Star Wars only in Canada — consumer content
    4799593: 'irrelevant',  # Disappointing update on One Piece Season 2 — entertainment content
    4771517: 'irrelevant',  # Netflix alternative at $23/life — consumer shopping comparison
    4219537: 'relevant',    # Netflix fires CCO over use of racial slur — executive departure (C-suite)
    4171874: 'irrelevant',  # Adyen eyeing IPO, Netflix mentioned as client — Adyen is subject
    4597751: 'irrelevant',  # Woody Harrelson joins Kate at Netflix — entertainment casting
    4493027: 'irrelevant',  # Women-led production leads to Netflix success — lifestyle/opinion
    4746702: 'irrelevant',  # Chrissy Teigen on Meghan Markle Netflix series — entertainment content
    4690592: 'irrelevant',  # 911 supervisor played Netflix movie — individual incident
    4096223: 'irrelevant',  # Josh Hartnett Netflix series casting — entertainment content
    # Nike Inc.
    4710631: 'irrelevant',  # Nordstrom promo codes on Nike — consumer shopping guide
    11773296:'irrelevant',  # Adidas leads Nike in World Cup shirts — sports/consumer content
    4439706: 'relevant',    # Texas to sign 15-year deal with Nike, richest in college sports — major contract
    4433652: 'irrelevant',  # Dear Nike when can we expect Back to the Future shoes — consumer/entertainment
    4677172: 'relevant',    # Nike drops Antonio Brown amid sexual assault lawsuit — contract termination
    4815509: 'irrelevant',  # Rare Nike sneakers key to catching riot suspect — individual crime
    4724107: 'irrelevant',  # Nike sneakers representing city jobs — consumer/lifestyle
    4588217: 'relevant',    # Backlash after Colin Kaepernick named face of Nike ad — major brand campaign with documented business impact
    11663508:'irrelevant',  # Nike plans to sell self-lacing Back to the Future sneakers — minor consumer product
    11764105:'irrelevant',  # Premarket China tariffs Facebook Nike earnings — market roundup
    # Nvidia
    3644222: 'relevant',    # Nvidia GeForce Now game streaming on Shield — major product/service launch
    3853011: 'irrelevant',  # Nvidia Shield 2019 review — consumer product review
    3964370: 'irrelevant',  # How to invest in Magnificent Seven like Nvidia — investment advice
    3719843: 'irrelevant',  # Groq eyes $6B valuation amid AI chip demand — competitor article
    3745227: 'relevant',    # What analysts expect from Nvidia earnings — earnings preview
    3668199: 'irrelevant',  # Baidu partners with Ford, Nvidia, others — Baidu is subject
    3719396: 'irrelevant',  # Nvidia first company valued at $4 trillion — market cap milestone, not corporate event
    3852262: 'relevant',    # EU regulators open probe into Nvidia $54B ARM bid — M&A regulatory
    3916920: 'irrelevant',  # AI stock up 254% and it's not Nvidia — Nvidia as comparison
    3938501: 'irrelevant',  # How much Cathie Wood missed by selling Nvidia — investor decisions
    # Pfizer
    5767578: 'relevant',    # Pfizer/BioNTech seek full US approval for COVID vaccine — regulatory filing
    5674390: 'relevant',    # Pfizer/BioNTech start combined trials in Japan — major clinical milestone
    5704722: 'relevant',    # Amnesty International: Pfizer misleading poor countries — significant allegation/regulatory
    5828042: 'irrelevant',  # Pfizer CEO says vaccine-resistant variant likely — macro health prediction
    5557595: 'relevant',    # Pfizer to roll back drug prices after Trump discussion — major pricing decision
    5713169: 'relevant',    # Pfizer booster 95.6% efficacy vs Delta — major clinical data announcement
    5723008: 'relevant',    # Pfizer expects vaccine data for children in late September — clinical milestone/guidance
    5967480: 'irrelevant',  # Tomi Lahren Taylor Swift remark after Travis Kelce Pfizer uproar — social media/consumer
    5653221: 'irrelevant',  # Corporate scientific leadership key to Pfizer vaccine success — opinion/lifestyle
    5961409: 'irrelevant',  # Cresemba sales by Pfizer trigger payments to Basilea — Basilea is subject
    # Procter & Gamble
    7477826: 'irrelevant',  # P&G Gillette woes good news for consumers — consumer content
    7530497: 'relevant',    # P&G raised prices 10%, volume fell more than expected — earnings/revenue impact
    7475141: 'relevant',    # P&G may drop USA Gymnastics sponsorship — major contract/sponsorship decision
    7481244: 'irrelevant',  # JPMorgan downgrades P&G — analyst firm prediction
    7504900: 'relevant',    # Shoppers turn to smaller brands cutting into P&G profits — revenue/earnings
    7486767: 'irrelevant',  # Banned Christina Hendricks P&G ad — consumer/advertising content
    7521281: 'irrelevant',  # P&G Ventures teams with The Riveter — minor partnership/event
    7462070: 'relevant',    # Cramer on remarkable quarter for P&G and DuPont — earnings analysis
    7530200: 'irrelevant',  # Is P&G stock appropriately priced? — valuation analysis
    7486408: 'irrelevant',  # P&G files to trademark millennial phrases — minor marketing/PR
    # Salesforce
    2707065: 'irrelevant',  # Zscaler earnings, wants to be the Salesforce of cybersecurity — Zscaler subject
    2727571: 'relevant',    # Salesforce delivers, stock falls 6.3% post-earnings — earnings
    2703135: 'irrelevant',  # Stocks biggest premarket movers roundup — stock roundup
    2703809: 'irrelevant',  # Inside Salesforce's summit on inclusive AI — consumer event/lifestyle
    2629171: 'relevant',    # Salesforce aims to expand developer ecosystem — major strategic initiative
    2739564: 'relevant',    # Salesforce sees a fresh round of layoffs — major layoffs
    2718779: 'irrelevant',  # Former Salesforce exec joins Automation Anywhere — individual career move
    2629029: 'relevant',    # Salesforce to use Amazon's cloud to expand in Canada and Australia — major partnership/expansion
    2671242: 'relevant',    # 50 women sue Salesforce over sex trafficking — major legal action
    2745321: 'irrelevant',  # Salesforce CEO demands net zero commitment at Davos — public statement at conference
    # Starbucks
    9276031: 'relevant',    # Starbucks hit with boycott after pledging to hire refugees — documented boycott with business impact
    1780069: 'relevant',    # Second Starbucks location in Mesa votes to unionize — labor/union organizing
    9524246: 'relevant',    # Starbucks violated labor law in Buffalo union drive — legal ruling
    1808691: 'irrelevant',  # Your Starbucks order reveals Netflix protagonist — entertainment/consumer
    1669406: 'irrelevant',  # Starbucks robbery suspect arrested — individual crime
    9263812: 'relevant',    # Starbucks departing chairman backs China prospects — executive departure + guidance
    1649571: 'irrelevant',  # One trader made big bet on Starbucks pain — market speculation
    9091228: 'relevant',    # Starbucks partners with Spotify for loyalty — major partnership
    9341092: 'irrelevant',  # Hidden camera in Starbucks restroom — individual incident
    9297189: 'irrelevant',  # 5-year-old finds camera in Starbucks bathroom — individual incident
    # Tesla Inc.
    5650101: 'irrelevant',  # I drove 8 hours on Tesla Autopilot — consumer/lifestyle
    5877196: 'irrelevant',  # Tesla Musk overpromising on self-driving — opinion piece
    5930076: 'irrelevant',  # April Fool's jokes that backfired including Tesla — entertainment
    5929727: 'irrelevant',  # Vandalism at Michigan Tesla site — individual crime
    5889608: 'irrelevant',  # Musk whacks pylon in Cybertruck — social media/consumer
    5400530: 'irrelevant',  # Stock movers Tesla IBM Quantum Computing — stock roundup
    5688218: 'irrelevant',  # This Tesla caught fire — individual incident
    5944298: 'irrelevant',  # Retail investors stick with Tesla after $800B wipeout — investor sentiment
    5634770: 'irrelevant',  # Musk shares last photo of Starman in Tesla — social media/consumer
    5766182: 'irrelevant',  # Cramer says Musk should take medical leave — analyst opinion
    # Uber
    6402643: 'irrelevant',  # Uber drivers/riders can now text in-app — minor app feature
    6614772: 'irrelevant',  # Uber/Goodwill donation service — minor community program
    6189123: 'irrelevant',  # Why own a car? Google/Apple/Uber asking — industry analysis
    6696307: 'relevant',    # Uber/Lyft drivers plan to strike in dozen+ cities — major labor strike
    6722751: 'irrelevant',  # The Creature From Uber Island — lifestyle/opinion
    6763033: 'relevant',    # Uber faces UK tax challenge — regulatory/legal
    6664776: 'relevant',    # Uber faces legal challenges in France/South Korea/Germany — legal challenges
    6117758: 'irrelevant',  # Robot cars won't rescue Uber from clash with drivers — opinion
    6453417: 'irrelevant',  # Cities with Uber have lower ambulance usage — research study
    6109046: 'irrelevant',  # Uber plans helicopter rides through app — minor app feature addition
    # UnitedHealth Group
    7952030: 'relevant',    # UnitedHealth suspends efforts to sell Brazilian unit — M&A/divestiture
    7933839: 'relevant',    # DA: 40 UnitedHealthcare execs got bodyguards after CEO murder — CEO death/security crisis
    7898083: 'relevant',    # UnitedHealth earnings rise 6% — earnings
    7949595: 'relevant',    # UnitedHealth class action lawsuit filed by investors — legal action
    7934243: 'irrelevant',  # UnitedHealth spent $1.6M on executive security — operational expenditure detail
    7934048: 'irrelevant',  # UnitedHealth longtime bull throws in towel — analyst sentiment
    7906993: 'relevant',    # UnitedHealth Medicare plan must cover sex reassignment surgery — regulatory ruling
    7928410: 'relevant',    # UnitedHealthcare increases housing investments — major strategic investment
    7901613: 'irrelevant',  # UnitedHealth to hire 1,700 in Twin Cities — minor hiring
    7922028: 'relevant',    # UnitedHealth buys into hearing aid benefits — M&A/investment
    # Visa Inc.
    8325658: 'relevant',    # Nigerian unicorn Interswitch sells stake to Visa — M&A investment
    8045560: 'irrelevant',  # US suspends fast processing for H-1B visas — immigration policy
    776847:  'irrelevant',  # Iranian infant needs visa waiver — immigration/medical
    8331418: 'irrelevant',  # Trump outlet tried to get US visa for corrupt Ukrainian — immigration/political
    743201:  'irrelevant',  # H-1B visa program opinion — immigration
    968856:  'irrelevant',  # DHS refuses to investigate fast-tracked visas — immigration/political
    8400342: 'irrelevant',  # Africans travel across continent without visas — travel/immigration
    7950996: 'irrelevant',  # Visa windfall gives European banks Q2 boost — European banks are subject
    8352317: 'irrelevant',  # University of Colorado reinstates student visas — student visa/immigration
    8308080: 'irrelevant',  # Russia denies visas to US senators — travel/political
    # Walmart
    496551:  'irrelevant',  # What consumers buy most at Walmart — consumer/lifestyle
    6674040: 'relevant',    # Walmart to pay $16B for Flipkart — M&A
    17500:   'irrelevant',  # Walmart selling wireless earbuds at discount — consumer deal
    378562:  'irrelevant',  # Walmart greeter with cerebral palsy — individual employee story
    304783:  'irrelevant',  # 18-year-old hurls racial slurs at Walmart — individual crime
    381291:  'relevant',    # Walmart deploys 17,000 Oculus Go headsets for training — major operational initiative
    106101:  'irrelevant',  # Christmas Eve hours for Walmart/Target/etc — consumer shopping guide
    114492:  'relevant',    # Walmart Q4 2016 best quarter in 4 years — earnings
    199370:  'irrelevant',  # Walmart making its own meal kits — minor product addition
    355292:  'irrelevant',  # Save big on Vizio TVs at Walmart — consumer deal
    # Walt Disney Company
    2885048: 'irrelevant',  # LGBTQ+ community needs partnership from Disney — opinion/social
    2885097: 'relevant',    # Comcast and Walt Disney announce content carriage agreement — major partnership/deal
    2884972: 'irrelevant',  # Controversial Disney animatronic debuts — entertainment/consumer content
    2885232: 'relevant',    # Hollywood skeptical Disney CEO Iger will step down, successor search — CEO succession
    2885250: 'irrelevant',  # Hyundai reveals IONIQ 5 Disney100 concept — Hyundai is subject
    2885294: 'relevant',    # Marvel delays Deadpool 3/Captain America 4 post-strike — major product delays
    2885082: 'relevant',    # Walt Disney Company annual 10-K report — major corporate disclosure
    2885150: 'relevant',    # Disneyland requires non-union employees vaccinated, union talks — labor/HR
    2885124: 'relevant',    # Anaheim living wage ordinance doesn't apply to Disneyland — legal ruling on labor costs
    2884886: 'irrelevant',  # Petition to silence Trump animatronic — consumer/social petition
    # Wells Fargo
    8318869: 'relevant',    # Did Wells Fargo exploit workers who exploited customers — major legal/ethics scandal
    8491719: 'relevant',    # Wells Fargo executives/board must face lawsuit over fake accounts — legal action
    8509304: 'relevant',    # Wells Fargo pays $385M to settle car loan lawsuit — legal settlement
    8292405: 'relevant',    # Wells Fargo lowers targets for returns on equity/assets — earnings/guidance
    8447983: 'irrelevant',  # AT&T/Comcast/Wells Fargo promise bonuses after tax cut — multiple companies
    8428683: 'irrelevant',  # Wells Fargo put family in victim protection at risk — individual lawsuit
    8322694: 'irrelevant',  # Wells Fargo scandal points to industry-wide problem — macro/industry analysis
    8362998: 'relevant',    # Wells Fargo found liable in abusive tax shelter scheme — legal ruling
    8618336: 'irrelevant',  # Wells Fargo Championship live odds — golf tournament
    8324288: 'relevant',    # Wells Fargo CEO getting only seconds to speak — CEO testifying at congressional hearing
}

pool = pd.read_csv('relevance_classifier_v2/gold_annotation_pool.csv')
pool['id'] = pool['id'].astype(int)

# Map labels
pool['label'] = pool['id'].map(LABELS)
missing = pool[pool['label'].isna()]
if len(missing) > 0:
    print(f"WARNING: {len(missing)} articles missing labels:")
    print(missing[['id', 'title', 'company_name']].to_string())
else:
    print("All articles labeled.")

pool['source'] = 'gold_manual'
pool['annotator'] = 'claude_gold'

out = pool[['id', 'title', 'company_id', 'company_name', 'label', 'source', 'annotator']]
out.to_csv('relevance_classifier_v2/gold_a1.csv', index=False)

rel = (out['label'] == 'relevant').sum()
irr = (out['label'] == 'irrelevant').sum()
print(f"\nSaved relevance_classifier_v2/gold_a1.csv")
print(f"Total: {len(out)} | Relevant: {rel} ({100*rel/len(out):.1f}%) | Irrelevant: {irr} ({100*irr/len(out):.1f}%)")
print("\nPer-company breakdown:")
print(out.groupby('company_name')['label'].value_counts().unstack(fill_value=0).to_string())
