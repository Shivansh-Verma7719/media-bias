"""
Build a comprehensive company name list for the pre-filter step,
combining full names, short/common names, and aliases.
"""

companies = [
    ('PGR','Progressive'), ('MRK','Merck'), ('ETN','Eaton'),
    ('AMD','Advanced Micro Devices'), ('ANET','Arista Networks'), ('CRM','Salesforce'),
    ('CSCO','Cisco'), ('MU','Micron Technology'), ('COP','ConocoPhillips'), ('AXP','American Express'),
    ('PM','Philip Morris'), ('UNH','UnitedHealth'), ('RTX','RTX'),
    ('TMUS','T-Mobile'), ('ABT','Abbott'), ('TMO','Thermo Fisher'),
    ('GE','GE Aerospace'), ('GS','Goldman Sachs'), ('INTC','Intel'), ('CVX','Chevron'),
    ('INTU','Intuit'), ('VZ','Verizon'), ('IBM','IBM'), ('COF','Capital One'), ('WFC','Wells Fargo'),
    ('BRK.B','Berkshire Hathaway'), ('AVGO','Broadcom'), ('C','Citigroup'), ('TSLA','Tesla'),
    ('ISRG','Intuitive Surgical'), ('AMAT','Applied Materials'), ('PEP','PepsiCo'), ('T','AT&T'),
    ('DIS','Disney'), ('HON','Honeywell'), ('CEG','Constellation Energy'), ('CRWD','CrowdStrike'),
    ('PLD','Prologis'), ('HCA','HCA Healthcare'), ('HOOD','Robinhood'),
    ('CB','Chubb'), ('VRTX','Vertex Pharmaceuticals'), ('BX','Blackstone'), ('PH','Parker Hannifin'),
    ('MCK','McKesson'), ('META','Meta'), ('UNP','Union Pacific'), ('LOW','Lowe\'s'),
    ('MDT','Medtronic'), ('ADBE','Adobe'), ('DE','John Deere'), ('MS','Morgan Stanley'),
    ('CAT','Caterpillar'), ('PANW','Palo Alto Networks'), ('ADI','Analog Devices'),
    ('MCD','McDonald\'s'), ('APH','Amphenol'), ('NOW','ServiceNow'),
    ('SCHW','Charles Schwab'), ('BLK','BlackRock'), ('GM','General Motors'), ('GEV','GE Vernova'),
    ('DHR','Danaher'), ('GILD','Gilead Sciences'), ('UBER','Uber'), ('LLY','Eli Lilly'),
    ('BKNG','Booking Holdings'), ('ACN','Accenture'), ('PFE','Pfizer'),
    ('TXN','Texas Instruments'), ('SPGI','S&P Global'), ('BSX','Boston Scientific'),
    ('SYK','Stryker'), ('JPM','JPMorgan Chase'), ('ABBV','AbbVie'),
    ('COST','Costco'), ('PLTR','Palantir'), ('QCOM','Qualcomm'), ('BAC','Bank of America'),
    ('NEE','NextEra Energy'), ('WMT','Walmart'), ('HD','Home Depot'), ('FCX','Freeport-McMoRan'),
    ('XOM','ExxonMobil'), ('PG','Procter & Gamble'), ('KO','Coca-Cola'), ('PSX','Phillips 66'),
    ('MMM','3M'), ('DELL','Dell'), ('GD','General Dynamics'), ('SBUX','Starbucks'),
    ('NKE','Nike'), ('NDAQ','Nasdaq'), ('NEM','Newmont'), ('EA','Electronic Arts'),
    ('MO','Altria'), ('CMCSA','Comcast'), ('DUK','Duke Energy'), ('AAPL','Apple'),
    ('SLB','Schlumberger'), ('VLO','Valero Energy'), ('TFC','Truist Financial'),
    ('LMT','Lockheed Martin'), ('CME','CME Group'), ('CVS','CVS Health'),
    ('BMY','Bristol Myers Squibb'), ('MMC','Marsh McLennan'), ('F','Ford'), ('LHX','L3Harris'),
    ('URI','United Rentals'), ('WM','Waste Management'), ('MCO','Moody\'s'),
    ('MPC','Marathon Petroleum'), ('ZTS','Zoetis'), ('ROST','Ross Stores'),
    ('WDAY','Workday'), ('MSI','Motorola Solutions'), ('DDOG','Datadog'),
    ('BDX','Becton Dickinson'), ('WDC','Western Digital'), ('AFL','Aflac'), ('CI','Cigna'),
    ('EMR','Emerson Electric'), ('CTAS','Cintas'), ('EQIX','Equinix'),
    ('WMB','Williams Companies'), ('RCL','Royal Caribbean'), ('GLW','Corning'),
    ('COIN','Coinbase'), ('JCI','Johnson Controls'), ('CMI','Cummins'),
    ('ABNB','Airbnb'), ('HLT','Hilton'), ('AZO','AutoZone'), ('DASH','DoorDash'),
    ('REGN','Regeneron'), ('ECL','Ecolab'), ('MAR','Marriott'), ('NOC','Northrop Grumman'),
    ('UPS','UPS'), ('BK','BNY Mellon'), ('TDG','TransDigm'), ('APO','Apollo Global'),
    ('PNC','PNC Financial'), ('AON','Aon'), ('ELV','Elevance Health'),
    ('NSC','Norfolk Southern'), ('OKE','ONEOK'), ('LVS','Las Vegas Sands'),
    ('KR','Kroger'), ('MET','MetLife'), ('TGT','Target'), ('PYPL','PayPal'),
    ('BKR','Baker Hughes'), ('NFLX','Netflix'), ('CMG','Chipotle'),
    ('DAL','Delta Air Lines'), ('OXY','Occidental Petroleum'), ('CTSH','Cognizant'),
    ('AIG','AIG'), ('SHW','Sherwin-Williams'), ('LIN','Linde'), ('LRCX','Lam Research'),
    ('NVDA','Nvidia'), ('AMGN','Amgen'), ('AMZN','Amazon'), ('ADSK','Autodesk'),
    ('MSFT','Microsoft'), ('FDX','FedEx'), ('CL','Colgate-Palmolive'),
    ('GOOGL','Google'), ('GOOG','Alphabet'), ('MA','Mastercard'),
    ('KKR','KKR'), ('JNJ','Johnson & Johnson'), ('ORCL','Oracle'), ('V','Visa'), ('BA','Boeing'),
    ('FTNT','Fortinet'), ('KDP','Keurig Dr Pepper'), ('GRMN','Garmin'),
    ('TTWO','Take-Two Interactive'), ('YUM','Yum! Brands'), ('CARR','Carrier'),
    ('ROK','Rockwell Automation'), ('FAST','Fastenal'), ('FANG','Diamondback Energy'),
    ('STX','Seagate'), ('ICE','Intercontinental Exchange'), ('D','Dominion Energy'),
    ('SO','Southern Company'), ('EXC','Exelon'), ('AMT','American Tower'),
    ('NXPI','NXP Semiconductors'), ('IBKR','Interactive Brokers'),
]

aliases = [
    # Big tech
    'Alphabet', 'Google', 'Meta', 'Facebook', 'Amazon', 'Apple', 'Microsoft', 'Tesla',
    'Netflix', 'Nvidia', 'Walt Disney', 'JPMorgan', 'Goldman', 'Citi',
    # Finance
    'Bank of America', 'Wells Fargo', 'Citigroup', 'American Express', 'Amex',
    'Berkshire', 'Blackstone', 'BlackRock', 'Palantir', 'Coinbase', 'Robinhood',
    'PayPal', 'Visa', 'Mastercard', 'Moody\'s',
    # Auto
    'Ford', 'General Motors', 'Boeing',
    # Retail
    'Walmart', 'Target', 'Costco', 'Home Depot', 'Starbucks', 'Nike',
    # Food/Bev
    'Coca-Cola', 'Pepsi', 'McDonald\'s', 'Chipotle', 'Yum',
    # Telecom
    'AT&T', 'Verizon', 'Comcast', 'T-Mobile', 'Motorola',
    # Tech
    'IBM', 'Intel', 'Cisco', 'Oracle', 'Salesforce', 'Adobe', 'Qualcomm',
    'Broadcom', 'AMD', 'Micron', 'Uber', 'Airbnb', 'DoorDash', 'Booking',
    'ServiceNow', 'Workday', 'Datadog', 'CrowdStrike', 'Fortinet', 'Palo Alto',
    'Seagate', 'Western Digital', 'Autodesk', 'Garmin', 'NXP',
    # Healthcare
    'Pfizer', 'Merck', 'Johnson & Johnson', 'J&J', 'AbbVie', 'Eli Lilly',
    'Amgen', 'Gilead', 'UnitedHealth', 'CVS', 'Cigna', 'Medtronic', 'Abbott',
    'Bristol Myers', 'Regeneron', 'Becton Dickinson', 'Stryker', 'Zoetis',
    # Energy
    'ExxonMobil', 'Exxon', 'Chevron', 'ConocoPhillips', 'FedEx', 'UPS',
    'Schlumberger', 'Valero', 'Phillips 66', 'Marathon Petroleum', 'Baker Hughes',
    'Diamondback', 'Occidental', 'Dominion', 'NextEra', 'Duke Energy', 'Exelon',
    # Defense
    'Lockheed Martin', 'Northrop Grumman', 'General Dynamics', 'Raytheon',
    # Industrial
    'Honeywell', 'Caterpillar', 'John Deere', 'Emerson', '3M', 'Lowe\'s',
    'Eaton', 'Parker Hannifin', 'Cummins', 'Fastenal', 'Cintas', 'Carrier',
    'Rockwell Automation', 'TransDigm', 'Illinois Tool Works',
    # Travel/Hotels
    'Royal Caribbean', 'Marriott', 'Hilton', 'Delta', 'Las Vegas Sands',
    # Nifty 50
    'Reliance', 'Reliance Industries', 'TCS', 'Tata Consultancy', 'Infosys',
    'HDFC', 'ICICI', 'SBI', 'Wipro', 'HCL', 'Bajaj', 'Maruti', 'Adani',
    'ITC', 'Axis Bank', 'Kotak', 'Mahindra', 'Tata Motors', 'NTPC',
]

names = set(name for _, name in companies)
names.update(aliases)

with open('company_names.txt', 'w') as f:
    f.write('\n'.join(sorted(names)))

print(f'Total company name entries: {len(names)}')
