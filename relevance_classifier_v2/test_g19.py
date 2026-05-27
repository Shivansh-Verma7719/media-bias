import rule_adjuster

test_cases = [
    ("ICE agents at the border", "Visa Inc.", 0.8),
    ("Border crossing surges in December", "Visa", 0.7),
    ("Deportation numbers rise after policy change", "Visa", 0.9),
    ("Visa launches new payment system in UK", "Visa", 0.85),
    ("Border security concerns for tech companies", "Intel", 0.8),
]

print(f"{'Title':<45} | {'Company':<10} | {'Original':<8} | {'Adjusted':<8} | {'Rule'}")
print("-" * 100)

for title, co, prob in test_cases:
    adj_prob, rule = rule_adjuster.adjust(title, co, prob)
    print(f"{title[:44]:<45} | {co:<10} | {prob:<8.2f} | {adj_prob:<8.2f} | {rule}")
