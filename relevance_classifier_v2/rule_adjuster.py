"""
Rule-based post-processor for relevance classifier.

Adjusts p_rel scores after model inference based on linguistically distinctive
patterns that correspond to known FP categories. Applied before thresholding.

Rules are conservative — only fire on patterns that are NEVER relevant.
Each rule has a documented category and rationale.
"""
import re


def _contains(title: str, *phrases) -> bool:
    t = title.lower()
    return any(p.lower() in t for p in phrases)


def _matches(title: str, pattern: str) -> bool:
    return bool(re.search(pattern, title, re.IGNORECASE))


def adjust(title: str, company_name: str, p_rel: float) -> tuple[float, str]:
    """
    Returns (adjusted_p_rel, fired_rule_name).
    Returns (p_rel, '') if no rule fires.
    """
    t = title.strip()
    co = (company_name or "").strip().lower()

    # G19 — Visa contamination: travel/immigration visa articles tagged to Visa Inc.
    # These never concern Visa the payment company.
    if "visa" in co:
        travel_visa_signals = [
            "visa liberalization", "visa waiver", "visa program",
            "visa backlog", "visa requirements", "visa processing",
            "student visa", "work visa", "h-1b", "tourist visa",
            "eu visa", "visa-free", "visa applicat", "visa subpoena",
            "visa rules", "visa delays", "visa reform", "biometric visa",
            "immigration visa", "travel visa", "ice", "deportation", "border",
        ]
        if _contains(t, *travel_visa_signals):
            return (0.05, "G19:visa_contamination")

    # G12 — Analyst earnings preview (not an actual earnings release)
    if _contains(t,
                 "what to expect from", "what to watch", "earnings preview",
                 "what analysts are watching", "analyst estimates",
                 "consensus estimates", "what wall street expects",
                 "ahead of earnings", "before it reports", "preview:",
                 "earnings: what", "q1 preview", "q2 preview",
                 "q3 preview", "q4 preview"):
        return (max(0.05, p_rel - 0.40), "G12:analyst_preview")

    # G13 — Quantified philanthropy: [Company] Foundation + dollar commitment
    if _matches(t, r"foundation.{0,60}\$\d") or \
       _matches(t, r"\$[\d,.]+\s*(million|billion|M|B).{0,40}(fund|relief|grant|pledge|donat|scholar|commit|initiative)") or \
       _contains(t, "pledges $", "commits $", "donates $", "charitable", "philanthrop"):
        if not _contains(t, "fine", "penalt", "settl", "lawsuit", "regulat"):
            return (max(0.05, p_rel - 0.40), "G13:philanthropy")

    # G15 — Mid-level executive appointment (not C-suite)
    # "BRIEF-" prefix = wire service short, typically minor appointments
    if t.startswith("BRIEF-") or t.lower().startswith("brief-"):
        return (max(0.05, p_rel - 0.35), "G15:brief_wire")
    mid_level_roles = [
        "head of investor relations", "head of communications",
        "head of marketing", "head of sustainability", "head of digital",
        "head of government affairs", "head of supply chain",
        "director of investor", "director of communications",
        "vp of human", "vp of marketing", "vp of communications",
        "chief diversity officer", "chief of staff",
        "regional vice president", "regional director",
        "head of global partnerships", "head of corporate communications",
    ]
    if _contains(t, *mid_level_roles):
        # Only fire if it's an appointment, not a departure or action
        if _contains(t, "names", "appoints", "promotes", "hires", "taps", "joins", "recruits"):
            return (max(0.05, p_rel - 0.35), "G15:midlevel_appointment")

    # G17 — Company name used as brand comparator, not primary subject
    # "The Netflix of China", "The Uber of X", etc.
    comparator_pattern = r"the\s+" + re.escape(t.split()[0] if t.split() else "") + r"\s+of\s+"
    if _matches(t, r"\bthe\s+\w+\s+of\s+(china|india|europe|asia|africa|latin america|brazil|japan|korea|"
                    r"germany|france|uk|canada|australia|mexico|indonesia|southeast asia)\b"):
        return (max(0.05, p_rel - 0.40), "G17:brand_comparator")
    if _contains(t, "'s answer to", "answer to the", "version of", "rival to"):
        if _matches(t, r"answer to (the\s+)?\w+|version of \w+"):
            return (max(0.05, p_rel - 0.35), "G17:brand_comparator")

    # G20 — Company mentioned as industry disruption catalyst, article is about the industry
    if _matches(t, r"^competition with \w"):
        return (max(0.05, p_rel - 0.40), "G20:disruption_catalyst")
    if _matches(t, r"\b(taxi|hotel|cable|newspaper|travel agency|video rental)\b.{0,40}(struggle|brace|pivot|face|audit|lend)"):
        return (max(0.05, p_rel - 0.35), "G20:disruption_catalyst")

    # G11 — Speculative/preliminary: "considers", "mulls", "exploring" with no decision
    # Be conservative — only fire when clearly speculative with no corporate action taken
    if _matches(t, r"\bconsiders (importing|exporting|entering|exiting|selling|buying|moving)\b"):
        return (max(0.10, p_rel - 0.25), "G11:speculative")
    if _matches(t, r"\b(mulls|weighing|exploring potential|may consider|studying options)\b"):
        if _contains(t, "no deal", "no decision", "sources say", "people familiar",
                     "early stage", "preliminary"):
            return (max(0.10, p_rel - 0.25), "G11:speculative")

    # G16 — Local economic impact story (city/region as primary beneficiary)
    if _matches(t, r"^([\w\s]+)\s+a big winner in"):
        return (max(0.05, p_rel - 0.35), "G16:local_economic")
    if _matches(t, r"^([\w\s]+)\s+(gains|gets|lands|wins).{0,30}(jobs|economic|boost|investment)"):
        return (max(0.10, p_rel - 0.25), "G16:local_economic")

    # G18 — Small lawsuit ($1M–$49M) — immaterial at S&P 500 scale
    small_suit = re.search(
        r'\$([1-9]|[1-3]\d|4[0-9])\s*(million|M)\b.{0,40}(lawsuit|suit|claim|allege)',
        t, re.IGNORECASE
    )
    if small_suit:
        return (max(0.05, p_rel - 0.30), "G18:small_lawsuit")
    # Also catch "[amount]m lawsuit alleges" phrasing
    if _matches(t, r'\$[1-9]\d?m\s+lawsuit') or _matches(t, r'lawsuit alleges.{0,30}\$[1-4]\d?\s*m'):
        return (max(0.05, p_rel - 0.30), "G18:small_lawsuit")

    return (p_rel, "")


def adjust_batch(titles, company_names, p_rels):
    """Adjust a list of predictions. Returns (adjusted_probs, fired_rules)."""
    results = [adjust(t, c, p) for t, c, p in zip(titles, company_names, p_rels)]
    adjusted = [r[0] for r in results]
    rules = [r[1] for r in results]
    return adjusted, rules
