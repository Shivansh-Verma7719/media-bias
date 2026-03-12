import re

def find_company_in_title(title: str, company_name: str) -> str | None:
    """Return company name if found in title via word-boundary match, or None."""
    title_lower = title.lower()
    if re.search(r'\b' + re.escape(company_name.lower())[:10] + r'\b', title_lower) or company_name.lower() in title_lower:
        return company_name
    return None


def get_first_paragraph(content: str) -> str:
    """Extract first substantial paragraph from content (capped at 1000 chars)."""
    if not content:
        return ""
    for para in re.split(r'\n{2,}|\r\n\r\n', content.strip()):
        para = para.strip()
        if len(para) > 80:
            return para[:1000]
    return content.strip()[:500]


def split_on_aspect(text: str, aspect: str):
    """
    Split text into (left, aspect, right) around the aspect string.
    NewsMTSC requires all three to be non-empty strings — uses single space if empty.
    """
    first_word = aspect.split()[0] if aspect else ""
    m = re.search(re.escape(aspect), text, re.IGNORECASE)
    if not m and first_word:
        m = re.search(re.escape(first_word), text, re.IGNORECASE)
        
    if m:
        left  = text[:m.start()].strip() or " "
        mid   = text[m.start():m.end()]
        right = text[m.end():].strip()  or " "
        return left, mid, right
    # Aspect not found in text — use full text as left context
    return text.strip() or " ", aspect, " "


def choose_input_text(title: str, first_para: str, company_name: str) -> str:
    """
    Prefer first_paragraph if it contains the company name and is long enough.
    Fall back to title (which is guaranteed to contain the company).
    """
    first_word = company_name.split()[0] if company_name else ""
    if (first_para and len(first_para) > 100 and
            (re.search(re.escape(company_name), first_para, re.IGNORECASE) or 
             (first_word and re.search(re.escape(first_word), first_para, re.IGNORECASE)))):
        return first_para
    return title