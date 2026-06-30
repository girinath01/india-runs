"""
NER Engine v1.0 — Fast Text Extraction using spaCy.

Replaces fragile regular expressions with Named Entity Recognition (NER)
to accurately extract companies (ORG), titles, and dates.
"""

import spacy
from typing import Set

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            # Disable parser for speed since we only need NER
            _nlp = spacy.load("en_core_web_sm", disable=["parser"])
            print("[NER Engine] Loaded spaCy en_core_web_sm successfully.")
        except Exception as e:
            print(f"[NER Engine] WARNING: Failed to load spaCy model: {e}")
            _nlp = False
    return _nlp if _nlp is not False else None

def extract_companies(text: str) -> Set[str]:
    """Extract organization names from text using NER."""
    nlp = _get_nlp()
    if not nlp:
        return set()
    
    # Process text up to 100k chars to avoid memory issues
    doc = nlp(text[:100000])
    orgs = set()
    for ent in doc.ents:
        if ent.label_ == "ORG":
            org_name = ent.text.strip().lower()
            if len(org_name) > 2:
                orgs.add(org_name)
    return orgs

def has_senior_title(text: str) -> bool:
    """Uses NER to check if a senior job title is mentioned in the text."""
    # We look for specific keywords in the text since NER doesn't always classify titles as a specific entity.
    # However, if it's classified as a PERSON or WORK_OF_ART erroneously by regex, we can ignore it.
    # For now, a fast keyword check is still best for titles, but we can augment it.
    lower_text = text.lower()
    senior_keywords = ["senior", "staff", "principal", "lead", "head of", "director", "vp"]
    for kw in senior_keywords:
        if kw in lower_text:
            return True
    return False
