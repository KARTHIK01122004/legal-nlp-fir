# tools/ner_tool.py

import re


def extract_entities(text: str) -> dict:
    """
    Rule-based NER for FIR intake.
    Extracts PERSON, LOCATION, DATE, and CONTACT from complaint text.
    """
    entities = {"PERSON": [], "LOCATION": [], "DATE": [], "CONTACT": []}
    text_lower = text.lower()
    words = text.split()

    skip_words = {
        "yesterday", "today", "tomorrow", "evening", "morning",
        "night", "the", "a", "an", "this", "that", "my", "i",
        "he", "she", "they", "we", "on", "at", "in", "from",
    }

    # PERSON — first capitalised word not in skip list
    # FIX #8 — Original code used words.index(word) which returns the FIRST
    # occurrence of that value in the list. If a word appears multiple times
    # (e.g. "My name is Kumar Kumar"), index() returns index 0 for both,
    # so `idx + 1` always points to the wrong next word.
    # Fix: use enumerate() to get the correct positional index directly.
    for idx, word in enumerate(words):
        clean = word.strip(".,;:\"'")
        if (
            clean
            and clean[0].isupper()
            and clean.lower() not in skip_words
            and len(clean) > 2
            and not clean.isdigit()
        ):
            if idx + 1 < len(words):
                next_word = words[idx + 1].strip(".,;:\"'")
                if next_word and next_word[0].isupper() and next_word.lower() not in skip_words:
                    entities["PERSON"].append(f"{clean} {next_word}")
                else:
                    entities["PERSON"].append(clean)
            else:
                entities["PERSON"].append(clean)
            break

    # LOCATION — known city names + common location suffixes + residence keywords
    city_names = [
        "chennai", "delhi", "mumbai", "bangalore", "hyderabad", "kolkata",
        "pune", "ahmedabad", "jaipur", "surat", "lucknow", "kanpur",
        "nagpur", "indore", "thane", "bhopal", "visakhapatnam", "coimbatore",
    ]
    location_suffixes = [
        "nagar", "road", "street", "signal", "market", "station",
        "colony", "area", "village", "town", "cross", "layout", "park",
    ]
    residence_keywords = ["house", "home", "flat", "apartment", "shop", "office", "school", "temple", "hospital"]

    for city in city_names:
        if city in text_lower:
            entities["LOCATION"].append(city.title())
            break

    if not entities["LOCATION"]:
        # FIX #8 — same fix applied here: use enumerate() instead of index()
        for i, word in enumerate(words):
            clean = word.strip(".,;:\"'").lower()
            if any(clean.endswith(sfx) or clean == sfx for sfx in location_suffixes):
                if i > 0:
                    prev = words[i - 1].strip(".,;:\"'")
                    entities["LOCATION"].append(f"{prev} {word.strip('.,;:')}".title())
                else:
                    entities["LOCATION"].append(word.strip(".,;:").title())
                break

    if not entities["LOCATION"]:
        for kw in residence_keywords:
            match = re.search(rf'\b(\w+\s+)?{kw}\b', text_lower)
            if match:
                entities["LOCATION"].append(match.group().strip().title())
                break

    # DATE — order matters: most specific patterns first
    date_patterns = [
        r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}",
        r"\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}",
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}",
        r"\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}",
        r"\w+\s+\d{1,2},?\s+\d{4}",
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            entities["DATE"].append(match.group())
            break

    for word in ["yesterday", "today", "tomorrow"]:
        if word in text_lower and not entities["DATE"]:
            entities["DATE"].append(word)

    # CONTACT — phone numbers with optional separators
    contact_matches = re.findall(r"\+?\d[\d\s\-.]{8,20}\d", text)
    for match in contact_matches:
        digits = re.sub(r"[^0-9]", "", match)
        if len(digits) == 10 or len(digits) == 11 or len(digits) == 12:
            entities["CONTACT"].append(match.strip())
            break

    return entities