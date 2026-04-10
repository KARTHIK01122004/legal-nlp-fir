"""
legal_writer.py
---------------
Rewrites raw complaint text into formal legal FIR-style language.
Uses crime-type detection to tailor the narrative appropriately.
"""

from datetime import datetime

# Crime type profiles: keywords → narrative template fragments
CRIME_PROFILES = {
    "theft": {
        "keywords": ["stolen", "steal", "theft", "missing", "pickpocket", "snatched", "robbed", "looted"],
        "nature": "theft/robbery",
        "action_verb": "dishonestly and unlawfully took possession of",
        "urgency": "Recovery of stolen property and identification of the accused is urgently required.",
    },
    "assault": {
        "keywords": ["attack", "beat", "hit", "slap", "punch", "kick", "hurt", "injured", "assault", "fight", "brawl"],
        "nature": "physical assault and causing hurt",
        "action_verb": "unlawfully assaulted and caused bodily harm to",
        "urgency": "Medical evidence should be preserved and the accused apprehended without delay.",
    },
    "fraud": {
        "keywords": ["fraud", "cheat", "scam", "deceive", "fake", "false", "trick", "online", "upi", "transfer", "investment"],
        "nature": "cheating and criminal fraud",
        "action_verb": "fraudulently and dishonestly induced",
        "urgency": "Digital/financial evidence including transaction records must be preserved immediately.",
    },
    "threat": {
        "keywords": ["threat", "threaten", "blackmail", "intimidate", "scare", "warn", "kill threat", "death threat"],
        "nature": "criminal intimidation and threatening",
        "action_verb": "unlawfully threatened and intimidated",
        "urgency": "All communication records (messages, calls) should be preserved as evidence.",
    },
    "domestic": {
        "keywords": ["husband", "wife", "domestic", "dowry", "in-laws", "cruelty", "marital", "family"],
        "nature": "domestic violence and cruelty",
        "action_verb": "subjected the complainant to cruelty and harassment",
        "urgency": "Medical examination of the complainant and witness statements must be recorded promptly.",
    },
    "sexual": {
        "keywords": ["rape", "sexual", "molest", "modesty", "grope", "outrage", "harass", "eve tease", "stalk"],
        "nature": "sexual assault/harassment",
        "action_verb": "committed an act of sexual violence/harassment against",
        "urgency": "Medical examination and preservation of forensic evidence is critical.",
    },
    "property_damage": {
        "keywords": ["damage", "destroy", "vandal", "broke", "smashed", "burned", "fire", "trespass"],
        "nature": "malicious damage to property and trespass",
        "action_verb": "willfully and maliciously caused damage to the property of",
        "urgency": "Photographic evidence of the damage should be collected immediately.",
    },
}

DEFAULT_PROFILE = {
    "nature": "cognizable offence",
    "action_verb": "committed an unlawful act against",
    "urgency": "A thorough investigation is required to establish the facts of the case.",
}


def detect_crime_type(text: str) -> dict:
    """Detects the most likely crime type from the complaint text."""
    text_lower = text.lower()
    best_match = None
    best_count = 0

    for crime_type, profile in CRIME_PROFILES.items():
        count = sum(1 for kw in profile["keywords"] if kw in text_lower)
        if count > best_count:
            best_count = count
            best_match = profile

    return best_match if best_match else DEFAULT_PROFILE


def rewrite_legal_style(description: str, complainant: str = "the complainant") -> str:
    """
    Rewrites a plain complaint description into formal FIR-style legal language.
    Adapts the narrative based on detected crime type.
    """
    profile = detect_crime_type(description)
    sentences = description.strip().rstrip(".")

    legal_text = f"""
It is respectfully submitted that the complainant, {complainant}, approached the police station 
to lodge a formal complaint regarding an incident of {profile['nature']}.

According to the statement furnished by the complainant, the accused person(s) 
{profile['action_verb']} {complainant} in the following manner:

"{sentences}."

The said act constitutes a cognizable offence under the relevant provisions of the Indian Penal 
Code. The complainant avers that the occurrence took place under circumstances that warrant 
immediate police intervention and legal action.

{profile['urgency']}

The complainant has provided this statement of their own free will and requests that an FIR be 
registered and appropriate action be initiated against the accused under the applicable sections 
of the IPC.
""".strip()

    return legal_text