"""
legal_writer.py - Enhanced Legal Document Rewriting
=====================================================

Converts raw complaint text into formal FIR-style legal language.
Uses intelligent crime-type detection to tailor narratives appropriately.

Author: Legal NLP System
"""

from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

CRIME_PROFILES = {
    "theft": {
        "keywords": ["stolen", "steal", "theft", "missing", "pickpocket", "snatched", "robbed", "looted", "burglary"],
        "nature": "theft/robbery",
        "action_verb": "dishonestly and unlawfully took possession of",
        "urgency": "Recovery of stolen property and identification of the accused is urgently required.",
        "evidence_type": ["CCTV footage", "bank statements", "receipts"],
        "ipc_sections": ["379", "380", "381"],
    },
    "assault": {
        "keywords": ["attack", "beat", "hit", "slap", "punch", "kick", "hurt", "injured", "assault", "grievous"],
        "nature": "physical assault and causing hurt",
        "action_verb": "unlawfully assaulted and caused bodily harm to",
        "urgency": "Medical evidence should be preserved and the accused apprehended without delay.",
        "evidence_type": ["medical examination", "photographs of injuries", "medical reports"],
        "ipc_sections": ["323", "324", "325", "326"],
    },
    "fraud": {
        "keywords": ["fraud", "cheat", "scam", "deceive", "fake", "false", "trick", "online", "upi", "transfer"],
        "nature": "cheating and criminal fraud",
        "action_verb": "fraudulently and dishonestly induced",
        "urgency": "Digital/financial evidence including transaction records must be preserved immediately.",
        "evidence_type": ["bank statements", "transaction records", "digital communication"],
        "ipc_sections": ["419", "420", "406"],
    },
    "threat": {
        "keywords": ["threat", "threaten", "blackmail", "intimidate", "scare", "warn", "death threat", "extort"],
        "nature": "criminal intimidation and threatening",
        "action_verb": "unlawfully threatened and intimidated",
        "urgency": "All communication records (messages, calls) should be preserved as evidence.",
        "evidence_type": ["SMS/WhatsApp messages", "call logs", "email records"],
        "ipc_sections": ["503", "504", "506"],
    },
    "domestic": {
        "keywords": ["husband", "wife", "domestic", "dowry", "in-laws", "cruelty", "marital", "family", "torture"],
        "nature": "domestic violence and cruelty",
        "action_verb": "subjected the complainant to cruelty and harassment",
        "urgency": "Medical examination of the complainant and witness statements must be recorded promptly.",
        "evidence_type": ["medical examination", "witness statements", "photographs"],
        "ipc_sections": ["498A", "337", "338"],
    },
    "sexual": {
        "keywords": ["rape", "sexual", "molest", "modesty", "grope", "outrage", "harass", "eve tease", "stalk"],
        "nature": "sexual assault/harassment",
        "action_verb": "committed an act of sexual violence/harassment against",
        "urgency": "Medical examination and preservation of forensic evidence is critical and time-sensitive.",
        "evidence_type": ["medical forensic examination", "CCTV footage", "witness statements"],
        "ipc_sections": ["354", "375", "376", "509"],
    },
    "property_damage": {
        "keywords": ["damage", "destroy", "vandal", "broke", "smashed", "burned", "fire", "trespass"],
        "nature": "malicious damage to property and trespass",
        "action_verb": "willfully and maliciously caused damage to the property of",
        "urgency": "Photographic evidence of the damage should be collected immediately.",
        "evidence_type": ["photographs", "repair estimates", "witness statements"],
        "ipc_sections": ["425", "426", "427"],
    },
    "cyber": {
        "keywords": ["hacking", "cyber", "website", "data", "password", "account", "phishing", "malware"],
        "nature": "cybercrime and unauthorized access",
        "action_verb": "unlawfully accessed and manipulated digital assets belonging to",
        "urgency": "Digital evidence must be preserved by an authorized technician immediately.",
        "evidence_type": ["digital forensics", "server logs", "IP records"],
        "ipc_sections": ["66", "66B", "66C", "66D"],
    },
}

DEFAULT_PROFILE = {
    "nature": "cognizable offence",
    "action_verb": "committed an unlawful act against",
    "urgency": "A thorough investigation is required to establish the facts of the case.",
    "evidence_type": [],
    "ipc_sections": [],
}


def detect_crime_type(text: str) -> Tuple[Dict, float]:
    """Detects crime type with confidence score."""
    if not text or not text.strip():
        return DEFAULT_PROFILE, 0.0
        
    text_lower = text.lower()
    best_match = None
    best_count = 0

    for crime_type, profile in CRIME_PROFILES.items():
        count = sum(1 for kw in profile["keywords"] if kw in text_lower)
        if count > best_count:
            best_count = count
            best_match = profile

    confidence = (best_count / len(best_match["keywords"]) * 100) if best_match else 0.0
    return (best_match if best_match else DEFAULT_PROFILE), confidence


def rewrite_legal_style(description: str, complainant: str = "the complainant", include_evidence: bool = True) -> str:
    """Converts complaint into formal FIR-style legal narrative."""
    if not description or not description.strip():
        return "No description provided."
    
    try:
        profile, confidence = detect_crime_type(description)
        sentences = description.strip().rstrip(".")
        
        evidence_section = ""
        if include_evidence and profile.get("evidence_type"):
            evidence_items = ", ".join(profile["evidence_type"][:3])
            evidence_section = f"\n\nRecommended Evidence: {evidence_items}."
        
        legal_text = f"""It is respectfully submitted that the complainant, {complainant}, approached the police station to lodge a formal complaint regarding an incident of {profile['nature']}.

According to the statement furnished by the complainant, the accused person(s) {profile['action_verb']} {complainant} in the following manner:

"{sentences}."

The said act constitutes a cognizable offence under the relevant provisions of the Indian Penal Code. The complainant avers that the occurrence took place under circumstances that warrant immediate police intervention and legal action.

{profile['urgency']}{evidence_section}

The complainant has provided this statement of their own free will and requests that an FIR be registered and appropriate action be initiated against the accused."""
        
        return legal_text.strip()
    except Exception as e:
        logger.error(f"Error in rewrite_legal_style: {e}")
        return description


def get_applicable_ipc_sections(description: str) -> list:
    """Get applicable IPC sections based on crime type."""
    profile, _ = detect_crime_type(description)
    return profile.get("ipc_sections", [])


def extract_evidence_recommendations(description: str) -> list:
    """Extract recommended evidence collection items."""
    profile, _ = detect_crime_type(description)
    return profile.get("evidence_type", [])


def get_crime_type_confidence(description: str) -> Dict:
    """Get crime type and confidence details."""
    profile, confidence = detect_crime_type(description)
    return {
        "nature": profile.get("nature", "unknown"),
        "confidence": round(confidence, 2),
        "urgency": profile.get("urgency", ""),
        "ipc_sections": profile.get("ipc_sections", []),
    }


def classify_intent(text: str) -> str:
    """
    Classify incident intent and return appropriate incident type string.
    
    Maps detected crime types to standardized incident categories.
    """
    if not text or not text.strip():
        return "Other"
    
    profile, confidence = detect_crime_type(text)
    
    # Map crime profile keys to INCIDENT_OPTIONS
    crime_to_incident = {
        "theft": "Theft / Robbery",
        "assault": "Assault", 
        "fraud": "Cyber Crime / Fraud",
        "threat": "Criminal Intimidation",
        "domestic": "Domestic Violence",
        "sexual": "Sexual Offence",
        "murder": "Murder / Attempt to Murder",
        "kidnapping": "Missing Person / Kidnapping",
        "property": "Property Damage / Trespass",
        "harassment": "Harassment",
    }
    
    # Try to match by crime type key
    crime_key = None
    for key in CRIME_PROFILES.keys():
        if key in profile.get("nature", "").lower():
            crime_key = key
            break
    
    if crime_key and crime_key in crime_to_incident:
        return crime_to_incident[crime_key]
    
    # Fallback: check keywords for better matching
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["sexual", "rape", "molest"]):
        return "Sexual Offence"
    elif any(kw in text_lower for kw in ["murder", "kill", "attempt to murder"]):
        return "Murder / Attempt to Murder"
    elif any(kw in text_lower for kw in ["kidnap", "missing", "abduct"]):
        return "Missing Person / Kidnapping"
    elif any(kw in text_lower for kw in ["damage", "vandal", "trespass"]):
        return "Property Damage / Trespass"
    
    return "Other"