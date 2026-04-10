# utils/legal_ai_writer.py


def ai_legal_rewrite(text: str, complainant: str = "the complainant") -> str:
    """
    Rewrites a plain complaint into a formal FIR-style first-person narrative.
    Detects crime type and tailors the legal language accordingly.
    Called from app.py for Section 3 of the FIR document.
    """
    text = text.strip()
    t = text.lower()

    # ── Detect crime type and build narrative fragments ───────────
    if any(w in t for w in ["rape", "sexual assault", "molest", "outrage"]):
        crime_type = "sexual assault"
        action     = "subjected to sexual assault and harassment by the accused"
        urgency    = "Medical examination and preservation of forensic evidence is critical and must be done immediately."

    elif any(w in t for w in ["murder", "killed", "stabbed", "dead", "shot"]):
        crime_type = "murder / attempt to murder"
        action     = "attacked by the accused with intent to cause death"
        urgency    = "Immediate apprehension of the accused and preservation of the crime scene is required."

    elif any(w in t for w in ["snatched", "robbery", "robbed", "looted", "chain"]):
        crime_type = "robbery"
        action     = "subjected to robbery by the accused who forcibly snatched property from my person"
        urgency    = "Identification and arrest of the accused and recovery of stolen property is urgently required."

    elif any(w in t for w in ["stolen", "theft", "pickpocket", "burglary", "broke into"]):
        crime_type = "theft"
        action     = "had property dishonestly and unlawfully taken by an unknown accused person"
        urgency    = "Recovery of stolen property and identification of the accused is required."

    elif any(w in t for w in ["attack", "assault", "beat", "hit", "punch", "hurt", "injured", "slap"]):
        crime_type = "assault causing hurt"
        action     = "physically assaulted and caused grievous bodily harm by the accused"
        urgency    = "Medical evidence must be preserved and the accused apprehended without delay."

    elif any(w in t for w in ["fraud", "scam", "cheat", "fake", "upi", "online", "transfer", "otp"]):
        crime_type = "cheating and criminal fraud"
        action     = "cheated and defrauded through fraudulent and dishonest means by the accused"
        urgency    = "Digital and financial evidence including transaction records must be preserved immediately."

    elif any(w in t for w in ["threat", "blackmail", "intimidate", "warn", "death threat"]):
        crime_type = "criminal intimidation"
        action     = "unlawfully threatened and intimidated by the accused causing me serious alarm"
        urgency    = "All communication records including messages and call logs must be preserved as evidence."

    elif any(w in t for w in ["husband", "wife", "dowry", "domestic", "in-laws", "cruelty", "marital"]):
        crime_type = "domestic violence and cruelty"
        action     = "subjected to continuous cruelty, harassment and domestic violence by the accused"
        urgency    = "Medical examination of the complainant and statements of witnesses must be recorded promptly."

    elif any(w in t for w in ["missing", "kidnap", "abduct", "disappeared"]):
        crime_type = "missing person / kidnapping"
        action     = "wrongfully confined or kidnapped by unknown persons"
        urgency    = "Immediate search and rescue operation must be initiated without delay."

    else:
        crime_type = "cognizable offence"
        action     = "subjected to an unlawful act by the accused"
        urgency    = "A thorough investigation is required to establish the full facts of the case."

    # ── Extract a location hint from text ─────────────────────────
    location_hint = "an unspecified location"
    suffixes = ["nagar", "road", "signal", "market", "station", "street",
                "colony", "area", "village", "town", "cross"]
    words = t.split()
    for i, word in enumerate(words):
        clean = word.strip(".,")
        if any(sfx in clean for sfx in suffixes) and i > 0:
            prev = words[i - 1].strip(".,")
            location_hint = f"{prev} {clean}"
            break

    # ── Build formal narrative ─────────────────────────────────────
    narrative = (
        f"I, {complainant}, do hereby solemnly state and submit as follows:\n\n"
        f"That I approached this police station to lodge a formal complaint regarding an incident "
        f"of {crime_type} perpetrated against me.\n\n"
        f"That on the date and time of the incident, I was {action}. "
        f"The incident occurred at {location_hint}. "
        f"The full facts of the incident, as narrated by me, are as follows:\n\n"
        f'"{text}"\n\n'
        f"That the said act constitutes a cognizable offence under the relevant provisions of the "
        f"Indian Penal Code and I hereby request that an FIR be registered and appropriate legal "
        f"action be initiated against the accused.\n\n"
        f"{urgency}\n\n"
        f"I hereby declare that the above statement is true and correct to the best of my knowledge "
        f"and belief, and nothing material has been concealed therefrom."
    )

    return narrative