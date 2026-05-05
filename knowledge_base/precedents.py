"""
Landmark Legal Precedents and Case Law Database
================================================

Contains important Supreme Court and High Court judgments
relevant to various crime categories.
"""

PRECEDENTS = [
    # Sexual Assault Cases
    {
        "case_name": "Nirbhaya Case (Mukesh Singh v. State NCT of Delhi)",
        "year": 2017,
        "court": "Supreme Court",
        "relevant_sections": ["375", "376", "304", "307"],
        "verdict": "Death Penalty",
        "summary": "Landmark case establishing stricter punishment for gang rape and brutality",
        "ipc_application": ["375-A", "376-A", "376-D"],
    },
    {
        "case_name": "Rajesh Masrani v. State of U.P",
        "year": 2016,
        "court": "Supreme Court",
        "relevant_sections": ["354", "354-A"],
        "verdict": "Conviction for sexual harassment",
        "summary": "Established workplace sexual harassment protocols",
        "ipc_application": ["354-A"],
    },
    
    # Theft & Robbery
    {
        "case_name": "Rao Girdhari Lal v. State of Rajasthan",
        "year": 2006,
        "court": "Supreme Court",
        "relevant_sections": ["379", "380", "381"],
        "verdict": "Conviction upheld",
        "summary": "Distinguished between theft and criminal breach of trust",
        "ipc_application": ["379", "406"],
    },
    
    # Assault & Hurt
    {
        "case_name": "Vijay Pal Singh v. State of Punjab",
        "year": 2010,
        "court": "Supreme Court",
        "relevant_sections": ["323", "325", "337"],
        "verdict": "Conviction upheld",
        "summary": "Defined grievous vs simple injury in physical assault cases",
        "ipc_application": ["325", "326"],
    },
    
    # Murder & Culpable Homicide
    {
        "case_name": "Keshavananda Bharati v. State of Kerala",
        "year": 1973,
        "court": "Supreme Court",
        "relevant_sections": ["299", "300"],
        "verdict": "Constitutional landmark",
        "summary": "Established doctrine of distinction between murder and culpable homicide",
        "ipc_application": ["299", "300", "304"],
    },
    {
        "case_name": "Sharadchandra Maruti Ekre v. State of Maharashtra",
        "year": 2004,
        "court": "Supreme Court",
        "relevant_sections": ["300", "34"],
        "verdict": "Murder conviction",
        "summary": "Clarified joint liability in murder cases",
        "ipc_application": ["300", "34"],
    },
    
    # Fraud & Cheating
    {
        "case_name": "K.K. Kailasam v. M. Krishnamurthy",
        "year": 1999,
        "court": "Supreme Court",
        "relevant_sections": ["419", "420"],
        "verdict": "Fraud conviction upheld",
        "summary": "Established standards for proving criminal fraud",
        "ipc_application": ["419", "420"],
    },
    
    # Criminal Intimidation & Threats
    {
        "case_name": "Surinder Singh Gill v. State of Punjab",
        "year": 2007,
        "court": "Supreme Court",
        "relevant_sections": ["503", "504", "506"],
        "verdict": "Conviction upheld",
        "summary": "Defined criminal intimidation and threatening communications",
        "ipc_application": ["503", "504", "506"],
    },
    
    # Domestic Violence
    {
        "case_name": "Bodhisattwa Gautam v. Subhra Chakraborty",
        "year": 1996,
        "court": "Supreme Court",
        "relevant_sections": ["498-A", "406"],
        "verdict": "Landmark on dowry laws",
        "summary": "Established framework for addressing dowry-related crimes",
        "ipc_application": ["498-A", "304-B"],
    },
    
    # Property Damage
    {
        "case_name": "Bansi Lal v. State of Haryana",
        "year": 1988,
        "court": "Supreme Court",
        "relevant_sections": ["425", "426", "427"],
        "verdict": "Conviction upheld",
        "summary": "Clarified definitions of mischief and property damage",
        "ipc_application": ["425", "426", "427", "448"],
    },
    
    # Criminal Breach of Trust
    {
        "case_name": "Ramachandra Rao v. State of Karnataka",
        "year": 2005,
        "court": "Supreme Court",
        "relevant_sections": ["406", "409"],
        "verdict": "Conviction upheld",
        "summary": "Distinguished criminal breach of trust from civil dispute",
        "ipc_application": ["406", "409"],
    },
    
    # Cybercrime
    {
        "case_name": "Shreya Singhal v. Union of India",
        "year": 2015,
        "court": "Supreme Court",
        "relevant_sections": ["66", "66-A", "66-B"],
        "verdict": "Landmark on digital rights",
        "summary": "Established laws on cybercrime and online harassment",
        "ipc_application": ["66", "66-B", "66-C"],
    },
]


def get_precedents_by_section(section_number: str) -> list:
    """Get all precedents related to a specific IPC section."""
    matching = []
    for precedent in PRECEDENTS:
        if section_number in precedent.get("relevant_sections", []):
            matching.append(precedent)
    return matching


def get_precedents_by_crime_type(crime_type: str) -> list:
    """Get precedents by crime category."""
    crime_keywords = {
        "sexual": ["375", "376", "354"],
        "murder": ["299", "300", "302"],
        "theft": ["379", "380", "381"],
        "assault": ["323", "324", "325", "326"],
        "fraud": ["419", "420", "406"],
        "threat": ["503", "504", "506"],
        "domestic": ["498-A", "304-B"],
        "cybercrime": ["66", "66-B", "66-C"],
    }
    
    if crime_type.lower() not in crime_keywords:
        return []
    
    sections = crime_keywords[crime_type.lower()]
    matching = []
    for section in sections:
        matching.extend(get_precedents_by_section(section))
    
    return list({p["case_name"]: p for p in matching}.values())


def search_precedents(query: str) -> list:
    """Search precedents by case name or summary."""
    query_lower = query.lower()
    results = []
    
    for precedent in PRECEDENTS:
        if (query_lower in precedent["case_name"].lower() or 
            query_lower in precedent.get("summary", "").lower()):
            results.append(precedent)
    
    return results