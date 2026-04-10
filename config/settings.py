import datetime

FIR_TEMPLATE = """
FIRST INFORMATION REPORT

Complainant: {complainant}
Date: {date}
Location: {location}

Incident:
{description}

Applicable IPC Sections:
{ipc_sections}
"""

COMPLAINT_TEMPLATE = """
LEGAL COMPLAINT

Name: {complainant}
Date: {date}

Complaint Details:
{description}

Relevant IPC Sections:
{ipc_sections}
"""

LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Bengali": "bn",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Odia": "or",
    "Assamese": "as"
}

DEFAULTS = {
    "description": "",
    "auto_name": "",
    "auto_location": "",
    "auto_date": datetime.date.today(),
}