# utils/document_generator.py

import datetime
import random
from utils.legal_writer import rewrite_legal_style


def generate_fir_number() -> str:
    year = datetime.datetime.now().year
    return f"FIR-{year}-{random.randint(1000, 9999)}"


def generate_document(data: dict) -> str:
    """
    Generates a plain-text FIR document from structured case data.
    Used as a standalone document generator (separate from app.py FIR builder).

    Expected keys: complainant, contact, date, location, description, ipc_sections
    """
    narrative = rewrite_legal_style(data.get("description", ""))

    return f"""
╔══════════════════════════════════════════════════════════════════╗
║              FIRST INFORMATION REPORT (FIR)                    ║
║                  Under Section 154 Cr.P.C                      ║
╚══════════════════════════════════════════════════════════════════╝

  FIR Number    : {generate_fir_number()}
  Date Filed    : {datetime.date.today().strftime('%d %B %Y')}

══════════════════════════════════════════════════════════════════
  SECTION 1 — COMPLAINANT DETAILS
══════════════════════════════════════════════════════════════════

  Name          : {data.get('complainant', 'N/A')}
  Contact       : {data.get('contact', 'Not provided')}
  Incident Date : {data.get('date', 'N/A')}
  Location      : {data.get('location', 'N/A')}

══════════════════════════════════════════════════════════════════
  SECTION 2 — COMPLAINT NARRATIVE
══════════════════════════════════════════════════════════════════

{narrative}

══════════════════════════════════════════════════════════════════
  SECTION 3 — APPLICABLE IPC SECTIONS
══════════════════════════════════════════════════════════════════

{data.get('ipc_sections', 'Not specified')}

══════════════════════════════════════════════════════════════════
  SECTION 4 — DECLARATION
══════════════════════════════════════════════════════════════════

  I hereby declare that the above statement is true to the best
  of my knowledge and belief.

  Signature : {data.get('complainant', '')}
  Date      : {datetime.date.today().strftime('%d/%m/%Y')}

══════════════════════════════════════════════════════════════════
  * Requires officer verification and official stamp.
══════════════════════════════════════════════════════════════════
""".strip()