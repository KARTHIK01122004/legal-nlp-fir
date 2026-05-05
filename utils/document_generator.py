# utils/document_generator.py
"""
Advanced Document Generation System
====================================

Generates formal FIR documents with comprehensive validation
and error handling.
"""

import datetime
import random
import logging
from typing import Dict, Optional, List, Tuple
from utils.legal_writer import (
    rewrite_legal_style,
    get_applicable_ipc_sections,
    extract_evidence_recommendations,
    get_crime_type_confidence,
)

logger = logging.getLogger(__name__)

# FIR document counter for unique numbering
_fir_counter = random.randint(1000, 9999)


def generate_fir_number() -> str:
    """Generate a unique FIR number in format FIR-YEAR-NNNN."""
    global _fir_counter
    year = datetime.datetime.now().year
    _fir_counter += 1
    if _fir_counter > 9999:
        _fir_counter = 1000
    return f"FIR-{year}-{_fir_counter:04d}"


def validate_document_data(data: Dict) -> Tuple[bool, List[str]]:
    """
    Validates required fields in document data.
    
    Args:
        data: Document data dictionary
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    required_fields = {
        "complainant": "Complainant name",
        "description": "Complaint description",
    }
    
    optional_fields = {
        "contact": "Contact information",
        "date": "Incident date",
        "location": "Incident location",
    }
    
    # Check required fields
    for field, label in required_fields.items():
        if not data.get(field) or not str(data.get(field)).strip():
            errors.append(f"Missing required field: {label}")
    
    # Log warnings for optional fields
    for field, label in optional_fields.items():
        if not data.get(field):
            logger.warning(f"Optional field not provided: {label}")
    
    return len(errors) == 0, errors


def generate_document(data: Dict, include_evidence: bool = True) -> Tuple[str, Dict]:
    """
    Generates a complete FIR document from structured case data.
    
    Args:
        data: Document data with keys:
            - complainant (required): Complainant name
            - description (required): Complaint description
            - contact (optional): Contact information
            - date (optional): Incident date
            - location (optional): Incident location
            - ipc_sections (optional): IPC sections string
            
        include_evidence: Include evidence recommendations
        
    Returns:
        Tuple of (document_string, metadata_dict)
    """
    # Validate input data
    is_valid, errors = validate_document_data(data)
    if not is_valid:
        logger.error(f"Document validation failed: {errors}")
        error_msg = "Validation Error:\n" + "\n".join(errors)
        return error_msg, {"success": False, "errors": errors}
    
    try:
        # Extract key fields with defaults
        complainant = str(data.get("complainant", "N/A")).strip()
        description = str(data.get("description", "")).strip()
        contact = str(data.get("contact", "Not provided")).strip()
        incident_date = str(data.get("date", "Not specified")).strip()
        location = str(data.get("location", "Not specified")).strip()
        
        # Generate legal narrative
        narrative = rewrite_legal_style(
            description,
            complainant=complainant,
            include_evidence=include_evidence
        )
        
        # Get applicable sections if not provided
        ipc_sections_str = data.get("ipc_sections", "")
        if not ipc_sections_str:
            sections = get_applicable_ipc_sections(description)
            ipc_sections_str = ", ".join(sections) if sections else "To be determined"
        
        # Get crime type confidence
        crime_info = get_crime_type_confidence(description)
        
        # Generate evidence recommendations
        evidence_list = ""
        if include_evidence:
            evidence = extract_evidence_recommendations(description)
            if evidence:
                evidence_list = "\n  Evidence Items:\n  " + "\n  ".join(
                    f"• {item}" for item in evidence
                )
        
        # Generate FIR number and timestamps
        fir_number = generate_fir_number()
        filed_date = datetime.date.today().strftime("%d %B %Y")
        filed_date_short = datetime.date.today().strftime("%d/%m/%Y")
        
        # Construct document
        document = f"""
╔══════════════════════════════════════════════════════════════════╗
║              FIRST INFORMATION REPORT (FIR)                    ║
║                  Under Section 154 Cr.P.C                      ║
╚══════════════════════════════════════════════════════════════════╝

  FIR Number         : {fir_number}
  Date Filed         : {filed_date}
  Report Type        : {crime_info.get("nature", "Cognizable Offence")}
  Urgency Level      : {crime_info.get("urgency", "Normal")}

══════════════════════════════════════════════════════════════════
  SECTION 1 — COMPLAINANT DETAILS
══════════════════════════════════════════════════════════════════

  Name               : {complainant}
  Contact            : {contact}
  Incident Date      : {incident_date}
  Location           : {location}

══════════════════════════════════════════════════════════════════
  SECTION 2 — COMPLAINT NARRATIVE
══════════════════════════════════════════════════════════════════

{narrative}

══════════════════════════════════════════════════════════════════
  SECTION 3 — APPLICABLE IPC SECTIONS
══════════════════════════════════════════════════════════════════

  Sections          : {ipc_sections_str}
  Crime Type        : {crime_info.get("nature", "Unknown")}
  Confidence Score  : {crime_info.get("confidence", 0)}%

══════════════════════════════════════════════════════════════════
  SECTION 4 — RECOMMENDED EVIDENCE
══════════════════════════════════════════════════════════════════

{evidence_list if evidence_list else "  No specific evidence items identified"}

══════════════════════════════════════════════════════════════════
  SECTION 5 — DECLARATION
══════════════════════════════════════════════════════════════════

  I hereby declare that the above statement is true to the best
  of my knowledge and belief. The facts stated herein are within
  my personal knowledge and are true and correct.

  Signature / Mark  : {complainant}
  Date             : {filed_date_short}

══════════════════════════════════════════════════════════════════
  SECTION 6 — INVESTIGATING OFFICER NOTES
══════════════════════════════════════════════════════════════════

  [ To be filled by IO ]
  
  Officer Name      : ________________________
  Badge Number      : ________________________
  Station           : ________________________
  Date              : ________________________

══════════════════════════════════════════════════════════════════
Generated by Legal NLP FIR System - {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}
"""
        
        metadata = {
            "success": True,
            "fir_number": fir_number,
            "crime_type": crime_info.get("nature"),
            "confidence": crime_info.get("confidence"),
            "ipc_sections": ipc_sections_str,
            "generated_at": filed_date_short,
        }
        
        logger.info(f"FIR generated successfully: {fir_number}")
        return document.strip(), metadata

    except Exception as e:
        logger.error(f"Error generating FIR document: {e}", exc_info=True)
        return f"Error generating document: {str(e)}", {
            "success": False,
            "error": str(e),
        }


def generate_summary(data: Dict) -> str:
    """Generate a brief human-readable summary of the complaint."""
    crime_info  = get_crime_type_confidence(data.get("description", ""))
    complainant = data.get("complainant", "Unknown")
    return (
        f"COMPLAINT SUMMARY\n"
        f"═════════════════\n"
        f"Complainant : {complainant}\n"
        f"Crime Type  : {crime_info.get('nature', 'Unknown')}\n"
        f"Location    : {data.get('location', 'Not specified')}\n"
        f"Date        : {data.get('date', 'Not specified')}\n"
    )