# tools/ipc_search_tool.py
"""
Enhanced IPC Section Matching Tool
===================================

Provides intelligent matching of complaint text to Indian Penal Code sections
using keyword scoring, fuzzy matching, and caching for performance.
"""

from knowledge_base.ipc_sections import IPC_SECTIONS
from difflib import SequenceMatcher
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# Simple cache for search results
_search_cache = {}


def _similarity_ratio(a: str, b: str) -> float:
    """Calculate fuzzy match ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def search_ipc(query: str, use_fuzzy: bool = True) -> dict:
    """
    Keyword-based IPC section search with optional fuzzy matching.
    
    Args:
        query: Search query text
        use_fuzzy: Enable fuzzy matching for typos
        
    Returns:
        Best matching IPC section dictionary
    """
    if not query:
        return _get_default_section()
    
    # Check cache
    cache_key = f"search:{query}:{use_fuzzy}"
    if cache_key in _search_cache:
        return _search_cache[cache_key]
    
    query_lower = query.lower()
    best_match = None
    best_score = 0

    for section in IPC_SECTIONS:
        score = 0
        
        # Match against keywords (highest weight)
        for kw in section.get("keywords", []):
            if kw in query_lower:
                score += 3
            elif use_fuzzy and _similarity_ratio(kw, query_lower) > 0.7:
                score += 2
        
        # Match against description
        for word in query_lower.split():
            if len(word) > 3 and word in section["description"].lower():
                score += 1
        
        # Match against title
        if query_lower in section["title"].lower():
            score += 5

        if score > best_score:
            best_score = score
            best_match = section

    result = best_match if best_match else _get_default_section()
    _search_cache[cache_key] = result
    return result


def match_top_ipc(text: str, top_n: int = 3, use_fuzzy: bool = True) -> List[Dict]:
    """
    Returns top_n IPC sections scored by keyword + title + description match.
    
    Args:
        text: Complaint text to analyze
        top_n: Number of top matches to return
        use_fuzzy: Enable fuzzy matching
        
    Returns:
        List of matched IPC sections with scores
    """
    if not text:
        return []
    
    # Check cache
    cache_key = f"match_top:{text[:50]}:{top_n}:{use_fuzzy}"
    if cache_key in _search_cache:
        return _search_cache[cache_key]
    
    text_lower = text.lower()
    scored = []

    for sec in IPC_SECTIONS:
        score = 0
        
        # Keyword matches (highest weight)
        for kw in sec.get("keywords", []):
            if kw in text_lower:
                score += 3
            elif use_fuzzy and _similarity_ratio(kw, text_lower) > 0.75:
                score += 1
        
        # Title matches
        for word in sec["title"].lower().split():
            if len(word) > 3 and word in text_lower:
                score += 2
        
        # Description matches
        for word in sec["description"].lower().split():
            if len(word) > 4 and word in text_lower:
                score += 1
        
        # Category match bonus
        if sec.get("category") and any(
            cat_word in text_lower 
            for cat_word in sec["category"].lower().split()
        ):
            score += 1
        
        if score > 0:
            scored.append({**sec, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    result = scored[:top_n]
    
    _search_cache[cache_key] = result
    return result


def format_ipc_for_fir(matches: List[Dict]) -> str:
    """
    Formats a list of IPC matches into a readable FIR section string.
    
    Args:
        matches: List of matched IPC sections
        
    Returns:
        Formatted string for FIR document
    """
    if not matches:
        return "  No specific IPC section identified. Manual review required."
    
    formatted_sections = []
    for m in matches:
        section_text = (
            f"  Section {m['section']} IPC — {m['title']}\n"
            f"  {m['description']}\n"
            f"  Punishment: {m['punishment']}"
        )
        if m.get("score"):
            section_text += f"\n  [Match Score: {m['score']}]"
        formatted_sections.append(section_text)
    
    return "\n\n".join(formatted_sections)


def _get_default_section() -> Dict:
    """Return a default section for no match scenarios."""
    return {
        "section": "0",
        "title": "General Complaint",
        "description": "No matching IPC section found. Manual categorization required.",
        "punishment": "N/A",
        "keywords": [],
        "category": "Unclassified",
    }


def clear_search_cache():
    """Clear the search cache."""
    global _search_cache
    _search_cache.clear()
    logger.info("IPC search cache cleared")


def search_by_section_number(section_number: str) -> Dict:
    """Search for a specific IPC section by number."""
    section_num = str(section_number).strip()
    for section in IPC_SECTIONS:
        if section["section"] == section_num:
            return section
    return _get_default_section()


def search_by_category(category: str) -> List[Dict]:
    """Get all sections in a specific category."""
    cat_lower = category.lower()
    return [
        sec for sec in IPC_SECTIONS 
        if cat_lower in sec.get("category", "").lower()
    ]