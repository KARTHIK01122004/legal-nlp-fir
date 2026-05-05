# utils/preprocessing.py
"""Advanced text preprocessing utilities for FIR system."""

import re
from typing import List, Optional


def clean_text(text: str, preserve_punctuation: bool = False) -> str:
    """
    Cleans raw complaint text for IPC keyword matching.
    
    Args:
        text: Raw text to clean
        preserve_punctuation: If True, keeps punctuation for context
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    
    if not preserve_punctuation:
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    else:
        text = re.sub(r"[^a-zA-Z0-9\s.,!?-]", "", text)
    
    return text.strip()


def tokenize_text(text: str) -> List[str]:
    """Split text into meaningful tokens."""
    cleaned = clean_text(text)
    return [token for token in cleaned.split() if len(token) > 2]


def remove_stopwords(text: str, stopwords: Optional[List[str]] = None) -> str:
    """Remove common words that don't add meaning."""
    if stopwords is None:
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'is', 'was', 'are', 'were', 'be', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might'
        }
    
    tokens = text.lower().split()
    return " ".join([t for t in tokens if t not in stopwords])


def normalize_text(text: str) -> str:
    """Comprehensive text normalization."""
    text = clean_text(text, preserve_punctuation=True)
    text = remove_stopwords(text)
    return text.strip()


def extract_sentences(text: str) -> List[str]:
    """Extract individual sentences from text."""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def get_text_statistics(text: str) -> dict:
    """Get basic statistics about text."""
    words = text.split()
    chars = len(text)
    unique_words = len(set(words))
    avg_word_length = chars / len(words) if words else 0
    
    return {
        "word_count": len(words),
        "character_count": chars,
        "unique_words": unique_words,
        "avg_word_length": round(avg_word_length, 2),
        "sentence_count": len(extract_sentences(text))
    }