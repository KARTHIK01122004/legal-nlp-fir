# utils/preprocessing.py

import re


def clean_text(text: str) -> str:
    """
    Cleans raw complaint text for IPC keyword matching.
    Lowercases, removes extra whitespace and special characters.
    Called before passing text to search_ipc() in ipc_search_tool.py.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()