# tools/ipc_search_tool.py

from knowledge_base.ipc_sections import IPC_SECTIONS


def search_ipc(query: str) -> dict:
    """
    Keyword-based IPC section search.
    Scores each section by how many query words appear in its description + keywords.
    Returns the best matching section dict.
    """
    query = query.lower()
    best_match = None
    best_score = 0

    for section in IPC_SECTIONS:
        score = 0
        # Match against description
        score += sum(1 for word in query.split() if word in section["description"].lower())
        # Match against keywords (weighted higher)
        score += sum(2 for kw in section.get("keywords", []) if kw in query)

        if score > best_score:
            best_score = score
            best_match = section

    if best_match is None:
        return {
            "section": "0",
            "title": "General Complaint",
            "description": "No matching IPC section found",
            "punishment": "N/A",
        }

    return best_match


def match_top_ipc(text: str, top_n: int = 3) -> list:
    """
    Returns top_n IPC sections scored by keyword + title + description match.
    Used for displaying multiple applicable sections in the FIR.
    """
    text_lower = text.lower()
    scored = []

    for sec in IPC_SECTIONS:
        score = 0
        for kw in sec.get("keywords", []):
            if kw in text_lower:
                score += 3
        for word in sec["title"].lower().split():
            if len(word) > 3 and word in text_lower:
                score += 2
        for word in sec["description"].lower().split():
            if len(word) > 4 and word in text_lower:
                score += 1
        if score > 0:
            scored.append({**sec, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]


def format_ipc_for_fir(matches: list) -> str:
    """Formats a list of IPC matches into a readable FIR section string."""
    if not matches:
        return "  No specific IPC section identified. Manual review required."
    return "\n\n".join(
        f"  Section {m['section']} IPC — {m['title']}\n"
        f"  {m['description']}\n"
        f"  Punishment: {m['punishment']}"
        for m in matches
    )