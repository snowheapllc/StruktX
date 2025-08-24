from __future__ import annotations

MEMORY_EXTRACTION_PROMPT_TEMPLATE = """You will extract useful, durable memory entries from a user interaction.
Existing memories (avoid duplicates):
{existing_context}
Input text: {text}
Final response: {response}

STRICT CRITERIA:
- Only extract durable user information (e.g., preferences, recurring behaviors, stable locations).
- Do NOT extract transient questions, requests, or external facts (e.g., 'best place...', 'what is...').
- Do NOT extract memories that duplicate or closely resemble existing memories listed above.
- If nothing durable is present, return an empty list.

Return JSON with an array 'items', where each item has: category (one of location, preference, behavior, context, other),
key (short identifier), value (brief content), and optional context. If none, return items: []."""
