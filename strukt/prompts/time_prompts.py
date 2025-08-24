from __future__ import annotations

TIME_PROBE_PROMPT_TEMPLATE = """You are assisting a time query.
Known facts (from memory):
{mem_block}

Task: If any item indicates the user's city or timezone, return exactly one line: MEM_PROBE: <city-or-timezone>.
If not enough information, return exactly: MEM_PROBE: none."""
