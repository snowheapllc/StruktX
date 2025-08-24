from __future__ import annotations

from typing import Any, Dict


def format_prompt_preserving_json(template: str, variables: Dict[str, Any]) -> str:
    """Safely replace placeholders like {name} without breaking JSON examples.

    This performs a conservative literal replacement for exact tokens of the form
    '{key}', leaving any other braces untouched (including JSON braces). This avoids
    Python str.format() pitfalls with unescaped braces inside code examples.
    """
    result = template
    for key, value in variables.items():
        token = "{" + str(key) + "}"
        result = result.replace(token, str(value))
    return result


def render_prompt_with_safe_braces(template: str, variables: Dict[str, Any]) -> str:
    """Render a prompt where only known variables are formatted.

    Algorithm:
    1) Escape every brace by doubling it to neutralize formatting.
    2) For each variable key, revert '{{key}}' back to '{key}'.
    3) Call str.format(**variables) to substitute only allowed variables.

    This preserves JSON examples and any other brace usage while still allowing
    specific placeholders to be replaced. It avoids 'missing variables' errors
    in PromptTemplate-style engines encountering stray braces.
    """
    # Step 1: escape all braces
    escaped = template.replace("{", "{{").replace("}", "}}")
    # Step 2: un-escape allowed placeholders
    for key in variables.keys():
        escaped = escaped.replace("{{" + str(key) + "}}", "{" + str(key) + "}")
    # Step 3: format
    try:
        return escaped.format(**variables)
    except Exception:
        # Fallback to conservative replacer if format fails for any reason
        return format_prompt_preserving_json(template, variables)
