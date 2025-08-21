from __future__ import annotations

from typing import List, Optional
from datetime import datetime, timezone

from pydantic import BaseModel

from ..interfaces import Classifier, LLMClient
from ..types import InvocationState, QueryClassification, StruktQueryEnum


class ClassificationOut(BaseModel):
    query_types: List[str]
    confidences: List[float]
    parts: List[str]


DEFAULT_CLASSIFIER_TEMPLATE = """
Current UTC: {current_time}

You are an intent classifier.

Goal:
- Decide if the user input contains one or several distinct, actionable requests
- Extract minimal, non-overlapping spans for each request
- Assign the best-fitting type to each span

Constraints:
- Use only types from {allowed_types}. If none fits, use "general".
- Return at most {max_parts} parts.
- Output arrays must have equal length and aligned indices.
- Confidence values must be within [0.0, 1.0].
- Do not include any prose or explanation or other text in your response.

Policies:
- Prefer minimal parts; avoid creating multiple parts for closely related statements about the same fact or the same request. (e.g. "I live in Beirut" and "What's the time in Beirut?" should be one part)
- Discard parts that are not actionable or not relevant to the user's request. (e.g. "I live in Dubai" is not actionable, so it should be discarded)
- Parts that may lead to useful memory extraction should be kept and classified as "memory_extraction". (e.g. "I like Pizza" should be kept)
- Do not over-fragment tightly coupled instructions aimed at a single target.
- Prefer spans that can be executed or answered directly.
- If prior user preferences reasonably disambiguate a vague request, refine the span text accordingly.
- If the user asks about their own stored information, preferences, routines, or habits, classify as "no-op-memory-response".

Output specification (must be valid JSON):
- query_types: array of strings
- confidences: array of numbers between 0 and 1
- parts: array of strings
- The three arrays must have the same length and aligned indices.

User request:
{text}
"""


class DefaultLLMClassifier(Classifier):
    def __init__(
        self,
        *,
        llm: LLMClient,
        prompt_template: Optional[str] = None,
        allowed_types: Optional[List[str]] = None,
        max_parts: int = 5,
    ) -> None:
        self._llm = llm
        self._template = prompt_template or DEFAULT_CLASSIFIER_TEMPLATE
        self._allowed_types = allowed_types or [StruktQueryEnum.GENERAL]
        self._max_parts = max_parts

    def classify(self, state: InvocationState) -> QueryClassification:
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
        prompt = self._template.format(
            text=state.text,
            allowed_types=self._allowed_types,
            max_parts=self._max_parts,
            current_time=current_time,
        )

        try:
            out: ClassificationOut = self._llm.structured(
                prompt,
                ClassificationOut,
                context=state.context,
                query_hint=state.text,
            )
            qt = list(out.query_types or [])
            cf = list(out.confidences or [])
            pr = list(out.parts or [])

            # Normalize lengths
            n = min(len(qt), len(cf), len(pr)) or 0
            if n == 0:
                return QueryClassification(
                    query_types=[StruktQueryEnum.GENERAL],
                    confidences=[1.0],
                    parts=[state.text],
                )
            return QueryClassification(
                query_types=qt[:n], confidences=cf[:n], parts=pr[:n]
            )
        except Exception:
            # Fallback if LLM parsing fails
            return QueryClassification(
                query_types=[StruktQueryEnum.GENERAL],
                confidences=[0.7],
                parts=[state.text],
            )
