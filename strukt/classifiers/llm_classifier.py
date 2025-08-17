from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel

from ..interfaces import Classifier, LLMClient
from ..types import InvocationState, QueryClassification


class ClassificationOut(BaseModel):
    query_types: List[str]
    confidences: List[float]
    parts: List[str]


DEFAULT_CLASSIFIER_TEMPLATE = (
    "You are an intent classifier. Given a user request, you must: \n"
    "1) Determine one or more intent types from the allowed list.\n"
    "2) Split the input into minimal, meaningful parts (at most {max_parts}) where each part maps to exactly one type.\n"
    "3) Return a strict JSON with fields: query_types, confidences, parts.\n\n"
    "Rules:\n"
    "- Use only types from: {allowed_types}. If none fits, use 'general'.\n"
    "- Keep query_types, confidences, and parts the same length and aligned by index.\n"
    "- confidences should be between 0 and 1.\n"
    "- Do not include any prose or explanation.\n\n"
    "User text:\n"
    "{text}\n"
)


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
        self._allowed_types = allowed_types or ["general"]
        self._max_parts = max_parts

    def classify(self, state: InvocationState) -> QueryClassification:
        prompt = self._template.format(
            text=state.text,
            allowed_types=self._allowed_types,
            max_parts=self._max_parts,
        )

        try:
            out: ClassificationOut = self._llm.structured(prompt, ClassificationOut)
            qt = list(out.query_types or [])
            cf = list(out.confidences or [])
            pr = list(out.parts or [])

            # Normalize lengths
            n = min(len(qt), len(cf), len(pr)) or 0
            if n == 0:
                return QueryClassification(query_types=["general"], confidences=[1.0], parts=[state.text])
            return QueryClassification(query_types=qt[:n], confidences=cf[:n], parts=pr[:n])
        except Exception:
            # Fallback if LLM parsing fails
            return QueryClassification(query_types=["general"], confidences=[0.7], parts=[state.text])



