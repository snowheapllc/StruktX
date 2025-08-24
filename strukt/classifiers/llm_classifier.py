from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel

from ..interfaces import Classifier, LLMClient
from ..prompts import DEFAULT_CLASSIFIER_TEMPLATE
from ..types import InvocationState, QueryClassification, StruktQueryEnum


class ClassificationOut(BaseModel):
    query_types: List[str]
    confidences: List[float]
    parts: List[str]


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
                augment_source="classifier",
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
