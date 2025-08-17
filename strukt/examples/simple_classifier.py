from __future__ import annotations

from ..interfaces import Classifier
from ..types import InvocationState, QueryClassification


class SimpleKeywordClassifier(Classifier):
    def classify(self, state: InvocationState) -> QueryClassification:
        text = state.text.lower()
        if "time" in text or "clock" in text:
            return QueryClassification(query_types=["time_service"], confidences=[0.9], parts=[state.text])
        return QueryClassification(query_types=["general"], confidences=[0.6], parts=[state.text])



