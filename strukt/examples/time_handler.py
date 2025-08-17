from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from pydantic import BaseModel

from ..interfaces import Handler, LLMClient
from ..types import InvocationState, HandlerResult
from ..langchain_helpers import create_structured_chain


class TimeResponse(BaseModel):
    response: str


TIME_PROMPT = (
    "You are a helpful assistant. The current UTC time is {now}. "
    "User request: {text}. "
    "Answer with a concise sentence that includes the time in the requested city if specified."
)


class TimeHandler(Handler):
    def __init__(self, llm: LLMClient, prompt_template: str | None = None):
        self._llm = llm
        self._prompt = prompt_template or TIME_PROMPT

    def handle(self, state: InvocationState, parts: List[str]) -> HandlerResult:
        text = parts[0] if parts else state.text
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

        try:
            chain = create_structured_chain(
                llm_client=self._llm,
                prompt_template=self._prompt,
                output_model=TimeResponse,
                input_variables=["now", "text"],
            )
            out: TimeResponse = chain.invoke({"now": now_str, "text": text})
            return HandlerResult(response=out.response, status="time_service")
        except Exception:
            # Fallback using simple formatting
            return HandlerResult(
                response=f"As of {now_str}, here's what I can say about your request: {text}",
                status="time_service",
            )


