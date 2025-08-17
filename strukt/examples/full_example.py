"""Builds a full example app with the StruktX framework."""


from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import BaseModel

import sys
import os
from pathlib import Path


current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
sys.path.insert(0, str(parent_dir))

from strukt import (
    create,
    StruktConfig,
    HandlersConfig,
    LLMClientConfig,
    ClassifierConfig,
)
from strukt.interfaces import LLMClient, Handler
from strukt.types import InvocationState, HandlerResult, QueryClassification
from strukt.classifiers.llm_classifier import (
    DefaultLLMClassifier,
    DEFAULT_CLASSIFIER_TEMPLATE,
)

from strukt.examples.middleware import ApprovalMiddleware, LoggingMiddleware
from strukt.middleware import Middleware

from dotenv import load_dotenv
load_dotenv()



#! -------------------------------- LLM Client --------------------------------
class OpenAILLM(LLMClient):
    """OpenAI LLM client using langchain_openai."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4.1-mini"):
        try:
            from langchain_openai import ChatOpenAI
            self._client = ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=0.1,
            )
        except ImportError:
            raise ImportError("Please install langchain-openai: pip install langchain-openai")

    def invoke(self, prompt: str, **kwargs: Any) -> Any:
        return self._client.invoke(prompt, **kwargs)

    def structured(self, prompt: str, output_model: type[BaseModel], **kwargs: Any) -> Any:
        try:
            from langchain_core.output_parsers import PydanticOutputParser
            parser = PydanticOutputParser(pydantic_object=output_model)
            full_prompt = prompt + "\n" + parser.get_format_instructions()
            response = self._client.invoke(full_prompt, **kwargs)
            return parser.parse(response.content)
        except Exception as e:
            print(f"OpenAI structured call failed: {e}")
            # Fallback to simple invoke and try to parse
            try:
                response = self._client.invoke(prompt, **kwargs)
                # Try to extract JSON from response
                import json
                import re
                content = response.content
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    return output_model(**data)
            except Exception:
                pass
            # Final fallback
            return output_model.model_construct()

#! -------------------------------- Middleware --------------------------------

class CustomPreClassifyMiddleware(Middleware):
    """Example middleware showing all the different hooks."""
    
    def before_classify(self, state: InvocationState) -> InvocationState:
        print(f"🔍 [CustomPreClassify] before_classify: '{state.text}'")
        # You can modify the state before classification
        # For example, add context or modify the text
        state.context["pre_classify_timestamp"] = datetime.now(timezone.utc).isoformat()
        return state
    
    def after_classify(self, state: InvocationState, classification: QueryClassification) -> tuple[InvocationState, QueryClassification]:
        print(f"🎯 [CustomPreClassify] after_classify: types={classification.query_types}")
        # You can modify the classification results
        # For example, boost confidence for certain types
        return state, classification
    
    def before_handle(self, state: InvocationState, query_type: str, parts: List[str]) -> tuple[InvocationState, List[str]]:
        print(f"⚙️ [CustomPreClassify] before_handle[{query_type}]: parts={parts}")
        # You can modify parts or add context before handling
        if query_type == "time_service":
            # Add timezone info to context for time handlers
            state.context["timezone"] = "UTC"
        return state, parts
    
    def after_handle(self, state: InvocationState, query_type: str, result: HandlerResult) -> HandlerResult:
        print(f"✅ [CustomPreClassify] after_handle[{query_type}]: status={result.status}")
        # You can modify the handler result
        # For example, add metadata or modify the response
        if query_type == "time_service" and "approval" not in result.status.lower():
            result.response = f"[Enhanced] {result.response}"
        return result

#! -------------------------------- Handlers --------------------------------

class TimeResponse(BaseModel):
    response: str

class ExampleTimeHandler(Handler):
    """Simple handler that uses the LLMClient.structured without LangChain."""

    def __init__(self, llm: LLMClient, prompt_template: str | None = None) -> None:
        self._llm = llm
        self._prompt = (
            prompt_template
            or "You are a time helper. Now is {now}. The user said: {text}. "
            "Reply with a single sentence summarizing the time context."
        )

    def handle(self, state: InvocationState, parts: List[str]) -> HandlerResult:
        text = parts[0] if parts else state.text
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
        prompt = self._prompt.format(now=now_str, text=text)
        out: TimeResponse = self._llm.structured(prompt, TimeResponse)
        return HandlerResult(response=out.response, status="time_service")

#! -------------------------------- App --------------------------------


def build_app():
    custom_prompt = (
        "You are a careful intent classifier. Allowed types: {allowed_types}.\n"
        "If the user asks about time, use 'time_service'; else 'general'.\n"
        "Return JSON with query_types, confidences, parts.\n\n"
        "User text:\n{text}\n"
    )

    return create(
        StruktConfig(
            llm=LLMClientConfig(
                factory=lambda **_: OpenAILLM(
                    api_key=None,  # Will use OPENAI_API_KEY env var
                    model="gpt-4.1-mini"
                )
            ),
            classifier=ClassifierConfig(
                factory=lambda llm, **_: DefaultLLMClassifier(
                    llm=llm,
                    prompt_template=custom_prompt or DEFAULT_CLASSIFIER_TEMPLATE,
                    allowed_types=["time_service", "general"],
                    max_parts=3,
                )
            ),
            handlers=HandlersConfig(
                registry={
                    "time_service": lambda llm, **_: ExampleTimeHandler(llm),
                },
                default_route="general",
            ),
            middleware=[
                # Logging middleware - shows the flow
                lambda **_: LoggingMiddleware(verbose=True),
                # Approval middleware - blocks time requests
                lambda **_: ApprovalMiddleware(
                    rule=lambda state, query_type, parts: "time" not in state.text.lower()
                ),
                # Custom middleware that modifies state before classification
                lambda **_: CustomPreClassifyMiddleware(),
            ]
        )
    )


if __name__ == "__main__":
    app = build_app()
    result = app.invoke("what is the time in Beirut?", context={"user_id": "u1"})
    print("Response:", result.response)
    print("Query types:", result.query_types)
    print("Parts:", result.parts)


