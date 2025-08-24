from __future__ import annotations

from typing import Any, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from .interfaces import LLMClient


class LangChainLLMClient(LLMClient):
    """Adapter that makes a LangChain ChatModel/Runnable usable as an `LLMClient`.

    Accepts any LangChain Runnable (e.g., `ChatOpenAI`).
    """

    def __init__(self, chat_model: Runnable | BaseChatModel) -> None:
        self._chat_model = chat_model
        self._format_instructions = "\n{format_instructions}"

    def invoke(self, prompt: str, **kwargs: Any) -> Any:
        return self._chat_model.invoke(prompt, **kwargs)

    def structured(
        self, prompt: str, output_model: Type[BaseModel], **kwargs: Any
    ) -> Any:
        try:
            from langchain_core.output_parsers import PydanticOutputParser
            from langchain_core.prompts import PromptTemplate
        except Exception as exc:
            raise RuntimeError(
                "LangChain not installed; cannot use LangChainLLMClient. Install 'langchain-core'."
            ) from exc

        parser = PydanticOutputParser(pydantic_object=output_model)
        # Always wrap the entire prompt as a single variable to avoid brace conflicts in JSON examples
        safe_template = "{prompt}" + self._format_instructions
        prompt_tmpl = PromptTemplate(
            template=safe_template,
            input_variables=["prompt"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt_tmpl | self._chat_model | parser  # type: ignore[operator]
        return chain.invoke({"prompt": prompt})


def _is_langchain_runnable(obj: Any) -> bool:
    try:
        from langchain_core.runnables import Runnable

        return isinstance(obj, Runnable)
    except Exception:
        # Best-effort duck-typing: common Runnable attributes
        return hasattr(obj, "invoke") and (
            hasattr(obj, "astream") or hasattr(obj, "batch")
        )


def adapt_to_llm_client(obj: Any) -> LLMClient:
    """Return an object that satisfies `LLMClient`.

    - If `obj` already satisfies `LLMClient`, return it as-is.
    - If `obj` is a LangChain Runnable, wrap it with `LangChainLLMClient`.
    - Otherwise raise `TypeError`.
    """
    from .interfaces import LLMClient as LLMClientProtocol

    if isinstance(obj, LLMClientProtocol):
        return obj
    if _is_langchain_runnable(obj):
        return LangChainLLMClient(obj)
    raise TypeError(
        "Provided LLM object does not implement LLMClient and is not a LangChain Runnable"
    )


def create_structured_chain(
    *,
    llm_client: LLMClient,
    prompt_template: str,
    output_model: Type[BaseModel],
) -> Runnable:
    """Build a simple structured output chain using LangChain-compatible client.

    Note: For this helper to work, `llm_client` should be compatible with the LangChain
    `ChatOpenAI` interface. If you're using a custom client, you can provide an adapter
    that exposes the same API.
    """
    # Best-effort import; users must include langchain in their env when using this helper
    from langchain_core.output_parsers import PydanticOutputParser

    parser = PydanticOutputParser(pydantic_object=output_model)

    # Use a safe wrapper so JSON braces in prompt_template do not become variables
    prompt = PromptTemplate(
        template="{prompt}\n{format_instructions}",
        input_variables=["prompt"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # If using our adapter, unwrap to underlying runnable for best LC integration
    underlying: Any = getattr(llm_client, "_chat_model", llm_client)
    return prompt | underlying | parser  # type: ignore[operator]


def parse_models_list(models_str: str) -> list[str]:
    """
    Parse a string into a list of model names.
    Accepts:
      - Comma-separated: 'model1,model2,model3'
      - JSON-style: '["model1", "model2", "model3"]'
      - Loosely quoted: '"model1", \'model2\', "model3"'
    Returns a clean list of model names with all extraneous quotes removed.
    """
    import json
    import re

    models_str = models_str.strip()
    if not models_str:
        return []

    # Try JSON parsing first
    try:
        models = json.loads(models_str)
        if isinstance(models, list):
            return [str(m).strip().strip("'\"") for m in models]
    except Exception:
        pass

    # Fallback: split by comma, strip whitespace and any surrounding quotes
    # Handles: 'model1', "model2", 'model3'
    models = [
        m.strip().strip("'\"")
        for m in re.split(r",(?![^[]*\])", models_str)
        if m.strip().strip("'\"")
    ]
    return [m for m in models if m]
