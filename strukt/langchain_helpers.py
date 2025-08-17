from __future__ import annotations

from typing import List, Type

from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from .interfaces import LLMClient


def create_structured_chain(
    *,
    llm_client: LLMClient,
    prompt_template: str,
    output_model: Type[BaseModel],
    input_variables: List[str] | None = None,
) -> Runnable:
    """Build a simple structured output chain using LangChain-compatible client.

    Note: For this helper to work, `llm_client` should be compatible with the LangChain
    `ChatOpenAI` interface. If you're using a custom client, you can provide an adapter
    that exposes the same API.
    """
    # Best-effort import; users must include langchain in their env when using this helper
    from langchain_core.output_parsers import PydanticOutputParser

    parser = PydanticOutputParser(pydantic_object=output_model)

    if input_variables is None:
        import re
        input_variables = re.findall(r"\{([^}]+)\}", prompt_template)
        input_variables = [v for v in input_variables if v != "format_instructions"]

    prompt = PromptTemplate(
        template=prompt_template + "\n{format_instructions}",
        input_variables=input_variables,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Chain: prompt -> llm_client -> parser
    return prompt | llm_client | parser  # type: ignore[operator]



