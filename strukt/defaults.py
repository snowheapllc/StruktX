from __future__ import annotations

from typing import Any, List, Type, Dict, Optional, Protocol, runtime_checkable

import httpx
import json as _json
from datetime import datetime

from pydantic import BaseModel

from .interfaces import Classifier, Handler, LLMClient, MemoryEngine
from .prompts import render_prompt_with_safe_braces
from .types import HandlerResult, InvocationState, QueryClassification, StruktQueryEnum
from .logging import get_logger


class DateTimeEncoder(_json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class UniversalLLMLogger(LLMClient):
    """
    Universal wrapper that automatically logs all LLM operations to Weave.

    This wrapper intercepts ALL LLM calls (invoke, structured, ainvoke, etc.)
    and captures both the inputs and outputs for comprehensive logging.
    Works with any underlying LLM client implementation.
    """

    def __init__(self, base: LLMClient) -> None:
        self._base = base
        self._logger = get_logger("universal-llm-logger")
        self._weave_available = self._logger.is_weave_available()

    def _create_contextual_operation_name(
        self, base_name: str, context: dict | None
    ) -> str:
        """Create contextual operation name with user info."""
        if not context:
            return base_name

        user_id = (
            str(context.get("user_id", ""))
            if isinstance(context.get("user_id"), (str, int))
            else ""
        )
        unit_id = (
            str(context.get("unit_id", ""))
            if isinstance(context.get("unit_id"), (str, int))
            else ""
        )
        unit_name = context.get("unit_name")

        parts: list[str] = []
        if user_id:
            parts.append(f"user:{user_id}")
        if unit_id:
            parts.append(f"unit:{unit_id}")
        if unit_name:
            clean = str(unit_name).replace(" ", "_").replace("-", "_")[:20]
            parts.append(f"apt:{clean}")

        return f"{base_name}[{','.join(parts)}]" if parts else base_name

    def _safe_context(self, d: dict | None) -> dict:
        """Create safe context dict for logging."""
        if not isinstance(d, dict):
            return {}
        safe = dict(d)
        # Remove potentially large fields that might clutter logs
        for k in [
            "input_prompt",
            "prompt",
            "messages",
            "docs",
            "documents",
            "timestamp",
        ]:
            safe.pop(k, None)
        return safe

    def _safe_serialize_output(self, output: Any) -> Any:
        """Safely serialize output for JSON compatibility."""
        try:
            # If it's a Pydantic model, use model_dump
            if hasattr(output, "model_dump"):
                return output.model_dump()
            elif hasattr(output, "dict"):
                return output.dict()
            # If it's a simple type, return as-is
            elif isinstance(output, (str, int, float, bool, list, dict, type(None))):
                return output
            # For other objects, try to convert to dict
            elif hasattr(output, "__dict__"):
                return {
                    k: v for k, v in output.__dict__.items() if not k.startswith("_")
                }
            # Last resort: convert to string
            else:
                return str(output)
        except Exception:
            # If all else fails, return a safe representation
            return {
                "type": type(output).__name__,
                "repr": str(output)[:200] + "..."
                if len(str(output)) > 200
                else str(output),
            }

    def _log_llm_call(
        self, operation: str, inputs: dict, output: Any, context: dict | None = None
    ) -> None:
        """Log LLM call to Weave with full context."""
        if not self._weave_available:
            return

        try:
            import weave
            import time

            # Safely serialize output for JSON compatibility
            safe_output = self._safe_serialize_output(output)

            # Prepare attributes - these will be attached to the current call
            attributes = {
                "llm.wrapper": "UniversalLLMLogger",
                "llm.wrapped_type": type(self._base).__name__,
                "llm.operation": operation,
                "llm.context": self._safe_context(context),
                "llm.timestamp": time.time(),
                "llm.inputs": inputs,
                "llm.output": safe_output,
            }

            # Add attributes to current trace
            with weave.attributes(attributes):
                # Attributes are attached to the current call
                self._logger.debug(f"LLM operation {operation} logged")

        except Exception as e:
            self._logger.warn(f"Failed to log LLM operation {operation}: {e}")

    def invoke(self, prompt: str, **kwargs: Any) -> Any:
        """Intercept and log invoke calls."""
        # Extract context from kwargs
        context = (
            kwargs.get("context") if isinstance(kwargs.get("context"), dict) else {}
        )

        # Prepare inputs for logging
        inputs = {
            "prompt": prompt,
            "kwargs": {k: v for k, v in kwargs.items() if k != "context"},
        }

        # Execute the actual LLM call
        result = self._base.invoke(prompt, **kwargs)

        # Log the call
        self._log_llm_call("invoke", inputs, result, context)

        return result

    def structured(
        self, prompt: str, output_model: Type[BaseModel], **kwargs: Any
    ) -> Any:
        """Intercept and log structured calls."""
        # Extract context from kwargs
        context = (
            kwargs.get("context") if isinstance(kwargs.get("context"), dict) else {}
        )

        # Prepare inputs for logging - safely serialize output_model
        model_name = "unknown"
        try:
            if hasattr(output_model, "__name__"):
                model_name = output_model.__name__
            elif hasattr(output_model, "__class__") and hasattr(
                output_model.__class__, "__name__"
            ):
                model_name = output_model.__class__.__name__
            else:
                model_name = str(type(output_model).__name__)
        except Exception:
            model_name = "unknown_model"

        inputs = {
            "prompt": prompt,
            "output_model": model_name,
            "kwargs": {k: v for k, v in kwargs.items() if k != "context"},
        }

        # Execute the actual LLM call
        result = self._base.structured(prompt, output_model, **kwargs)

        # Log the call
        self._log_llm_call("structured", inputs, result, context)

        return result

    async def ainvoke(self, prompt: str, **kwargs: Any) -> Any:
        """Intercept and log async invoke calls."""
        # Extract context from kwargs
        context = (
            kwargs.get("context") if isinstance(kwargs.get("context"), dict) else {}
        )

        # Prepare inputs for logging
        inputs = {
            "prompt": prompt,
            "kwargs": {k: v for k, v in kwargs.items() if k != "context"},
        }

        # Execute the actual LLM call
        result = await self._base.ainvoke(prompt, **kwargs)

        # Log the call
        self._log_llm_call("ainvoke", inputs, result, context)

        return result

    async def astructured(
        self, prompt: str, output_model: Type[BaseModel], **kwargs: Any
    ) -> Any:
        """Intercept and log async structured calls."""
        # Extract context from kwargs
        context = (
            kwargs.get("context") if isinstance(kwargs.get("context"), dict) else {}
        )

        # Prepare inputs for logging - safely serialize output_model
        model_name = "unknown"
        try:
            if hasattr(output_model, "__name__"):
                model_name = output_model.__name__
            elif hasattr(output_model, "__class__") and hasattr(
                output_model.__class__, "__name__"
            ):
                model_name = output_model.__class__.__name__
            else:
                model_name = str(type(output_model).__name__)
        except Exception:
            model_name = "unknown_model"

        inputs = {
            "prompt": prompt,
            "output_model": model_name,
            "kwargs": {k: v for k, v in kwargs.items() if k != "context"},
        }

        # Execute the actual LLM call
        result = await self._base.astructured(prompt, output_model, **kwargs)

        # Log the call
        self._log_llm_call("astructured", inputs, result, context)

        return result

    # Delegate all other attributes to the base client
    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


@runtime_checkable
class RequestSigner(Protocol):
    def sign(
        self, *, method: str, url: str, headers: Dict[str, str]
    ) -> Dict[str, str]: ...


class AwsSigV4Signer:
    """AWS SigV4 signer using default boto3 credential resolution.

    This signer is optional and only used when composed with transports that
    require SigV4.
    """

    def __init__(
        self, *, service: str = "execute-api", region: str = "us-east-1"
    ) -> None:
        self._service = service
        self._region = region

    def sign(self, *, method: str, url: str, headers: Dict[str, str]) -> Dict[str, str]:
        # Lazy imports so this module remains importable without AWS deps
        import boto3  # type: ignore
        from botocore.auth import SigV4Auth  # type: ignore
        from botocore.awsrequest import AWSRequest  # type: ignore

        session = boto3.Session()
        credentials = session.get_credentials()
        if not credentials:
            raise ValueError("No AWS credentials found for SigV4 signing")

        # Create AWSRequest with headers that include the payload hash
        # The X-Amz-Content-Sha256 header should already be set by the transport
        request = AWSRequest(method=method, url=url, headers=headers)
        SigV4Auth(credentials, self._service, self._region).add_auth(request)
        return dict(request.headers)


class BaseAWSTransport:
    """Base AWS transport class that provides common AWS SigV4 functionality.
    This class can be inherited by specific transport implementations that need
    AWS authentication and common HTTP operations.
    """

    def __init__(
        self,
        *,
        base_url: str,
        user_header: str = "x-user-id",
        unit_header: str = "x-unit-id",
        content_type: str = "application/json",
        client: Optional[httpx.Client] = None,
        signer: Optional[RequestSigner] = None,
        log_responses: bool = False,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._user_header = user_header
        self._unit_header = unit_header
        self._content_type = content_type
        self._client = client or httpx.Client(timeout=20.0)
        self._signer = signer
        self._log_responses = log_responses
        self._log = get_logger(f"{self.__class__.__name__.lower()}")

    def _signed_headers(
        self,
        *,
        method: str,
        url: str,
        user_id: str,
        unit_id: str,
        body: Optional[bytes] = None,
    ) -> Dict[str, str]:
        """Generate signed headers for AWS requests."""
        headers: Dict[str, str] = {
            self._user_header: user_id,
            self._unit_header: unit_id,
            "Content-Type": self._content_type,
        }
        if self._signer is not None:
            # Include body hash header to ensure SigV4 covers the payload
            try:
                import hashlib

                sha256 = hashlib.sha256(body or b"").hexdigest()
                headers["X-Amz-Content-Sha256"] = sha256
            except Exception:
                pass
            headers = self._signer.sign(method=method, url=url, headers=headers)  # type: ignore[arg-type]
        return headers

    def _make_request(
        self,
        *,
        method: str,
        endpoint: str = "",
        user_id: str,
        unit_id: str,
        body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> httpx.Response:
        """Make a signed HTTP request to the AWS endpoint."""
        url = f"{self._base_url}{endpoint}"

        # Prepare body
        body_bytes = None
        if body is not None:
            body_str = _json.dumps(
                body, separators=(",", ":"), sort_keys=True, cls=DateTimeEncoder
            )
            body_bytes = body_str.encode("utf-8")

        # Get signed headers
        request_headers = self._signed_headers(
            method=method, url=url, user_id=user_id, unit_id=unit_id, body=body_bytes
        )

        # Merge with additional headers
        if headers:
            request_headers.update(headers)

        # Log request details if enabled
        if self._log_responses:
            self._log_request_details(method, url, request_headers, body)

        # Make request
        if method.upper() == "GET":
            response = self._client.get(url, headers=request_headers)
        elif method.upper() == "POST":
            response = self._client.post(
                url, headers=request_headers, content=body_bytes
            )
        elif method.upper() == "PATCH":
            response = self._client.patch(
                url, headers=request_headers, content=body_bytes
            )
        elif method.upper() == "DELETE":
            response = self._client.delete(url, headers=request_headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        response.raise_for_status()

        # Log response details if enabled
        if self._log_responses:
            self._log_response_details(response)

        return response

    def _log_request_details(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: Optional[Dict[str, Any]],
    ) -> None:
        """Log request details for debugging."""
        try:
            safe_headers = {
                k: ("***" if k.lower() == "authorization" else v)
                for k, v in headers.items()
            }
            self._log.json(f"HTTP {method} - Headers", safe_headers)
            if body:
                self._log.json(f"HTTP {method} - Payload", body)
        except Exception:
            pass

    def _log_response_details(self, response: httpx.Response) -> None:
        """Log response details for debugging."""
        try:
            if response.status_code == 200:
                self._log.json(
                    "HTTP Response - Status", {"status": response.status_code}
                )
            else:
                self._log.json("HTTP Response", response.json())
        except Exception:
            self._log.info(f"HTTP Response - Status {response.status_code}")

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._client.close()
        except Exception:
            pass


class SimpleLLMClient(LLMClient):
    """A minimal LLM client placeholder. Users should supply their own.

    For development, this client just echoes prompts.
    """

    def __init__(self):
        self._logger = get_logger("simple-llm-client")

    def invoke(self, prompt: str, **kwargs: Any) -> Any:
        """Simple echo implementation for development."""
        return type("Resp", (), {"content": prompt})

    def structured(
        self, prompt: str, output_model: Type[BaseModel], **kwargs: Any
    ) -> Any:
        """Simple structured response for development."""
        try:
            # Try to return an empty-but-valid object for common fields
            return output_model.model_validate(
                {
                    "query_types": [],
                    "confidences": [],
                    "parts": [],
                }
            )
        except Exception:
            # Fallback for other models
            try:
                return output_model.model_construct()
            except Exception:
                # Last resort: return a mock object
                return type("MockResponse", (), {})()


class MemoryAugmentedLLMClient(LLMClient):
    """Decorator that injects retrieved memory context into prompts."""

    def __init__(
        self, base: LLMClient, memory: MemoryEngine, *, top_k: int = 5
    ) -> None:
        self._base = base
        self._memory = memory
        self._top_k = top_k
        self._logger = get_logger("memory-augmented-llm-client")

    def _augment(
        self,
        prompt: str,
        *,
        context: dict | None = None,
        query_hint: str | None = None,
        source: str | None = None,
    ) -> str:
        try:
            # Prefer sync retrieval via new API; fallback to legacy method names if present
            docs: List[str] = []
            user_id = None
            unit_id = None
            if context:
                try:
                    user_id = (
                        str(context.get("user_id"))
                        if context.get("user_id") is not None
                        else None
                    )
                    unit_id = (
                        str(context.get("unit_id"))
                        if context.get("unit_id") is not None
                        else None
                    )
                except Exception:
                    user_id = None
                    unit_id = None
            # Try using store for scope-aware listing if present
            try:
                store = getattr(self._memory, "store", None)
            except Exception:
                store = None
            if store is not None and (user_id or unit_id):
                try:
                    docs = store.list_engine_memories_for_scope(
                        user_id=user_id, unit_id=unit_id, limit=self._top_k
                    )  # type: ignore[attr-defined]
                except Exception:
                    docs = []
            if not docs:
                retrieval_query = query_hint or prompt
                if hasattr(self._memory, "get_scoped") and callable(
                    self._memory.get_scoped
                ):
                    docs = self._memory.get_scoped(
                        retrieval_query,
                        user_id=user_id,
                        unit_id=unit_id,
                        top_k=self._top_k,
                    )  # type: ignore[call-arg]
                elif hasattr(self._memory, "get") and callable(self._memory.get):
                    docs = self._memory.get(retrieval_query, self._top_k)  # type: ignore[arg-type]
                elif hasattr(self._memory, "retrieve") and callable(
                    self._memory.retrieve
                ):
                    docs = self._memory.retrieve(retrieval_query, self._top_k)  # type: ignore[call-arg]
        except Exception:
            docs = []
        if not docs:
            return prompt
        mem_block = "\n".join(f"- {d}" for d in docs)
        # Log memory injection using our structured logger with source context
        try:
            from .logging import get_logger

            _log = get_logger("memory")
            src = source or "unknown"
            _log.memory_injection(src, len(docs))
        except Exception:
            pass
        return f"Relevant memory:\n{mem_block}\n\n{prompt}"

    def invoke(self, prompt: str, **kwargs: Any) -> Any:
        """Augment prompt with memory and delegate to base client."""
        # Execute memory augmentation
        ctx = kwargs.get("context") if isinstance(kwargs, dict) else None
        qh = kwargs.get("query_hint") if isinstance(kwargs, dict) else None
        src = None
        if isinstance(kwargs, dict):
            src = kwargs.get("augment_source") or kwargs.get("caller")

        augmented = self._augment(prompt, context=ctx, query_hint=qh, source=src)

        # Remove augmentation-only kwargs before delegating
        clean_kwargs = dict(kwargs)
        if "context" in clean_kwargs:
            clean_kwargs.pop("context", None)
        if "query_hint" in clean_kwargs:
            clean_kwargs.pop("query_hint", None)
        if "augment_source" in clean_kwargs:
            clean_kwargs.pop("augment_source", None)
        if "caller" in clean_kwargs:
            clean_kwargs.pop("caller", None)

        # Execute base LLM client
        return self._base.invoke(augmented, **clean_kwargs)

    def structured(self, prompt: str, output_model: Any, **kwargs: Any) -> Any:
        ctx = kwargs.get("context") if isinstance(kwargs, dict) else None
        qh = kwargs.get("query_hint") if isinstance(kwargs, dict) else None
        src = None
        if isinstance(kwargs, dict):
            src = kwargs.get("augment_source") or kwargs.get("caller")
        augmented = self._augment(prompt, context=ctx, query_hint=qh, source=src)
        clean_kwargs = dict(kwargs)
        if "context" in clean_kwargs:
            clean_kwargs.pop("context", None)
        if "query_hint" in clean_kwargs:
            clean_kwargs.pop("query_hint", None)
        if "augment_source" in clean_kwargs:
            clean_kwargs.pop("augment_source", None)
        if "caller" in clean_kwargs:
            clean_kwargs.pop("caller", None)
        return self._base.structured(augmented, output_model, **clean_kwargs)


class SimpleClassifier(Classifier):
    def __init__(self):
        self._logger = get_logger("simple-classifier")
        self._weave_available = self._logger.is_weave_available()

    def _create_contextual_operation_name(
        self, base_name: str, user_context: dict = None
    ) -> str:
        """Create a contextual operation name that includes user information."""
        if not user_context:
            return base_name

        # Extract user identifiers
        user_id = user_context.get("user_id", "")
        unit_id = user_context.get("unit_id", "")
        unit_name = user_context.get("unit_name", "")

        # Build contextual name
        context_parts = []
        if user_id:
            context_parts.append(f"user:{user_id}")
        if unit_id:
            context_parts.append(f"unit:{unit_id}")
        if unit_name:
            # Clean unit name for use in operation names
            clean_unit_name = unit_name.replace(" ", "_").replace("-", "_")[:20]
            context_parts.append(f"apt:{clean_unit_name}")

        if context_parts:
            return f"{base_name}[{','.join(context_parts)}]"

        return base_name

    def classify(self, state: InvocationState) -> QueryClassification:
        """Simple classification - always returns GENERAL."""
        return QueryClassification(
            query_types=[StruktQueryEnum.GENERAL], confidences=[1.0], parts=[state.text]
        )


class GeneralHandler(Handler):
    """General handler that consults the configured LLM for a friendly reply.

    If the app enabled memory augmentation, the LLM is already wrapped to include
    relevant context. On failure, we return the user text as-is.
    """

    def __init__(self, llm: LLMClient, prompt_template: str | None = None) -> None:
        self._llm = llm
        self._prompt = prompt_template
        self._logger = get_logger("general-handler")
        self._weave_available = self._logger.is_weave_available()

    def _create_contextual_operation_name(
        self, base_name: str, user_context: dict = None
    ) -> str:
        """Create a contextual operation name that includes user information."""
        if not user_context:
            return base_name

        # Extract user identifiers
        user_id = user_context.get("user_id", "")
        unit_id = user_context.get("unit_id", "")
        unit_name = user_context.get("unit_name", "")

        # Build contextual name
        context_parts = []
        if user_id:
            context_parts.append(f"user:{user_id}")
        if unit_id:
            context_parts.append(f"unit:{unit_id}")
        if unit_name:
            # Clean unit name for use in operation names
            clean_unit_name = unit_name.replace(" ", "_").replace("-", "_")[:20]
            context_parts.append(f"apt:{clean_unit_name}")

        if context_parts:
            return f"{base_name}[{','.join(context_parts)}]"

        return base_name

    def handle(self, state: InvocationState, parts: List[str]) -> HandlerResult:
        """Handle general queries."""
        # Preserve the user's full question for general responses
        text = state.text
        return self._execute_handle_logic(state, parts, text)

    def _execute_handle_logic(
        self, state: InvocationState, parts: List[str], text: str
    ) -> HandlerResult:
        """Execute the actual handler logic."""
        try:
            if self._prompt:
                # Safely format only known variables, preserving any JSON braces in templates
                prompt = render_prompt_with_safe_braces(
                    self._prompt, {"text": text, "context": state.context}
                )
            else:
                prompt = (
                    "You are a helpful assistant. Provide a concise, friendly, and personalized answer.\n"
                    "Incorporate any 'Relevant memory' context provided above to avoid asking for information we already know.\n"
                    "Prefer to use known preferences (e.g., food, location) to answer directly.\n\n"
                    f"User: {text}\n"
                    "Assistant:"
                )
            # Extract user context from state and pass as kwargs
            user_context = {}
            if hasattr(state, "context") and state.context:
                if "user_id" in state.context:
                    user_context["user_id"] = state.context["user_id"]
                if "unit_id" in state.context:
                    user_context["unit_id"] = state.context["unit_id"]
                if "unit_name" in state.context:
                    user_context["unit_name"] = state.context["unit_name"]

            resp = self._llm.invoke(
                prompt, context=state.context, query_hint=state.text, **user_context
            )
            content = getattr(resp, "content", None)
            response = str(content) if content is not None else str(resp)
            if not response:
                response = text
            return HandlerResult(response=response, status=StruktQueryEnum.GENERAL)
        except Exception:
            return HandlerResult(response=text, status=StruktQueryEnum.GENERAL)
