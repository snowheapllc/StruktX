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

    def invoke(self, prompt: str, **kwargs: Any) -> Any:
        # No formatting here, this client is for dev only and echoes back
        return type("Resp", (), {"content": prompt})

    def structured(
        self, prompt: str, output_model: Type[BaseModel], **kwargs: Any
    ) -> Any:
        # Try to return an empty-but-valid object for common fields to avoid raising
        try:
            return output_model.model_validate(
                {
                    "query_types": [],
                    "confidences": [],
                    "parts": [],
                }
            )
        except Exception:
            # Best-effort pydantic v2 -> v1 fallback paths
            try:
                return output_model.model_construct(
                    query_types=[], confidences=[], parts=[]
                )  # type: ignore[attr-defined]
            except Exception:
                try:
                    return output_model.construct(
                        query_types=[], confidences=[], parts=[]
                    )  # type: ignore[attr-defined]
                except Exception:
                    return output_model()  # may still raise if strict


class MemoryAugmentedLLMClient(LLMClient):
    """Decorator that injects retrieved memory context into prompts."""

    def __init__(
        self, base: LLMClient, memory: MemoryEngine, *, top_k: int = 5
    ) -> None:
        self._base = base
        self._memory = memory
        self._top_k = top_k

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
    def classify(self, state: InvocationState) -> QueryClassification:
        # Default: route everything to 'general'
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

    def handle(self, state: InvocationState, parts: List[str]) -> HandlerResult:
        # Preserve the user's full question for general responses
        text = state.text
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
            resp = self._llm.invoke(
                prompt, context=state.context, query_hint=state.text
            )
            content = getattr(resp, "content", None)
            response = str(content) if content is not None else str(resp)
            if not response:
                response = text
            return HandlerResult(response=response, status=StruktQueryEnum.GENERAL)
        except Exception:
            return HandlerResult(response=text, status=StruktQueryEnum.GENERAL)
