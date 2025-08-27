from __future__ import annotations

from typing import Callable, List, Tuple

from ..middleware import Middleware
from ..types import HandlerResult, InvocationState


class ApprovalMiddleware(Middleware):
    """Example human-approval style middleware.

    The rule returns True if the request should proceed without approval.
    Otherwise, we mark the part as pending and override the handler result.
    """

    def __init__(self, rule: Callable[[InvocationState, str, List[str]], bool]) -> None:
        self._rule = rule

    def before_handle(
        self, state: InvocationState, query_type: str, parts: List[str]
    ) -> Tuple[InvocationState, List[str]]:
        approved = False
        try:
            approved = self._rule(state, query_type, parts)
        except Exception:
            approved = False
        approvals = state.context.setdefault("_approvals", {})
        approvals[query_type] = "approved" if approved else "pending"
        return state, parts

    def after_handle(
        self, state: InvocationState, query_type: str, result: HandlerResult
    ) -> HandlerResult:
        status = (state.context.get("_approvals", {}) or {}).get(query_type)
        if status == "pending":
            return HandlerResult(
                response="This action requires approval. Please approve to proceed.",
                status=f"{query_type}_PENDING_APPROVAL",
            )
        return result
