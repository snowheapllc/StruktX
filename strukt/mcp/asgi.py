from __future__ import annotations

from typing import Any, Dict, Optional

from ..config import StruktConfig, ensure_config_types
from ..ai import Strukt
from .server import MCPServerApp
from .adapters import build_tools_from_handlers
from .auth import APIKeyAuthConfig


def build_fastapi_app(
    strukt_app: Strukt,
    cfg: StruktConfig,
    *,
    app: Any | None = None,
    prefix: str = "/mcp",
):
    """Create or extend a FastAPI app exposing unified MCP endpoint(s).

    - GET {prefix} -> list tools
    - POST {prefix} -> { op: "list_tools" | "call_tool", tool_name?, args? }

    If an existing FastAPI app is provided via `app`, routes are added under
    `prefix` using an APIRouter.
    """
    try:
        from fastapi import FastAPI, Header, HTTPException, Body, APIRouter
        from pydantic import BaseModel
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "FastAPI is required to build MCP HTTP endpoints. Install fastapi."
        ) from e

    cfg = ensure_config_types(cfg)

    class MCPRequest(BaseModel):
        op: str
        tool_name: Optional[str] = None
        args: Optional[Dict[str, Any]] = None

    if app is None:
        app = FastAPI()
    router = APIRouter()
    mcp: Optional[MCPServerApp] = None

    @app.on_event("startup")
    def _init_mcp() -> None:
        nonlocal mcp
        handlers = getattr(strukt_app._engine, "_handlers", {})  # type: ignore[attr-defined]
        memory = strukt_app.get_memory()
        mcp = MCPServerApp(
            server_name=cfg.mcp.server_name or "struktmcp",
            handlers=handlers,
            include_handlers=cfg.mcp.include_handlers,
            memory=memory,
            api_key_auth=APIKeyAuthConfig(
                header_name=cfg.mcp.auth_api_key.header_name,
                env_var=cfg.mcp.auth_api_key.env_var,
            ),
        )
        # Build tools from config
        mcp._tools = build_tools_from_handlers(
            handlers=handlers,
            include=cfg.mcp.include_handlers,
            mcp_config=cfg.mcp,
        )

    @router.get("")
    def mcp_list(x_api_key: Optional[str] = Header(default=None)):
        if not mcp:
            raise HTTPException(status_code=503, detail="MCP app not ready")
        header_name = cfg.mcp.auth_api_key.header_name
        if not mcp.check_api_key({header_name: x_api_key or ""}):
            raise HTTPException(status_code=401, detail="Unauthorized")
        return {"name": mcp.server_name, "tools": mcp.list_tools()}

    @router.post("")
    def mcp_post(
        payload: Dict[str, Any] = Body(...),
        x_api_key: Optional[str] = Header(default=None),
    ):
        if not mcp:
            raise HTTPException(status_code=503, detail="MCP app not ready")
        header_name = cfg.mcp.auth_api_key.header_name
        if not mcp.check_api_key({header_name: x_api_key or ""}):
            raise HTTPException(status_code=401, detail="Unauthorized")
        try:
            # Backward/compat: accept either {op,...} or {tool_name,args}
            op = payload.get("op")
            if not op:
                op = "call_tool" if "tool_name" in payload else "list_tools"

            if op == "list_tools":
                return {"name": mcp.server_name, "tools": mcp.list_tools()}
            if op == "call_tool":
                tool_name = payload.get("tool_name")
                if not tool_name:
                    raise HTTPException(
                        status_code=400, detail="tool_name is required for call_tool"
                    )
                args = payload.get("args") or {}
                return mcp.call_tool(tool_name=tool_name, args=args)
            raise HTTPException(status_code=400, detail="Unsupported op")
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover
            raise HTTPException(status_code=400, detail=str(e))

    # Mount router
    app.include_router(router, prefix=prefix)
    return app
