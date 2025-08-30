#!/usr/bin/env python3
"""
StruktX AI CLI - Command line interface for the StruktX AI framework
"""

import argparse
import asyncio
import sys
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .ai import Strukt
from .config import StruktConfig, ensure_config_types
from .logging import get_logger
from .ai import _build_handlers as build_handlers  # reuse existing builder
from .ai import _build_memory as build_memory  # reuse existing builder
from .ai import _build_llm as build_llm  # may exist in ai.py
from .mcp.server import MCPServerApp
from .mcp.auth import APIKeyAuthConfig

console = Console()
logger = get_logger(__name__)


def print_banner():
    """Print the StruktX AI banner"""
    banner = Text("StruktX AI", style="bold blue")
    subtitle = Text("Configurable AI Framework", style="italic")

    panel = Panel(f"{banner}\n{subtitle}", border_style="blue", padding=(1, 2))
    console.print(panel)


async def run_chat(config_path: Optional[str] = None, message: Optional[str] = None):
    """Run a chat session with StruktX AI"""
    try:
        # Load configuration
        if config_path:
            config = StruktConfig.from_file(config_path)
        else:
            config = StruktConfig()

        # Initialize AI
        ai = Strukt(config)

        if message:
            # Single message mode
            response = await ai.chat(message)
            console.print(f"[green]AI Response:[/green] {response}")
        else:
            # Interactive mode
            console.print(
                "[yellow]Entering interactive mode. Type 'quit' to exit.[/yellow]"
            )
            while True:
                try:
                    user_input = input("\n[blue]You:[/blue] ")
                    if user_input.lower() in ["quit", "exit", "q"]:
                        break

                    if user_input.strip():
                        response = await ai.chat(user_input)
                        console.print(f"[green]AI:[/green] {response}")

                except KeyboardInterrupt:
                    console.print("\n[yellow]Exiting...[/yellow]")
                    break

    except Exception as e:
        logger.error(f"Error running chat: {e}")
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="StruktX AI - Configurable AI Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  struktx-ai --config config.yaml --message "Hello, how are you?"
  struktx-ai --interactive
  struktx-ai --version
        """,
    )

    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")

    parser.add_argument(
        "--message", "-m", type=str, help="Single message to send to AI"
    )

    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Start interactive chat mode"
    )

    parser.add_argument(
        "--version", "-v", action="version", version="struktx-ai 0.0.1-beta"
    )

    subparsers = parser.add_subparsers(dest="command")
    mcp_parser = subparsers.add_parser("mcp", help="MCP operations")
    mcp_sub = mcp_parser.add_subparsers(dest="mcp_cmd")
    mcp_serve = mcp_sub.add_parser(
        "serve", help="Serve MCP tools (fast-agent host can attach)"
    )
    mcp_serve.add_argument(
        "--stdio", action="store_true", help="Print tools as JSON for stdio host"
    )
    mcp_serve.add_argument("--list", action="store_true", help="List tools and exit")

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Determine mode
    if args.command == "mcp" and args.mcp_cmd == "serve":
        # Load config and validate MCP settings
        cfg = StruktConfig()
        if args.config:
            cfg = StruktConfig.from_file(args.config)
        cfg = ensure_config_types(cfg)
        if not cfg.mcp.enabled:
            console.print("[red]MCP is disabled in config[/red]")
            sys.exit(2)
        if not cfg.mcp.server_name:
            console.print("[red]mcp.server_name must be set in config[/red]")
            sys.exit(2)
        # Build core components to get handlers and memory
        llm = build_llm(cfg)  # type: ignore[misc]
        memory = build_memory(cfg, llm)  # type: ignore[misc]
        handlers = build_handlers(cfg, llm, memory)
        app = MCPServerApp(
            server_name=cfg.mcp.server_name,
            handlers=handlers,
            include_handlers=cfg.mcp.include_handlers,
            memory=memory,
            api_key_auth=APIKeyAuthConfig(
                header_name=cfg.mcp.auth_api_key.header_name,
                env_var=cfg.mcp.auth_api_key.env_var,
            ),
        )
        # Rebuild tools with MCP config for consent/schema/multiple-tools
        from .mcp.adapters import build_tools_from_handlers

        tools_map = build_tools_from_handlers(
            handlers=handlers,
            include=cfg.mcp.include_handlers,
            mcp_config=cfg.mcp,
        )
        # Monkey-assign built tools into app instance for listing
        app._tools = tools_map  # type: ignore[attr-defined]
        tools = app.list_tools()
        if args.list:
            from json import dumps

            console.print(dumps(tools, indent=2))
            sys.exit(0)
        # For now we only list/stdio; actual fast-agent hosting will import app
        from json import dumps

        if args.stdio:
            print(dumps({"name": cfg.mcp.server_name, "tools": tools}))
            sys.exit(0)
        console.print(
            "[green]MCP app initialized[/green]\nServer: {}\nTools: {}".format(
                cfg.mcp.server_name, ", ".join([t["name"] for t in tools])
            )
        )
        sys.exit(0)

    if args.message:
        # Single message mode
        asyncio.run(run_chat(args.config, args.message))
    elif args.interactive or not args.message:
        # Interactive mode
        asyncio.run(run_chat(args.config))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
