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
from .config import StruktConfig
from .logging import get_logger

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

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Determine mode
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
