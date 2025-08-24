from __future__ import annotations

import datetime
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel

from ..interfaces import Handler, LLMClient
from ..logging import get_logger
from ..types import HandlerResult, InvocationState


class TimeHandler(Handler):
    def __init__(
        self,
        llm: LLMClient,
        store: object | None = None,
        prompt_template: str | None = None,
    ):
        # llm is accepted for signature compatibility; not used in this simplified handler
        self._prompt = None
        self._store = store
        self.console = Console()
        self._llm = llm
        self._log = get_logger("examples.time")

    def _extract_iana_timezone(self, text: str) -> Optional[str]:
        # Look for explicit IANA timezone like Region/City
        import re

        match = re.search(r"\b[A-Za-z]+/[A-Za-z_]+\b", text)
        if match:
            return match.group(0)
        return None

    def _city_to_tz(self, text: str) -> Optional[str]:
        # Minimal common city mapping; can be extended/configured
        city_map = {
            "beirut": "Asia/Beirut",
            "dubai": "Asia/Dubai",
            "abu dhabi": "Asia/Dubai",
            "tokyo": "Asia/Tokyo",
            "london": "Europe/London",
            "paris": "Europe/Paris",
            "berlin": "Europe/Berlin",
            "amsterdam": "Europe/Amsterdam",
            "new york": "America/New_York",
            "los angeles": "America/Los_Angeles",
            "san francisco": "America/Los_Angeles",
            "chicago": "America/Chicago",
            "sydney": "Australia/Sydney",
            "melbourne": "Australia/Melbourne",
            "singapore": "Asia/Singapore",
            "hong kong": "Asia/Hong_Kong",
            "mumbai": "Asia/Kolkata",
            "delhi": "Asia/Kolkata",
            "shanghai": "Asia/Shanghai",
        }
        lowered = text.lower()
        # Check multi-word keys first
        for city in sorted(city_map.keys(), key=lambda s: -len(s)):
            if city in lowered:
                return city_map[city]
        return None

    def _find_country_code(self, text: str) -> Optional[str]:
        try:
            import pycountry  # type: ignore
        except Exception:
            return None

        lowered = text.lower()
        for country in pycountry.countries:
            names = {country.name}
            if hasattr(country, "official_name"):
                names.add(country.official_name)  # type: ignore[attr-defined]
            if hasattr(country, "common_name"):
                names.add(country.common_name)  # type: ignore[attr-defined]
            for name in names:
                if name and name.lower() in lowered:
                    return country.alpha_2
        return None

    def handle(self, state: InvocationState, parts: List[str]) -> HandlerResult:
        text = parts[0] if parts else state.text

        # Always probe the LLM so memory augmentation runs and scoped retrieval is exercised
        # We also include injected docs explicitly to make the task unambiguous
        injected_list = []
        # Inspect engine-backed docs for this scope (debug aid)
        user_id = str(state.context.get("user_id", ""))
        unit_id = str(state.context.get("unit_id", ""))
        injected_docs: List[str] = []
        try:
            if self._store and hasattr(self._store, "list_engine_memories_for_scope"):
                injected_docs = self._store.list_engine_memories_for_scope(
                    user_id=user_id, unit_id=unit_id, limit=5
                )  # type: ignore[attr-defined]
        except Exception:
            injected_docs = []
        injected_list = [f"- {d}" for d in injected_docs]
        mem_block = ("\n".join(injected_list)) if injected_list else "- none"
        from ..prompts import TIME_PROBE_PROMPT_TEMPLATE

        probe_prompt = TIME_PROBE_PROMPT_TEMPLATE.format(mem_block=mem_block)
        try:
            self._log.prompt("Time Probe Prompt", probe_prompt)
            probe_resp = self._llm.invoke(
                probe_prompt,
                context=state.context,
                query_hint=text,
                augment_source="time_handler.probe",
            )
            probe_line = str(getattr(probe_resp, "content", "")).strip() or str(
                probe_resp
            )
        except Exception:
            probe_line = "MEM_PROBE: none"

        # Fallback: if LLM returned none or unexpected, derive from injected docs
        try:
            import re as _re

            m = _re.match(r"^\s*MEM_PROBE:\s*(.+?)\s*$", probe_line)
            inferred = (m.group(1) if m else "").strip().lower()
            if not inferred or inferred == "none":
                # Look for a city memory
                for d in injected_docs:
                    if ":city=" in d:
                        city = d.split(":city=")[-1].strip()
                        if city:
                            probe_line = f"MEM_PROBE: {city}"
                            break
            # Extract final probed value (may come from LLM or fallback above)
            m2 = _re.match(r"^\s*MEM_PROBE:\s*(.+?)\s*$", probe_line)
            probed_value = (m2.group(1) if m2 else "").strip()
        except Exception:
            probed_value = ""

        # Priority 0: user preference from memory store (if available)
        tz_name = ""
        try:
            if self._store and hasattr(self._store, "find_nodes"):
                user_id = str(state.context.get("user_id", ""))
                unit_id = str(state.context.get("unit_id", ""))
                pref_nodes = self._store.find_nodes(
                    category="preference", key="timezone"
                )  # type: ignore[attr-defined]

                # Prefer exact user+unit match, then user-only, then any
                def _pick(nodes):
                    for n in nodes:
                        if (
                            user_id
                            and unit_id
                            and n.user_id == user_id
                            and n.unit_id == unit_id
                        ):
                            return n.value
                    for n in nodes:
                        if user_id and n.user_id == user_id:
                            return n.value
                    return nodes[0].value if nodes else ""

                tz_name = str(_pick(pref_nodes) or "")
        except Exception:
            tz_name = ""

        # Priority 1: explicit IANA timezone in the text
        # But first, apply probed value if available (city or explicit timezone)
        probe_tz = ""
        try:
            if probed_value and probed_value.lower() != "none":
                if "/" in probed_value:
                    probe_tz = probed_value
                else:
                    probe_tz = self._city_to_tz(probed_value) or ""
        except Exception:
            probe_tz = ""
        tz_name = tz_name or probe_tz or self._extract_iana_timezone(text) or ""

        # Priority 2a: if no preference, infer from stored location nodes
        if not tz_name and self._store and hasattr(self._store, "find_nodes"):
            try:
                user_id = str(state.context.get("user_id", ""))
                unit_id = str(state.context.get("unit_id", ""))
                loc_nodes = self._store.find_nodes(category="location")  # type: ignore[attr-defined]

                # Prefer unit-specific, then user-specific, then any
                def _pick_loc(nodes):
                    for n in nodes:
                        if (
                            user_id
                            and unit_id
                            and str(n.user_id) == user_id
                            and str(n.unit_id) == unit_id
                        ):
                            return n.value
                    for n in nodes:
                        if user_id and str(n.user_id) == user_id:
                            return n.value
                    return nodes[0].value if nodes else ""

                loc = str(_pick_loc(loc_nodes) or "").strip()
                if loc:
                    tz_name = self._city_to_tz(loc) or ""
            except Exception:
                pass

        # Priority 2b: common city mapping from query text
        if not tz_name:
            tz_name = self._city_to_tz(text) or ""

        # Priority 3: country-based first timezone
        code = None
        if not tz_name:
            code = self._find_country_code(text)
        try:
            import pytz  # type: ignore
        except Exception:
            tz_name = "UTC"

        if not tz_name and code:
            try:
                from pytz import country_timezones  # type: ignore

                tz_list = country_timezones.get(code, [])  # type: ignore[attr-defined]
                if tz_list:
                    tz_name = tz_list[0]
            except Exception:
                pass
        tz_name = tz_name or "UTC"

        try:
            import pytz  # type: ignore

            tz = pytz.timezone(tz_name)
            local_dt = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
            message = f"{local_dt} in {tz_name} | {probe_line}"
            self._log.info(f"Resolved time for tz={tz_name} -> {local_dt}")

            # Pretty print the time result
            panel = Panel(
                f"[cyan]Query Part:[/cyan] {text}\n"
                f"[yellow]Timezone:[/yellow] {tz_name}\n"
                f"[green]Local Time:[/green] {local_dt} ({tz_name})\n"
                f"[magenta]{probe_line}[/magenta]\n"
                f"[blue]Injected Docs:[/blue] {len(injected_docs)}\n"
                + (
                    "\n".join(f"[dim]- {d}[/dim]" for d in injected_docs)
                    if injected_docs
                    else ""
                ),
                title="🕐 [bold green]Time Service Result[/bold green]",
                border_style="green",
            )
            self.console.print(panel)

            return HandlerResult(response=message, status="time_service")
        except Exception:
            # Fallback: UTC time only
            utc_dt = datetime.datetime.now(datetime.UTC).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
            self._log.warn("Failed to resolve timezone; falling back to UTC")

            panel = Panel(
                f"[red]Failed to resolve timezone for:[/red] {text}\n"
                f"[yellow]Fallback to UTC:[/yellow] {utc_dt}\n"
                f"[magenta]{probe_line}[/magenta]\n"
                f"[blue]Injected Docs:[/blue] {len(injected_docs)}\n"
                + (
                    "\n".join(f"[dim]- {d}[/dim]" for d in injected_docs)
                    if injected_docs
                    else ""
                ),
                title="⚠️ [bold red]Timezone Resolution Failed[/bold red]",
                border_style="red",
            )
            self.console.print(panel)

            return HandlerResult(
                response=f"{utc_dt} (UTC) | {probe_line}", status="time_service"
            )
