from __future__ import annotations

import json
from typing import List

from strukt.interfaces import Handler, LLMClient
from strukt.logging import get_logger
from strukt.prompts import render_prompt_with_safe_braces
from strukt.types import HandlerResult, InvocationState

from .models import DeviceControlResponse
from .prompts import (
    DEVICE_CONTROL_HANDLER_PROMPT_TEMPLATE,
    determine_device_providers,
    get_device_instruction_for_provider,
    get_mixed_provider_instruction,
)
from .toolkit import DeviceToolkit


class DeviceControlHandler(Handler):
    """LLM-driven device control handler for StruktX.

    This handler uses an injected LLM client to produce a structured response
    (`DeviceControlResponse`) which is then validated and executed by the toolkit.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        toolkit: DeviceToolkit,
        prompt_template: str | None = None,
    ) -> None:
        self._llm = llm
        self._toolkit = toolkit
        self._prompt_template = prompt_template or self._default_prompt()
        self._log = get_logger("devices.handler")

    def handle(self, state: InvocationState, parts: List[str]) -> HandlerResult:
        try:
            user_id = (
                str(state.context.get("user_id"))
                if state.context.get("user_id") is not None
                else None
            )
            unit_id = (
                str(state.context.get("unit_id"))
                if state.context.get("unit_id") is not None
                else None
            )
            if not user_id or not unit_id:
                return HandlerResult(
                    response="Missing user or unit information for device control.",
                    status="DEVICE_CONTROL_ERROR",
                )

            devices = self._toolkit.list_devices(user_id=user_id, unit_id=unit_id)

            # Build provider-aware instruction
            providers = determine_device_providers(devices)
            user_context = state.context.get("user_context") or state.context
            if len(providers) > 1:
                device_instruction = get_mixed_provider_instruction(
                    providers, user_context
                )
            else:
                provider = providers[0] if providers else "lifesmart"
                device_instruction = get_device_instruction_for_provider(
                    provider, user_context
                )

            # Build requests list string (handle multiple parts if provided)
            requests_list = parts if parts else [state.text]
            requests_list_str = "\n".join(f"- {p}" for p in requests_list)

            # Compose final prompt using generic template (preserve JSON examples)
            self._log.prompt_template(
                "Device Control (provider-aware)",
                DEVICE_CONTROL_HANDLER_PROMPT_TEMPLATE,
            )
            # Build final prompt for the LLM with full devices
            prompt = render_prompt_with_safe_braces(
                DEVICE_CONTROL_HANDLER_PROMPT_TEMPLATE,
                {
                    "device_instruction": device_instruction,
                    "requests_list_str": requests_list_str,
                    "devices": json.dumps(devices),
                },
            )
            # Build a log-friendly prompt with summarized devices to avoid flooding the console
            try:
                devices_summary = {
                    "count": len(devices) if isinstance(devices, list) else None,
                    "sample": devices[:1] if isinstance(devices, list) else None,
                }
            except Exception:
                devices_summary = {"count": None, "sample": None}
            log_prompt = render_prompt_with_safe_braces(
                DEVICE_CONTROL_HANDLER_PROMPT_TEMPLATE,
                {
                    "device_instruction": device_instruction,
                    "requests_list_str": requests_list_str,
                    "devices": json.dumps(devices_summary),
                },
            )
            self._log.prompt("Device Control - Final Prompt", log_prompt)

            structured: DeviceControlResponse = self._llm.structured(
                prompt, DeviceControlResponse, augment_source="devices.handler"
            )
            self._log.json(
                "Structured DeviceControlResponse",
                (
                    structured.model_dump()
                    if hasattr(structured, "model_dump")
                    else structured
                ),
            )  # type: ignore[attr-defined]
            if not getattr(structured, "commands", None):
                self._log.warn(
                    "LLM returned zero device commands. Check prompt, devices, and provider instructions."
                )
            validation = self._toolkit.validate(
                commands=structured.commands, devices=devices
            )
            if not validation["valid"]:
                self._log.warn(f"Validation failed: {validation['error_message']}")
                return HandlerResult(
                    response=f"Invalid device commands: {validation['error_message']}",
                    status="DEVICE_CONTROL_ERROR",
                )

            result = self._toolkit.execute(
                commands=structured.commands, user_id=user_id, unit_id=unit_id
            )
            if result.get("status") == "success":
                return HandlerResult(
                    response=structured.response, status="DEVICE_CONTROL_SUCCESS"
                )
            self._log.error(f"Execution failed: {result}")
            return HandlerResult(
                response=f"Device control failed: {result.get('message', 'unknown error')}",
                status="DEVICE_CONTROL_ERROR",
            )
        except Exception as e:
            self._log.exception(f"Unhandled exception in DeviceControlHandler: {e}")
            return HandlerResult(
                response="I encountered an error while processing your device control request. Please try again.",
                status="DEVICE_CONTROL_ERROR",
            )

    def _default_prompt(self) -> str:
        return DEVICE_CONTROL_HANDLER_PROMPT_TEMPLATE
