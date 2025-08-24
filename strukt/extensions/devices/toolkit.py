from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, List, Optional

from strukt.logging import get_logger

from .models import DeviceCommand
from .transport import DeviceTransport
from .validation import DeviceCommandValidator


class DeviceToolkit:
    """Framework-agnostic toolkit for device discovery, validation, and execution.

    This toolkit is transport-driven and contains provider-agnostic validation
    logic that can be extended as needed by applications.
    """

    def __init__(
        self, *, transport: DeviceTransport, log_devices_summary: bool = False
    ) -> None:
        self._transport = transport
        self._cache: dict[tuple[str, str], List[Dict[str, Any]]] = {}
        self._log = get_logger("devices.toolkit")
        self._log_devices_summary = log_devices_summary

    def list_devices(
        self, *, user_id: str, unit_id: str, use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        cache_key = (user_id, unit_id)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        if self._log_devices_summary:
            self._log.info(f"Fetching devices for user_id={user_id} unit_id={unit_id}")
        devices = self._transport.list_devices(user_id=user_id, unit_id=unit_id)
        self._cache[cache_key] = devices
        # Optional minimal summary only
        if self._log_devices_summary:
            try:
                names = []
                for d in devices[:3]:
                    if isinstance(d, dict):
                        names.append(str(d.get("name")))
                self._log.json(
                    "Devices Cache Refresh", {"count": len(devices), "names": names}
                )
            except Exception:
                self._log.json("Devices Cache Refresh", {"count": len(devices)})
        return devices

    def validate(
        self,
        *,
        commands: Iterable[DeviceCommand],
        devices: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[str] = None,
        unit_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        cmds = list(commands)
        if devices is None and user_id and unit_id:
            devices = self.list_devices(user_id=user_id, unit_id=unit_id)
        if not devices:
            return {
                "valid": False,
                "error_message": "No devices available for validation",
                "invalid_indices": list(range(len(cmds))),
            }

        is_valid, error_message, invalid_indices = (
            DeviceCommandValidator.validate_commands(cmds, devices)
        )
        return {
            "valid": is_valid,
            "error_message": error_message,
            "invalid_indices": invalid_indices,
        }

    def execute(
        self, *, commands: Iterable[DeviceCommand], user_id: str, unit_id: str
    ) -> Dict[str, Any]:
        self._log.json("Executing Commands", [c.model_dump() for c in commands])
        result = self._transport.execute(
            commands=commands, user_id=user_id, unit_id=unit_id
        )
        self._log.json("Execute Result", result)
        return result

    def _find_device(
        self, devices: List[Dict[str, Any]], device_id: str
    ) -> Optional[Dict[str, Any]]:
        for d in devices:
            identifier = (
                d.get("identifier")
                or d.get("attributes", {}).get("identifier")
                or d.get("id")
            )
            if identifier == device_id:
                return d
        return None

    def _is_valid_action(self, action_type: str) -> bool:
        valid_actions = {
            "updateStatus",
            "updateCurrentValue",
            "updateModes",
            "updateACWinds",
        }
        return action_type in valid_actions
