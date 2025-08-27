from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from strukt.logging import get_logger

from .models import DeviceCommand

_log = get_logger("devices.validation")


class DeviceCommandValidator:
    """Validates device commands against defined rules for multiple providers."""

    # Valid action types by provider
    PROVIDER_ACTION_TYPES: Dict[str, set[str]] = {
        "lifesmart": {
            "updateStatus",
            "updateCurrentValue",
            "updateLightColor",
            "updateModes",
            "updateACWinds",
        },
        "philips_hue": {
            "powerOn",
            "brightness",
            "color",
        },
        "samsung_smartthings": {
            "switch",
            "setLevel",
            "setTemperature",
        },
    }

    # Fan speed values (example set; adjust based on provider docs)
    VALID_FAN_SPEEDS = {"0", "15", "45", "75"}

    # Valid light colors (ARGB without 0x prefix)
    VALID_LIGHT_COLORS = {
        "FFFF0000",  # Red
        "FF00FF00",  # Green
        "FF000000",  # Black
        "FF00A5FF",  # Light Blue
        "FFFFFF00",  # Yellow
    }

    @staticmethod
    def validate_commands(
        commands: List[DeviceCommand], available_devices: Any
    ) -> Tuple[bool, str, List[int]]:
        """Validate device commands based on their provider.

        Returns (is_valid, error_message, invalid_indices)
        """
        try:
            _log.debug(f"Validating commands: {[c.model_dump() for c in commands]}")
        except Exception:
            _log.debug("Validating commands (non-pydantic)")
        # Log only a compact summary of available devices (avoid full dumps)
        try:
            count = None
            summary: list[dict[str, str | None]] = []
            if isinstance(available_devices, list):
                count = len(available_devices)
                for d in available_devices[:2]:
                    if isinstance(d, dict):
                        attrs = (
                            (d.get("attributes", {}) or {})
                            if isinstance(d.get("attributes"), dict)
                            else {}
                        )
                        summary.append(
                            {
                                "name": str(d.get("name")),
                                "brand": str(d.get("brand")),
                                "attr.identifier": str(
                                    attrs.get("identifier")
                                    if attrs.get("identifier") is not None
                                    else attrs.get("me")
                                ),
                            }
                        )
            elif isinstance(available_devices, dict):
                devs = available_devices.get("devices")
                if isinstance(devs, list):
                    count = len(devs)
                    for d in devs[:2]:
                        if isinstance(d, dict):
                            attrs = (
                                (d.get("attributes", {}) or {})
                                if isinstance(d.get("attributes"), dict)
                                else {}
                            )
                            summary.append(
                                {
                                    "name": str(d.get("name")),
                                    "brand": str(d.get("brand")),
                                    "attr.identifier": str(
                                        attrs.get("identifier")
                                        if attrs.get("identifier") is not None
                                        else attrs.get("me")
                                    ),
                                }
                            )
            _log.json(
                "Available devices (summary)", {"count": count, "sample": summary}
            )
        except Exception:
            pass

        error_messages: List[str] = []
        invalid_indices: List[int] = []

        for i, command in enumerate(commands):
            device_id = str(command.deviceId or "").strip()
            device = DeviceCommandValidator._get_device(device_id, available_devices)

            if not device:
                error_msg = f"Device ID not found: {device_id} at index {i}"
                _log.warn(error_msg)
                error_messages.append(error_msg)
                invalid_indices.append(i)
                continue

            # Get device provider/brand
            device_provider = str(
                device.get("brand", "lifesmart") or "lifesmart"
            ).lower()

            # Validate action type and ID based on device provider
            action_type = str(command.actionType or "")
            action_id = command.actionId
            target_value = str(command.targetValue or "")

            _log.debug(
                f"Validating action - Provider: {device_provider}, Type: {action_type}, ID: {action_id}, Value: {target_value}"
            )

            # Check if action type is valid for this provider
            valid_actions = DeviceCommandValidator.PROVIDER_ACTION_TYPES.get(
                device_provider,
                DeviceCommandValidator.PROVIDER_ACTION_TYPES["lifesmart"],
            )

            if action_type not in valid_actions:
                error_msg = (
                    f"Invalid action type '{action_type}' for {device_provider} device {device_id} at index {i}. "
                    f"Valid actions: {sorted(list(valid_actions))}"
                )
                error_messages.append(error_msg)
                invalid_indices.append(i)
                continue

            # Provider-specific validation
            if device_provider == "lifesmart":
                error_msg = DeviceCommandValidator._validate_lifesmart_command(
                    command, device, i
                )
            elif device_provider == "philips_hue":
                error_msg = DeviceCommandValidator._validate_philips_hue_command(
                    command, device, i
                )
            elif device_provider == "samsung_smartthings":
                error_msg = (
                    DeviceCommandValidator._validate_samsung_smartthings_command(
                        command, device, i
                    )
                )
            else:
                # Default to generic error for unknown providers
                error_msg = DeviceCommandValidator._validate_unknown_provider(
                    command, device, i
                )

            if error_msg:
                error_messages.append(error_msg)
                invalid_indices.append(i)

        is_valid = len(error_messages) == 0
        error_message = "; ".join(error_messages) if error_messages else ""
        _log.debug(
            f"Validation result - Valid: {is_valid}, Errors: {error_message}, Invalid indices: {invalid_indices}"
        )
        return is_valid, error_message, invalid_indices

    # --- Provider-specific validators ---
    @staticmethod
    def _validate_lifesmart_command(
        command: DeviceCommand, device: Dict[str, Any], index: int
    ) -> Optional[str]:
        action_type = command.actionType
        action_id = command.actionId
        target_value = command.targetValue

        if action_type == "updateStatus":
            if target_value not in ["0", "1"]:
                return f"Invalid target value for updateStatus: {target_value} at index {index}"

        elif action_type == "updateCurrentValue":
            try:
                value = float(target_value)
                max_val = float(device.get("maxValue", 100))
                min_val = float(device.get("minValue", 0))
                if not (min_val <= value <= max_val):
                    return f"Target value {value} out of range [{min_val}, {max_val}] at index {index}"
            except ValueError:
                return f"Invalid number format for target value: {target_value} at index {index}"

        elif action_type == "updateModes":
            modes = device.get("modes") or device.get("actions", {}).get("modes") or []
            mode_ids = [m.get("id") for m in modes if isinstance(m, dict)]
            if action_id not in mode_ids:
                return (
                    f"Invalid mode ID {action_id} for device {command.deviceId} at index {index}. "
                    f"Available modes: {mode_ids}"
                )

        elif action_type == "updateACWinds":
            winds = device.get("wind", [])
            wind_ids = [w.get("id") for w in winds if isinstance(w, dict)]
            if action_id not in wind_ids:
                return (
                    f"Invalid wind ID {action_id} for device {command.deviceId} at index {index}. "
                    f"Available winds: {wind_ids}"
                )

        return None

    @staticmethod
    def _validate_philips_hue_command(
        command: DeviceCommand, device: Dict[str, Any], index: int
    ) -> Optional[str]:
        action_type = command.actionType
        target_value = command.targetValue

        if action_type == "powerOn":
            if target_value not in ["true", "false"]:
                return (
                    f"Invalid target value for powerOn: {target_value} at index {index}. "
                    "Must be 'true' or 'false'"
                )

        elif action_type == "brightness":
            try:
                value = int(target_value)
                if not (1 <= value <= 254):
                    return f"Brightness value {value} out of range [1, 254] at index {index}"
            except ValueError:
                return f"Invalid brightness value: {target_value} at index {index}"

        elif action_type == "color":
            # Require JSON object as a string, minimal validation
            if not (target_value.startswith("{") and target_value.endswith("}")):
                return f"Invalid color format: {target_value} at index {index}. Must be JSON object"

        return None

    @staticmethod
    def _validate_samsung_smartthings_command(
        command: DeviceCommand, device: Dict[str, Any], index: int
    ) -> Optional[str]:
        action_type = command.actionType
        target_value = command.targetValue

        if action_type == "switch":
            if target_value not in ["on", "off"]:
                return f"Invalid target value for switch: {target_value} at index {index}. Must be 'on' or 'off'"

        elif action_type == "setLevel":
            try:
                value = int(target_value)
                if not (0 <= value <= 100):
                    return f"Level value {value} out of range [0, 100] at index {index}"
            except ValueError:
                return f"Invalid level value: {target_value} at index {index}"

        elif action_type == "setTemperature":
            try:
                value = float(target_value)
                if not (-20 <= value <= 50):  # Celsius range
                    return f"Temperature value {value} out of reasonable range [-20, 50] at index {index}"
            except ValueError:
                return f"Invalid temperature value: {target_value} at index {index}"

        return None

    @staticmethod
    def _validate_unknown_provider(
        command: DeviceCommand, device: Dict[str, Any], index: int
    ) -> Optional[str]:
        return f"Unknown device provider for device {command.deviceId} at index {index}"

    # --- Helpers ---
    @staticmethod
    def _normalize_token(value: Any) -> str:
        # Lowercase and strip non-alphanumeric for robust matching
        import re as _re

        s = str(value or "").strip().lower()
        return "".join(_re.findall(r"[a-z0-9]", s))

    @staticmethod
    def _normalize_device_id(value: Any) -> str:
        return DeviceCommandValidator._normalize_token(value)

    @staticmethod
    def _candidate_ids_from_device(device: Dict[str, Any]) -> List[str]:
        ids: List[str] = []
        try:
            attrs = device.get("attributes", {}) if isinstance(device, dict) else {}
            for key in (
                "identifier",
                "id",
                "deviceId",
                "device_id",
                "uniqueId",
                "uuid",
                "me",
                "thingName",
            ):
                if key in device:
                    ids.append(
                        DeviceCommandValidator._normalize_device_id(device.get(key))
                    )
                if key in attrs:
                    ids.append(
                        DeviceCommandValidator._normalize_device_id(attrs.get(key))
                    )
        except Exception:
            pass
        # Remove empties and duplicates
        return [i for i in dict.fromkeys(ids) if i]

    @staticmethod
    def _validate_device_id(device_id: str) -> bool:
        """Validate device ID format (3 or 4 lowercase alphanumeric characters)."""
        if not device_id:
            return False
        return bool(re.match(r"^[a-z0-9]{3,4}$", device_id))

    @staticmethod
    def _get_device(device_id: str, available_devices: Any) -> Optional[Dict[str, Any]]:
        """Get device config by ID. Accepts list or dict with devices list."""
        try:
            norm_id = DeviceCommandValidator._normalize_device_id(device_id)
            # Handle different input formats
            if isinstance(available_devices, dict):
                if "devices" in available_devices:
                    device_list = available_devices["devices"]
                else:
                    device_list = available_devices.get("config", {}).get("devices", [])
            elif isinstance(available_devices, list):
                device_list = available_devices
            else:
                _log.error(f"Invalid devices format: {type(available_devices)}")
                return None

            # Search for device by ID across multiple candidate fields with normalization
            for device in device_list or []:
                if not isinstance(device, dict):
                    continue
                candidates = DeviceCommandValidator._candidate_ids_from_device(device)
                if norm_id in candidates:
                    _log.debug(
                        f"Found device {device_id}: {device.get('name', 'Unknown')} ({device.get('brand', '')})"
                    )
                    return device

            # For debugging: show a small sample of known IDs
            try:
                sample_ids: List[str] = []
                for d in device_list[:10]:
                    if isinstance(d, dict):
                        sample_ids.extend(
                            DeviceCommandValidator._candidate_ids_from_device(d)
                        )
                sample_ids = [s for s in dict.fromkeys(sample_ids) if s]
                _log.warn(
                    f"Device {device_id} not found in available devices | known ids (sample): {sample_ids[:10]}"
                )
            except Exception:
                _log.warn(f"Device {device_id} not found in available devices")
            return None
        except Exception as e:
            _log.error(f"Error getting device: {str(e)}")
            return None
