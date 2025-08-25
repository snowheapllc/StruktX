from __future__ import annotations

from collections.abc import Iterable
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
)

import httpx

from strukt.defaults import BaseAWSTransport, RequestSigner

from .models import DeviceCommand


class DeviceTransport(Protocol):
    """Abstract transport for device discovery and command execution."""

    def list_devices(self, *, user_id: str, unit_id: str) -> List[Dict[str, Any]]: ...

    def execute(
        self, *, commands: Iterable[DeviceCommand], user_id: str, unit_id: str
    ) -> Dict[str, Any]: ...


class AWSSignedHttpTransport(BaseAWSTransport):
    """HTTP transport for a state manager-like API that uses AWS SigV4.

    The base_url and header keys are configurable. This transport is generic and
    is suitable for any backend that accepts per-request identifiers via headers.
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
        payload_builder: Optional[
            Callable[[List[Dict[str, Any]], str, str], Dict[str, Any]]
        ] = None,
        log_devices_response: bool = False,
    ) -> None:
        super().__init__(
            base_url=base_url,
            user_header=user_header,
            unit_header=unit_header,
            content_type=content_type,
            client=client,
            signer=signer,
            log_responses=log_devices_response,
        )
        self._payload_builder = payload_builder

    def list_devices(self, *, user_id: str, unit_id: str) -> List[Dict[str, Any]]:
        """List devices for a user/unit."""
        response = self._make_request(method="GET", user_id=user_id, unit_id=unit_id)
        data = response.json()

        # Flexible response shape: prefer top-level 'devices', else passthrough list
        if isinstance(data, dict) and "devices" in data:
            return data["devices"]
        if isinstance(data, list):
            return data
        return []

    def execute(
        self, *, commands: Iterable[DeviceCommand], user_id: str, unit_id: str
    ) -> Dict[str, Any]:
        """Execute device commands."""
        # Build device list
        device_list = [
            {
                "deviceId": c.deviceId,
                "actionType": c.actionType,
                "actionId": c.actionId,
                "targetValue": c.targetValue,
            }
            for c in commands
        ]

        # Allow custom request shape via payload_builder; default to {user_id, unit_id, data: {devices}}
        if self._payload_builder is not None:
            try:
                wrapped = self._payload_builder(device_list, user_id, unit_id)
            except Exception as e:
                self._log.warn(
                    f"payload_builder failed: {e}; falling back to default shape"
                )
                wrapped = {
                    "user_id": user_id,
                    "unit_id": unit_id,
                    "data": {"devices": device_list},
                }
        else:
            wrapped = {
                "user_id": user_id,
                "unit_id": unit_id,
                "data": {"devices": device_list},
            }

        self._make_request(
            method="POST", user_id=user_id, unit_id=unit_id, body=wrapped
        )

        return {
            "status": "success",
            "message": f"Executed {len(device_list)} device command(s)",
        }
