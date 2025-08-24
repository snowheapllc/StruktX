from __future__ import annotations

from collections.abc import Iterable
import json as _json
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

import httpx

from strukt.logging import get_logger

from .models import DeviceCommand


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
        # Note: AWSRequest will compute payload hash from body if provided via headers['X-Amz-Content-Sha256']
        request = AWSRequest(method=method, url=url, headers=headers)
        SigV4Auth(credentials, self._service, self._region).add_auth(request)
        return dict(request.headers)


class DeviceTransport(Protocol):
    """Abstract transport for device discovery and command execution."""

    def list_devices(self, *, user_id: str, unit_id: str) -> List[Dict[str, Any]]: ...

    def execute(
        self, *, commands: Iterable[DeviceCommand], user_id: str, unit_id: str
    ) -> Dict[str, Any]: ...


class AWSSignedHttpTransport:
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
        self._base_url = base_url.rstrip("/")
        self._user_header = user_header
        self._unit_header = unit_header
        self._content_type = content_type
        self._client = client or httpx.Client(timeout=20.0)
        self._signer = signer
        self._log = get_logger("devices.transport")
        self._payload_builder = payload_builder
        self._log_devices_response = log_devices_response

    def _signed_headers(
        self,
        *,
        method: str,
        url: str,
        user_id: str,
        unit_id: str,
        body: Optional[bytes] = None,
    ) -> Dict[str, str]:
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

    def list_devices(self, *, user_id: str, unit_id: str) -> List[Dict[str, Any]]:
        url = self._base_url
        headers = self._signed_headers(
            method="GET", url=url, user_id=user_id, unit_id=unit_id, body=None
        )
        if self._log_devices_response:
            try:
                safe_headers = {
                    k: ("***" if k.lower() == "authorization" else v)
                    for k, v in headers.items()
                }
                self._log.json("HTTP GET Devices - Headers", safe_headers)
            except Exception:
                pass
        resp = self._client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        # Optionally log a tiny summary of the devices response
        if self._log_devices_response:
            try:
                count = None
                sample = None
                if isinstance(data, list):
                    count = len(data)
                    sample = data[:1]
                elif isinstance(data, dict) and "devices" in data:
                    devs = data.get("devices", [])
                    count = len(devs) if isinstance(devs, list) else None
                    sample = devs[:1] if isinstance(devs, list) else None
                self._log.json(
                    "HTTP GET Devices - Response",
                    {"status": resp.status_code, "count": count, "sample": sample},
                )
            except Exception:
                self._log.info(f"HTTP GET Devices - Status {resp.status_code}")
        # Flexible response shape: prefer top-level 'devices', else passthrough list
        if isinstance(data, dict) and "devices" in data:
            return data["devices"]
        if isinstance(data, list):
            return data
        return []

    def execute(
        self, *, commands: Iterable[DeviceCommand], user_id: str, unit_id: str
    ) -> Dict[str, Any]:
        url = self._base_url
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
        # Deterministic JSON for signing
        body_str = _json.dumps(wrapped, separators=(",", ":"), sort_keys=True)
        body_bytes = body_str.encode("utf-8")
        headers = self._signed_headers(
            method="POST", url=url, user_id=user_id, unit_id=unit_id, body=body_bytes
        )
        # Concise logs (no full payload)
        if self._log_devices_response:
            self._log.json(
                "HTTP POST Devices - Payload (summary)",
                {
                    "count": len(device_list),
                    "first": (device_list[0] if device_list else None),
                },
            )
            try:
                safe_headers = {
                    k: ("***" if k.lower() == "authorization" else v)
                    for k, v in headers.items()
                }
                self._log.json("HTTP POST Devices - Headers", safe_headers)
            except Exception:
                pass
        resp = self._client.post(url, headers=headers, content=body_bytes)
        resp.raise_for_status()
        if self._log_devices_response:
            try:
                if resp.status_code == 200:
                    self._log.json("HTTP POST Devices - Response", {"status": 200})
                else:
                    self._log.json("HTTP POST Devices - Response", resp.json())
            except Exception:
                self._log.info(f"HTTP POST Devices - Status {resp.status_code}")
        return {
            "status": "success",
            "message": f"Executed {len(device_list)} device command(s)",
        }

    def cleanup(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
