from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ErrorInfo(BaseModel):
    """Error information container."""

    code: str = Field(..., description="Error code identifier")
    message: str = Field(..., description="Human-readable error message")
    extra: Dict[str, Any] | None = Field(
        default=None, description="Optional structured details"
    )


class DeviceCommand(BaseModel):
    """Generic device command for any provider/transport."""

    deviceId: str = Field(..., description="Device identifier as known by the provider")
    actionType: str = Field(
        ..., description="Action type (transport/provider specific)"
    )
    actionId: int = Field(
        ...,
        description="Action identifier for the action type; use 0 if not applicable",
    )
    targetValue: str = Field(
        ..., description="Target value as string; transports may coerce types"
    )


class DeviceRequest(BaseModel):
    """A set of device commands to execute in order."""

    devices: List[DeviceCommand] = Field(
        ..., min_length=1, description="Commands to execute"
    )


class DeviceControlResponse(BaseModel):
    """LLM-structured response for device control flows."""

    response: str = Field(
        ..., description="Natural language response explaining actions"
    )
    commands: List[DeviceCommand] = Field(..., description="Commands to execute")


class DeviceDescriptor(BaseModel):
    """Optional descriptor for devices returned by transports/toolkits.

    This is intentionally loose to accommodate heterogeneous provider payloads
    while keeping a common surface for common fields.
    """

    identifier: str = Field(
        ..., description="Stable provider identifier for the device"
    )
    name: str | None = Field(default=None)
    type: str | None = Field(default=None)
    room: str | None = Field(default=None)
    status: Any | None = Field(default=None)
    currentValue: Any | None = Field(default=None)
    minValue: Any | None = Field(default=None)
    maxValue: Any | None = Field(default=None)
    extra: Dict[str, Any] | None = Field(
        default=None, description="Provider-specific fields"
    )
