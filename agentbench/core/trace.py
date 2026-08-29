"""Trace and event model for agent trajectories."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class TraceEventType(StrEnum):
    TASK_START = "task_start"
    TASK_END = "task_end"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RETRIEVAL = "retrieval"
    EVALUATOR = "evaluator"
    RETRY = "retry"
    ERROR = "error"
    EXTERNAL_SERVICE = "external_service"
    STREAM = "stream"


class TraceEvent(BaseModel):
    """Normalized trace event capturing agent trajectory."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    task_id: str
    event_type: TraceEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    agent_id: str | None = None
    parent_agent_id: str | None = None
    parent_event_id: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    tool: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: Any | None = None
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    error: str | None = None
    retry_count: int = 0
    termination_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_otel_attributes(self) -> dict[str, str | int | float | bool]:
        """Convert to OpenTelemetry span attributes."""
        attrs: dict[str, str | int | float | bool] = {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "event_type": self.event_type.value,
        }
        for key in (
            "agent_id",
            "parent_agent_id",
            "model",
            "prompt_version",
            "tool",
            "termination_reason",
            "error",
        ):
            val = getattr(self, key)
            if val is not None:
                attrs[key] = val
        for key in ("latency_ms", "input_tokens", "output_tokens", "estimated_cost_usd", "retry_count"):
            val = getattr(self, key)
            if val is not None:
                attrs[key] = val
        return attrs


class TraceSpan(BaseModel):
    """Hierarchical span for reconstructing agent trajectories."""

    span_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_span_id: str | None = None
    name: str
    event_type: TraceEventType
    start_time: datetime
    end_time: datetime | None = None
    events: list[TraceEvent] = Field(default_factory=list)
    children: list[TraceSpan] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() * 1000
