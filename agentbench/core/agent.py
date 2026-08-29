"""Agent adapter interface and base implementations."""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from agentbench.core.task import Task
from agentbench.core.trace import TraceEvent


@dataclass
class AgentRunContext:
    """Context passed to agents during task execution."""

    run_id: str
    task: Task
    prompt_version: str = "default"
    model: str = "mock"
    temperature: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Normalized result from agent task execution."""

    output: str
    success: bool = True
    traces: list[TraceEvent] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    termination_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentAdapter(abc.ABC):
    """Standard adapter interface for agent implementations."""

    @abc.abstractmethod
    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the agent with configuration."""

    @abc.abstractmethod
    async def run_task(self, context: AgentRunContext) -> AgentResult:
        """Execute a task and return normalized result."""

    @abc.abstractmethod
    async def stream_events(self, context: AgentRunContext) -> AsyncIterator[TraceEvent]:
        """Stream trace events during execution."""
        yield TraceEvent(  # pragma: no cover
            run_id=context.run_id,
            task_id=context.task.task_id,
            event_type="stream",  # type: ignore[arg-type]
        )

    @abc.abstractmethod
    async def terminate(self) -> None:
        """Clean up agent resources."""

    @abc.abstractmethod
    def collect_metadata(self) -> dict[str, Any]:
        """Return agent metadata for reproducibility."""

    @property
    @abc.abstractmethod
    def agent_id(self) -> str:
        """Unique identifier for this agent configuration."""

    @property
    @abc.abstractmethod
    def agent_type(self) -> str:
        """Agent architecture type (single, planner_executor, etc.)."""
