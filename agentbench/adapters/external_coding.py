"""External coding agent adapter boundary.

This adapter defines the integration interface for external agentic coding
assistants (e.g., Claude Code, Cursor Agent, Aider). It does NOT fake an
integration — concrete implementations require the external tool's API/CLI.

To integrate Claude Code or similar:
1. Subclass ExternalCodingAgentAdapter
2. Implement _invoke_external_agent() using the tool's documented API
3. Map external events to normalized TraceEvents

See docs/architecture.md for integration guidelines.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from typing import Any

from agentbench.core.agent import AgentAdapter, AgentResult, AgentRunContext
from agentbench.core.trace import TraceEvent, TraceEventType


class ExternalCodingAgentAdapter(AgentAdapter):
    """Base adapter for external agentic coding assistants."""

    def __init__(self, agent_id: str = "external_coding_agent") -> None:
        self._agent_id = agent_id
        self._config: dict[str, Any] = {}
        self._available = False

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def agent_type(self) -> str:
        return "external_coding"

    @property
    def is_available(self) -> bool:
        """Whether the external agent is configured and reachable."""
        return self._available

    async def initialize(self, config: dict[str, Any]) -> None:
        self._config = config
        self._available = await self._check_availability()

    @abc.abstractmethod
    async def _check_availability(self) -> bool:
        """Check if external agent is available."""

    @abc.abstractmethod
    async def _invoke_external_agent(self, context: AgentRunContext) -> AgentResult:
        """Invoke the external coding agent."""

    async def run_task(self, context: AgentRunContext) -> AgentResult:
        if not self._available:
            return AgentResult(
                output="",
                success=False,
                traces=[TraceEvent(
                    run_id=context.run_id,
                    task_id=context.task.task_id,
                    event_type=TraceEventType.ERROR,
                    agent_id=self._agent_id,
                    error="External coding agent not available. Configure API/CLI access.",
                )],
                termination_reason="external_agent_unavailable",
            )
        return await self._invoke_external_agent(context)

    async def stream_events(self, context: AgentRunContext) -> AsyncIterator[TraceEvent]:
        result = await self.run_task(context)
        for event in result.traces:
            yield event

    async def terminate(self) -> None:
        pass

    def collect_metadata(self) -> dict[str, Any]:
        return {
            "agent_type": self.agent_type,
            "agent_id": self._agent_id,
            "available": self._available,
            "config_keys": list(self._config.keys()),
        }


class ClaudeCodeAdapter(ExternalCodingAgentAdapter):
    """Adapter stub for Claude Code integration.

    Requires ANTHROPIC_API_KEY and Claude Code CLI/API access.
    This is a documented integration boundary, not a fake implementation.
    """

    def __init__(self) -> None:
        super().__init__(agent_id="claude_code")

    async def _check_availability(self) -> bool:
        import os
        return bool(os.environ.get("ANTHROPIC_API_KEY")) and self._config.get("cli_path") is not None

    async def _invoke_external_agent(self, context: AgentRunContext) -> AgentResult:
        # Integration point: invoke Claude Code CLI/API here
        # Example: subprocess.run([cli_path, "--task", context.task.input])
        raise NotImplementedError(
            "Claude Code integration requires CLI/API setup. "
            "Subclass and implement _invoke_external_agent with your deployment's interface."
        )
