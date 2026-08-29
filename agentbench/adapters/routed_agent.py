"""Agent with configurable tool routing."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from agentbench.adapters.mock_agent import MockAgentAdapter
from agentbench.adapters.tool_router import ToolRouter, ToolRoutingStrategy
from agentbench.core.agent import AgentResult, AgentRunContext
from agentbench.core.cost import estimate_cost
from agentbench.core.trace import TraceEvent, TraceEventType


class RoutedAgentAdapter(MockAgentAdapter):
    """Agent adapter with tool routing strategy evaluation."""

    def __init__(
        self,
        routing_strategy: ToolRoutingStrategy = ToolRoutingStrategy.UNRESTRICTED,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(agent_type="routed", agent_id=agent_id or f"routed_{routing_strategy.value}")
        self._router = ToolRouter(routing_strategy)

    async def run_task(self, context: AgentRunContext) -> AgentResult:
        start = time.perf_counter()
        task = context.task
        selected_tools = self._router.select_tools(
            task.category.value, task.available_tools, task.input
        )

        traces: list[TraceEvent] = [
            TraceEvent(
                run_id=context.run_id, task_id=task.task_id,
                event_type=TraceEventType.AGENT_START, agent_id=self._agent_id,
                metadata={"routing_strategy": self._router.strategy.value},
            ),
        ]

        invalid_calls = 0
        for tool in selected_tools:
            is_valid = tool in task.available_tools
            if not is_valid:
                invalid_calls += 1
            traces.append(TraceEvent(
                run_id=context.run_id, task_id=task.task_id,
                event_type=TraceEventType.TOOL_CALL, agent_id=self._agent_id,
                tool=tool, tool_input={"query": task.input[:30]},
                error=None if is_valid else "invalid_tool",
                latency_ms=40,
            ))

        success_rate = self._config.get("success_rate", 0.75)
        # Routing strategy affects success rate simulation
        strategy_bonus = {
            ToolRoutingStrategy.UNRESTRICTED: 0.0,
            ToolRoutingStrategy.RULE_BASED: 0.05,
            ToolRoutingStrategy.MODEL_SELECTED: 0.03,
            ToolRoutingStrategy.SPECIALIZED_AGENT: 0.08,
        }
        adjusted_rate = min(1.0, success_rate + strategy_bonus.get(self._router.strategy, 0))
        task_hash = hash(task.task_id) % 100
        success = task_hash < int(adjusted_rate * 100) and invalid_calls == 0

        expected = task.expected_behavior.answer
        output = expected if success and expected else f"Result for: {task.input[:40]}"

        traces.append(TraceEvent(
            run_id=context.run_id, task_id=task.task_id,
            event_type=TraceEventType.AGENT_END, agent_id=self._agent_id,
            termination_reason="completed" if success else "failure",
        ))

        input_tokens = len(task.input.split()) * 2
        output_tokens = len(output.split()) * 2
        return AgentResult(
            output=output,
            success=success,
            traces=traces,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=len(selected_tools),
            latency_ms=(time.perf_counter() - start) * 1000,
            estimated_cost_usd=estimate_cost(context.model, input_tokens, output_tokens),
            metadata={"invalid_tool_calls": invalid_calls, "routing": self._router.metadata()},
        )

    async def stream_events(self, context: AgentRunContext) -> AsyncIterator[TraceEvent]:
        result = await self.run_task(context)
        for event in result.traces:
            yield event

    def collect_metadata(self) -> dict[str, Any]:
        meta = super().collect_metadata()
        meta["routing"] = self._router.metadata()
        return meta
