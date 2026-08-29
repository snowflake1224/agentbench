"""Mock deterministic agent for CI and local testing."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from agentbench.core.agent import AgentAdapter, AgentResult, AgentRunContext
from agentbench.core.cost import estimate_cost
from agentbench.core.trace import TraceEvent, TraceEventType


class MockAgentAdapter(AgentAdapter):
    """Deterministic mock agent for testing without API calls."""

    def __init__(self, agent_type: str = "single", agent_id: str | None = None) -> None:
        self._agent_type = agent_type
        self._agent_id = agent_id or f"mock_{agent_type}"
        self._config: dict[str, Any] = {}
        self._tools: dict[str, Any] = {}

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def agent_type(self) -> str:
        return self._agent_type

    async def initialize(self, config: dict[str, Any]) -> None:
        self._config = config
        self._tools = config.get("tools", {})

    async def run_task(self, context: AgentRunContext) -> AgentResult:
        start = time.perf_counter()
        traces: list[TraceEvent] = []
        task = context.task
        expected = task.expected_behavior.answer or ""

        # Simulate architecture-specific behavior
        if self._agent_type == "single":
            traces.extend(await self._run_single(context))
        elif self._agent_type == "planner_executor":
            traces.extend(await self._run_planner_executor(context))
        elif self._agent_type == "planner_executor_critic":
            traces.extend(await self._run_planner_executor_critic(context))
        elif self._agent_type == "multi_agent":
            traces.extend(await self._run_multi_agent(context))
        else:
            traces.extend(await self._run_single(context))

        # Determine output based on task metadata success_rate simulation
        success_rate = self._config.get("success_rate", 0.8)
        task_hash = hash(task.task_id) % 100
        simulated_success = task_hash < int(success_rate * 100)

        if simulated_success and expected:
            output = expected
        elif simulated_success:
            output = f"Completed: {task.input[:50]}"
        else:
            output = f"Failed attempt for: {task.input[:50]}"

        input_tokens = len(task.input.split()) * 2
        output_tokens = len(output.split()) * 2
        tool_calls = sum(1 for t in traces if t.event_type == TraceEventType.TOOL_CALL)
        latency_ms = (time.perf_counter() - start) * 1000

        return AgentResult(
            output=output,
            success=simulated_success,
            traces=traces,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
            latency_ms=latency_ms,
            estimated_cost_usd=estimate_cost(context.model, input_tokens, output_tokens),
            termination_reason="completed" if simulated_success else "failure",
        )

    async def _run_single(self, context: AgentRunContext) -> list[TraceEvent]:
        return [
            TraceEvent(
                run_id=context.run_id,
                task_id=context.task.task_id,
                event_type=TraceEventType.AGENT_START,
                agent_id=self._agent_id,
                model=context.model,
                prompt_version=context.prompt_version,
            ),
            TraceEvent(
                run_id=context.run_id,
                task_id=context.task.task_id,
                event_type=TraceEventType.MODEL_CALL,
                agent_id=self._agent_id,
                model=context.model,
                input_tokens=50,
                output_tokens=30,
                latency_ms=100,
            ),
            TraceEvent(
                run_id=context.run_id,
                task_id=context.task.task_id,
                event_type=TraceEventType.AGENT_END,
                agent_id=self._agent_id,
                termination_reason="completed",
            ),
        ]

    async def _run_planner_executor(self, context: AgentRunContext) -> list[TraceEvent]:
        traces = []
        planner_id = f"{self._agent_id}_planner"
        executor_id = f"{self._agent_id}_executor"

        traces.append(TraceEvent(
            run_id=context.run_id, task_id=context.task.task_id,
            event_type=TraceEventType.AGENT_START, agent_id=planner_id,
            model=context.model, prompt_version=context.prompt_version,
        ))
        traces.append(TraceEvent(
            run_id=context.run_id, task_id=context.task.task_id,
            event_type=TraceEventType.MODEL_CALL, agent_id=planner_id,
            model=context.model, input_tokens=80, output_tokens=40, latency_ms=150,
        ))
        traces.append(TraceEvent(
            run_id=context.run_id, task_id=context.task.task_id,
            event_type=TraceEventType.AGENT_END, agent_id=planner_id,
        ))

        traces.append(TraceEvent(
            run_id=context.run_id, task_id=context.task.task_id,
            event_type=TraceEventType.AGENT_START, agent_id=executor_id,
            parent_agent_id=planner_id, model=context.model,
        ))

        for tool in context.task.available_tools[:2]:
            traces.append(TraceEvent(
                run_id=context.run_id, task_id=context.task.task_id,
                event_type=TraceEventType.TOOL_CALL, agent_id=executor_id,
                tool=tool, tool_input={"query": context.task.input[:30]},
                latency_ms=50,
            ))

        traces.append(TraceEvent(
            run_id=context.run_id, task_id=context.task.task_id,
            event_type=TraceEventType.MODEL_CALL, agent_id=executor_id,
            model=context.model, input_tokens=120, output_tokens=60, latency_ms=200,
        ))
        traces.append(TraceEvent(
            run_id=context.run_id, task_id=context.task.task_id,
            event_type=TraceEventType.AGENT_END, agent_id=executor_id,
            termination_reason="completed",
        ))
        return traces

    async def _run_planner_executor_critic(self, context: AgentRunContext) -> list[TraceEvent]:
        traces = await self._run_planner_executor(context)
        critic_id = f"{self._agent_id}_critic"
        traces.append(TraceEvent(
            run_id=context.run_id, task_id=context.task.task_id,
            event_type=TraceEventType.AGENT_START, agent_id=critic_id,
            model=context.model,
        ))
        traces.append(TraceEvent(
            run_id=context.run_id, task_id=context.task.task_id,
            event_type=TraceEventType.MODEL_CALL, agent_id=critic_id,
            model=context.model, input_tokens=60, output_tokens=20, latency_ms=80,
        ))
        traces.append(TraceEvent(
            run_id=context.run_id, task_id=context.task.task_id,
            event_type=TraceEventType.AGENT_END, agent_id=critic_id,
            termination_reason="approved",
        ))
        return traces

    async def _run_multi_agent(self, context: AgentRunContext) -> list[TraceEvent]:
        traces = await self._run_planner_executor_critic(context)
        specialist_id = f"{self._agent_id}_specialist"
        traces.append(TraceEvent(
            run_id=context.run_id, task_id=context.task.task_id,
            event_type=TraceEventType.AGENT_START, agent_id=specialist_id,
            model=context.model,
        ))
        traces.append(TraceEvent(
            run_id=context.run_id, task_id=context.task.task_id,
            event_type=TraceEventType.TOOL_CALL, agent_id=specialist_id,
            tool="specialist_search", tool_input={"domain": context.task.category.value},
            latency_ms=75,
        ))
        traces.append(TraceEvent(
            run_id=context.run_id, task_id=context.task.task_id,
            event_type=TraceEventType.AGENT_END, agent_id=specialist_id,
        ))
        return traces

    async def stream_events(self, context: AgentRunContext) -> AsyncIterator[TraceEvent]:
        result = await self.run_task(context)
        for event in result.traces:
            yield event

    async def terminate(self) -> None:
        pass

    def collect_metadata(self) -> dict[str, Any]:
        return {
            "agent_type": self._agent_type,
            "agent_id": self._agent_id,
            "config": self._config,
        }
