"""Agent adapter registry and factory."""

from __future__ import annotations

from typing import Any

from agentbench.adapters.external_coding import ClaudeCodeAdapter, ExternalCodingAgentAdapter
from agentbench.adapters.mock_agent import MockAgentAdapter
from agentbench.adapters.model_agent import ModelAgentAdapter
from agentbench.adapters.routed_agent import RoutedAgentAdapter
from agentbench.adapters.tool_router import ToolRoutingStrategy
from agentbench.core.agent import AgentAdapter

AGENT_REGISTRY: dict[str, type[AgentAdapter]] = {
    "mock_single": MockAgentAdapter,
    "mock_planner_executor": lambda **kw: MockAgentAdapter(agent_type="planner_executor", **kw),  # type: ignore[dict-item]
    "mock_planner_executor_critic": lambda **kw: MockAgentAdapter(
        agent_type="planner_executor_critic", **kw
    ),  # type: ignore[dict-item]
    "mock_multi_agent": lambda **kw: MockAgentAdapter(agent_type="multi_agent", **kw),  # type: ignore[dict-item]
    "model_single": ModelAgentAdapter,
    "claude_code": ClaudeCodeAdapter,
    "external_coding": ExternalCodingAgentAdapter,
}


def create_agent(config: dict[str, Any]) -> AgentAdapter:
    """Factory to create agent adapters from configuration."""
    agent_type = config.get("type", "mock_single")
    agent_id = config.get("agent_id")

    if agent_type == "mock_single":
        return MockAgentAdapter(agent_type="single", agent_id=agent_id)
    if agent_type == "model_single":
        return ModelAgentAdapter(agent_id=agent_id or "model_single")
    if agent_type == "planner_executor":
        return MockAgentAdapter(agent_type="planner_executor", agent_id=agent_id)
    if agent_type == "planner_executor_critic":
        return MockAgentAdapter(agent_type="planner_executor_critic", agent_id=agent_id)
    if agent_type == "multi_agent":
        return MockAgentAdapter(agent_type="multi_agent", agent_id=agent_id)
    if agent_type.startswith("routed_"):
        strategy_name = agent_type.replace("routed_", "")
        strategy = ToolRoutingStrategy(strategy_name)
        return RoutedAgentAdapter(routing_strategy=strategy, agent_id=agent_id)
    if agent_type == "claude_code":
        return ClaudeCodeAdapter()
    if agent_type in AGENT_REGISTRY and isinstance(AGENT_REGISTRY[agent_type], type):
        cls = AGENT_REGISTRY[agent_type]
        return cls()  # type: ignore[call-arg]

    raise ValueError(f"Unknown agent type: {agent_type}")
