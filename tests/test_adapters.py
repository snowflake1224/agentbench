"""Tests for agent adapters."""

import pytest

from agentbench.adapters.mock_agent import MockAgentAdapter
from agentbench.adapters.registry import create_agent
from agentbench.adapters.routed_agent import RoutedAgentAdapter
from agentbench.adapters.tool_router import ToolRouter, ToolRoutingStrategy
from agentbench.core.agent import AgentRunContext
from agentbench.core.task import Task, TaskCategory


def _make_context(task_id: str = "t1") -> AgentRunContext:
    return AgentRunContext(
        run_id="run-1",
        task=Task(
            task_id=task_id,
            category=TaskCategory.GENERAL,
            input="What is 2+2?",
            available_tools=["calculator", "search"],
        ),
    )


@pytest.mark.asyncio
async def test_mock_single_agent():
    agent = MockAgentAdapter(agent_type="single")
    await agent.initialize({"success_rate": 0.9})
    result = await agent.run_task(_make_context())
    assert result.output
    assert len(result.traces) > 0
    await agent.terminate()


@pytest.mark.asyncio
async def test_planner_executor_agent():
    agent = MockAgentAdapter(agent_type="planner_executor")
    await agent.initialize({"success_rate": 0.9})
    result = await agent.run_task(_make_context())
    agents = {t.agent_id for t in result.traces if t.agent_id}
    assert len(agents) >= 2
    await agent.terminate()


@pytest.mark.asyncio
async def test_create_agent_factory():
    agent = create_agent({"type": "planner_executor_critic"})
    assert agent.agent_type == "planner_executor_critic"


def test_tool_router_rule_based():
    router = ToolRouter(ToolRoutingStrategy.RULE_BASED)
    tools = router.select_tools("coding", ["read_file", "write_file", "api_call"])
    assert "read_file" in tools
    assert "api_call" not in tools


@pytest.mark.asyncio
async def test_routed_agent():
    agent = RoutedAgentAdapter(ToolRoutingStrategy.SPECIALIZED_AGENT)
    await agent.initialize({"success_rate": 0.9})
    result = await agent.run_task(_make_context())
    assert "routing" in result.metadata
    await agent.terminate()
