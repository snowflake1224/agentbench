"""Tests for evaluators."""

from agentbench.core.agent import AgentResult
from agentbench.core.task import ExpectedBehavior, Task, TaskCategory
from agentbench.core.trace import TraceEvent, TraceEventType
from agentbench.evaluators.base import (
    ExactMatchEvaluator,
    ExpectedToolCallEvaluator,
    TaskLevelEvaluator,
    TrajectoryEvaluator,
)


def _make_task(**kwargs) -> Task:
    defaults = {
        "task_id": "t1",
        "category": TaskCategory.GENERAL,
        "input": "test input",
    }
    defaults.update(kwargs)
    return Task(**defaults)


def _make_result(output: str = "42", traces=None) -> AgentResult:
    return AgentResult(output=output, traces=traces or [])


def test_exact_match_pass():
    ev = ExactMatchEvaluator()
    task = _make_task(expected_behavior=ExpectedBehavior(answer="42"))
    result = _make_result("42")
    r = ev.evaluate(task, result)
    assert r.passed
    assert r.score == 1.0


def test_exact_match_fail():
    ev = ExactMatchEvaluator()
    task = _make_task(expected_behavior=ExpectedBehavior(answer="42"))
    result = _make_result("43")
    r = ev.evaluate(task, result)
    assert not r.passed
    assert r.score == 0.0


def test_expected_tool_call():
    ev = ExpectedToolCallEvaluator()
    task = _make_task(
        available_tools=["search", "calculator"],
        expected_behavior=ExpectedBehavior(expected_tools=["search"]),
    )
    traces = [
        TraceEvent(run_id="r", task_id="t1", event_type=TraceEventType.TOOL_CALL, tool="search"),
    ]
    result = _make_result(traces=traces)
    r = ev.evaluate(task, result)
    assert r.passed


def test_trajectory_loop_detection():
    ev = TrajectoryEvaluator()
    task = _make_task()
    traces = [
        TraceEvent(run_id="r", task_id="t1", event_type=TraceEventType.TOOL_CALL, tool="search")
        for _ in range(6)
    ]
    result = _make_result(traces=traces)
    r = ev.evaluate(task, result)
    assert "looping" in r.failure_categories or r.score < 1.0


def test_task_level_aggregation():
    ev = TaskLevelEvaluator()
    task = _make_task(expected_behavior=ExpectedBehavior(answer="42"))
    result = _make_result("42")
    r = ev.evaluate(task, result)
    assert r.passed
    assert r.score > 0.5
