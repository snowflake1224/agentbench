"""Tests for regression detection."""

from agentbench.core.experiment import ExperimentConfig, ExperimentRun, TaskResult
from agentbench.core.regression import RegressionThresholds, check_regression


def _make_run(successes: list[bool]) -> ExperimentRun:
    return ExperimentRun(
        config=ExperimentConfig(name="test", benchmark="local", agent={}),
        task_results=[
            TaskResult(task_id=f"t{i}", success=s, score=1.0 if s else 0.0, latency_ms=100)
            for i, s in enumerate(successes)
        ],
    )


def test_regression_pass():
    baseline = _make_run([True, True, True, False])
    candidate = _make_run([True, True, True, True])
    result = check_regression(baseline, candidate)
    assert result.passed


def test_regression_fail():
    baseline = _make_run([True, True, True, True, True])
    candidate = _make_run([False, False, True, False, False])
    result = check_regression(baseline, candidate, RegressionThresholds(max_success_decrease=0.05))
    assert not result.passed
