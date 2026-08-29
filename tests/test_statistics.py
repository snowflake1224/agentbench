"""Tests for statistical comparison."""

import numpy as np

from agentbench.core.experiment import ExperimentConfig, ExperimentRun, TaskResult
from agentbench.statistics.comparison import bootstrap_ci, compare_runs, compute_distribution, run_summary


def _make_run(name: str, successes: list[bool], latencies: list[float]) -> ExperimentRun:
    return ExperimentRun(
        config=ExperimentConfig(name=name, benchmark="local", agent={"type": "mock"}),
        task_results=[
            TaskResult(
                task_id=f"t{i}",
                success=s,
                score=1.0 if s else 0.0,
                latency_ms=lat,
            )
            for i, (s, lat) in enumerate(zip(successes, latencies))
        ],
    )


def test_bootstrap_ci():
    data = np.array([0.8, 0.9, 0.7, 0.85, 0.75, 0.95, 0.8, 0.9, 0.85, 0.8])
    mean, lo, hi = bootstrap_ci(data, seed=42)
    assert lo <= mean <= hi
    assert 0.5 < mean < 1.0


def test_compute_distribution():
    dist = compute_distribution([1.0, 2.0, 3.0, 4.0, 5.0], seed=42)
    assert dist.mean == 3.0
    assert dist.n == 5
    assert dist.ci_lower <= dist.mean <= dist.ci_upper


def test_compare_runs():
    run_a = _make_run("a", [True, True, False, True], [100, 200, 150, 120])
    run_b = _make_run("b", [True, True, True, True], [80, 90, 100, 110])
    comp = compare_runs(run_a, run_b, seed=42)
    assert comp.success_rate.value_b > comp.success_rate.value_a
    assert comp.latency.value_b < comp.latency.value_a


def test_run_summary():
    run = _make_run("test", [True, False, True], [100, 200, 150])
    summary = run_summary(run)
    assert summary["n_tasks"] == 3
    assert 0 < summary["success_rate"]["mean"] < 1
