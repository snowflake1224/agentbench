"""Regression testing gate."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentbench.core.experiment import ExperimentRun
from agentbench.statistics.comparison import compare_runs, run_summary


class RegressionThresholds(BaseModel):
    max_success_decrease: float = 0.05
    max_latency_increase_pct: float = 0.20
    max_cost_increase_pct: float = 0.30
    max_tool_error_rate: float = 0.10
    critical_categories: list[str] = Field(default_factory=lambda: ["coding"])


class RegressionResult(BaseModel):
    passed: bool
    checks: list[dict]
    baseline_run_id: str
    candidate_run_id: str
    summary: dict


def check_regression(
    baseline: ExperimentRun,
    candidate: ExperimentRun,
    thresholds: RegressionThresholds | None = None,
) -> RegressionResult:
    """Compare candidate against baseline for regression detection."""
    thresholds = thresholds or RegressionThresholds()
    comparison = compare_runs(baseline, candidate)
    checks: list[dict] = []

    # Success rate check
    success_diff = comparison.success_rate.difference
    success_ok = success_diff >= -thresholds.max_success_decrease
    checks.append({
        "check": "success_rate",
        "passed": success_ok,
        "baseline": comparison.success_rate.value_a,
        "candidate": comparison.success_rate.value_b,
        "difference": success_diff,
        "threshold": -thresholds.max_success_decrease,
    })

    # Latency check
    if comparison.latency.value_a > 0:
        latency_increase = (comparison.latency.value_b - comparison.latency.value_a) / comparison.latency.value_a
        latency_ok = latency_increase <= thresholds.max_latency_increase_pct
    else:
        latency_increase = 0
        latency_ok = True
    checks.append({
        "check": "latency",
        "passed": latency_ok,
        "baseline_ms": comparison.latency.value_a,
        "candidate_ms": comparison.latency.value_b,
        "increase_pct": latency_increase,
        "threshold_pct": thresholds.max_latency_increase_pct,
    })

    # Cost check
    if comparison.cost.value_a > 0:
        cost_increase = (comparison.cost.value_b - comparison.cost.value_a) / comparison.cost.value_a
        cost_ok = cost_increase <= thresholds.max_cost_increase_pct
    else:
        cost_increase = 0
        cost_ok = True
    checks.append({
        "check": "cost",
        "passed": cost_ok,
        "baseline_usd": comparison.cost.value_a,
        "candidate_usd": comparison.cost.value_b,
        "increase_pct": cost_increase,
        "threshold_pct": thresholds.max_cost_increase_pct,
    })

    # Tool error rate
    candidate_errors = sum(
        comparison.failure_distribution_b.get(f, 0)
        for f in ["invalid_tool_arguments", "tool_failure", "incorrect_tool_selection"]
    )
    total_tasks = len(candidate.task_results) or 1
    tool_error_rate = candidate_errors / total_tasks
    tool_ok = tool_error_rate <= thresholds.max_tool_error_rate
    checks.append({
        "check": "tool_error_rate",
        "passed": tool_ok,
        "rate": tool_error_rate,
        "threshold": thresholds.max_tool_error_rate,
    })

    all_passed = all(c["passed"] for c in checks)
    return RegressionResult(
        passed=all_passed,
        checks=checks,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        summary={
            "baseline": run_summary(baseline),
            "candidate": run_summary(candidate),
        },
    )
