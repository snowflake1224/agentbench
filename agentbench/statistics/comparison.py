"""Statistical comparison engine for experiment evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats

from agentbench.core.experiment import ExperimentRun


class DistributionStats(BaseModel):
    mean: float
    std: float
    median: float
    min: float
    max: float
    ci_lower: float
    ci_upper: float
    n: int


class ComparisonResult(BaseModel):
    run_a_id: str
    run_b_id: str
    metric: str
    value_a: float
    value_b: float
    difference: float
    ci_lower: float
    ci_upper: float
    p_value: float | None = None
    significant: bool = False
    method: str = "bootstrap"


class ExperimentComparison(BaseModel):
    run_a: str
    run_b: str
    success_rate: ComparisonResult
    latency: ComparisonResult
    cost: ComparisonResult
    tool_calls: ComparisonResult
    failure_distribution_a: dict[str, int] = Field(default_factory=dict)
    failure_distribution_b: dict[str, int] = Field(default_factory=dict)
    efficiency: dict[str, dict[str, float]] = Field(default_factory=dict)


def bootstrap_ci(
    data: np.ndarray,
    statistic: str = "mean",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0

    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        if statistic == "mean":
            boot_stats.append(np.mean(sample))
        elif statistic == "median":
            boot_stats.append(np.median(sample))
        else:
            boot_stats.append(np.mean(sample))

    boot_arr = np.array(boot_stats)
    alpha = (1 - confidence) / 2
    return (
        float(np.mean(data)),
        float(np.percentile(boot_arr, alpha * 100)),
        float(np.percentile(boot_arr, (1 - alpha) * 100)),
    )


def compute_distribution(values: list[float], seed: int = 42) -> DistributionStats:
    """Compute distribution statistics with bootstrap CI."""
    if not values:
        return DistributionStats(mean=0, std=0, median=0, min=0, max=0, ci_lower=0, ci_upper=0, n=0)
    arr = np.array(values)
    mean, ci_lower, ci_upper = bootstrap_ci(arr, seed=seed)
    return DistributionStats(
        mean=mean,
        std=float(np.std(arr)),
        median=float(np.median(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n=len(values),
    )


def compare_runs(
    run_a: ExperimentRun, run_b: ExperimentRun, seed: int = 42
) -> ExperimentComparison:
    """Statistically compare two experiment runs."""

    def extract(run: ExperimentRun, attr: str) -> list[float]:
        return [getattr(r, attr) for r in run.task_results]

    def compare_metric(metric: str, transform=None) -> ComparisonResult:
        vals_a = extract(run_a, metric)
        vals_b = extract(run_b, metric)
        if transform:
            vals_a = [transform(v) for v in vals_a]
            vals_b = [transform(v) for v in vals_b]

        arr_a = np.array(vals_a) if vals_a else np.array([0.0])
        arr_b = np.array(vals_b) if vals_b else np.array([0.0])

        mean_a, ci_a_lo, ci_a_hi = bootstrap_ci(arr_a, seed=seed)
        mean_b, ci_b_lo, ci_b_hi = bootstrap_ci(arr_b, seed=seed)
        diff = mean_b - mean_a

        # Paired comparison if same tasks
        p_value = None
        significant = False
        if len(vals_a) == len(vals_b) and len(vals_a) > 1:
            task_ids_a = [r.task_id for r in run_a.task_results]
            task_ids_b = [r.task_id for r in run_b.task_results]
            if task_ids_a == task_ids_b:
                if np.allclose(arr_a, arr_b):
                    p_value = 1.0
                else:
                    try:
                        _, p_value = stats.wilcoxon(arr_a, arr_b)
                        significant = p_value < 0.05
                    except ValueError:
                        pass

        return ComparisonResult(
            run_a_id=run_a.run_id,
            run_b_id=run_b.run_id,
            metric=metric,
            value_a=mean_a,
            value_b=mean_b,
            difference=diff,
            ci_lower=ci_b_lo - ci_a_hi,
            ci_upper=ci_b_hi - ci_a_lo,
            p_value=p_value,
            significant=significant,
        )

    def failure_dist(run: ExperimentRun) -> dict[str, int]:
        dist: dict[str, int] = {}
        for r in run.task_results:
            for f in r.failure_categories:
                dist[f] = dist.get(f, 0) + 1
        return dist

    def efficiency(run: ExperimentRun) -> dict[str, float]:
        n = len(run.task_results)
        if n == 0:
            return {}
        success = sum(1 for r in run.task_results if r.success) / n
        total_cost = sum(r.estimated_cost_usd for r in run.task_results)
        total_tokens = sum(r.input_tokens + r.output_tokens for r in run.task_results)
        total_latency = sum(r.latency_ms for r in run.task_results) / 1000
        return {
            "success_per_dollar": success / total_cost if total_cost > 0 else float("inf"),
            "success_per_1k_tokens": success / (total_tokens / 1000) if total_tokens > 0 else 0,
            "success_per_second": success / total_latency if total_latency > 0 else 0,
        }

    return ExperimentComparison(
        run_a=run_a.run_id,
        run_b=run_b.run_id,
        success_rate=compare_metric("success", transform=lambda x: float(x)),
        latency=compare_metric("latency_ms"),
        cost=compare_metric("estimated_cost_usd"),
        tool_calls=compare_metric("tool_calls"),
        failure_distribution_a=failure_dist(run_a),
        failure_distribution_b=failure_dist(run_b),
        efficiency={"a": efficiency(run_a), "b": efficiency(run_b)},
    )


def run_summary(run: ExperimentRun, seed: int = 42) -> dict[str, Any]:
    """Generate statistical summary for a single run."""
    results = run.task_results
    if not results:
        return {"run_id": run.run_id, "n_tasks": 0}

    successes = [float(r.success) for r in results]
    latencies = [r.latency_ms for r in results]
    costs = [r.estimated_cost_usd for r in results]
    tokens = [r.input_tokens + r.output_tokens for r in results]
    tool_calls = [float(r.tool_calls) for r in results]

    failure_dist: dict[str, int] = {}
    for r in results:
        for f in r.failure_categories:
            failure_dist[f] = failure_dist.get(f, 0) + 1

    return {
        "run_id": run.run_id,
        "name": run.config.name,
        "n_tasks": len(results),
        "success_rate": compute_distribution(successes, seed).model_dump(),
        "latency_ms": compute_distribution(latencies, seed).model_dump(),
        "cost_usd": compute_distribution(costs, seed).model_dump(),
        "tokens": compute_distribution([float(t) for t in tokens], seed).model_dump(),
        "tool_calls": compute_distribution(tool_calls, seed).model_dump(),
        "failure_distribution": failure_dist,
        "git_revision": run.git_revision,
        "benchmark_version": run.benchmark_version,
    }
