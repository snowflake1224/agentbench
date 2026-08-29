"""Prometheus metrics for AgentBench."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

EXPERIMENT_RUNS = Counter(
    "agentbench_experiment_runs_total",
    "Total experiment runs",
    ["benchmark", "agent_type", "status"],
)

TASK_RESULTS = Counter(
    "agentbench_task_results_total",
    "Task evaluation results",
    ["benchmark", "success"],
)

TASK_LATENCY = Histogram(
    "agentbench_task_latency_seconds",
    "Task execution latency",
    ["benchmark", "agent_type"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300],
)

TASK_COST = Histogram(
    "agentbench_task_cost_usd",
    "Estimated task cost in USD",
    ["model"],
    buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)

TOOL_CALLS = Counter(
    "agentbench_tool_calls_total",
    "Tool invocations",
    ["tool", "success"],
)

ACTIVE_RUNS = Gauge(
    "agentbench_active_runs",
    "Currently running experiments",
)

FAILURE_CATEGORIES = Counter(
    "agentbench_failures_total",
    "Failure categories",
    ["category"],
)

FRAMEWORK_INFO = Info("agentbench", "AgentBench framework info")
