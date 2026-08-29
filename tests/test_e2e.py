"""End-to-end smoke tests."""

import pytest

from agentbench.benchmarks.registry import get_benchmark
from agentbench.core.experiment import ExperimentConfig
from agentbench.core.runner import ExperimentRunner
from agentbench.storage.repository import InMemoryRunRepository


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_full_pipeline():
    """Smoke test: benchmark -> agent -> evaluator -> storage."""
    benchmark = get_benchmark("local")
    tasks = benchmark.load_tasks(split="dev")
    assert len(tasks) >= 5

    repo = InMemoryRunRepository()
    runner = ExperimentRunner(repo)
    config = ExperimentConfig(
        name="e2e_smoke",
        benchmark="local",
        task_split="dev",
        agent={"type": "planner_executor", "success_rate": 0.85},
        task_subset=[t.task_id for t in tasks[:3]],
    )
    run = await runner.run(config)
    assert run.status.value == "completed"
    assert len(run.task_results) == 3

    traces = await repo.get_traces(run.run_id)
    assert len(traces) > 0

@pytest.mark.e2e
def test_all_benchmarks_load():
    for name in ["local", "gaia", "toolbench", "swebench"]:
        benchmark = get_benchmark(name)
        tasks = benchmark.load_tasks()
        assert len(tasks) > 0, f"{name} returned no tasks"
