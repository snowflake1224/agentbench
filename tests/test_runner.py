"""Tests for experiment runner."""

import pytest

from agentbench.core.experiment import ExperimentConfig
from agentbench.core.runner import ExperimentRunner
from agentbench.storage.repository import InMemoryRunRepository


@pytest.mark.asyncio
async def test_experiment_runner():
    repo = InMemoryRunRepository()
    runner = ExperimentRunner(repo)
    config = ExperimentConfig(
        name="test_run",
        benchmark="local",
        task_split="dev",
        agent={"type": "mock_single", "success_rate": 0.8},
        concurrency=2,
    )
    run = await runner.run(config)
    assert run.status.value == "completed"
    assert len(run.task_results) > 0
    saved = await repo.get_run(run.run_id)
    assert saved is not None


@pytest.mark.asyncio
async def test_traces_persisted():
    repo = InMemoryRunRepository()
    runner = ExperimentRunner(repo)
    config = ExperimentConfig(
        name="trace_test",
        benchmark="local",
        task_split="dev",
        agent={"type": "mock_single"},
        task_subset=["local_001"],
    )
    run = await runner.run(config)
    traces = await repo.get_traces(run.run_id)
    assert len(traces) > 0
