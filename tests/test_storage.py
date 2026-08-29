"""Tests for storage repository."""

import pytest

from agentbench.core.experiment import ExperimentConfig, ExperimentRun
from agentbench.core.trace import TraceEvent, TraceEventType
from agentbench.storage.repository import FileRunRepository, InMemoryRunRepository


@pytest.mark.asyncio
async def test_in_memory_repo():
    repo = InMemoryRunRepository()
    run = ExperimentRun(config=ExperimentConfig(name="test", benchmark="local", agent={}))
    await repo.save_run(run)
    loaded = await repo.get_run(run.run_id)
    assert loaded is not None
    assert loaded.config.name == "test"


@pytest.mark.asyncio
async def test_trace_storage(tmp_path):
    repo = FileRunRepository(base_dir=tmp_path)
    run = ExperimentRun(config=ExperimentConfig(name="test", benchmark="local", agent={}))
    await repo.save_run(run)
    traces = [
        TraceEvent(run_id=run.run_id, task_id="t1", event_type=TraceEventType.TOOL_CALL, tool="calc"),
    ]
    await repo.save_traces(run.run_id, traces)
    loaded = await repo.get_traces(run.run_id)
    assert len(loaded) == 1
    assert loaded[0].tool == "calc"
