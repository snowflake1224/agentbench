"""Tests for storage repository."""

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agentbench.core.experiment import ExperimentConfig, ExperimentRun
from agentbench.core.trace import TraceEvent, TraceEventType
from agentbench.storage.models import Base
from agentbench.storage.postgres import PostgresRunRepository
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
        TraceEvent(
            run_id=run.run_id, task_id="t1", event_type=TraceEventType.TOOL_CALL, tool="calc"
        ),
    ]
    await repo.save_traces(run.run_id, traces)
    loaded = await repo.get_traces(run.run_id)
    assert len(loaded) == 1
    assert loaded[0].tool == "calc"


@pytest.mark.asyncio
async def test_sql_repository_contract_with_async_database():
    """Exercise the PostgreSQL repository through SQLAlchemy using SQLite in CI."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    repo = PostgresRunRepository(engine=engine, session_factory=sessions)

    run = ExperimentRun(config=ExperimentConfig(name="sql-test", benchmark="local", agent={}))
    await repo.save_run(run)
    trace = TraceEvent(
        run_id=run.run_id,
        task_id="t1",
        event_type=TraceEventType.MODEL_CALL,
        model="test-model",
    )
    await repo.save_traces(run.run_id, [trace])

    loaded = await repo.get_run(run.run_id)
    traces = await repo.get_traces(run.run_id, task_id="t1")
    listed = await repo.list_runs()
    assert loaded is not None and loaded.run_id == run.run_id
    assert traces == [trace]
    assert listed[0].config.name == "sql-test"
    await engine.dispose()
