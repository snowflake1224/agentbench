"""Async SQLAlchemy repository for PostgreSQL."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from agentbench.config import settings
from agentbench.core.experiment import ExperimentRun
from agentbench.core.trace import TraceEvent
from agentbench.storage.models import ExperimentRunRecord, TraceEventRecord
from agentbench.storage.repository import RunRepository


class PostgresRunRepository(RunRepository):
    """Durable run and trace persistence backed by PostgreSQL."""

    def __init__(
        self,
        database_url: str | None = None,
        *,
        engine: AsyncEngine | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._owns_engine = engine is None and session_factory is None
        self._engine = engine or (
            create_async_engine(
                database_url or settings.database_url,
                pool_pre_ping=True,
            )
            if session_factory is None
            else None
        )
        self._sessions = session_factory or async_sessionmaker(self._engine, expire_on_commit=False)

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[AsyncSession]:
        async with self._sessions() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def save_run(self, run: ExperimentRun) -> None:
        payload = run.model_dump(mode="json")
        record = ExperimentRunRecord(
            run_id=run.run_id,
            status=run.status.value,
            benchmark=run.config.benchmark,
            created_at=run.created_at,
            completed_at=run.completed_at,
            payload=payload,
        )
        async with self._session() as session:
            await session.merge(record)

    async def get_run(self, run_id: str) -> ExperimentRun | None:
        async with self._session() as session:
            record = await session.get(ExperimentRunRecord, run_id)
            return ExperimentRun.model_validate(record.payload) if record is not None else None

    async def list_runs(self, limit: int = 50) -> list[ExperimentRun]:
        statement = (
            select(ExperimentRunRecord).order_by(ExperimentRunRecord.created_at.desc()).limit(limit)
        )
        async with self._session() as session:
            records = (await session.scalars(statement)).all()
            return [ExperimentRun.model_validate(record.payload) for record in records]

    async def save_traces(self, run_id: str, traces: list[TraceEvent]) -> None:
        async with self._session() as session:
            await session.execute(delete(TraceEventRecord).where(TraceEventRecord.run_id == run_id))
            session.add_all(
                [
                    TraceEventRecord(
                        event_id=trace.event_id,
                        run_id=run_id,
                        task_id=trace.task_id,
                        event_type=trace.event_type.value,
                        timestamp=trace.timestamp,
                        payload=trace.model_dump(mode="json"),
                    )
                    for trace in traces
                ]
            )

    async def get_traces(self, run_id: str, task_id: str | None = None) -> list[TraceEvent]:
        statement = select(TraceEventRecord).where(TraceEventRecord.run_id == run_id)
        if task_id is not None:
            statement = statement.where(TraceEventRecord.task_id == task_id)
        statement = statement.order_by(TraceEventRecord.timestamp)
        async with self._session() as session:
            records = (await session.scalars(statement)).all()
            return [TraceEvent.model_validate(record.payload) for record in records]

    async def close(self) -> None:
        if self._owns_engine and self._engine is not None:
            await self._engine.dispose()
