"""Storage repository abstraction."""

from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Any

from agentbench.config import settings
from agentbench.core.experiment import ExperimentRun, RunStatus
from agentbench.core.trace import TraceEvent


class RunRepository(abc.ABC):
    """Abstract repository for experiment runs."""

    @abc.abstractmethod
    async def save_run(self, run: ExperimentRun) -> None: ...

    @abc.abstractmethod
    async def get_run(self, run_id: str) -> ExperimentRun | None: ...

    @abc.abstractmethod
    async def list_runs(self, limit: int = 50) -> list[ExperimentRun]: ...

    @abc.abstractmethod
    async def save_traces(self, run_id: str, traces: list[TraceEvent]) -> None: ...

    @abc.abstractmethod
    async def get_traces(self, run_id: str, task_id: str | None = None) -> list[TraceEvent]: ...


class FileRunRepository(RunRepository):
    """File-based repository for development and testing."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base = base_dir or settings.artifacts_dir / "runs"
        self._base.mkdir(parents=True, exist_ok=True)

    def _run_path(self, run_id: str) -> Path:
        return self._base / run_id

    async def save_run(self, run: ExperimentRun) -> None:
        path = self._run_path(run.run_id)
        path.mkdir(parents=True, exist_ok=True)
        (path / "run.json").write_text(run.model_dump_json(indent=2))

    async def get_run(self, run_id: str) -> ExperimentRun | None:
        run_file = self._run_path(run_id) / "run.json"
        if not run_file.exists():
            return None
        return ExperimentRun.model_validate_json(run_file.read_text())

    async def list_runs(self, limit: int = 50) -> list[ExperimentRun]:
        runs = []
        for d in sorted(self._base.iterdir(), reverse=True):
            if d.is_dir() and (d / "run.json").exists():
                runs.append(ExperimentRun.model_validate_json((d / "run.json").read_text()))
                if len(runs) >= limit:
                    break
        return runs

    async def save_traces(self, run_id: str, traces: list[TraceEvent]) -> None:
        path = self._run_path(run_id)
        path.mkdir(parents=True, exist_ok=True)
        trace_data = [t.model_dump(mode="json") for t in traces]
        (path / "traces.json").write_text(json.dumps(trace_data, indent=2, default=str))

    async def get_traces(self, run_id: str, task_id: str | None = None) -> list[TraceEvent]:
        trace_file = self._run_path(run_id) / "traces.json"
        if not trace_file.exists():
            return []
        data = json.loads(trace_file.read_text())
        events = [TraceEvent.model_validate(e) for e in data]
        if task_id:
            events = [e for e in events if e.task_id == task_id]
        return events


class InMemoryRunRepository(RunRepository):
    """In-memory repository for testing."""

    def __init__(self) -> None:
        self._runs: dict[str, ExperimentRun] = {}
        self._traces: dict[str, list[TraceEvent]] = {}

    async def save_run(self, run: ExperimentRun) -> None:
        self._runs[run.run_id] = run

    async def get_run(self, run_id: str) -> ExperimentRun | None:
        return self._runs.get(run_id)

    async def list_runs(self, limit: int = 50) -> list[ExperimentRun]:
        return list(self._runs.values())[:limit]

    async def save_traces(self, run_id: str, traces: list[TraceEvent]) -> None:
        self._traces[run_id] = traces

    async def get_traces(self, run_id: str, task_id: str | None = None) -> list[TraceEvent]:
        events = self._traces.get(run_id, [])
        if task_id:
            events = [e for e in events if e.task_id == task_id]
        return events
