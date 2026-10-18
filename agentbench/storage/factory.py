"""Storage backend selection."""

from __future__ import annotations

from agentbench.config import settings
from agentbench.storage.repository import FileRunRepository, RunRepository


def create_run_repository(backend: str | None = None) -> RunRepository:
    """Construct the configured storage backend."""
    selected = backend or settings.storage_backend
    if selected == "file":
        return FileRunRepository()
    if selected == "postgres":
        from agentbench.storage.postgres import PostgresRunRepository

        return PostgresRunRepository()
    raise ValueError(f"Unsupported storage backend: {selected}")
