"""Benchmark adapter framework."""

from __future__ import annotations

import abc
from typing import Any

from agentbench.core.task import Task


class BenchmarkAdapter(abc.ABC):
    """Base class for benchmark adapters."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Benchmark name identifier."""

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Benchmark version for reproducibility."""

    @abc.abstractmethod
    def load_tasks(self, split: str = "dev", subset: list[str] | None = None) -> list[Task]:
        """Load tasks from benchmark."""

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version}
