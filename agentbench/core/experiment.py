"""Experiment configuration and run models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentConfig(BaseModel):
    """Configuration-driven experiment specification."""

    name: str
    description: str = ""
    benchmark: str
    task_subset: list[str] | None = None
    task_split: str = "dev"
    agent: dict[str, Any]
    model: str = "mock"
    prompt_version: str = "default"
    tools: list[str] = Field(default_factory=list)
    evaluator_versions: list[str] = Field(default_factory=lambda: ["task_level"])
    seed: int = 42
    concurrency: int = 1
    timeout: float = 300.0
    budget: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskResult(BaseModel):
    """Result of evaluating a single task."""

    task_id: str
    success: bool
    score: float
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    estimated_cost_usd: float = 0.0
    failure_categories: list[str] = Field(default_factory=list)
    evaluator_scores: dict[str, float] = Field(default_factory=dict)
    output: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentRun(BaseModel):
    """Persisted experiment run with reproducibility metadata."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    config: ExperimentConfig
    status: RunStatus = RunStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    git_revision: str | None = None
    benchmark_version: str | None = None
    task_results: list[TaskResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if not self.task_results:
            return 0.0
        return sum(1 for r in self.task_results if r.success) / len(self.task_results)
