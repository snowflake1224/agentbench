"""Task abstraction for agent evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskCategory(StrEnum):
    GENERAL = "general"
    TOOL_USE = "tool_use"
    CODING = "coding"
    CUSTOM = "custom"


class Budget(BaseModel):
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_tool_calls: int | None = None
    max_latency_seconds: float | None = None


class ExpectedBehavior(BaseModel):
    """Declarative expectations for task completion."""

    answer: str | None = None
    answer_pattern: str | None = None
    json_schema: dict[str, Any] | None = None
    expected_tools: list[str] = Field(default_factory=list)
    expected_file_changes: list[str] = Field(default_factory=list)
    unit_test_command: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    """Standard task model for agent evaluation."""

    task_id: str
    category: TaskCategory
    input: str
    available_tools: list[str] = Field(default_factory=list)
    expected_behavior: ExpectedBehavior = Field(default_factory=ExpectedBehavior)
    evaluator: str = "task_level"
    timeout: float = 300.0
    budget: Budget = Field(default_factory=Budget)
    metadata: dict[str, Any] = Field(default_factory=dict)
    split: str = "dev"  # dev | validation | test

    model_config = {"frozen": True}
