"""Framework-independent model provider interfaces."""

from __future__ import annotations

import abc
from typing import Any

from pydantic import BaseModel, Field


class ModelUsage(BaseModel):
    """Normalized token usage returned by a model provider."""

    input_tokens: int = 0
    output_tokens: int = 0


class ModelResponse(BaseModel):
    """Normalized model response."""

    content: str
    model: str
    usage: ModelUsage = Field(default_factory=ModelUsage)
    finish_reason: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class ModelProvider(abc.ABC):
    """Async model-provider boundary used by agents and evaluators."""

    @abc.abstractmethod
    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        response_schema: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> ModelResponse:
        """Generate a completion, optionally constrained to a JSON schema."""

    async def close(self) -> None:
        """Release provider resources."""
        return None
