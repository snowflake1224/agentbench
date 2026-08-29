"""Prompt version management."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agentbench.config import settings


class PromptVersion(BaseModel):
    prompt_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_prompt_id: str | None = None
    name: str
    content: str
    experiment_id: str | None = None
    benchmark_results: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptStore:
    """Immutable prompt version store."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base = base_dir or settings.artifacts_dir / "prompts"
        self._base.mkdir(parents=True, exist_ok=True)
        self._index_file = self._base / "index.json"
        self._load_index()

    def _load_index(self) -> None:
        if self._index_file.exists():
            self._index: dict[str, str] = json.loads(self._index_file.read_text())
        else:
            self._index = {}

    def _save_index(self) -> None:
        self._index_file.write_text(json.dumps(self._index, indent=2))

    def save(self, prompt: PromptVersion) -> PromptVersion:
        """Save prompt version (never overwrites)."""
        path = self._base / f"{prompt.prompt_id}.json"
        if path.exists():
            raise ValueError(f"Prompt {prompt.prompt_id} already exists")
        path.write_text(prompt.model_dump_json(indent=2))
        self._index[prompt.prompt_id] = prompt.name
        self._save_index()
        return prompt

    def get(self, prompt_id: str) -> PromptVersion | None:
        path = self._base / f"{prompt_id}.json"
        if not path.exists():
            return None
        return PromptVersion.model_validate_json(path.read_text())

    def list_versions(self) -> list[PromptVersion]:
        versions = []
        for pid in self._index:
            p = self.get(pid)
            if p:
                versions.append(p)
        return sorted(versions, key=lambda p: p.created_at)

    def create_variant(self, parent_id: str, name: str, content: str, experiment_id: str | None = None) -> PromptVersion:
        parent = self.get(parent_id)
        if not parent:
            raise ValueError(f"Parent prompt {parent_id} not found")
        variant = PromptVersion(
            parent_prompt_id=parent_id,
            name=name,
            content=content,
            experiment_id=experiment_id,
        )
        return self.save(variant)
