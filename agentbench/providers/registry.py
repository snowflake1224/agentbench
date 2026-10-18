"""Model-provider construction."""

from __future__ import annotations

from typing import Any

from agentbench.config import settings
from agentbench.providers.base import ModelProvider
from agentbench.providers.openai_compatible import OpenAICompatibleProvider


def create_provider(config: dict[str, Any] | None = None) -> ModelProvider:
    """Create a configured model provider without exposing credentials."""
    config = config or {}
    provider_type = config.get("provider", "openai_compatible")
    if provider_type != "openai_compatible":
        raise ValueError(f"Unsupported model provider: {provider_type}")

    api_key = config.get("api_key") or settings.openai_api_key
    base_url = config.get("base_url") or settings.openai_base_url
    return OpenAICompatibleProvider(api_key=api_key, base_url=base_url)
