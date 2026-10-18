"""Application configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+asyncpg://agentbench:agentbench@localhost:5432/agentbench"
    database_url_sync: str = "postgresql://agentbench:agentbench@localhost:5432/agentbench"
    redis_url: str = "redis://localhost:6379/0"
    otel_exporter_otlp_endpoint: str = ""
    otel_service_name: str = "agentbench"
    artifacts_dir: Path = Path("./artifacts")
    sandbox_enabled: bool = True
    max_tool_output_bytes: int = 1_048_576
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str = ""
    storage_backend: str = "file"

    # Configurable pricing (USD per 1K tokens)
    pricing: dict[str, dict[str, float]] = {
        "mock": {"input": 0.0, "output": 0.0},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    }


settings = Settings()
