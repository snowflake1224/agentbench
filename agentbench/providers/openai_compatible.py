"""OpenAI-compatible HTTP model provider."""

from __future__ import annotations

from typing import Any

import httpx

from agentbench.providers.base import ModelProvider, ModelResponse, ModelUsage


class ProviderError(RuntimeError):
    """Raised when a model provider request fails."""


class OpenAICompatibleProvider(ModelProvider):
    """Call OpenAI or any compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("A model-provider API key is required")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        response_schema: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "agentbench_response",
                    "strict": True,
                    "schema": response_schema,
                },
            }

        try:
            response = await self._client.post("/chat/completions", json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})
            return ModelResponse(
                content=choice["message"]["content"] or "",
                model=data.get("model", model),
                usage=ModelUsage(
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                ),
                finish_reason=choice.get("finish_reason"),
                raw=data,
            )
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(f"Model request failed: {exc}") from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
