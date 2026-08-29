"""Cost estimation utilities."""

from __future__ import annotations

from agentbench.config import settings


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD based on configurable pricing metadata."""
    pricing = settings.pricing.get(model, {"input": 0.0, "output": 0.0})
    input_cost = (input_tokens / 1000) * pricing.get("input", 0.0)
    output_cost = (output_tokens / 1000) * pricing.get("output", 0.0)
    return round(input_cost + output_cost, 6)


def efficiency_metrics(success_rate: float, total_cost: float, total_tokens: int, total_latency_s: float) -> dict[str, float]:
    """Compute efficiency metrics."""
    return {
        "success_per_dollar": success_rate / total_cost if total_cost > 0 else float("inf"),
        "success_per_1k_tokens": success_rate / (total_tokens / 1000) if total_tokens > 0 else 0.0,
        "success_per_second": success_rate / total_latency_s if total_latency_s > 0 else 0.0,
    }
