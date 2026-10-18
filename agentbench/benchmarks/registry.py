"""Benchmark registry and factory."""

from __future__ import annotations

from typing import Any

from agentbench.benchmarks.base import BenchmarkAdapter
from agentbench.benchmarks.gaia import GAIABenchmarkAdapter
from agentbench.benchmarks.local import LocalBenchmarkAdapter
from agentbench.benchmarks.swebench import SWEBenchAdapter
from agentbench.benchmarks.toolbench import ToolBenchAdapter

BENCHMARK_REGISTRY: dict[str, type[BenchmarkAdapter]] = {
    "local": LocalBenchmarkAdapter,
    "gaia": GAIABenchmarkAdapter,
    "toolbench": ToolBenchAdapter,
    "swebench": SWEBenchAdapter,
}


def get_benchmark(name: str, **kwargs: Any) -> BenchmarkAdapter:
    """Get benchmark adapter by name."""
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(BENCHMARK_REGISTRY.keys())}")
    return BENCHMARK_REGISTRY[name](**kwargs)
