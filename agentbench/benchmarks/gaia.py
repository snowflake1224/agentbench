"""GAIA benchmark adapter.

GAIA (General AI Assistants) is a benchmark for general AI assistants.
Dataset: https://huggingface.co/datasets/gaia-benchmark/GAIA

This adapter loads a subset of GAIA-style tasks. Full dataset access
requires HuggingFace authentication and accepting dataset terms.
"""

from __future__ import annotations

from agentbench.benchmarks.base import BenchmarkAdapter
from agentbench.core.task import Budget, ExpectedBehavior, Task, TaskCategory

# Representative GAIA-style tasks (publicly described format)
GAIA_SAMPLE_TASKS = [
    {
        "task_id": "gaia_001",
        "input": "A paper about AI regulation was originally scheduled for May 2023. What month was it actually published?",
        "answer": "June 2023",
        "level": 1,
        "tools": ["search", "read_file"],
    },
    {
        "task_id": "gaia_002",
        "input": "According to the 2022 World Bank data, which country had the highest GDP per capita?",
        "answer": "Luxembourg",
        "level": 2,
        "tools": ["search", "calculator"],
    },
    {
        "task_id": "gaia_003",
        "input": "How many studio albums did the artist who released '1989' release before 2020?",
        "answer": "7",
        "level": 2,
        "tools": ["search"],
    },
    {
        "task_id": "gaia_004",
        "input": "What is the sum of the ASCII values of the characters in 'GAIA'?",
        "answer": "280",
        "level": 1,
        "tools": ["calculator"],
    },
    {
        "task_id": "gaia_005",
        "input": "In the Python standard library, which module would you use to parse JSON?",
        "answer": "json",
        "level": 1,
        "tools": ["read_file", "search"],
    },
]


class GAIABenchmarkAdapter(BenchmarkAdapter):
    """GAIA benchmark adapter with sample tasks.

    For full GAIA dataset, set HF_TOKEN and use load_from_huggingface=True.
  """

    def __init__(self, load_from_huggingface: bool = False) -> None:
        self._load_hf = load_from_huggingface

    @property
    def name(self) -> str:
        return "gaia"

    @property
    def version(self) -> str:
        return "1.0.0-sample"

    def load_tasks(self, split: str = "dev", subset: list[str] | None = None) -> list[Task]:
        if self._load_hf:
            return self._load_from_huggingface(split, subset)

        tasks = []
        for item in GAIA_SAMPLE_TASKS:
            if subset and item["task_id"] not in subset:
                continue
            tasks.append(Task(
                task_id=item["task_id"],
                category=TaskCategory.GENERAL,
                input=item["input"],
                available_tools=item["tools"],
                expected_behavior=ExpectedBehavior(answer=item["answer"]),
                timeout=300.0,
                budget=Budget(max_tool_calls=20, max_tokens=8000),
                metadata={"level": item["level"], "source": "gaia"},
                split=split,
            ))
        return tasks

    def _load_from_huggingface(self, split: str, subset: list[str] | None) -> list[Task]:
        """Load from HuggingFace datasets (requires HF_TOKEN)."""
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("Install datasets: pip install datasets") from None

        ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split=split)
        tasks = []
        for row in ds:
            task_id = f"gaia_{row.get('task_id', len(tasks))}"
            if subset and task_id not in subset:
                continue
            tasks.append(Task(
                task_id=task_id,
                category=TaskCategory.GENERAL,
                input=row["Question"],
                available_tools=["search", "calculator", "read_file"],
                expected_behavior=ExpectedBehavior(answer=str(row.get("Final answer", ""))),
                metadata={"level": row.get("Level"), "source": "gaia_hf"},
                split=split,
            ))
        return tasks
