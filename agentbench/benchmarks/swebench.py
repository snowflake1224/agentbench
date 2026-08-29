"""SWE-bench benchmark adapter.

SWE-bench evaluates coding agents on real GitHub issues.
Dataset: https://www.swebench.com/

Full SWE-bench requires cloning repositories and Docker for isolated execution.
This adapter provides representative tasks and documents external requirements.
"""

from __future__ import annotations

from agentbench.benchmarks.base import BenchmarkAdapter
from agentbench.core.task import Budget, ExpectedBehavior, Task, TaskCategory

SWEBENCH_SAMPLE = [
    {
        "task_id": "swe_001",
        "input": "Fix IndexError in pandas/core/indexing.py when using iloc with empty DataFrame",
        "repo": "pandas-dev/pandas",
        "tools": ["read_file", "write_file", "run_tests", "grep"],
        "answer": "patch_applied",
    },
    {
        "task_id": "swe_002",
        "input": "Resolve TypeError in numpy array concatenation with mixed dtypes",
        "repo": "numpy/numpy",
        "tools": ["read_file", "write_file", "run_tests", "grep"],
        "answer": "patch_applied",
    },
    {
        "task_id": "swe_003",
        "input": "Fix authentication bypass in login endpoint",
        "repo": "example/webapp",
        "tools": ["read_file", "write_file", "run_tests"],
        "answer": "auth_fixed",
    },
]


class SWEBenchAdapter(BenchmarkAdapter):
    """SWE-bench adapter for coding agent evaluation.

    For full SWE-bench evaluation:
    - Requires Docker for isolated container execution
    - Requires cloning target repositories
    - See https://github.com/princeton-nlp/SWE-bench for setup
    """

    def __init__(self, use_docker: bool = False) -> None:
        self._use_docker = use_docker

    @property
    def name(self) -> str:
        return "swebench"

    @property
    def version(self) -> str:
        return "1.0.0-sample"

    def load_tasks(self, split: str = "dev", subset: list[str] | None = None) -> list[Task]:
        tasks = []
        for item in SWEBENCH_SAMPLE:
            if subset and item["task_id"] not in subset:
                continue
            tasks.append(Task(
                task_id=item["task_id"],
                category=TaskCategory.CODING,
                input=item["input"],
                available_tools=item["tools"],
                expected_behavior=ExpectedBehavior(
                    answer=item["answer"],
                    unit_test_command="pytest tests/",
                ),
                timeout=600.0,
                budget=Budget(max_tool_calls=50, max_tokens=32000),
                metadata={"repo": item["repo"], "source": "swebench", "requires_docker": self._use_docker},
                split=split,
            ))
        return tasks
