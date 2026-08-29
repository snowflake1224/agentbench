"""Tool-use benchmark adapter (ToolBench-style).

ToolBench evaluates agents on API/tool usage tasks.
Reference: https://github.com/OpenBMB/ToolBench

This adapter provides representative tool-use tasks in ToolBench format.
Full ToolBench requires downloading the dataset separately.
"""

from __future__ import annotations

from agentbench.benchmarks.base import BenchmarkAdapter
from agentbench.core.task import Budget, ExpectedBehavior, Task, TaskCategory

TOOLBENCH_SAMPLE = [
    {
        "task_id": "tool_001",
        "input": "Find the current stock price of AAPL using the finance API.",
        "tools": ["finance_api", "calculator"],
        "expected_tools": ["finance_api"],
        "answer": "AAPL stock price",
    },
    {
        "task_id": "tool_002",
        "input": "Book a flight from NYC to LAX for next Monday.",
        "tools": ["flight_api", "calendar_api", "payment_api"],
        "expected_tools": ["flight_api", "calendar_api"],
        "answer": "flight_booked",
    },
    {
        "task_id": "tool_003",
        "input": "Send an email to team@example.com with the quarterly report attached.",
        "tools": ["email_api", "file_api", "database_query"],
        "expected_tools": ["email_api", "file_api"],
        "answer": "email_sent",
    },
    {
        "task_id": "tool_004",
        "input": "Get weather forecast for Tokyo for the next 3 days.",
        "tools": ["weather_api", "calendar_api"],
        "expected_tools": ["weather_api"],
        "answer": "tokyo_forecast",
    },
    {
        "task_id": "tool_005",
        "input": "Create a new user account with role 'editor' in the CMS.",
        "tools": ["cms_api", "database_query", "auth_api"],
        "expected_tools": ["cms_api", "auth_api"],
        "answer": "user_created",
    },
]


class ToolBenchAdapter(BenchmarkAdapter):
    """Tool-use benchmark adapter."""

    @property
    def name(self) -> str:
        return "toolbench"

    @property
    def version(self) -> str:
        return "1.0.0-sample"

    def load_tasks(self, split: str = "dev", subset: list[str] | None = None) -> list[Task]:
        tasks = []
        for item in TOOLBENCH_SAMPLE:
            if subset and item["task_id"] not in subset:
                continue
            tasks.append(Task(
                task_id=item["task_id"],
                category=TaskCategory.TOOL_USE,
                input=item["input"],
                available_tools=item["tools"],
                expected_behavior=ExpectedBehavior(
                    answer=item["answer"],
                    expected_tools=item["expected_tools"],
                ),
                timeout=120.0,
                budget=Budget(max_tool_calls=15),
                metadata={"source": "toolbench"},
                split=split,
            ))
        return tasks
