"""Deterministic local benchmark for CI and development."""

from __future__ import annotations

from agentbench.benchmarks.base import BenchmarkAdapter
from agentbench.core.task import Budget, ExpectedBehavior, Task, TaskCategory

FIXTURE_TASKS: list[dict] = [
    {
        "task_id": "local_001",
        "category": "general",
        "input": "What is 15 + 27?",
        "tools": ["calculator"],
        "answer": "42",
        "split": "dev",
    },
    {
        "task_id": "local_002",
        "category": "general",
        "input": "Find the capital of France.",
        "tools": ["search"],
        "answer": "Paris",
        "split": "dev",
    },
    {
        "task_id": "local_003",
        "category": "tool_use",
        "input": "Query the database for users with status active.",
        "tools": ["database_query", "api_call"],
        "answer": "active_users_list",
        "split": "dev",
    },
    {
        "task_id": "local_004",
        "category": "tool_use",
        "input": "Call the weather API for London.",
        "tools": ["api_call", "calculator"],
        "answer": "London weather data",
        "split": "validation",
    },
    {
        "task_id": "local_005",
        "category": "coding",
        "input": "Fix the failing test in test_auth.py",
        "tools": ["read_file", "write_file", "run_tests"],
        "answer": "test_auth_fixed",
        "split": "dev",
    },
    {
        "task_id": "local_006",
        "category": "coding",
        "input": "Implement a function to reverse a string.",
        "tools": ["read_file", "write_file", "run_tests", "grep"],
        "answer": "reverse_string_implemented",
        "split": "validation",
    },
    {
        "task_id": "local_007",
        "category": "general",
        "input": "Summarize the key points of machine learning.",
        "tools": ["search", "read_file"],
        "answer": "ML summary",
        "split": "test",
    },
    {
        "task_id": "local_008",
        "category": "tool_use",
        "input": "Calculate compound interest for $1000 at 5% for 3 years.",
        "tools": ["calculator"],
        "answer": "1157.63",
        "split": "test",
    },
    {
        "task_id": "local_009",
        "category": "coding",
        "input": "Add error handling to the API endpoint.",
        "tools": ["read_file", "write_file", "run_tests"],
        "answer": "error_handling_added",
        "split": "test",
    },
    {
        "task_id": "local_010",
        "category": "general",
        "input": "List three benefits of unit testing.",
        "tools": ["search"],
        "answer": "unit testing benefits",
        "split": "dev",
    },
    {
        "task_id": "local_011",
        "category": "tool_use",
        "input": "Fetch user profile from API endpoint /users/123.",
        "tools": ["api_call", "database_query"],
        "answer": "user_profile_123",
        "split": "dev",
    },
    {
        "task_id": "local_012",
        "category": "coding",
        "input": "Refactor duplicate code in utils module.",
        "tools": ["read_file", "write_file", "grep", "run_tests"],
        "answer": "utils_refactored",
        "split": "validation",
    },
    {
        "task_id": "local_013",
        "category": "general",
        "input": "Explain the difference between sync and async programming.",
        "tools": ["search", "read_file"],
        "answer": "sync vs async explanation",
        "split": "validation",
    },
    {
        "task_id": "local_014",
        "category": "tool_use",
        "input": "Convert 100 USD to EUR using exchange rate API.",
        "tools": ["api_call", "calculator"],
        "answer": "EUR equivalent",
        "split": "test",
    },
    {
        "task_id": "local_015",
        "category": "coding",
        "input": "Write unit tests for the authentication module.",
        "tools": ["read_file", "write_file", "run_tests"],
        "answer": "auth_tests_written",
        "split": "test",
    },
    {
        "task_id": "local_016",
        "category": "general",
        "input": "What are the SOLID principles?",
        "tools": ["search"],
        "answer": "SOLID principles",
        "split": "dev",
    },
    {
        "task_id": "local_017",
        "category": "tool_use",
        "input": "Search logs for ERROR entries in the last hour.",
        "tools": ["grep", "database_query"],
        "answer": "error_log_entries",
        "split": "dev",
    },
    {
        "task_id": "local_018",
        "category": "coding",
        "input": "Optimize the database query in reports.py.",
        "tools": ["read_file", "write_file", "run_tests"],
        "answer": "query_optimized",
        "split": "validation",
    },
    {
        "task_id": "local_019",
        "category": "general",
        "input": "Describe the CAP theorem.",
        "tools": ["search", "read_file"],
        "answer": "CAP theorem description",
        "split": "test",
    },
    {
        "task_id": "local_020",
        "category": "coding",
        "input": "Fix memory leak in data processing pipeline.",
        "tools": ["read_file", "write_file", "run_tests", "grep"],
        "answer": "memory_leak_fixed",
        "split": "test",
    },
]


class LocalBenchmarkAdapter(BenchmarkAdapter):
    """Deterministic local benchmark suite for CI and development."""

    @property
    def name(self) -> str:
        return "local"

    @property
    def version(self) -> str:
        return "1.0.0"

    def load_tasks(self, split: str = "dev", subset: list[str] | None = None) -> list[Task]:
        tasks = []
        for fixture in FIXTURE_TASKS:
            if fixture["split"] != split and split != "all":
                continue
            if subset and fixture["task_id"] not in subset:
                continue
            tasks.append(Task(
                task_id=fixture["task_id"],
                category=TaskCategory(fixture["category"]),
                input=fixture["input"],
                available_tools=fixture["tools"],
                expected_behavior=ExpectedBehavior(answer=fixture["answer"]),
                timeout=60.0,
                budget=Budget(max_tool_calls=10, max_tokens=4000),
                split=fixture["split"],
            ))
        return tasks
