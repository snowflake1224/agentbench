"""Structured failure taxonomy for agent evaluation."""

from __future__ import annotations

from enum import StrEnum


class FailureCategory(StrEnum):
    """Structured failure categories for agent evaluation."""

    MISUNDERSTANDING_TASK = "misunderstanding_task"
    HALLUCINATION = "hallucination"
    INCORRECT_TOOL_SELECTION = "incorrect_tool_selection"
    INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
    TOOL_FAILURE = "tool_failure"
    CONTEXT_FAILURE = "context_failure"
    PLANNING_FAILURE = "planning_failure"
    IMPLEMENTATION_FAILURE = "implementation_failure"
    VERIFICATION_FAILURE = "verification_failure"
    LOOPING = "looping"
    TIMEOUT = "timeout"
    BUDGET_EXHAUSTION = "budget_exhaustion"
    UNKNOWN = "unknown"


FAILURE_DESCRIPTIONS: dict[FailureCategory, str] = {
    FailureCategory.MISUNDERSTANDING_TASK: "Agent misinterpreted the task requirements",
    FailureCategory.HALLUCINATION: "Agent produced factually incorrect or fabricated content",
    FailureCategory.INCORRECT_TOOL_SELECTION: "Agent selected wrong tool for the task",
    FailureCategory.INVALID_TOOL_ARGUMENTS: "Agent provided invalid arguments to a tool",
    FailureCategory.TOOL_FAILURE: "Tool execution failed due to external or runtime error",
    FailureCategory.CONTEXT_FAILURE: "Agent lost or exceeded context window",
    FailureCategory.PLANNING_FAILURE: "Agent's plan was flawed or incomplete",
    FailureCategory.IMPLEMENTATION_FAILURE: "Agent's implementation was incorrect",
    FailureCategory.VERIFICATION_FAILURE: "Agent failed to verify its own output",
    FailureCategory.LOOPING: "Agent entered a repetitive action loop",
    FailureCategory.TIMEOUT: "Task exceeded time limit",
    FailureCategory.BUDGET_EXHAUSTION: "Task exceeded token or cost budget",
    FailureCategory.UNKNOWN: "Failure could not be categorized",
}
