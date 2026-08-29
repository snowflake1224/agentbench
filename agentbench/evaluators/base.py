"""Evaluator plugin framework."""

from __future__ import annotations

import abc
import re
from typing import Any

import jsonschema
from pydantic import BaseModel, Field

from agentbench.core.agent import AgentResult
from agentbench.core.failure import FailureCategory
from agentbench.core.task import Task
from agentbench.core.trace import TraceEvent, TraceEventType


class EvaluationResult(BaseModel):
    """Result from a single evaluator."""

    evaluator_id: str
    version: str = "v1"
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    failure_categories: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class Evaluator(abc.ABC):
    """Base evaluator interface."""

    @property
    @abc.abstractmethod
    def evaluator_id(self) -> str:
        pass

    @property
    def version(self) -> str:
        return "v1"

    @abc.abstractmethod
    def evaluate(self, task: Task, result: AgentResult) -> EvaluationResult:
        pass


class ExactMatchEvaluator(Evaluator):
    @property
    def evaluator_id(self) -> str:
        return "exact_match"

    def evaluate(self, task: Task, result: AgentResult) -> EvaluationResult:
        expected = task.expected_behavior.answer
        if expected is None:
            return EvaluationResult(evaluator_id=self.evaluator_id, version=self.version, passed=True, score=1.0)
        passed = result.output.strip().lower() == expected.strip().lower()
        return EvaluationResult(
            evaluator_id=self.evaluator_id,
            version=self.version,
            passed=passed,
            score=1.0 if passed else 0.0,
            failure_categories=[FailureCategory.IMPLEMENTATION_FAILURE.value] if not passed else [],
        )


class RegexEvaluator(Evaluator):
    @property
    def evaluator_id(self) -> str:
        return "regex"

    def evaluate(self, task: Task, result: AgentResult) -> EvaluationResult:
        pattern = task.expected_behavior.answer_pattern
        if not pattern:
            return EvaluationResult(evaluator_id=self.evaluator_id, version=self.version, passed=True, score=1.0)
        passed = bool(re.search(pattern, result.output, re.I))
        return EvaluationResult(
            evaluator_id=self.evaluator_id, version=self.version, passed=passed, score=1.0 if passed else 0.0,
        )


class JsonSchemaEvaluator(Evaluator):
    @property
    def evaluator_id(self) -> str:
        return "json_schema"

    def evaluate(self, task: Task, result: AgentResult) -> EvaluationResult:
        schema = task.expected_behavior.json_schema
        if not schema:
            return EvaluationResult(evaluator_id=self.evaluator_id, version=self.version, passed=True, score=1.0)
        import json
        try:
            data = json.loads(result.output)
            jsonschema.validate(data, schema)
            return EvaluationResult(evaluator_id=self.evaluator_id, version=self.version, passed=True, score=1.0)
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            return EvaluationResult(
                evaluator_id=self.evaluator_id, version=self.version, passed=False, score=0.0,
                details={"error": str(e)},
            )


class ExpectedToolCallEvaluator(Evaluator):
    @property
    def evaluator_id(self) -> str:
        return "expected_tool_call"

    def evaluate(self, task: Task, result: AgentResult) -> EvaluationResult:
        expected = set(task.expected_behavior.expected_tools)
        if not expected:
            return EvaluationResult(evaluator_id=self.evaluator_id, version=self.version, passed=True, score=1.0)
        actual = {e.tool for e in result.traces if e.event_type == TraceEventType.TOOL_CALL and e.tool}
        missing = expected - actual
        extra_invalid = {e.tool for e in result.traces if e.event_type == TraceEventType.TOOL_CALL and e.error}
        passed = not missing and not extra_invalid
        score = len(expected & actual) / len(expected) if expected else 1.0
        failures = []
        if missing:
            failures.append(FailureCategory.INCORRECT_TOOL_SELECTION.value)
        if extra_invalid:
            failures.append(FailureCategory.INVALID_TOOL_ARGUMENTS.value)
        return EvaluationResult(
            evaluator_id=self.evaluator_id, version=self.version, passed=passed, score=score,
            failure_categories=failures,
            details={"expected": list(expected), "actual": list(actual), "missing": list(missing)},
        )


class TrajectoryEvaluator(Evaluator):
    @property
    def evaluator_id(self) -> str:
        return "trajectory"

    def evaluate(self, task: Task, result: AgentResult) -> EvaluationResult:
        traces = result.traces
        tool_calls = [e for e in traces if e.event_type == TraceEventType.TOOL_CALL]
        failures: list[str] = []
        score = 1.0

        # Invalid tool calls
        invalid = sum(1 for e in tool_calls if e.error)
        if invalid:
            failures.append(FailureCategory.INVALID_TOOL_ARGUMENTS.value)
            score -= 0.2 * invalid

        # Repeated actions (loops)
        tool_sequence = [e.tool for e in tool_calls if e.tool]
        if len(tool_sequence) != len(set(tool_sequence)) and len(tool_sequence) > 3:
            repeats = len(tool_sequence) - len(set(tool_sequence))
            if repeats > 2:
                failures.append(FailureCategory.LOOPING.value)
                score -= 0.3

        # Retries
        retries = sum(e.retry_count for e in traces)
        if retries > 3:
            failures.append(FailureCategory.TOOL_FAILURE.value)
            score -= 0.1 * retries

        # Excessive tool calls
        max_calls = task.budget.max_tool_calls or 20
        if len(tool_calls) > max_calls:
            failures.append(FailureCategory.BUDGET_EXHAUSTION.value)
            score -= 0.2

        # Premature termination
        if result.termination_reason in ("timeout", "budget_exceeded"):
            failures.append(
                FailureCategory.TIMEOUT.value if "timeout" in (result.termination_reason or "")
                else FailureCategory.BUDGET_EXHAUSTION.value
            )
            score -= 0.5

        score = max(0.0, min(1.0, score))
        return EvaluationResult(
            evaluator_id=self.evaluator_id, version=self.version,
            passed=score >= 0.5, score=score, failure_categories=failures,
            details={"tool_calls": len(tool_calls), "invalid": invalid, "retries": retries},
        )


class LLMJudgeEvaluator(Evaluator):
    """LLM-as-judge with structured output schema.

    Requires API access. Falls back to heuristic scoring when unavailable.
    """

    CRITERIA = ["correctness", "relevance", "completeness", "instruction_following"]

    def __init__(self, model: str = "mock") -> None:
        self._model = model

    @property
    def evaluator_id(self) -> str:
        return "llm_judge"

    def evaluate(self, task: Task, result: AgentResult) -> EvaluationResult:
        if self._model == "mock":
            return self._heuristic_judge(task, result)
        # Real LLM judge would call API here with structured output schema
        return self._heuristic_judge(task, result)

    def _heuristic_judge(self, task: Task, result: AgentResult) -> EvaluationResult:
        scores = {}
        expected = task.expected_behavior.answer
        output = result.output.lower()

        scores["correctness"] = 1.0 if (not expected or expected.lower() in output) else 0.3
        scores["relevance"] = 1.0 if any(w in output for w in task.input.lower().split()[:3]) else 0.5
        scores["completeness"] = min(1.0, len(result.output) / max(len(task.input), 1) * 0.5)
        scores["instruction_following"] = 1.0 if result.success else 0.4

        avg_score = sum(scores.values()) / len(scores)
        return EvaluationResult(
            evaluator_id=self.evaluator_id, version=self.version,
            passed=avg_score >= 0.6, score=avg_score,
            details={"criteria_scores": scores, "judge_model": self._model},
        )


class TaskLevelEvaluator(Evaluator):
    """Aggregate sub-evaluations into final task score."""

    def __init__(self, evaluators: list[Evaluator] | None = None) -> None:
        self._evaluators = evaluators or [
            ExactMatchEvaluator(),
            ExpectedToolCallEvaluator(),
            TrajectoryEvaluator(),
        ]

    @property
    def evaluator_id(self) -> str:
        return "task_level"

    def evaluate(self, task: Task, result: AgentResult) -> EvaluationResult:
        sub_results = [ev.evaluate(task, result) for ev in self._evaluators]
        if not sub_results:
            return EvaluationResult(evaluator_id=self.evaluator_id, version=self.version, passed=False, score=0.0)

        # Weighted aggregation: exact match counts most
        weights = {"exact_match": 0.5, "expected_tool_call": 0.25, "trajectory": 0.15, "llm_judge": 0.1}
        total_weight = 0.0
        weighted_score = 0.0
        all_failures: list[str] = []
        sub_scores = {}

        for sr in sub_results:
            w = weights.get(sr.evaluator_id, 0.1)
            weighted_score += sr.score * w
            total_weight += w
            all_failures.extend(sr.failure_categories)
            sub_scores[sr.evaluator_id] = sr.score

        final_score = weighted_score / total_weight if total_weight > 0 else 0.0
        passed = final_score >= 0.6 and all(sr.passed for sr in sub_results if sr.evaluator_id == "exact_match")

        return EvaluationResult(
            evaluator_id=self.evaluator_id, version=self.version,
            passed=passed, score=final_score,
            failure_categories=list(set(all_failures)),
            details={"sub_scores": sub_scores},
        )


def get_evaluators(names: list[str] | None = None, model: str = "mock") -> list[Evaluator]:
    registry: dict[str, Evaluator] = {
        "exact_match": ExactMatchEvaluator(),
        "regex": RegexEvaluator(),
        "json_schema": JsonSchemaEvaluator(),
        "expected_tool_call": ExpectedToolCallEvaluator(),
        "trajectory": TrajectoryEvaluator(),
        "llm_judge": LLMJudgeEvaluator(model=model),
        "task_level": TaskLevelEvaluator(),
    }
    if names:
        evaluators = [registry[n] for n in names if n in registry]
        if evaluators:
            return evaluators
    return [TaskLevelEvaluator()]
