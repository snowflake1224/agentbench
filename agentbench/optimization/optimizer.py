"""Optimization framework for prompts, tools, and agent topology."""

from __future__ import annotations

import random
from typing import Any

from pydantic import BaseModel, Field

from agentbench.core.experiment import ExperimentConfig, ExperimentRun
from agentbench.core.runner import ExperimentRunner
from agentbench.statistics.comparison import run_summary
from agentbench.storage.prompts import PromptStore, PromptVersion


class OptimizationConfig(BaseModel):
    name: str
    benchmark: str
    agent: dict[str, Any]
    base_prompt: str
    optimization_target: str = "prompt"  # prompt | tools | topology | model
    dev_split: str = "dev"
    validation_split: str = "validation"
    test_split: str = "test"
    num_candidates: int = 3
    seed: int = 42
    metadata: dict[str, Any] = Field(default_factory=dict)


class OptimizationResult(BaseModel):
    config: OptimizationConfig
    baseline_run_id: str
    best_candidate_id: str
    best_validation_score: float
    test_run_id: str | None = None
    test_score: float | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    leakage_warning: str = (
        "Candidates were selected on validation set. Test set results are reported separately "
        "and were NOT used for selection. Do not iterate on test set."
    )


PROMPT_MUTATIONS = [
    ("detailed", "Be thorough and step-by-step in your reasoning. "),
    ("concise", "Be concise and direct. "),
    ("tool_focused", "Always use available tools before answering. "),
    ("verify", "Verify your answer before submitting. "),
]


class PromptOptimizer:
    """Baseline prompt optimization via candidate generation and validation."""

    def __init__(self, runner: ExperimentRunner, prompt_store: PromptStore) -> None:
        self._runner = runner
        self._prompt_store = prompt_store

    def generate_candidates(
        self, base_prompt: str, num: int, seed: int = 42
    ) -> list[PromptVersion]:
        rng = random.Random(seed)
        base = self._prompt_store.save(PromptVersion(name="baseline", content=base_prompt))
        candidates = [base]

        mutations = rng.sample(PROMPT_MUTATIONS, min(num, len(PROMPT_MUTATIONS)))
        for suffix, prefix in mutations:
            variant = self._prompt_store.create_variant(
                parent_id=base.prompt_id,
                name=f"variant_{suffix}",
                content=f"{prefix}{base_prompt}",
            )
            candidates.append(variant)

        return candidates

    async def optimize(self, config: OptimizationConfig) -> OptimizationResult:
        candidates = self.generate_candidates(
            config.base_prompt, config.num_candidates, config.seed
        )
        candidate_results: list[dict[str, Any]] = []

        # Evaluate each candidate on validation set
        best_score = -1.0
        best_candidate = candidates[0]

        for candidate in candidates:
            exp_config = ExperimentConfig(
                name=f"opt_{config.name}_{candidate.name}",
                benchmark=config.benchmark,
                task_split=config.validation_split,
                agent=config.agent,
                prompt_version=candidate.prompt_id,
                seed=config.seed,
            )
            run = await self._runner.run(exp_config)
            summary = run_summary(run)
            score = summary.get("success_rate", {}).get("mean", 0)
            candidate_results.append({
                "prompt_id": candidate.prompt_id,
                "name": candidate.name,
                "run_id": run.run_id,
                "validation_score": score,
            })

            if score > best_score:
                best_score = score
                best_candidate = candidate

            # Update prompt with benchmark results
            candidate.benchmark_results[config.validation_split] = score

        # Baseline run
        baseline_config = ExperimentConfig(
            name=f"opt_{config.name}_baseline",
            benchmark=config.benchmark,
            task_split=config.dev_split,
            agent=config.agent,
            seed=config.seed,
        )
        baseline_run = await self._runner.run(baseline_config)

        # Final test on untouched test set
        test_config = ExperimentConfig(
            name=f"opt_{config.name}_test",
            benchmark=config.benchmark,
            task_split=config.test_split,
            agent=config.agent,
            prompt_version=best_candidate.prompt_id,
            seed=config.seed,
        )
        test_run = await self._runner.run(test_config)
        test_summary = run_summary(test_run)
        test_score = test_summary.get("success_rate", {}).get("mean", 0)

        return OptimizationResult(
            config=config,
            baseline_run_id=baseline_run.run_id,
            best_candidate_id=best_candidate.prompt_id,
            best_validation_score=best_score,
            test_run_id=test_run.run_id,
            test_score=test_score,
            candidates=candidate_results,
        )


class FineTuningPipeline:
    """Optional fine-tuning dataset builder from agent failures.

    Requires optional finetune dependencies (peft, transformers, torch).
    The framework works completely without fine-tuning.
    """

    @staticmethod
    def build_dataset_from_failures(run: ExperimentRun, traces: list) -> list[dict[str, Any]]:
        """Extract training examples from failed trajectories."""
        examples = []
        failed_tasks = {r.task_id for r in run.task_results if not r.success}
        for trace in traces:
            if trace.task_id in failed_tasks and trace.event_type.value == "model_call":
                examples.append({
                    "task_id": trace.task_id,
                    "input": trace.metadata.get("input", ""),
                    "output": trace.tool_output or "",
                    "label": "failure",
                })
        return examples

    @staticmethod
    def export_for_training(examples: list[dict], output_path: str) -> None:
        """Export dataset in JSONL format for fine-tuning."""
        import json
        from pathlib import Path
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
