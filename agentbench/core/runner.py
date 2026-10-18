"""Configuration-driven experiment runner."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import yaml

from agentbench.adapters.registry import create_agent
from agentbench.benchmarks.registry import get_benchmark
from agentbench.core.agent import AgentRunContext
from agentbench.core.experiment import ExperimentConfig, ExperimentRun, RunStatus, TaskResult
from agentbench.core.reproducibility import get_git_revision
from agentbench.core.trace import TraceEvent
from agentbench.evaluators.base import get_evaluators
from agentbench.storage.repository import RunRepository
from agentbench.tracing.metrics import (
    ACTIVE_RUNS,
    EXPERIMENT_RUNS,
    FAILURE_CATEGORIES,
    TASK_COST,
    TASK_LATENCY,
    TASK_RESULTS,
)
from agentbench.tracing.otel import init_tracing, record_trace_event, trace_span


class ExperimentRunner:
    """Execute benchmark experiments with full observability."""

    def __init__(self, repository: RunRepository) -> None:
        self._repo = repository

    @staticmethod
    def load_config(path: Path) -> ExperimentConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return ExperimentConfig.model_validate(data)

    async def run(self, config: ExperimentConfig) -> ExperimentRun:
        init_tracing()

        run = ExperimentRun(
            config=config,
            status=RunStatus.RUNNING,
            git_revision=get_git_revision(),
        )

        benchmark = get_benchmark(config.benchmark)
        run.benchmark_version = benchmark.version
        tasks = benchmark.load_tasks(split=config.task_split, subset=config.task_subset)

        agent = create_agent(config.agent)
        await agent.initialize(config.agent)
        evaluators = get_evaluators(
            config.evaluator_versions,
            model=config.judge_model or config.model,
            provider_config=config.evaluator_config,
        )
        if not evaluators:
            evaluators = get_evaluators(["task_level"], model=config.model)

        ACTIVE_RUNS.inc()
        all_traces: list[TraceEvent] = []

        try:
            semaphore = asyncio.Semaphore(config.concurrency)

            async def run_task(task) -> TaskResult:
                async with semaphore:
                    with trace_span("task", {"task_id": task.task_id, "run_id": run.run_id}):
                        ctx = AgentRunContext(
                            run_id=run.run_id,
                            task=task,
                            prompt_version=config.prompt_version,
                            model=config.model,
                            temperature=config.temperature,
                        )
                        agent_result = await agent.run_task(ctx)

                        for trace in agent_result.traces:
                            record_trace_event(trace)
                            all_traces.append(trace)

                        eval_result = await evaluators[0].evaluate(task, agent_result)

                        TASK_RESULTS.labels(
                            benchmark=config.benchmark, success=str(eval_result.passed)
                        ).inc()
                        TASK_LATENCY.labels(
                            benchmark=config.benchmark, agent_type=agent.agent_type
                        ).observe(agent_result.latency_ms / 1000)
                        TASK_COST.labels(model=config.model).observe(
                            agent_result.estimated_cost_usd
                        )
                        for cat in eval_result.failure_categories:
                            FAILURE_CATEGORIES.labels(category=cat).inc()

                        return TaskResult(
                            task_id=task.task_id,
                            success=eval_result.passed,
                            score=eval_result.score,
                            latency_ms=agent_result.latency_ms,
                            input_tokens=agent_result.input_tokens,
                            output_tokens=agent_result.output_tokens,
                            tool_calls=agent_result.tool_calls,
                            estimated_cost_usd=agent_result.estimated_cost_usd,
                            failure_categories=eval_result.failure_categories,
                            evaluator_scores=eval_result.details.get("sub_scores", {}),
                            output=agent_result.output,
                        )

            results = await asyncio.gather(*[run_task(t) for t in tasks])
            run.task_results = list(results)
            run.status = RunStatus.COMPLETED
            run.completed_at = datetime.now(UTC)

            EXPERIMENT_RUNS.labels(
                benchmark=config.benchmark, agent_type=agent.agent_type, status="completed"
            ).inc()

        except Exception as e:
            run.status = RunStatus.FAILED
            run.metadata["error"] = str(e)
            EXPERIMENT_RUNS.labels(
                benchmark=config.benchmark,
                agent_type=config.agent.get("type", "unknown"),
                status="failed",
            ).inc()
        finally:
            ACTIVE_RUNS.dec()
            await agent.terminate()
            for evaluator in evaluators:
                await evaluator.close()
            await self._repo.save_run(run)
            await self._repo.save_traces(run.run_id, all_traces)

        return run
