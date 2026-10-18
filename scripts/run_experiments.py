#!/usr/bin/env python3
"""Run all required experiments and generate comparison report."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentbench.core.runner import ExperimentRunner
from agentbench.optimization.optimizer import OptimizationConfig, PromptOptimizer
from agentbench.statistics.comparison import compare_runs, run_summary
from agentbench.storage.factory import create_run_repository
from agentbench.storage.prompts import PromptStore

EXPERIMENTS = [
    "configs/experiments/single_agent_baseline.yaml",
    "configs/experiments/planner_executor.yaml",
    "configs/experiments/planner_executor_critic.yaml",
    "configs/experiments/tool_routing_unrestricted.yaml",
    "configs/experiments/tool_routing_rule_based.yaml",
    "configs/experiments/tool_routing_specialized.yaml",
]


async def main() -> None:
    repo = create_run_repository()
    runner = ExperimentRunner(repo)
    results: dict[str, str] = {}

    print("=" * 60)
    print("AgentBench Experiment Suite")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 60)

    # Run architecture comparison experiments
    for config_path in EXPERIMENTS:
        config = ExperimentRunner.load_config(Path(config_path))
        print(f"\nRunning: {config.name}...")
        run = await runner.run(config)
        summary = run_summary(run)
        sr = summary["success_rate"]
        print(f"  run_id: {run.run_id}")
        print(f"  success_rate: {sr['mean']:.1%} [{sr['ci_lower']:.1%}, {sr['ci_upper']:.1%}]")
        print(f"  latency: {summary['latency_ms']['mean']:.0f}ms")
        print(f"  cost: ${summary['cost_usd']['mean']:.4f}")
        print(f"  failures: {summary['failure_distribution']}")
        results[config.name] = run.run_id

    # Comparisons
    print("\n" + "=" * 60)
    print("COMPARISONS")
    print("=" * 60)

    comparisons = {}

    # A vs B: single vs planner_executor
    if "single_agent_baseline" in results and "planner_executor" in results:
        run_a = await repo.get_run(results["single_agent_baseline"])
        run_b = await repo.get_run(results["planner_executor"])
        if run_a and run_b:
            comp = compare_runs(run_a, run_b)
            comparisons["single_vs_planner_executor"] = comp.model_dump()
            print("\nSingle vs Planner+Executor:")
            print(
                f"  Success: {comp.success_rate.value_a:.1%} -> "
                f"{comp.success_rate.value_b:.1%} "
                f"(diff: {comp.success_rate.difference:+.1%})"
            )
            print(f"  Significant: {comp.success_rate.significant}")

    # B vs C: planner_executor vs planner_executor_critic
    if "planner_executor" in results and "planner_executor_critic" in results:
        run_a = await repo.get_run(results["planner_executor"])
        run_b = await repo.get_run(results["planner_executor_critic"])
        if run_a and run_b:
            comp = compare_runs(run_a, run_b)
            comparisons["planner_executor_vs_critic"] = comp.model_dump()
            print("\nPlanner+Executor vs +Critic:")
            print(
                f"  Success: {comp.success_rate.value_a:.1%} -> "
                f"{comp.success_rate.value_b:.1%} "
                f"(diff: {comp.success_rate.difference:+.1%})"
            )
            print(f"  Significant: {comp.success_rate.significant}")

    # Tool routing comparison
    routing_runs = {k: v for k, v in results.items() if k.startswith("tool_routing")}
    if len(routing_runs) >= 2:
        print("\nTool Routing Comparison:")
        routing_summaries = {}
        for name, run_id in routing_runs.items():
            run = await repo.get_run(run_id)
            if run:
                s = run_summary(run)
                routing_summaries[name] = s
                print(f"  {name}: {s['success_rate']['mean']:.1%}")

    # Prompt optimization
    print("\nRunning prompt optimization...")
    with open("configs/experiments/prompt_optimization.yaml") as config_file:
        opt_config = OptimizationConfig.model_validate(yaml.safe_load(config_file))
    optimizer = PromptOptimizer(runner, PromptStore())
    opt_result = await optimizer.optimize(opt_config)
    print(f"  Best validation score: {opt_result.best_validation_score:.1%}")
    print(f"  Test score: {opt_result.test_score:.1%}")
    print(f"  Candidates: {len(opt_result.candidates)}")

    # Failure taxonomy analysis
    print("\nFailure Taxonomy Analysis:")
    all_failures: dict[str, int] = {}
    for run_id in results.values():
        run = await repo.get_run(run_id)
        if run:
            for r in run.task_results:
                for f in r.failure_categories:
                    all_failures[f] = all_failures.get(f, 0) + 1
    for cat, count in sorted(all_failures.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Save report
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "runs": results,
        "comparisons": comparisons,
        "optimization": opt_result.model_dump(),
        "failure_taxonomy": all_failures,
        "note": "Results from deterministic mock agents on local benchmark. "
        "Real LLM-backed experiments require API keys and will produce different results.",
    }
    report_path = Path("artifacts/experiment_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport saved to: {report_path}")
    await repo.close()


if __name__ == "__main__":
    asyncio.run(main())
