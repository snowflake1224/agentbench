"""AgentBench CLI."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import click
import yaml
from rich.console import Console
from rich.table import Table

from agentbench.core.regression import RegressionThresholds, check_regression
from agentbench.core.runner import ExperimentRunner
from agentbench.optimization.optimizer import OptimizationConfig, PromptOptimizer
from agentbench.statistics.comparison import compare_runs, run_summary
from agentbench.storage.factory import create_run_repository
from agentbench.storage.prompts import PromptStore

console = Console()
repo = create_run_repository()
runner = ExperimentRunner(repo)


def async_cmd[**P, R](f: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, R]:
    """Run an async Click callback, preserving the command name and help text."""

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return asyncio.run(f(*args, **kwargs))

    command_name = getattr(f, "__name__", None)
    if isinstance(command_name, str):
        wrapper.__name__ = command_name
    command_doc = getattr(f, "__doc__", None)
    if command_doc is None or isinstance(command_doc, str):
        wrapper.__doc__ = command_doc
    return wrapper


@click.group()
def cli() -> None:
    """AgentBench: AI Agent Evaluation & Reliability Platform."""


@cli.command("run")
@click.argument("config", type=click.Path(exists=True))
@async_cmd
async def run(config: str) -> None:
    """Run an experiment from configuration file."""
    exp_config = ExperimentRunner.load_config(Path(config))
    console.print(f"[bold blue]Running experiment:[/] {exp_config.name}")
    result = await runner.run(exp_config)
    summary = run_summary(result)
    sr = summary.get("success_rate", {})
    console.print(f"\n[green]Completed[/] run_id={result.run_id}")
    mean = sr.get("mean", 0)
    lower = sr.get("ci_lower", 0)
    upper = sr.get("ci_upper", 0)
    console.print(f"  Success rate: {mean:.1%} [{lower:.1%}, {upper:.1%}]")
    console.print(f"  Tasks: {summary.get('n_tasks', 0)}")
    console.print(f"  Latency: {summary.get('latency_ms', {}).get('mean', 0):.0f}ms")
    console.print(f"  Cost: ${summary.get('cost_usd', {}).get('mean', 0):.4f}")


@cli.command("compare")
@click.argument("run_a")
@click.argument("run_b")
@async_cmd
async def compare(run_a: str, run_b: str) -> None:
    """Compare two experiment runs."""
    a = await repo.get_run(run_a)
    b = await repo.get_run(run_b)
    if not a or not b:
        console.print("[red]Run not found[/]")
        return

    comp = compare_runs(a, b)
    table = Table(title="Experiment Comparison")
    table.add_column("Metric")
    table.add_column("Run A")
    table.add_column("Run B")
    table.add_column("Difference")
    table.add_column("Significant")

    for metric in [comp.success_rate, comp.latency, comp.cost, comp.tool_calls]:
        sig = "✓" if metric.significant else ""
        table.add_row(
            metric.metric,
            f"{metric.value_a:.4f}",
            f"{metric.value_b:.4f}",
            f"{metric.difference:+.4f}",
            sig,
        )
    console.print(table)


@cli.command("trace")
@click.argument("run_id")
@async_cmd
async def trace(run_id: str) -> None:
    """Show traces for a run."""
    traces = await repo.get_traces(run_id)
    for t in traces[:50]:
        agent = t.agent_id or "-"
        tool = t.tool or ""
        latency = t.latency_ms or ""
        console.print(f"  [{t.event_type.value}] {agent} {tool} {latency}ms")


@cli.command("failures")
@click.argument("run_id")
@async_cmd
async def failures(run_id: str) -> None:
    """Show failures for a run."""
    run = await repo.get_run(run_id)
    if not run:
        console.print("[red]Run not found[/]")
        return
    for r in run.task_results:
        if not r.success:
            console.print(f"  [red]{r.task_id}[/]: {r.failure_categories} (score={r.score:.2f})")


@cli.command("report")
@click.argument("run_id")
@async_cmd
async def report(run_id: str) -> None:
    """Generate report for a run."""
    run = await repo.get_run(run_id)
    if not run:
        console.print("[red]Run not found[/]")
        return
    summary = run_summary(run)
    console.print(json.dumps(summary, indent=2, default=str))


@cli.command("regression")
@click.argument("config", type=click.Path(exists=True))
@async_cmd
async def regression(config: str) -> None:
    """Run regression check from config."""
    with open(config) as f:
        data = yaml.safe_load(f)
    baseline = await repo.get_run(data["baseline_run_id"])
    candidate = await repo.get_run(data["candidate_run_id"])
    if not baseline or not candidate:
        console.print("[red]Run not found[/]")
        return
    thresholds = RegressionThresholds(**data.get("thresholds", {}))
    result = check_regression(baseline, candidate, thresholds)
    status = "[green]PASS[/]" if result.passed else "[red]FAIL[/]"
    console.print(f"Regression check: {status}")
    for check in result.checks:
        c_status = "✓" if check["passed"] else "✗"
        console.print(f"  {c_status} {check['check']}")


@cli.command("optimize")
@click.argument("config", type=click.Path(exists=True))
@async_cmd
async def optimize(config: str) -> None:
    """Run prompt optimization experiment."""
    with open(config) as f:
        data = yaml.safe_load(f)
    opt_config = OptimizationConfig.model_validate(data)
    optimizer = PromptOptimizer(runner, PromptStore())
    result = await optimizer.optimize(opt_config)
    console.print("[green]Optimization complete[/]")
    console.print(f"  Best candidate: {result.best_candidate_id}")
    console.print(f"  Validation score: {result.best_validation_score:.1%}")
    if result.test_score is not None:
        console.print(f"  Test score: {result.test_score:.1%}")
    console.print(f"  [yellow]{result.leakage_warning}[/]")


if __name__ == "__main__":
    cli()
