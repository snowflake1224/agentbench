"""FastAPI application for AgentBench."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from prometheus_client import generate_latest
from pydantic import BaseModel
from starlette.responses import Response

from agentbench.core.experiment import ExperimentConfig, ExperimentRun
from agentbench.core.regression import RegressionThresholds, check_regression
from agentbench.core.runner import ExperimentRunner
from agentbench.statistics.comparison import compare_runs, run_summary
from agentbench.storage.prompts import PromptStore
from agentbench.storage.repository import FileRunRepository

repo = FileRunRepository()
runner = ExperimentRunner(repo)
prompt_store = PromptStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="AgentBench", version="0.1.0", lifespan=lifespan)


class RunRequest(BaseModel):
    config: ExperimentConfig


class CompareRequest(BaseModel):
    run_a: str
    run_b: str


class RegressionRequest(BaseModel):
    baseline_run_id: str
    candidate_run_id: str
    thresholds: RegressionThresholds | None = None


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/runs", response_model=ExperimentRun)
async def create_run(request: RunRequest) -> ExperimentRun:
    return await runner.run(request.config)


@app.get("/runs", response_model=list[ExperimentRun])
async def list_runs(limit: int = 50) -> list[ExperimentRun]:
    return await repo.list_runs(limit=limit)


@app.get("/runs/{run_id}", response_model=ExperimentRun)
async def get_run(run_id: str) -> ExperimentRun:
    run = await repo.get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return run


@app.get("/runs/{run_id}/summary")
async def get_run_summary(run_id: str) -> dict[str, Any]:
    run = await repo.get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return run_summary(run)


@app.get("/runs/{run_id}/traces")
async def get_traces(run_id: str, task_id: str | None = None) -> list[dict]:
    traces = await repo.get_traces(run_id, task_id=task_id)
    return [t.model_dump(mode="json") for t in traces]


@app.get("/runs/{run_id}/failures")
async def get_failures(run_id: str) -> list[dict]:
    run = await repo.get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return [
        {"task_id": r.task_id, "failure_categories": r.failure_categories, "score": r.score, "output": r.output}
        for r in run.task_results if not r.success
    ]


@app.post("/compare")
async def compare_experiments(request: CompareRequest) -> dict[str, Any]:
    run_a = await repo.get_run(request.run_a)
    run_b = await repo.get_run(request.run_b)
    if not run_a or not run_b:
        raise HTTPException(404, "One or both runs not found")
    return compare_runs(run_a, run_b).model_dump()


@app.post("/regression")
async def regression_check(request: RegressionRequest) -> dict[str, Any]:
    baseline = await repo.get_run(request.baseline_run_id)
    candidate = await repo.get_run(request.candidate_run_id)
    if not baseline or not candidate:
        raise HTTPException(404, "One or both runs not found")
    return check_regression(baseline, candidate, request.thresholds).model_dump()


@app.get("/prompts")
async def list_prompts() -> list[dict]:
    return [p.model_dump(mode="json") for p in prompt_store.list_versions()]


@app.get("/prompts/{prompt_id}")
async def get_prompt(prompt_id: str) -> dict:
    p = prompt_store.get(prompt_id)
    if not p:
        raise HTTPException(404, "Prompt not found")
    return p.model_dump(mode="json")


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> str:
    runs = await repo.list_runs(limit=20)
    rows = ""
    for run in runs:
        summary = run_summary(run)
        sr = summary.get("success_rate", {})
        rows += f"""<tr>
            <td><a href="/runs/{run.run_id}">{run.config.name}</a></td>
            <td>{run.run_id[:8]}</td>
            <td>{run.status.value}</td>
            <td>{sr.get('mean', 0):.1%}</td>
            <td>[{sr.get('ci_lower', 0):.1%}, {sr.get('ci_upper', 0):.1%}]</td>
            <td>{summary.get('latency_ms', {}).get('mean', 0):.0f}ms</td>
            <td>${summary.get('cost_usd', {}).get('mean', 0):.4f}</td>
            <td>{summary.get('n_tasks', 0)}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html><head><title>AgentBench Dashboard</title>
<style>
body {{ font-family: system-ui; margin: 2rem; background: #0f172a; color: #e2e8f0; }}
h1 {{ color: #38bdf8; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
th, td {{ border: 1px solid #334155; padding: 0.5rem 1rem; text-align: left; }}
th {{ background: #1e293b; }}
tr:nth-child(even) {{ background: #1e293b; }}
a {{ color: #38bdf8; }}
.metric {{ display: inline-block; background: #1e293b; padding: 1rem; margin: 0.5rem; border-radius: 8px; }}
</style></head><body>
<h1>AgentBench Dashboard</h1>
<p>AI Agent Evaluation & Reliability Platform</p>
<div>
  <span class="metric">Runs: {len(runs)}</span>
  <span class="metric"><a href="/metrics">Prometheus Metrics</a></span>
  <span class="metric"><a href="/docs">API Docs</a></span>
</div>
<h2>Recent Runs</h2>
<table>
<tr><th>Name</th><th>Run ID</th><th>Status</th><th>Success Rate</th><th>95% CI</th><th>Latency</th><th>Cost</th><th>Tasks</th></tr>
{rows}
</table>
</body></html>"""
