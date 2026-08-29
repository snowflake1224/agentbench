# Observability

## OpenTelemetry

AgentBench uses OpenTelemetry as the primary instrumentation approach.

### Span Types

| Span | Description |
|------|-------------|
| `task` | Full task execution |
| `agent` | Agent lifecycle |
| `model_call` | LLM inference |
| `tool_call` | Tool invocation |
| `evaluator` | Evaluation execution |
| `retrieval` | Context retrieval |
| `external_service` | External API calls |
| `retry` | Retry attempts |

### Configuration

```env
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=agentbench
```

Traces export to Jaeger (included in docker-compose) or any OTLP-compatible backend.

## Prometheus Metrics

| Metric | Type | Labels |
|--------|------|--------|
| `agentbench_experiment_runs_total` | Counter | benchmark, agent_type, status |
| `agentbench_task_results_total` | Counter | benchmark, success |
| `agentbench_task_latency_seconds` | Histogram | benchmark, agent_type |
| `agentbench_task_cost_usd` | Histogram | model |
| `agentbench_tool_calls_total` | Counter | tool, success |
| `agentbench_failures_total` | Counter | category |
| `agentbench_active_runs` | Gauge | — |

Access metrics at `http://localhost:8000/metrics`.

## Grafana Dashboard

Pre-configured dashboard at `http://localhost:3000` (admin/admin):
- Experiment run counts
- Task success rate gauge
- Latency heatmap
- Failure category pie chart

## Dashboard

Built-in web dashboard at `http://localhost:8000/dashboard`:
- Recent runs with success rates and confidence intervals
- Latency and cost per run
- Links to API docs and Prometheus metrics

## Trace Drill-Down

Via API:
```
GET /runs/{run_id}/traces?task_id={task_id}
```

Reconstructs: task → agent → model call → tool call → evaluator → final score.
