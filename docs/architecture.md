# Architecture

## Overview

AgentBench is designed as an **Agent Evaluation and AI Reliability Platform**. It provides rigorous measurement infrastructure for comparing agent systems across architectures, models, prompts, tools, and orchestration strategies.

## Why Agent Evaluation Is Difficult

1. **Non-determinism**: LLM outputs vary across runs, making single-shot evaluation unreliable
2. **Multi-step reasoning**: Agents make sequences of decisions; partial success is hard to score
3. **Tool interaction complexity**: Success depends on tool selection, argument correctness, and result interpretation
4. **Cost-latency tradeoffs**: Better accuracy may come at 10x cost; raw success rate hides efficiency
5. **Evaluation leakage**: Optimizing on test data produces inflated metrics
6. **Failure opacity**: "Failed" doesn't explain whether the agent misunderstood, hallucinated, or hit a tool error

## Core Components

### Task Model

Every benchmark task conforms to a standard interface:

```
Task
├── task_id, category, input
├── available_tools
├── expected_behavior (answer, patterns, schemas, expected tools)
├── evaluator, timeout, budget
└── metadata, split (dev/validation/test)
```

### Agent Adapter Interface

```
AgentAdapter
├── initialize(config)
├── run_task(context) → AgentResult
├── stream_events(context) → AsyncIterator[TraceEvent]
├── terminate()
├── collect_metadata()
└── agent_id, agent_type
```

Adapters normalize traces regardless of underlying framework (mock, LangGraph, custom Python, external coding agents).

### Trace Model

Hierarchical event model capturing the complete agent trajectory:

- Parent-child agent relationships for multi-agent systems
- Model calls with token counts and estimated cost
- Tool calls with inputs, outputs, errors
- Retries, termination reasons, latency

### Evaluator Plugin Architecture

```
Evaluator (base)
├── ExactMatchEvaluator
├── RegexEvaluator
├── JsonSchemaEvaluator
├── ExpectedToolCallEvaluator
├── TrajectoryEvaluator
├── LLMJudgeEvaluator (structured schema output)
└── TaskLevelEvaluator (weighted aggregation)
```

Scoring logic is versioned. Evaluators assign failure categories from the structured taxonomy.

### Statistical Engine

- Bootstrap confidence intervals (1000 resamples, 95% CI)
- Paired Wilcoxon signed-rank test for same-task comparisons
- Distribution statistics: mean, median, std, min, max
- Efficiency metrics: success per dollar, per 1K tokens, per second

### Experiment Runner

Configuration-driven engine that:
1. Loads benchmark tasks
2. Initializes agent adapter
3. Runs tasks with concurrency control
4. Evaluates results
5. Records traces via OpenTelemetry
6. Persists results with reproducibility metadata (git revision, benchmark version, config hash)

### Storage

Repository abstraction supporting:
- `FileRunRepository` (development)
- `InMemoryRunRepository` (testing)
- PostgreSQL (production, via SQLAlchemy — schema extensible)

Large artifacts (traces, outputs) stored separately from metadata.

## Observability

- **OpenTelemetry**: Spans for task, agent, model call, tool call, evaluator, retrieval, retry
- **Prometheus**: Counters for runs, task results, tool calls, failures; histograms for latency and cost
- **Grafana**: Pre-configured dashboard for success rates, latency, failure categories
- **Jaeger**: Distributed trace visualization

## Security

- Prompt injection detection in tool arguments
- Dangerous command pattern blocking
- Secret redaction in traces and outputs
- Output truncation to prevent context overflow
- Sandbox mode for coding agent evaluation (Docker containers)

## External Agent Integration

The `ExternalCodingAgentAdapter` defines a clean boundary for integrating agentic coding assistants:

```python
class ExternalCodingAgentAdapter(AgentAdapter):
    async def _check_availability(self) -> bool: ...
    async def _invoke_external_agent(self, context) -> AgentResult: ...
```

`ClaudeCodeAdapter` is a documented stub — it does NOT fake integration. Implement `_invoke_external_agent` with your deployment's CLI/API.

## Data Flow

```
Config YAML → ExperimentRunner → BenchmarkAdapter.load_tasks()
                                → AgentAdapter.run_task()
                                → Evaluator.evaluate()
                                → TraceEvent → OpenTelemetry
                                → RunRepository.save()
                                → Statistics → Comparison/Regression
```
