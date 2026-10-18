# AgentBench

**Production-oriented framework for evaluating, observing, and optimizing AI agents and multi-agent systems.**

AgentBench is an AI evaluation and research infrastructure project — not a chatbot. It answers:

> Given multiple agent architectures, models, prompts, tools, and orchestration strategies, how can we rigorously measure which system performs better, why it fails, what it costs, and whether an optimization actually improves performance without introducing regressions?

## Features

- **Agent benchmarking** across architectures (single, planner+executor, planner+executor+critic, multi-agent)
- **Multi-agent evaluation** with parent-child trace reconstruction
- **Tool-use evaluation** with routing strategy comparison
- **Trajectory analysis** (loops, retries, invalid calls, excessive context)
- **LLM-as-judge** using a real OpenAI-compatible API call and strict JSON schema
- **Deterministic evaluators** (exact match, regex, JSON schema, expected tool calls)
- **Statistical comparison** with bootstrap confidence intervals and paired Wilcoxon tests
- **Failure taxonomy** with structured categorization
- **Prompt optimization** with dev/validation/test split separation
- **Regression testing** gates for CI/CD integration
- **OpenTelemetry** tracing + Prometheus metrics + Grafana dashboards
- **Framework-independent** agent adapters (deterministic mock, real model-backed, external coding boundary)
- **Durable PostgreSQL storage** through async SQLAlchemy and Alembic migrations

## Quick Start

```bash
# Install
uv pip install -e ".[dev]"

# Run an experiment
agentbench run configs/experiments/single_agent_baseline.yaml

# Real model + structured LLM judge (requires OPENAI_API_KEY)
agentbench run configs/experiments/real_model_agent.yaml

# Compare two runs
agentbench compare <RUN_A> <RUN_B>

# View failures
agentbench failures <RUN_ID>

# Run full experiment suite
python scripts/run_experiments.py

# Start API + dashboard
uvicorn agentbench.api.app:app --reload
# Dashboard: http://localhost:8000/dashboard
# API docs: http://localhost:8000/docs
```

## Docker (Observability Stack)

```bash
docker compose up -d
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# Jaeger: http://localhost:16686
# API: http://localhost:8000
```

## Architecture

```
agentbench/
├── core/           # Task, trace, experiment models
├── adapters/       # Agent adapters (mock, real model, routed, external coding)
├── benchmarks/     # GAIA, ToolBench, SWE-bench, local fixtures
├── evaluators/     # Plugin-based evaluation framework
├── statistics/     # Bootstrap CI, paired comparison
├── optimization/   # Prompt optimization, fine-tuning pipeline
├── tracing/        # OpenTelemetry + Prometheus
├── storage/        # File/PostgreSQL repositories, prompt versioning
├── api/            # FastAPI REST API + dashboard
├── cli/            # CLI commands
└── configs/        # Experiment configurations
```

See [docs/architecture.md](docs/architecture.md) for detailed design.

## Benchmarks

| Benchmark | Status | Notes |
|-----------|--------|-------|
| `local` | ✅ Ready | Deterministic fixtures for CI |
| `gaia` | ✅ Sample | Full dataset requires HuggingFace access |
| `toolbench` | ✅ Sample | Representative tool-use tasks |
| `swebench` | ✅ Sample | Full eval requires Docker + repo cloning |

## CLI Commands

```
agentbench run CONFIG          # Run experiment
agentbench compare RUN_A RUN_B # Statistical comparison
agentbench trace RUN_ID        # View traces
agentbench failures RUN_ID     # Failure analysis
agentbench report RUN_ID       # Generate report
agentbench regression CONFIG   # Regression gate
agentbench optimize CONFIG     # Prompt optimization
```

## External Dependencies

- **LLM-backed agents/evaluators**: Require `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
- **Claude Code integration**: Documented adapter boundary; requires CLI/API setup
- **Full GAIA/SWE-bench**: Require dataset download and optional Docker
- **Fine-tuning**: Optional `pip install agentbench[finetune]` (PEFT/LoRA)
- **PostgreSQL/Redis**: Optional; file-based storage works for development

## Testing

```bash
pytest tests/ -v
pytest tests/ -m e2e  # End-to-end smoke tests
ruff check agentbench/
mypy agentbench/
```

## License

MIT
