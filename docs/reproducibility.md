# Reproducibility

Every experiment run is tied to metadata enabling historical reconstruction.

## Recorded Metadata

| Field | Source |
|-------|--------|
| `git_revision` | `git rev-parse HEAD` at run time |
| `benchmark_version` | Benchmark adapter version string |
| `task_ids` | Exact tasks evaluated |
| `model` | Model identifier |
| `prompt_version` | Prompt version ID |
| `tool_version` | Tool configuration hash |
| `evaluator_version` | Evaluator version(s) |
| `configuration` | Full experiment config (YAML) |
| `seed` | Random seed for stochastic components |

## Configuration Hashing

```python
config_hash(config_dict)  # SHA-256 of sorted JSON
```

## Artifact Storage

```
artifacts/
├── runs/{run_id}/
│   ├── run.json          # Full experiment run
│   └── traces.json       # All trace events
└── prompts/
    ├── index.json        # Prompt version index
    └── {prompt_id}.json  # Immutable prompt versions
```

## Reproducing a Run

```bash
# 1. Checkout the git revision
git checkout <git_revision>

# 2. Run with the same config
agentbench run <saved_config.yaml>

# 3. Compare results
agentbench compare <original_run_id> <new_run_id>
```

## Limitations

- LLM outputs are non-deterministic even with fixed seeds (unless temperature=0 and provider supports seeding)
- External API versions may change between runs
- Mock agents are fully deterministic; real LLM agents are not
- Bootstrap CI intervals vary slightly due to resampling randomness (controlled by seed)

## Best Practices

- Pin model versions in experiment config
- Record provider API version if available
- Use `seed` parameter for bootstrap and mock agents
- Store experiment configs in version control
- Never modify evaluator versions without bumping the version string
