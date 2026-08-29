# Optimization

## Supported Optimization Targets

- System prompts
- Task-specific prompts
- Tool descriptions
- Tool routing strategies
- Agent topology
- Model selection
- Temperature
- Context strategy

## Prompt Optimization

Baseline workflow:

```
baseline prompt
  → generate candidate variants (mutations)
  → evaluate on validation set
  → select best candidate
  → evaluate on untouched test set
```

### Leakage Prevention

**Critical**: Never optimize and evaluate on the same data.

| Split | Purpose |
|-------|---------|
| dev | Development and debugging |
| validation | Candidate selection |
| test | Final evaluation only |

The optimizer reports a `leakage_warning` reminding users that test set results were NOT used for selection.

### Prompt Versioning

All prompt versions are immutable:
- `prompt_id`, `parent_prompt_id`, `experiment_id`
- Benchmark results per split
- Previous versions are never overwritten

## Tool Routing Experiments

Compare strategies:
- `unrestricted`: All tools available
- `rule_based`: Category-based filtering
- `model_selected`: Keyword-based selection
- `specialized_agent`: Domain-specific routing

Metrics: successful tool calls, invalid calls, call count, latency, task success.

## Fine-Tuning Pathway (Optional)

```
agent runs → trajectories → failure filtering → dataset builder → training data → candidate model → benchmark evaluation → regression comparison
```

Requires optional dependencies: `pip install agentbench[finetune]`

The framework works completely without fine-tuning. The pipeline exports JSONL training data from failed trajectories; actual LoRA/PEFT training is left to the user's infrastructure.

## Regression Gates

After optimization, run regression check:

```yaml
baseline_run_id: <baseline>
candidate_run_id: <optimized>
thresholds:
  max_success_decrease: 0.05
  max_latency_increase_pct: 0.20
  max_cost_increase_pct: 0.30
  max_tool_error_rate: 0.10
```

Candidate must pass all thresholds to be accepted.
