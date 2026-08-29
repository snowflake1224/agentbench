# Evaluation

## Evaluator Types

### Deterministic Evaluators

| Evaluator | Description |
|-----------|-------------|
| `exact_match` | Case-insensitive string comparison |
| `regex` | Pattern matching against output |
| `json_schema` | JSON output validation |
| `expected_tool_call` | Verify correct tools were invoked |
| `unit_test` | Execute test command (coding tasks) |

### LLM-as-Judge

Structured evaluation across criteria:
- **correctness**: Factual accuracy
- **relevance**: Response addresses the question
- **completeness**: All aspects covered
- **instruction_following**: Adheres to task requirements

The judge outputs a structured schema with per-criterion scores, not arbitrary prose.

**Limitations:**
- LLM judges inherit model biases
- Self-preference bias when judge and agent share a model family
- Non-deterministic: run multiple times for stability
- Cannot reliably evaluate factual claims without grounding

When API access is unavailable, a heuristic fallback is used (clearly labeled in results).

### Trajectory Evaluators

Measure agent behavior patterns:
- Unnecessary tool calls
- Invalid tool calls (wrong tool or bad arguments)
- Repeated actions (loop detection)
- Retry counts
- Excessive context usage
- Premature termination

### Task-Level Aggregation

Weighted combination of sub-evaluators:
- exact_match: 50%
- expected_tool_call: 25%
- trajectory: 15%
- llm_judge: 10%

Pass threshold: score ≥ 0.6 AND exact_match passes.

## Failure Taxonomy

| Category | Description |
|----------|-------------|
| `misunderstanding_task` | Misinterpreted requirements |
| `hallucination` | Fabricated content |
| `incorrect_tool_selection` | Wrong tool chosen |
| `invalid_tool_arguments` | Bad tool arguments |
| `tool_failure` | External tool error |
| `context_failure` | Context window issues |
| `planning_failure` | Flawed plan |
| `implementation_failure` | Incorrect implementation |
| `verification_failure` | Failed self-check |
| `looping` | Repetitive actions |
| `timeout` | Exceeded time limit |
| `budget_exhaustion` | Exceeded token/cost budget |

Evaluators can assign multiple categories per task.

## Versioning

Evaluator versions are tracked in experiment config (`evaluator_versions: ["v1"]`). Score logic changes require version bumps to maintain comparability across runs.

## Limitations

- Exact match is too strict for open-ended tasks
- Regex patterns require careful crafting
- LLM judges are not ground truth
- Trajectory evaluators use heuristics that may miss subtle failures
- Unit test evaluators require sandboxed execution for safety
