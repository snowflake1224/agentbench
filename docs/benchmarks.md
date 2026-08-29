# Benchmarks

## Design Principle

Every benchmark is an **adapter** implementing `BenchmarkAdapter`. No benchmark-specific logic is hard-coded in the experiment runner or evaluators.

```python
class BenchmarkAdapter(ABC):
    @property
    def name(self) -> str: ...
    @property
    def version(self) -> str: ...
    def load_tasks(self, split, subset) -> list[Task]: ...
```

## Available Benchmarks

### Local (`local`)

Deterministic fixture benchmark for CI and development. 20 tasks across general, tool-use, and coding categories with dev/validation/test splits.

- No external dependencies
- Fully reproducible
- Used for all included experiment results

### GAIA (`gaia`)

General AI Assistants benchmark. Sample tasks included; full dataset requires HuggingFace authentication.

- Source: https://huggingface.co/datasets/gaia-benchmark/GAIA
- Levels 1-3 difficulty
- Requires web search and reasoning

```python
# Full dataset loading (requires HF_TOKEN)
benchmark = GAIABenchmarkAdapter(load_from_huggingface=True)
```

### ToolBench (`toolbench`)

Tool-use evaluation with API interaction tasks.

- Source: https://github.com/OpenBMB/ToolBench
- Sample tasks cover finance, travel, email, weather, CMS APIs
- Evaluates tool selection and argument correctness

### SWE-bench (`swebench`)

Coding agent evaluation on real GitHub issues.

- Source: https://www.swebench.com/
- Sample tasks included
- Full evaluation requires Docker for isolated container execution
- Requires cloning target repositories

```python
benchmark = SWEBenchAdapter(use_docker=True)
```

## Custom Benchmarks

Create a new adapter:

```python
class MyBenchmarkAdapter(BenchmarkAdapter):
    @property
    def name(self) -> str:
        return "my_benchmark"

    @property
    def version(self) -> str:
        return "1.0.0"

    def load_tasks(self, split="dev", subset=None) -> list[Task]:
        # Return list of Task objects
        ...
```

Register in `benchmarks/registry.py`.

## Split Management

Tasks are tagged with `split`: `dev`, `validation`, or `test`.

- **dev**: Development and debugging
- **validation**: Candidate selection during optimization
- **test**: Final evaluation only — never use for optimization

## Methodology Notes

- Benchmark versions are recorded in experiment metadata for reproducibility
- Task subsets can be specified in experiment config for focused evaluation
- Timeout and budget constraints are per-task, enforced by the runner
