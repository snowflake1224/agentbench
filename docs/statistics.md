# Statistics

## Methodology

AgentBench does not report single percentages. All comparisons include confidence intervals and significance testing.

### Bootstrap Confidence Intervals

For any metric (success rate, latency, cost), we compute:
1. Resample with replacement (n=1000 bootstrap samples)
2. Compute statistic on each sample
3. Report 2.5th and 97.5th percentiles as 95% CI

```python
mean, ci_lower, ci_upper = bootstrap_ci(data, n_bootstrap=1000, confidence=0.95)
```

### Paired Comparison

When two runs evaluate the same tasks (same task_ids in same order), we use the **Wilcoxon signed-rank test** for paired comparison. This accounts for task difficulty correlation between runs.

Significance threshold: p < 0.05.

### Reported Metrics

| Metric | Description |
|--------|-------------|
| Success rate | Fraction of tasks passing evaluation |
| Latency | Task execution time (ms) |
| Cost | Estimated USD based on configurable pricing |
| Tool calls | Number of tool invocations per task |
| Tokens | Input + output tokens per task |

### Efficiency Metrics

- **Success per dollar**: success_rate / total_cost
- **Success per 1K tokens**: success_rate / (total_tokens / 1000)
- **Success per second**: success_rate / total_latency_seconds

These are labeled as approximate when provider pricing is unavailable.

### Failure Distribution

Aggregate failure categories across runs for taxonomy analysis. Compare distributions between configurations to identify systematic failure mode shifts.

## Interpreting Results

- **CI overlap**: If confidence intervals overlap substantially, the difference may not be meaningful
- **Significance without magnitude**: A statistically significant 0.5% improvement may not be practically important
- **Sample size**: Small benchmarks (< 20 tasks) produce wide confidence intervals
- **Multiple comparisons**: Running many pairwise comparisons increases false positive risk; consider Bonferroni correction for formal analysis

## Example Output

```
Single vs Planner+Executor:
  Success: 75.0% -> 82.0% (diff: +7.0%)
  95% CI: [+2.1%, +11.9%]
  p-value: 0.032 (significant)
```

We do not preselect winners. Reports present data; interpretation is the user's responsibility.
