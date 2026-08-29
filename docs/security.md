# Security

Agent outputs and benchmark content are treated as **potentially malicious**.

## Threat Model

| Threat | Mitigation |
|--------|------------|
| Prompt injection | Pattern detection in tool arguments |
| Malicious tool arguments | Validation before execution |
| Unsafe command execution | Dangerous command pattern blocking |
| Arbitrary file access | Sandbox mode for coding agents |
| Secrets leakage | Automatic redaction in traces/outputs |
| Untrusted repository content | Container isolation for SWE-bench |
| Evaluator manipulation | Deterministic evaluators preferred; LLM judge isolated |

## Implementation

### Prompt Injection Detection

```python
detect_prompt_injection(text)  # Returns warnings for known patterns
```

Patterns include: "ignore previous instructions", system prompt markers, special tokens.

### Command Safety

```python
is_safe_command("rm -rf /")  # → (False, "Dangerous command pattern")
```

Blocked patterns: `rm -rf /`, `mkfs`, `dd if=`, pipe-to-shell, etc.

### Secret Redaction

API keys, tokens, and AWS credentials are redacted from trace output:
```python
redact_secrets("api_key=sk-abc123")  # → "api_key=[REDACTED]"
```

### Output Sanitization

Tool outputs are truncated to `MAX_TOOL_OUTPUT_BYTES` (default 1MB) to prevent context overflow attacks.

### Sandbox Mode

For coding agent evaluation:
```env
SANDBOX_ENABLED=true
```

SWE-bench full evaluation runs in Docker containers with network isolation. See `SWEBenchAdapter(use_docker=True)`.

## Recommendations

- Run agent evaluation in isolated environments
- Never pass production credentials to agent tool interfaces
- Review trace output before sharing externally
- Use deterministic evaluators when possible to avoid judge manipulation
- Rotate API keys used for LLM-backed evaluation regularly
