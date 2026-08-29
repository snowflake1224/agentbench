"""Security utilities for sandboxed agent execution."""

from __future__ import annotations

import re
from typing import Any

# Patterns that may indicate prompt injection attempts
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"disregard\s+(all\s+)?prior", re.I),
    re.compile(r"system\s*:\s*", re.I),
    re.compile(r"<\|.*?\|>", re.I),
]

# Dangerous shell command patterns
DANGEROUS_COMMANDS = [
    re.compile(r"rm\s+-rf\s+/"),
    re.compile(r"mkfs\."),
    re.compile(r"dd\s+if="),
    re.compile(r">\s*/dev/"),
    re.compile(r"curl.*\|\s*bash"),
    re.compile(r"wget.*\|\s*sh"),
]

SECRET_PATTERNS = [
    re.compile(r"(api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?\S+", re.I),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]


def detect_prompt_injection(text: str) -> list[str]:
    """Detect potential prompt injection patterns."""
    warnings = []
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            warnings.append(f"Potential injection: {pattern.pattern}")
    return warnings


def is_safe_command(command: str) -> tuple[bool, str | None]:
    """Check if a command is safe to execute in sandbox."""
    for pattern in DANGEROUS_COMMANDS:
        if pattern.search(command):
            return False, f"Dangerous command pattern: {pattern.pattern}"
    return True, None


def redact_secrets(text: str) -> str:
    """Redact potential secrets from text."""
    result = text
    for pattern in SECRET_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result


def sanitize_tool_output(output: Any, max_bytes: int = 1_048_576) -> Any:
    """Sanitize and truncate tool output."""
    if isinstance(output, str):
        if len(output.encode()) > max_bytes:
            return output[:max_bytes] + "\n[TRUNCATED]"
        return redact_secrets(output)
    if isinstance(output, dict):
        return {k: sanitize_tool_output(v, max_bytes) for k, v in output.items()}
    return output


def validate_tool_arguments(tool_name: str, args: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate tool arguments for safety."""
    for key, value in args.items():
        if isinstance(value, str):
            injections = detect_prompt_injection(value)
            if injections and tool_name not in ("evaluator", "judge"):
                return False, f"Suspicious input in {key}: {injections[0]}"
            if key in ("command", "cmd", "shell"):
                safe, reason = is_safe_command(value)
                if not safe:
                    return False, reason
    return True, None
