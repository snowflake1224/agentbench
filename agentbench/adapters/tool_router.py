"""Tool routing strategies for agent evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ToolRoutingStrategy(StrEnum):
    UNRESTRICTED = "unrestricted"
    RULE_BASED = "rule_based"
    MODEL_SELECTED = "model_selected"
    SPECIALIZED_AGENT = "specialized_agent"


class ToolRouter:
    """Route tools based on strategy configuration."""

    def __init__(self, strategy: ToolRoutingStrategy, rules: dict[str, list[str]] | None = None) -> None:
        self.strategy = strategy
        self.rules = rules or {
            "general": ["search", "calculator", "read_file"],
            "tool_use": ["api_call", "database_query", "calculator"],
            "coding": ["read_file", "write_file", "run_tests", "grep"],
        }

    def select_tools(self, category: str, available_tools: list[str], task_input: str = "") -> list[str]:
        if self.strategy == ToolRoutingStrategy.UNRESTRICTED:
            return available_tools

        if self.strategy == ToolRoutingStrategy.RULE_BASED:
            allowed = self.rules.get(category, available_tools)
            return [t for t in available_tools if t in allowed]

        if self.strategy == ToolRoutingStrategy.MODEL_SELECTED:
            # Simulated model-based selection: pick tools matching keywords
            keywords = task_input.lower().split()
            selected = []
            for tool in available_tools:
                if any(kw in tool.lower() for kw in keywords[:3]):
                    selected.append(tool)
            return selected or available_tools[:2]

        if self.strategy == ToolRoutingStrategy.SPECIALIZED_AGENT:
            # Route to category-specific subset
            category_tools = self.rules.get(category, [])
            return [t for t in available_tools if t in category_tools] or available_tools[:1]

        return available_tools

    def metadata(self) -> dict[str, Any]:
        return {"strategy": self.strategy.value, "rules": self.rules}
