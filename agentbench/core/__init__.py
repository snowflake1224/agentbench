"""Core domain models and interfaces."""

from agentbench.core.agent import AgentAdapter, AgentResult, AgentRunContext
from agentbench.core.experiment import ExperimentConfig, ExperimentRun
from agentbench.core.task import Budget, ExpectedBehavior, Task, TaskCategory
from agentbench.core.trace import TraceEvent, TraceEventType, TraceSpan

__all__ = [
    "AgentAdapter",
    "AgentResult",
    "AgentRunContext",
    "Budget",
    "ExpectedBehavior",
    "ExperimentConfig",
    "ExperimentRun",
    "Task",
    "TaskCategory",
    "TraceEvent",
    "TraceEventType",
    "TraceSpan",
]
