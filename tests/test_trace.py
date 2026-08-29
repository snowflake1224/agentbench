"""Tests for trace model."""

from agentbench.core.trace import TraceEvent, TraceEventType


def test_trace_event_creation():
    event = TraceEvent(
        run_id="run-1",
        task_id="task-1",
        event_type=TraceEventType.TOOL_CALL,
        tool="calculator",
        tool_input={"expr": "2+2"},
        latency_ms=50.0,
    )
    assert event.run_id == "run-1"
    assert event.event_type == TraceEventType.TOOL_CALL
    assert event.tool == "calculator"


def test_trace_event_otel_attributes():
    event = TraceEvent(
        run_id="run-1",
        task_id="task-1",
        event_type=TraceEventType.MODEL_CALL,
        model="mock",
        input_tokens=100,
        output_tokens=50,
    )
    attrs = event.to_otel_attributes()
    assert attrs["run_id"] == "run-1"
    assert attrs["model"] == "mock"
    assert attrs["input_tokens"] == 100


def test_trace_span_duration():
    from datetime import UTC, datetime, timedelta

    from agentbench.core.trace import TraceSpan

    start = datetime.now(UTC)
    end = start + timedelta(milliseconds=500)
    span = TraceSpan(
        name="test",
        event_type=TraceEventType.TASK_START,
        start_time=start,
        end_time=end,
    )
    assert span.duration_ms == 500.0
