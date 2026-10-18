"""OpenTelemetry instrumentation for AgentBench."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from agentbench.config import settings
from agentbench.core.trace import TraceEvent

_tracer: trace.Tracer | None = None
_initialized = False


def init_tracing(service_name: str | None = None, console: bool = False) -> None:
    """Initialize OpenTelemetry tracing."""
    global _tracer, _initialized
    if _initialized:
        return

    resource = Resource.create({"service.name": service_name or settings.otel_service_name})
    provider = TracerProvider(resource=resource)

    if console:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif settings.otel_exporter_otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("agentbench")
    _initialized = True


def get_tracer() -> trace.Tracer:
    if _tracer is None:
        init_tracing(console=True)
    return _tracer or trace.get_tracer("agentbench")


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
) -> Generator[trace.Span, None, None]:
    """Context manager for creating trace spans."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name, kind=kind) as span:
        if attributes:
            for k, v in attributes.items():
                if isinstance(v, (str, int, float, bool)):
                    span.set_attribute(k, v)
        yield span


def record_trace_event(event: TraceEvent) -> None:
    """Record a trace event as a span."""
    tracer = get_tracer()
    span_name = f"{event.event_type.value}"
    with tracer.start_as_current_span(span_name) as span:
        for k, v in event.to_otel_attributes().items():
            span.set_attribute(k, v)
        if event.error:
            span.set_status(trace.Status(trace.StatusCode.ERROR, event.error))
