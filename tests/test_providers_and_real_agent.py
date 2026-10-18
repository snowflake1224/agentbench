"""Tests for real provider integration boundaries without paid API calls."""

from __future__ import annotations

import json
from typing import Any

import httpx

from agentbench.adapters.model_agent import ModelAgentAdapter
from agentbench.core.agent import AgentResult, AgentRunContext
from agentbench.core.task import ExpectedBehavior, Task, TaskCategory
from agentbench.core.trace import TraceEventType
from agentbench.evaluators.base import LLMJudgeEvaluator
from agentbench.providers.base import ModelProvider, ModelResponse, ModelUsage
from agentbench.providers.openai_compatible import OpenAICompatibleProvider


class FakeProvider(ModelProvider):
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def complete(self, **kwargs: Any) -> ModelResponse:
        self.calls.append(kwargs)
        return ModelResponse(
            content=self.content,
            model=kwargs["model"],
            usage=ModelUsage(input_tokens=12, output_tokens=5),
            finish_reason="stop",
        )

    async def close(self) -> None:
        self.closed = True


async def test_openai_compatible_provider_sends_structured_schema() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [{"message": {"content": '{"answer":"42"}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://models.test/v1"
    )
    provider = OpenAICompatibleProvider(api_key="test", client=client)
    response = await provider.complete(
        model="test-model",
        messages=[{"role": "user", "content": "answer"}],
        response_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )
    assert captured["response_format"]["type"] == "json_schema"
    assert response.usage.input_tokens == 10
    await client.aclose()


async def test_model_agent_calls_provider() -> None:
    provider = FakeProvider("42")
    agent = ModelAgentAdapter(provider=provider)
    await agent.initialize({})
    task = Task(
        task_id="real-1",
        category=TaskCategory.GENERAL,
        input="What is 6 * 7?",
        expected_behavior=ExpectedBehavior(answer="42"),
    )
    result = await agent.run_task(AgentRunContext(run_id="run", task=task, model="test-model"))
    assert result.output == "42"
    assert result.input_tokens == 12
    assert [trace.event_type for trace in result.traces] == [
        TraceEventType.AGENT_START,
        TraceEventType.MODEL_CALL,
        TraceEventType.AGENT_END,
    ]
    assert provider.calls
    await agent.terminate()
    assert provider.closed


async def test_llm_judge_calls_provider_and_validates_schema() -> None:
    judgment = {
        name: {"score": 1.0, "rationale": "Meets the criterion."}
        for name in (
            "correctness",
            "relevance",
            "completeness",
            "instruction_following",
        )
    }
    judgment["failure_categories"] = []
    provider = FakeProvider(json.dumps(judgment))
    evaluator = LLMJudgeEvaluator(model="judge-model", provider=provider)
    task = Task(
        task_id="judge-1",
        category=TaskCategory.GENERAL,
        input="What is 6 * 7?",
        expected_behavior=ExpectedBehavior(answer="42"),
    )
    evaluation = await evaluator.evaluate(task, AgentResult(output="42"))
    assert evaluation.passed
    assert evaluation.score == 1.0
    assert provider.calls[0]["response_schema"]["type"] == "object"
    await evaluator.close()
