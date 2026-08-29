"""API integration tests."""

import pytest
from httpx import ASGITransport, AsyncClient

from agentbench.api.app import app


@pytest.mark.asyncio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_dashboard():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/dashboard")
        assert resp.status_code == 200
        assert "AgentBench" in resp.text


@pytest.mark.asyncio
async def test_list_runs():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/runs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
