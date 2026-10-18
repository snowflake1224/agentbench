FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
RUN uv pip install --system -e ".[dev]"

COPY . .

RUN mkdir -p artifacts

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn agentbench.api.app:app --host 0.0.0.0 --port 8000"]
