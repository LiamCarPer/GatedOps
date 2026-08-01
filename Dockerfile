# Builder: resolve the exact dependency set from uv.lock.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs

RUN uv sync --frozen --no-dev

# Runtime: a single image that can run either the MLflow registry server or
# the scoring API, so the served mlflow version always matches uv.lock.
FROM python:3.12-slim

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    MLFLOW_DISABLE_TELEMETRY=1

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/configs /app/configs

EXPOSE 8000

CMD ["uvicorn", "gatedops.serve.main:app", "--host", "0.0.0.0", "--port", "8000"]
