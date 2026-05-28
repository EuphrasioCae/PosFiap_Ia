FROM python:3.12-slim-bookworm

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev

COPY notebooks ./notebooks
COPY data ./data
COPY results ./results

CMD ["uv", "run", "python", "-m", "pipeline"]
