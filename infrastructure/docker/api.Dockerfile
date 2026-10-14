# VEYRONIX API image (also used for the MCP server).
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install deps first for better layer caching.
COPY pyproject.toml README.md ./
COPY veyronix ./veyronix
RUN pip install -U pip wheel && pip install ".[ml,ai,db,otel]"

# Non-root runtime user.
RUN useradd --create-home --uid 10001 veyronix && chown -R veyronix /app
USER veyronix

EXPOSE 8000
CMD ["uvicorn", "veyronix.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
