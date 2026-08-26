# Runtime image for the project-owned API and CPU/Celery worker processes.
# Isaac Sim, GMR/GVHMR and robot assets are intentionally mounted externally.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements-platform.txt pyproject.toml ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-platform.txt

COPY adapters ./adapters
COPY apps ./apps
COPY backend ./backend
COPY packages ./packages
COPY schemas ./schemas
COPY tools ./tools
COPY scripts ./scripts
COPY infra/migrations ./infra/migrations
COPY third_party/README.md ./third_party/README.md

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/.runtime \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
