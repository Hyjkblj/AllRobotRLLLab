"""ASGI entrypoint used by uvicorn."""

from backend.app.main import app

__all__ = ["app"]

