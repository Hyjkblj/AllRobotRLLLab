"""FastAPI application entry point.

Importing this module loads only contracts and lightweight XML adapters.  It
does not import Isaac Sim, MuJoCo, Celery, SQLAlchemy, or GPU libraries.
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .api.routes import router
from .config.settings import settings


def create_app() -> FastAPI:
    deployment_errors = settings.deployment_errors()
    if deployment_errors:
        raise RuntimeError("Invalid deployment configuration: " + "; ".join(deployment_errors))
    app = FastAPI(title="AllRobotRLLLab API", version="0.1.0")

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.request_id
        return response

    @app.exception_handler(HTTPException)
    async def structured_http_error(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": {"code": "HTTP_ERROR", "message": str(exc.detail)}, "request_id": getattr(request.state, "request_id", str(uuid.uuid4()))}
        return JSONResponse(status_code=exc.status_code, content=detail, headers={"x-request-id": getattr(request.state, "request_id", "")})

    @app.exception_handler(RequestValidationError)
    async def structured_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        errors = [{"loc": list(item.get("loc", ())), "msg": str(item.get("msg", "invalid value")), "type": item.get("type", "value_error")} for item in exc.errors()]
        return JSONResponse(status_code=422, content={"error": {"code": "SCHEMA_INVALID", "message": "request payload does not match the API contract", "stage": "api", "details": {"errors": errors}, "retryable": False}, "request_id": request_id}, headers={"x-request-id": request_id})

    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
