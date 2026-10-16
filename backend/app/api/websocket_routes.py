"""WebSocket status streams for durable asset and motion state."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.app.domain.contracts import Actor
from backend.app.application.motion_pipeline_service import MotionPipelineError
from backend.app.application.run_service import RunServiceError


def build_status_websocket_router(*, asset_service, motion_pipeline_service) -> APIRouter:
    router = APIRouter()

    @router.websocket("/ws/projects/{project_id}/assets")
    async def watch_project_assets(websocket: WebSocket, project_id: str) -> None:
        await websocket.accept()
        actor = Actor(user_id=websocket.headers.get("x-user-id", "local-user").strip() or "local-user", email=websocket.headers.get("x-user-email"))
        try:
            last = None
            while True:
                records = asset_service.list_assets(actor=actor, project_id=project_id)
                payload = [{"asset": asset.model_dump(mode="json"), "versions": [version.model_dump(mode="json") for version in versions]} for asset, versions in records]
                serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
                if serialized != last:
                    await websocket.send_json({"type": "assets.snapshot", "items": payload})
                    last = serialized
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            return
        except RunServiceError as exc:
            await websocket.close(code=4403 if exc.status_code == 403 else 4404, reason=exc.message)
        except Exception:
            await websocket.close(code=1011, reason="asset watcher failed")

    @router.websocket("/ws/motion-pipelines/{pipeline_id}")
    async def watch_motion_pipeline(websocket: WebSocket, pipeline_id: str) -> None:
        await websocket.accept()
        actor = Actor(user_id=websocket.headers.get("x-user-id", "local-user").strip() or "local-user", email=websocket.headers.get("x-user-email"))
        try:
            last = None
            while True:
                record = motion_pipeline_service.get(actor=actor, pipeline_id=pipeline_id)
                payload = record.model_dump(mode="json")
                serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
                if serialized != last:
                    await websocket.send_json({"type": "motion_pipeline.snapshot", "item": payload})
                    last = serialized
                if record.status in {"GVHMR_READY", "GMR_READY", "READY", "FAILED"}:
                    await websocket.close(code=1000, reason="pipeline reached a stable state")
                    return
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            return
        except MotionPipelineError as exc:
            await websocket.close(code=4403 if exc.status_code == 403 else 4404, reason=exc.message)
        except Exception:
            await websocket.close(code=1011, reason="pipeline watcher failed")

    return router


__all__ = ["build_status_websocket_router"]
