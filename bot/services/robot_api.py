"""Tiny HTTP control API for the robot, consumed by NAT agent tools.

Runs inside the bot process (daemon thread, own event loop) so the ReAct
agent can deliberately gesture: the NAT functions `play_animation` /
`look_at` call these endpoints. Binds to localhost by default; set
ROBOT_API_HOST=0.0.0.0 if the NAT server runs on another machine.
"""

import logging
import os
import threading

from fastapi import FastAPI
from pydantic import BaseModel

from .reachy_service import ReachyService

logger = logging.getLogger(__name__)

_started = False
_lock = threading.Lock()


class PlayAnimationRequest(BaseModel):
    name: str


class LookAtRequest(BaseModel):
    direction: str


def _build_app() -> FastAPI:
    app = FastAPI(title="sparky-robot-api")
    service = ReachyService.get_instance()

    @app.get("/robot/animations")
    def animations():
        return {"animations": service.list_animations()}

    @app.post("/robot/play_animation")
    def play_animation(req: PlayAnimationRequest):
        if service.animations.get(req.name) is None:
            return {"ok": False, "error": "unknown_animation", "animations": service.list_animations()}
        if not service.connected:
            return {"ok": False, "error": "robot_not_connected"}
        ok = service.play_animation(req.name)
        return {"ok": ok, "error": None if ok else "robot_error"}

    @app.post("/robot/look_at")
    def look_at(req: LookAtRequest):
        direction = req.direction.strip().lower()
        valid = {"left", "right", "up", "down", "front"}
        if direction not in valid:
            return {"ok": False, "valid_directions": sorted(valid)}
        service.look_at(direction)
        return {"ok": True}

    @app.get("/health")
    def health():
        return {"status": "ok", "robot_connected": service.connected}

    return app


def start_robot_api():
    """Start the robot control API once, in a background daemon thread."""
    global _started
    with _lock:
        if _started:
            return
        _started = True

    host = os.getenv("ROBOT_API_HOST", "127.0.0.1")
    port = int(os.getenv("ROBOT_API_PORT", "7861"))

    def _serve():
        import uvicorn

        uvicorn.run(_build_app(), host=host, port=port, log_level="warning")

    threading.Thread(target=_serve, daemon=True, name="robot-api").start()
    logger.info(f"Robot control API listening on http://{host}:{port}")
