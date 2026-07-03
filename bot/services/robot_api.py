"""Control panel + robot control API.

Runs inside the bot process (daemon thread, own event loop) and serves:
- the control panel single-page UI (GET /)
- typed conversation turns (POST /say) and verbatim speech (POST /speak)
- mic mute (POST /mute), live transcript (WS /ws), status (GET /status)
- robot actions used by the NAT agent tools (POST /robot/*)

The pipeline side registers itself via attach_session(); handlers inject
frames into the pipeline's event loop with run_coroutine_threadsafe.
Binds to localhost; the nginx TLS proxy fronts it at https://<host>/.
"""

import asyncio
import collections
import logging
import os
import threading
from pathlib import Path

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .reachy_service import ReachyService

logger = logging.getLogger(__name__)

_started = False
_lock = threading.Lock()

# Set by attach_session() once the pipeline exists
_session = {"loop": None, "task": None, "messages": None, "mic_gate": None}

# Transcript fan-out (owned by the API server's event loop)
_api_loop: asyncio.AbstractEventLoop | None = None
_history: collections.deque = collections.deque(maxlen=200)
_ws_queues: set = set()

# /status support (all touched only from the API server's event loop):
# short-TTL health cache so several open panels don't multiply probes into
# the live inference engines, plus one long-lived client for keep-alive.
_HEALTH_TTL_SECS = 5.0
_health_cache: dict = {"ts": 0.0, "results": {}}
_http_client: httpx.AsyncClient | None = None
_sink_name: str | None = None  # pactl sink discovery is stable per session

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def attach_session(loop, task, messages, mic_gate):
    """Called from the bot once the pipeline is built."""
    _session.update(loop=loop, task=task, messages=messages, mic_gate=mic_gate)
    logger.info("Control panel: session attached")


def push_transcript(item: dict):
    """Thread-safe transcript push (called from the pipeline loop)."""
    if _api_loop is None:
        return
    _api_loop.call_soon_threadsafe(_broadcast, item)


def _broadcast(item: dict):
    _history.append(item)
    for q in list(_ws_queues):
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            # wedged panel client: drop its oldest item rather than leak
            try:
                q.get_nowait()
                q.put_nowait(item)
            except asyncio.QueueEmpty:
                pass


def _queue_frames(frames) -> bool:
    loop, task = _session["loop"], _session["task"]
    if loop is None or task is None:
        return False
    asyncio.run_coroutine_threadsafe(task.queue_frames(frames), loop)
    return True


class TextRequest(BaseModel):
    text: str


class VolumeRequest(BaseModel):
    percent: int


def _reachy_sink() -> str | None:
    import subprocess

    global _sink_name
    if _sink_name is not None:
        return _sink_name
    try:
        out = subprocess.run(["pactl", "list", "short", "sinks"],
                             capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) > 1 and "reachy" in parts[1].lower():
                _sink_name = parts[1]
                return _sink_name
    except Exception as e:
        logger.warning(f"sink discovery failed: {e}")
    return None


class MuteRequest(BaseModel):
    muted: bool


class PlayAnimationRequest(BaseModel):
    name: str


class LookAtRequest(BaseModel):
    direction: str


def _build_app() -> FastAPI:
    from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame

    app = FastAPI(title="sparky-panel")
    service = ReachyService.get_instance()

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "panel.html")

    @app.post("/say")
    def say(req: TextRequest):
        text = req.text.strip()
        if not text:
            return {"ok": False, "error": "empty"}
        messages = _session["messages"]
        if messages is None:
            return {"ok": False, "error": "no_session"}
        messages.append({"role": "user", "content": text})
        ok = _queue_frames([LLMRunFrame()])
        if ok:
            push_transcript({"role": "user", "text": f"{text}  (typed)"})
        return {"ok": ok}

    @app.post("/speak")
    def speak(req: TextRequest):
        text = req.text.strip()
        if not text:
            return {"ok": False, "error": "empty"}
        ok = _queue_frames([TTSSpeakFrame(text)])
        return {"ok": ok}

    @app.post("/mute")
    def mute(req: MuteRequest):
        gate = _session["mic_gate"]
        if gate is None:
            return {"ok": False, "error": "no_session"}
        gate.set_muted(req.muted)
        return {"ok": True, "muted": req.muted}

    @app.get("/volume")
    def get_volume():
        import re
        import subprocess

        sink = _reachy_sink()
        if not sink:
            return {"ok": False, "error": "no_sink"}
        try:
            out = subprocess.run(["pactl", "get-sink-volume", sink],
                                 capture_output=True, text=True, timeout=5).stdout
            m = re.search(r"(\d+)%", out)
            return {"ok": True, "percent": int(m.group(1)) if m else None}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/volume")
    def set_volume(req: VolumeRequest):
        import subprocess

        sink = _reachy_sink()
        if not sink:
            return {"ok": False, "error": "no_sink"}
        percent = max(0, min(150, req.percent))
        try:
            subprocess.run(["pactl", "set-sink-volume", sink, f"{percent}%"],
                           check=True, timeout=5)
            return {"ok": True, "percent": percent}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.get("/status")
    async def status():
        def health_targets():
            targets = {}
            nat = os.getenv("NAT_BASE_URL", "http://localhost:8001/v1").removesuffix("/v1")
            targets["nat"] = f"{nat}/docs"
            for role, env in [("agent-llm", "AGENT_LLM_BASE_URL"),
                              ("router-llm", "ROUTER_LLM_BASE_URL"),
                              ("vision-llm", "VISION_LLM_BASE_URL")]:
                base = os.getenv(env)
                if base:
                    targets[role] = base.removesuffix("/v1") + "/health"
            kokoro = os.getenv("KOKORO_BASE_URL", "http://localhost:8880/v1").removesuffix("/v1")
            targets["tts"] = f"{kokoro}/v1/models"
            riva_host = os.getenv("RIVA_SERVER", "localhost:50051").split(":")[0]
            targets["stt"] = f"http://{riva_host}:9000/v1/health/ready"
            wiki = os.getenv("WIKI_BASE_URL")
            if wiki:
                targets["wiki"] = f"{wiki.rstrip('/')}/health"
            return targets

        import time as _t

        global _http_client
        if _http_client is None:
            _http_client = httpx.AsyncClient(timeout=3.0)

        if _t.monotonic() - _health_cache["ts"] < _HEALTH_TTL_SECS:
            results = _health_cache["results"]
        else:
            # de-duplicate identical URLs (unified profile points several roles at one engine)
            targets = health_targets()
            unique = {}
            for name, url in targets.items():
                unique.setdefault(url, []).append(name)

            results = {}

            async def check(url, names):
                try:
                    r = await _http_client.get(url)
                    ok = r.status_code == 200
                except Exception:
                    ok = False
                for n in names:
                    results[n] = ok
            await asyncio.gather(*(check(u, ns) for u, ns in unique.items()))
            _health_cache.update(ts=_t.monotonic(), results=results)

        gate = _session["mic_gate"]
        try:
            import time as _time

            from .local_audio import AUDIO_STATS, GATE
            audio = {}
            if AUDIO_STATS["mic_last_frame_ts"] > 0:  # instrumented transport active
                audio = dict(AUDIO_STATS)
                audio["mic_frame_age_secs"] = round(_time.time() - audio.pop("mic_last_frame_ts"), 1)
                # gate visibility: a robot that hears nothing while 'unmuted'
                # was undiagnosable without this
                audio["speech_gated"] = bool(GATE["bot_speaking"])
                audio["gate_muted"] = bool(GATE["muted"])
        except Exception:
            audio = {}
        return {
            "services": results,
            "robot_connected": service.connected,
            "muted": bool(gate.muted) if gate else False,
            "audio": audio,
            "session_active": _session["task"] is not None,
            "models": {
                "agent": os.getenv("AGENT_LLM_MODEL", "?"),
                "router": os.getenv("ROUTER_LLM_MODEL", "?"),
                "vision": os.getenv("VISION_LLM_MODEL", "?"),
            },
        }

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        _ws_queues.add(q)
        try:
            await websocket.send_json({"type": "history", "items": list(_history)})
            while True:
                item = await q.get()
                await websocket.send_json({"type": "transcript", "item": item})
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            _ws_queues.discard(q)

    # --- robot actions (also used by the NAT agent tools) ---

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

    @app.post("/estop")
    def estop():
        """Emergency stop: halt motion, release torque, mute the mic."""
        results = {}
        try:
            gate = _session["mic_gate"]
            if gate:
                gate.set_muted(True)
            results["muted"] = True
        except Exception as e:
            results["muted"] = str(e)
        try:
            if service.motion_manager:
                service.motion_manager.stop()
            results["motion_stopped"] = True
        except Exception as e:
            results["motion_stopped"] = str(e)
        try:
            robot = service.robot
            if robot is not None:
                for meth in ("disable_motors", "turn_off"):
                    fn = getattr(robot, meth, None)
                    if fn:
                        fn()
                        results["torque_released"] = meth
                        break
        except Exception as e:
            results["torque_released"] = str(e)
        service.connected = False
        logger.warning(f"EMERGENCY STOP: {results}")
        return {"ok": True, **results}

    @app.get("/health")
    def health():
        return {"status": "ok", "robot_connected": service.connected}

    return app


def start_robot_api():
    """Start the panel/API server once, in a background daemon thread."""
    global _started
    with _lock:
        if _started:
            return
        _started = True

    host = os.getenv("ROBOT_API_HOST", "127.0.0.1")
    port = int(os.getenv("ROBOT_API_PORT", "7861"))

    def _serve():
        global _api_loop
        import uvicorn

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _api_loop = loop
        config = uvicorn.Config(_build_app(), host=host, port=port, log_level="warning", loop="asyncio")
        server = uvicorn.Server(config)
        loop.run_until_complete(server.serve())

    threading.Thread(target=_serve, daemon=True, name="robot-api").start()
    logger.info(f"Control panel listening on http://{host}:{port}")
