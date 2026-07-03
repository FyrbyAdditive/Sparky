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

# Speaker-name registry (diarization): 0-based speaker tag -> known name.
# Written by POST /speakers (panel or the NAT remember-speaker tool), read
# by the pipeline's SpeakerLabelerProcessor (plain cross-thread dict read,
# same pattern as _session).
_speaker_names: dict[int, str] = {}


def get_speaker_name(tag: int) -> str | None:
    """Known name for a 0-based diarization speaker tag, else None."""
    return _speaker_names.get(tag)


def _camera_state() -> dict:
    try:
        from .camera_service import CAMERA, CAMERA_RESOLUTIONS
        return {"resolution": CAMERA["resolution"], "options": CAMERA_RESOLUTIONS}
    except Exception:
        return {}


def set_speaker_name(tag: int, name: str):
    """Register a name for a speaker tag (pipeline-side helper)."""
    _speaker_names[tag] = name
    logger.info(f"Speaker registry: Speaker {tag + 1} -> {name}")

# /status support (all touched only from the API server's event loop):
# short-TTL health cache so several open panels don't multiply probes into
# the live inference engines, plus one long-lived client for keep-alive.
_HEALTH_TTL_SECS = 5.0
_health_cache: dict = {"ts": 0.0, "results": {}}
_http_client: httpx.AsyncClient | None = None

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




class MuteRequest(BaseModel):
    muted: bool


class PlayAnimationRequest(BaseModel):
    name: str


class SpeakerNameRequest(BaseModel):
    # display-number form as users see it: 1, "1", "Speaker 1"
    speaker: int | str
    name: str = ""


class CameraRequest(BaseModel):
    resolution: str


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
        # software gain in the bot's output path: platform-independent
        # (pactl only existed on Linux and the slider vanished on macOS)
        from .local_audio import VOLUME
        return {"ok": True, "percent": VOLUME["percent"]}

    @app.post("/volume")
    def set_volume(req: VolumeRequest):
        from .local_audio import VOLUME
        percent = max(0, min(120, req.percent))
        VOLUME["percent"] = percent
        logger.info(f"Speaker volume set to {percent}%")
        return {"ok": True, "percent": percent}

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
                hb_ts = audio.pop("mic_helper_last_hb_ts", 0.0)
                audio["mic_helper_hb_age_secs"] = (
                    round(_time.time() - hb_ts, 1) if hb_ts > 0 else None)
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
            "speakers": {str(t + 1): n for t, n in sorted(_speaker_names.items())},
            "camera": _camera_state(),
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

    # --- camera capture settings + live stream ---

    @app.get("/camera")
    def camera():
        from .camera_service import CAMERA, CAMERA_RESOLUTIONS
        return {"ok": True, "resolution": CAMERA["resolution"],
                "options": CAMERA_RESOLUTIONS}

    @app.post("/camera")
    def set_camera(req: CameraRequest):
        from .camera_service import CAMERA, CAMERA_RESOLUTIONS
        value = req.resolution.strip().lower()
        if value not in CAMERA_RESOLUTIONS:
            return {"ok": False, "error": "unknown resolution",
                    "options": CAMERA_RESOLUTIONS}
        CAMERA["resolution"] = value  # capture reopens on next grab
        logger.info(f"Camera capture resolution set to {value}")
        return {"ok": True, "resolution": value}

    @app.get("/camera/stream")
    async def camera_stream():
        """Live MJPEG feed for the panel (~8fps). The generator dies with the
        client connection, so an unwatched stream costs nothing."""
        from fastapi.responses import StreamingResponse

        from . import camera_service

        async def frames():
            loop = asyncio.get_running_loop()
            while True:
                jpeg = await loop.run_in_executor(None, camera_service.grab_jpeg)
                if jpeg is None:
                    await asyncio.sleep(1.0)  # camera unavailable; keep trying
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                       + jpeg + b"\r\n")
                await asyncio.sleep(0.12)

        return StreamingResponse(frames(),
                                 media_type="multipart/x-mixed-replace; boundary=frame")

    # --- speaker registry (diarization) ---

    @app.get("/speakers")
    def speakers():
        return {"ok": True,
                "speakers": {str(t + 1): n for t, n in sorted(_speaker_names.items())}}

    @app.post("/speakers")
    def set_speaker(req: SpeakerNameRequest):
        import re

        m = re.search(r"(\d+)", str(req.speaker))
        if m:
            display = int(m.group(1))
            if not 1 <= display <= 8:
                return {"ok": False, "error": "speaker number out of range"}
            tag = display - 1
        else:
            # already-named speaker referenced by name (panel rename)
            wanted = str(req.speaker).strip().lower()
            tag = next((t for t, n in _speaker_names.items() if n.lower() == wanted), None)
            if tag is None:
                return {"ok": False, "error": "speaker must be a number, 'Speaker N', or a known name"}
            display = tag + 1
        name = req.name.strip().strip("'\"")
        # a "name" that is itself a speaker label is always a confused caller
        if re.fullmatch(r"speaker\s*\d*", name, re.IGNORECASE):
            return {"ok": False, "error": "that is a label, not a name"}
        if name and name.lower() not in ("forget", "none", "clear", "unknown"):
            _speaker_names[tag] = name
            logger.info(f"Speaker registry: Speaker {display} -> {name}")
        else:
            _speaker_names.pop(tag, None)
            logger.info(f"Speaker registry: Speaker {display} forgotten")
        return {"ok": True,
                "speakers": {str(t + 1): n for t, n in sorted(_speaker_names.items())}}

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
