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
_session = {"loop": None, "task": None, "messages": None, "mic_gate": None,
            "tts": None}

# Transcript fan-out (owned by the API server's event loop)
_api_loop: asyncio.AbstractEventLoop | None = None
_history: collections.deque = collections.deque(maxlen=200)
_ws_queues: set = set()

# Speaker-name registry (diarization): 0-based speaker tag -> known name.
# Written by POST /speakers (panel or the NAT remember-speaker tool), read
# by the pipeline's SpeakerLabelerProcessor (plain cross-thread dict read,
# same pattern as _session).
_speaker_names: dict[int, str] = {}

# Optional-tools registry: capabilities the NAT agent may use only when
# switched on in the panel. The catalog is code-seeded here (labels and
# descriptions never go stale in a state file); only the enabled bits
# persist in ~/.sparky/tools.json. NAT tools GET /tools before acting and
# fail closed, so a toggle applies to the very next agent turn with no
# restarts. Adding a future optional tool = one entry here + a gated NAT
# function + config.yml wiring. Only ever touched from the API server's
# event loop (NAT reads over HTTP; the pipeline never reads it), so no
# cross-thread concern.
_TOOLS_FILE = Path.home() / ".sparky" / "tools.json"
OPTIONAL_TOOLS: dict[str, dict] = {
    "web_search": {
        "label": "Web search (DuckDuckGo)",
        "description": "Let the assistant search the live internet and read web pages.",
        "enabled": os.getenv("WEB_SEARCH_ENABLED", "0").strip() == "1",
    },
    "animation_sounds": {
        "label": "Animation sounds",
        "description": "Play the sound effects that ship with some animations.",
        # default OFF: the bundled sfx grate quickly (Tim's call)
        "enabled": os.getenv("ANIMATION_SOUNDS_ENABLED", "0").strip() == "1",
    },
    "speaking_animations": {
        "label": "Speaking animations",
        "description": "Keep gently animating while Sparky talks through longer replies.",
        "enabled": os.getenv("SPEAKING_ANIMATIONS_ENABLED", "1").strip() != "0",
    },
    "idle_animations": {
        "label": "Idle animations",
        "description": "Play a gentle animation now and then when Sparky is idle.",
        "enabled": os.getenv("IDLE_ANIMATIONS_ENABLED", "1").strip() != "0",
        # numeric settings rendered as inputs in the panel; values are
        # clamped to [min, max] on write and persist alongside enabled
        "params": {
            "still_secs": {"label": "Still after activity", "unit": "s",
                           "value": float(os.getenv("IDLE_ANIM_STILL_SECS", "60")),
                           "min": 10, "max": 600},
            "min_gap_secs": {"label": "Min gap", "unit": "s",
                             "value": float(os.getenv("IDLE_ANIM_MIN_SECS", "45")),
                             "min": 10, "max": 900},
            "max_gap_secs": {"label": "Max gap", "unit": "s",
                             "value": float(os.getenv("IDLE_ANIM_MAX_SECS", "120")),
                             "min": 10, "max": 1800},
        },
    },
}


def _load_tool_states():
    """Overlay persisted state onto the code-seeded catalog. Values are
    either a bare bool (legacy) or {"enabled": bool, "params": {k: num}}.
    Missing/corrupt file must never break the bot."""
    try:
        import json

        saved = json.loads(_TOOLS_FILE.read_text())
        for name, state in saved.items():
            tool = OPTIONAL_TOOLS.get(name)
            if tool is None:
                continue
            if isinstance(state, bool):
                tool["enabled"] = state
            elif isinstance(state, dict):
                if isinstance(state.get("enabled"), bool):
                    tool["enabled"] = state["enabled"]
                for k, v in (state.get("params") or {}).items():
                    p = tool.get("params", {}).get(k)
                    if p is not None and isinstance(v, (int, float)):
                        p["value"] = max(p["min"], min(p["max"], float(v)))
    except Exception:
        pass


def _save_tool_states():
    try:
        import json

        _TOOLS_FILE.parent.mkdir(parents=True, exist_ok=True)
        state = {}
        for n, t in OPTIONAL_TOOLS.items():
            if t.get("params"):
                state[n] = {"enabled": t["enabled"],
                            "params": {k: p["value"] for k, p in t["params"].items()}}
            else:
                state[n] = t["enabled"]
        tmp = _TOOLS_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=1))
        os.replace(tmp, _TOOLS_FILE)
    except Exception as e:
        logger.warning(f"tools registry: could not persist state: {e}")


def tool_param(name: str, key: str) -> float:
    """Live numeric setting for an optional feature (scheduler-side read)."""
    return float(OPTIONAL_TOOLS[name]["params"][key]["value"])


_load_tool_states()


def get_speaker_name(tag: int) -> str | None:
    """Known name for a 0-based diarization speaker tag, else None."""
    return _speaker_names.get(tag)


def _director_status() -> dict:
    try:
        from .animation_director import get_director
        return get_director().status()
    except Exception:
        return {}


def _camera_state() -> dict:
    try:
        from .camera_service import CAMERA, CAMERA_RESOLUTIONS
        return {"resolution": CAMERA["resolution"], "options": CAMERA_RESOLUTIONS}
    except Exception:
        return {}


# TTS voice: current selection (seeded from env, boot default) plus the
# option list fetched once from the Kokoro server itself — the panel offers
# only voices that are actually installed there. No persistence, matching
# volume/camera semantics: KOKORO_VOICE is the default on every boot.
VOICE = {"voice": os.getenv("KOKORO_VOICE", "af_heart"), "options": []}


async def _kokoro_voices() -> list[str]:
    """Installed voices from the Kokoro server, cached after first success."""
    if VOICE["options"]:
        return VOICE["options"]
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=3.0)
    try:
        base = os.getenv("KOKORO_BASE_URL", "http://localhost:8880/v1").rstrip("/")
        r = await _http_client.get(f"{base}/audio/voices")
        r.raise_for_status()
        VOICE["options"] = sorted(v["id"] for v in r.json().get("voices", []))
    except Exception as e:
        logger.warning(f"voice list: could not query Kokoro: {e}")
    return VOICE["options"]


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
ANIMATIONS_DIR = Path(os.getenv("ANIMATIONS_DIR",
                                Path(__file__).resolve().parent.parent / "animations"))

# Animation soundtrack playback: the authored clips ship with audio that was
# never played. "sfx" (default) plays only the <name>_sfx.wav sound effects
# (camera shutter, wake chime, beep); "full" also plays the <name>.wav
# voice/soundtrack files; "off" disables.
ANIMATION_AUDIO = os.getenv("ANIMATION_AUDIO", "sfx").strip().lower()
_OUT_RATE = 24000


def _load_wav_pcm24k(path: Path) -> bytes | None:
    """Load a wav as 24kHz mono s16 PCM (numpy linear resample)."""
    import wave

    import numpy as np

    try:
        with wave.open(str(path), "rb") as w:
            rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if width != 2:
            return None
        samples = np.frombuffer(raw, dtype=np.int16)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
        if rate != _OUT_RATE:
            n_out = int(len(samples) * _OUT_RATE / rate)
            x_old = np.linspace(0.0, 1.0, len(samples), endpoint=False)
            x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
            samples = np.interp(x_new, x_old, samples.astype(np.float32)).astype(np.int16)
        return samples.tobytes()
    except Exception as e:
        logger.warning(f"animation audio: could not load {path.name}: {e}")
        return None


def _animation_audio_frames(name: str, force: bool = False) -> list:
    """OutputAudioRawFrames for an animation's audio, per ANIMATION_AUDIO mode.
    Played through the normal output path, so the software gain applies and
    the bot-speaking gate keeps the mic from hearing the robot's own sfx.
    Gated by the panel's Animation-sounds toggle (default off); force=True
    bypasses it for the e-stop confirmation beep, which is a safety cue,
    not an animation sound effect."""
    from pipecat.frames.frames import OutputAudioRawFrame

    if not force and not OPTIONAL_TOOLS["animation_sounds"]["enabled"]:
        return []
    if ANIMATION_AUDIO == "off":
        return []
    candidates = [ANIMATIONS_DIR / name / f"{name}_sfx.wav"]
    if ANIMATION_AUDIO == "full":
        candidates.append(ANIMATIONS_DIR / name / f"{name}.wav")
    frames = []
    chunk = _OUT_RATE * 2 // 2  # 500ms of s16 mono
    for path in candidates:
        if not path.exists():
            continue
        pcm = _load_wav_pcm24k(path)
        if not pcm:
            continue
        for i in range(0, len(pcm), chunk):
            frames.append(OutputAudioRawFrame(
                audio=pcm[i:i + chunk], sample_rate=_OUT_RATE, num_channels=1))
    return frames


def attach_session(loop, task, messages, mic_gate, tts=None):
    """Called from the bot once the pipeline is built."""
    _session.update(loop=loop, task=task, messages=messages, mic_gate=mic_gate,
                    tts=tts)
    logger.info("Control panel: session attached")


# Idle-animation clock: monotonic timestamp of the last conversational
# activity (any transcript line: user final, typed turn, assistant reply).
import time as _time_mod

LAST_INTERACTION = {"ts": _time_mod.monotonic()}


def push_transcript(item: dict):
    """Thread-safe transcript push (called from the pipeline loop)."""
    LAST_INTERACTION["ts"] = _time_mod.monotonic()
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
    # None = random for mirrorable clips; True/False forces (testing)
    mirror: bool | None = None


class SpeakerNameRequest(BaseModel):
    # display-number form as users see it: 1, "1", "Speaker 1"
    speaker: int | str
    name: str = ""


class OptionalToolRequest(BaseModel):
    name: str
    enabled: bool
    # optional numeric settings {key: value}, clamped server-side
    params: dict[str, float] | None = None


class CameraRequest(BaseModel):
    resolution: str


class VoiceRequest(BaseModel):
    voice: str


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
        # Inject directly at the TTS stage: frames queued at the pipeline
        # source stall behind the LLM stage for the whole agent turn, so
        # search announcements spoke AFTER the answer. Direct injection
        # synthesizes immediately even mid-turn.
        tts, loop = _session["tts"], _session["loop"]
        if tts is not None and loop is not None:
            asyncio.run_coroutine_threadsafe(
                tts.queue_frame(TTSSpeakFrame(text)), loop)
            return {"ok": True}
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
            # de-duplicate identical URLs (several roles share the one shodan engine)
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
            "tools": OPTIONAL_TOOLS,
            "camera": _camera_state(),
            "animation": _director_status(),
            "voice": {"voice": VOICE["voice"],
                      "options": await _kokoro_voices()},
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

    # --- TTS voice selection (applies from the next spoken sentence) ---

    @app.get("/voice")
    async def voice():
        return {"ok": True, "voice": VOICE["voice"],
                "options": await _kokoro_voices()}

    @app.post("/voice")
    async def set_voice(req: VoiceRequest):
        from pipecat.frames.frames import TTSUpdateSettingsFrame
        from pipecat.services.openai import tts as openai_tts
        from pipecat.services.openai.tts import OpenAITTSService

        value = req.voice.strip()
        options = await _kokoro_voices()
        if value not in options:
            return {"ok": False, "error": "unknown_voice", "options": options}
        # the constructor only registered the boot voice; keep the OpenAI
        # base class's client-side validation happy for the new one
        openai_tts.VALID_VOICES[value] = value
        ok = _queue_frames([TTSUpdateSettingsFrame(
            delta=OpenAITTSService.Settings(voice=value))])
        if not ok:
            return {"ok": False, "error": "no_session"}
        VOICE["voice"] = value
        logger.info(f"TTS voice set to {value}")
        return {"ok": True, "voice": value, "options": options}

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

    @app.get("/tools")
    def tools():
        return {"ok": True, "tools": OPTIONAL_TOOLS}

    @app.post("/tools")
    def set_tool(req: OptionalToolRequest):
        if req.name not in OPTIONAL_TOOLS:
            return {"ok": False, "error": "unknown_tool"}
        tool = OPTIONAL_TOOLS[req.name]
        tool["enabled"] = req.enabled
        for k, v in (req.params or {}).items():
            p = tool.get("params", {}).get(k)
            if p is not None:
                p["value"] = max(p["min"], min(p["max"], float(v)))
        # a gap window must stay ordered
        params = tool.get("params", {})
        if "min_gap_secs" in params and "max_gap_secs" in params:
            if params["max_gap_secs"]["value"] < params["min_gap_secs"]["value"]:
                params["max_gap_secs"]["value"] = params["min_gap_secs"]["value"]
        _save_tool_states()
        logger.info(f"Optional tool '{req.name}' "
                    f"{'enabled' if req.enabled else 'disabled'}")
        return {"ok": True, "tools": OPTIONAL_TOOLS}

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
        # flat names kept for the NAT tool; catalog powers the panel browser
        return {"animations": service.list_animations(),
                "catalog": service.animations.catalog()}

    @app.post("/robot/play_animation")
    def play_animation(req: PlayAnimationRequest):
        from .animation_director import get_director

        if service.animations.get(req.name) is None:
            # exact shape the NAT tool parses to self-correct with the list
            return {"ok": False, "error": "unknown_animation", "animations": service.list_animations()}
        if not service.connected:
            return {"ok": False, "error": "robot_not_connected"}
        res = get_director().request("user", clip=req.name, mirror=req.mirror)
        if res["accepted"]:
            audio_frames = _animation_audio_frames(req.name)
            if audio_frames:
                _queue_frames(audio_frames)
        return {"ok": res["accepted"], "error": res["reason"]}

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
        # audible acknowledgment (output path keeps running; only the mic
        # and motors stop)
        beep = _animation_audio_frames("beep", force=True)
        if beep:
            _queue_frames(beep)
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
        from .animation_director import get_director
        loop.create_task(get_director().tick_task())
        loop.run_until_complete(server.serve())

    threading.Thread(target=_serve, daemon=True, name="robot-api").start()
    logger.info(f"Control panel listening on http://{host}:{port}")
