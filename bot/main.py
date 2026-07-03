#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
# Robot-native fork: audio flows through the Reachy Mini's own mic and
# speaker (PipeWire echo-cancelled by default), everything runs on local
# hardware, and a control panel replaces the browser playground.


import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.services.nvidia.stt import NvidiaSTTService
from pipecat.transports.local.audio import LocalAudioTransportParams

from services.local_audio import DeepBufferedLocalAudioTransport

from nat_vision_llm import NATVisionLLMService
from services.emotion import EmotionReactorProcessor
from services.kokoro_tts import KokoroTTSService
from services.mic_gate import MicGateProcessor
from services.reachy_service import ReachyService
from services.processor import ReachyWobblerProcessor
from services.robot_api import start_robot_api, attach_session, push_transcript
from services.speaker_labels import SpeakerLabelerProcessor
from services.robot_camera import RobotCameraResponder
from services.transcript_tap import TranscriptTap


load_dotenv(override=True)

# Loguru's default sink is stderr at DEBUG, so pipecat's frame-level DEBUG
# detail floods the console on the realtime audio path. Default to INFO;
# BOT_LOG_LEVEL=DEBUG restores full diagnostics.
import sys as _sys

logger.remove()
logger.add(_sys.stderr, level=os.getenv("BOT_LOG_LEVEL", "INFO").upper())


def _acquire_single_instance_lock():
    """Refuse to run two bots: duplicate instances fight over the robot
    (conflicting motion commands) and over audio devices."""
    import fcntl

    lock_path = os.path.expanduser("~/.sparky-bot.lock")
    lock = open(lock_path, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("Another Sparky bot instance is already running on this "
                         "machine (lock: ~/.sparky-bot.lock). Stop it first.")
    lock.write(str(os.getpid()))
    lock.flush()
    return lock


_instance_lock = _acquire_single_instance_lock()

# Control panel + agent robot tools (play_animation / look_at)
start_robot_api()

PERSONA = (
    "You are Sparky, a small expressive robot assistant with a physical body: "
    "a head that moves and two antennas. You run entirely on local hardware — "
    "no cloud. Your replies are spoken aloud, so keep them short (one to three "
    "sentences), natural and conversational. Never use emojis, bullet points or "
    "special characters. Vary your phrasing; never repeat earlier sentences or "
    "reintroduce yourself. Stay aware of the whole conversation and refer back "
    "to things the user said. You can see through your camera when asked about "
    "the surroundings, and you can move: nod, look around, wiggle your antennas. "
    "You may hear several different people. Each line of user speech is prefixed "
    "with who said it, like 'Speaker 1:' or their name once known. Keep track of "
    "who said what and address people by name when you know it. When someone "
    "tells you their name, or names another speaker, use your remember-speaker "
    "tool to store it. Do not claim to recognize voices beyond these labels."
)


def _scan(pa, name_substring: str, want_input: bool) -> int | None:
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        channels = info.get("maxInputChannels" if want_input else "maxOutputChannels", 0)
        if channels > 0 and name_substring.lower() in str(info.get("name", "")).lower():
            return i
    return None


def _resolve_audio_devices():
    """Find the robot's audio devices, waiting for hotplug if absent.

    Sound must go in and out of the robot — never silently fall back to
    this machine's mic/speakers. If the named devices aren't present, poll
    (PortAudio caches enumeration, so each retry uses a fresh instance) so
    the robot can be plugged in after launch. AUDIO_STRICT=false restores
    the old default-device fallback for bench/dev setups.

    Returns (pyaudio_instance, input_index, output_index).
    """
    import pyaudio

    in_name = os.getenv("AUDIO_IN_DEVICE", "").strip()
    out_name = os.getenv("AUDIO_OUT_DEVICE", "").strip()
    strict = os.getenv("AUDIO_STRICT", "true").strip().lower() != "false"
    wait_cap = float(os.getenv("AUDIO_WAIT_SECS", "0")) or None  # None = wait forever

    started = None
    while True:
        pa = pyaudio.PyAudio()
        in_idx = _scan(pa, in_name, True) if in_name else None
        out_idx = _scan(pa, out_name, False) if out_name else None
        in_ok = (not in_name) or in_idx is not None
        out_ok = (not out_name) or out_idx is not None
        if in_ok and out_ok:
            if in_idx is not None:
                logger.info(f"Audio input: [{in_idx}] {pa.get_device_info_by_index(in_idx)['name']}")
            if out_idx is not None:
                logger.info(f"Audio output: [{out_idx}] {pa.get_device_info_by_index(out_idx)['name']}")
            return pa, in_idx, out_idx

        missing = [n for n, ok in ((in_name, in_ok), (out_name, out_ok)) if n and not ok]
        available = sorted({str(pa.get_device_info_by_index(i).get("name")) for i in range(pa.get_device_count())})
        pa.terminate()

        if not strict:
            logger.warning(f"Audio device(s) {missing} not found, using defaults (AUDIO_STRICT=false)")
            import pyaudio as _pa
            return _pa.PyAudio(), None, None

        import time as _time
        if started is None:
            started = _time.monotonic()
            logger.warning(f"Waiting for robot audio device(s) {missing} — is the robot plugged in? "
                           f"Devices seen: {available}")
        elif int(_time.monotonic() - started) % 15 < 3:
            logger.warning(f"Still waiting for robot audio device(s) {missing}...")
        if wait_cap and (_time.monotonic() - started) > wait_cap:
            raise SystemExit(f"Robot audio device(s) {missing} did not appear within "
                             f"{wait_cap:.0f}s. Devices seen: {available}")
        _time.sleep(3)


async def run_bot():
    logger.info("Starting Sparky (robot-native audio)")

    pa, in_idx, out_idx = _resolve_audio_devices()
    transport = DeepBufferedLocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=int(os.getenv("AUDIO_IN_SAMPLE_RATE", "16000")),
            audio_out_sample_rate=int(os.getenv("AUDIO_OUT_SAMPLE_RATE", "24000")),
            # Room-mic turn taking: natural pauses must not end the turn.
            # Silero marks candidate stops; smart-turn decides semantically.
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(stop_secs=float(os.getenv("VAD_STOP_SECS", "0.6")))
            ),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
            input_device_index=in_idx,
            output_device_index=out_idx,
        )
    )

    # Streaming ASR from the local Nemotron ASR NIM (Riva protocol).
    stt = NvidiaSTTService(
        server=os.getenv("RIVA_SERVER", "localhost:50051"),
        use_ssl=False,
        model_function_map={"function_id": "", "model_name": os.getenv("RIVA_MODEL", "")},
        # pipecat defaults stop_history to 320 (ms), telling Riva to
        # finalize at every ~1/3s pause — sentences shred into fragment
        # finals ("The / Fox jumps over / Do"). -1 = server-default
        # endpointing, verified to yield whole-utterance finals here.
        # The EOU pair applies to models with end-of-utterance detection
        # (Nemotron ASR streaming); -1 keeps server defaults there too.
        stop_history=int(os.getenv("RIVA_STOP_HISTORY_MS", "-1")),
        stop_history_eou=int(os.getenv("RIVA_STOP_HISTORY_EOU_MS", "-1")),
        stop_threshold_eou=float(os.getenv("RIVA_STOP_THRESHOLD_EOU", "-1.0")),
        # The NIM embeds the streaming sortformer diarizer: tag every word
        # with a speaker (stable per stream, up to 4). SpeakerLabelerProcessor
        # turns the tags into "Speaker N:"/name prefixes downstream.
        settings=NvidiaSTTService.Settings(
            speaker_diarization=os.getenv("SPEAKER_DIARIZATION", "1").strip() != "0",
            diarization_max_speakers=int(os.getenv("DIARIZATION_MAX_SPEAKERS", "4")),
            word_time_offsets=True,  # tags ride on words[]; keep it populated
        ),
    )

    # Streaming TTS from the local Kokoro-FastAPI server.
    tts = KokoroTTSService(
        api_key="EMPTY",
        base_url=os.getenv("KOKORO_BASE_URL", "http://localhost:8880/v1"),
        model=os.getenv("KOKORO_MODEL", "kokoro"),
        voice=os.getenv("KOKORO_VOICE", "af_heart"),
    )

    # The NAT router service (local), fanning out to local vLLM endpoints.
    llm = NATVisionLLMService(
        api_key="EMPTY",
        base_url=os.getenv("NAT_BASE_URL", "http://localhost:8001/v1"),
    )
    llm.set_user_id("local")  # vision requests answered by the robot camera

    # Date + timezone context (no clock time: a per-turn timestamp would
    # bust the LLM prefix cache; date-only keeps it stable all day).
    import datetime

    now = datetime.datetime.now().astimezone()
    date_context = (f" Today's date is {now.strftime('%A %d %B %Y')} and the local "
                    f"timezone is {now.tzname()}. Mention these only when relevant.")

    messages = [{"role": "system", "content": PERSONA + date_context}]
    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    mic_gate = MicGateProcessor()

    pipeline = Pipeline(
        [
            transport.input(),
            RobotCameraResponder(),  # answer image requests from the robot camera
            mic_gate,  # panel mute + optional speak-time gating
            stt,
            EmotionReactorProcessor(),  # sentiment on RAW text (before labels)
            SpeakerLabelerProcessor(),  # bake "Speaker N:"/name into finals
            TranscriptTap("user", push_transcript),
            context_aggregator.user(),
            llm,
            tts,
            TranscriptTap("assistant", push_transcript),
            ReachyWobblerProcessor(),
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[TranscriptionLogObserver()],
        # Kiosk: never kill a quiet session. 0 disables.
        idle_timeout_secs=(int(os.getenv("BOT_IDLE_TIMEOUT_SECS", "0")) or None),
    )

    # Hand the panel what it needs to inject typed turns / verbatim speech.
    attach_session(
        loop=asyncio.get_running_loop(),
        task=task,
        messages=messages,
        mic_gate=mic_gate,
    )

    # Greet on startup through the robot speaker. Must be a USER turn: a
    # conversation with only a system message 400s on the chat endpoint
    # ("No user query found"), which made the robot's first words a spoken
    # apology at every launch.
    messages.append({"role": "user",
                     "content": "(startup) You just powered on: greet the user briefly."})
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(run_bot())
