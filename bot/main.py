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

from services.local_audio import ResilientLocalAudioTransport

from nat_vision_llm import NATVisionLLMService
from services.emotion import EmotionReactorProcessor
from services.kokoro_tts import KokoroTTSService
from services.mic_gate import MicGateProcessor
from services.reachy_service import ReachyService
from services.processor import ReachyWobblerProcessor
from services.robot_api import start_robot_api, attach_session, push_transcript
from services.robot_camera import RobotCameraResponder
from services.transcript_tap import TranscriptTap


load_dotenv(override=True)

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
    "the surroundings, and you can move: nod, look around, wiggle your antennas."
)


def _find_device_index(pa, name_substring: str, want_input: bool) -> int | None:
    """Resolve a PyAudio device index by name substring; None = default."""
    if not name_substring:
        return None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        channels = info.get("maxInputChannels" if want_input else "maxOutputChannels", 0)
        if channels > 0 and name_substring.lower() in str(info.get("name", "")).lower():
            logger.info(f"Audio {'input' if want_input else 'output'}: [{i}] {info.get('name')}")
            return i
    logger.warning(f"No audio {'input' if want_input else 'output'} device matching "
                   f"'{name_substring}', using default")
    return None


async def run_bot():
    logger.info("Starting Sparky (robot-native audio)")

    import pyaudio

    pa = pyaudio.PyAudio()
    transport = ResilientLocalAudioTransport(
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
            input_device_index=_find_device_index(pa, os.getenv("AUDIO_IN_DEVICE", ""), True),
            output_device_index=_find_device_index(pa, os.getenv("AUDIO_OUT_DEVICE", ""), False),
        )
    )

    # Streaming ASR from the local Riva/Parakeet NIM.
    stt = NvidiaSTTService(
        server=os.getenv("RIVA_SERVER", "localhost:50051"),
        use_ssl=False,
        model_function_map={"function_id": "", "model_name": os.getenv("RIVA_MODEL", "")},
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

    messages = [{"role": "system", "content": PERSONA}]
    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    mic_gate = MicGateProcessor()

    pipeline = Pipeline(
        [
            transport.input(),
            RobotCameraResponder(),  # answer image requests from the robot camera
            mic_gate,  # panel mute + optional speak-time gating
            stt,
            TranscriptTap("user", push_transcript),
            EmotionReactorProcessor(),  # react to user sentiment with animations
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

    # Greet on startup through the robot speaker.
    messages[0]["content"] += " You just powered on: start by greeting the user briefly."
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(run_bot())
