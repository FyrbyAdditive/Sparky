#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
# Local-offline fork: all speech and language services run on local hardware
# (DGX Spark or LAN endpoints) — no cloud APIs, no API keys.


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
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIObserver
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import (
    create_transport,
    get_transport_client_id,
    maybe_capture_participant_camera,
)
from pipecat.services.nvidia.stt import NvidiaSTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams

from nat_vision_llm import NATVisionLLMService
from services.emotion import EmotionReactorProcessor
from services.reachy_service import ReachyService
from services.processor import ReachyWobblerProcessor
from services.robot_api import start_robot_api


load_dotenv(override=True)

# Control endpoint for NAT agent tools (play_animation / look_at)
start_robot_api()


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    # Streaming ASR from a local Riva/NIM server (e.g. Parakeet NIM on a Spark).
    # An empty model name selects the server's default model.
    stt = NvidiaSTTService(
        server=os.getenv("RIVA_SERVER", "localhost:50051"),
        use_ssl=False,
        model_function_map={"function_id": "", "model_name": os.getenv("RIVA_MODEL", "")},
    )

    # OpenAI-compatible TTS from a local Kokoro-FastAPI server.
    tts = OpenAITTSService(
        api_key="EMPTY",
        base_url=os.getenv("KOKORO_BASE_URL", "http://localhost:8880/v1"),
        model=os.getenv("KOKORO_MODEL", "kokoro"),
        voice=os.getenv("KOKORO_VOICE", "af_heart"),
    )

    # The NAT router service (local), which fans out to local vLLM endpoints.
    llm = NATVisionLLMService(
        api_key="EMPTY",
        base_url=os.getenv("NAT_BASE_URL", "http://localhost:8001/v1"),
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points. Respond to what the user said in a creative and helpful way. You are able to describe images from the user camera.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)
    rtvi = RTVIProcessor()

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI protocol processor
            stt,  # STT
            EmotionReactorProcessor(),  # React to user sentiment with animations
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            ReachyWobblerProcessor(),
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi), TranscriptionLogObserver()],
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")

        await maybe_capture_participant_camera(transport, client)

        client_id = get_transport_client_id(transport, client)

        # Set the user_id for automatic image fetching
        llm.set_user_id(client_id)

        # Kick off the conversation.
        messages.append(
            {
                "role": "system",
                "content": f"Say hello!",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
