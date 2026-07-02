"""Mic gate: mute control + robot-speech gating state.

The actual audio silencing happens inside the capture callback
(services/local_audio.py) which zeroes mic audio while gated — that runs
BEFORE the transport's VAD, so the robot's own speech can't trigger
interruptions. This processor just maintains the shared gate state from
the pipeline's speaking events and exposes the panel's mute toggle.
"""

import time

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from .local_audio import GATE

SPEECH_TAIL_SECS = 0.4


class MicGateProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        logger.info("MicGate: robot speech gates the mic at the capture callback")

    @property
    def muted(self) -> bool:
        return GATE["muted"]

    def set_muted(self, muted: bool):
        GATE["muted"] = muted
        logger.info(f"MicGate: {'muted' if muted else 'unmuted'}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            GATE["bot_speaking"] = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            GATE["bot_speaking"] = False
            GATE["tail_until"] = time.monotonic() + SPEECH_TAIL_SECS

        await self.push_frame(frame, direction)
