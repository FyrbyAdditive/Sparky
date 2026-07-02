"""Mic gate: mute control and (optional) speech-time gating.

Always provides the panel's mute toggle. With ECHO_MODE=gate it also drops
mic audio while the robot is speaking (+ a short tail) as a fallback when
PipeWire echo cancellation isn't available — sacrificing barge-in for
guaranteed no self-hearing.
"""

import os
import time

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

SPEECH_TAIL_SECS = 0.3


class MicGateProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.muted = False
        self._gate_while_speaking = os.getenv("ECHO_MODE", "aec").strip().lower() == "gate"
        self._bot_speaking = False
        self._bot_stopped_at = 0.0
        if self._gate_while_speaking:
            logger.info("MicGate: ECHO_MODE=gate — mic muted while the robot speaks")

    def set_muted(self, muted: bool):
        self.muted = muted
        logger.info(f"MicGate: {'muted' if muted else 'unmuted'}")

    def _in_speech_window(self) -> bool:
        if self._bot_speaking:
            return True
        return (time.monotonic() - self._bot_stopped_at) < SPEECH_TAIL_SECS

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._bot_stopped_at = time.monotonic()

        if isinstance(frame, InputAudioRawFrame):
            if self.muted or (self._gate_while_speaking and self._in_speech_window()):
                return  # drop mic audio

        await self.push_frame(frame, direction)
