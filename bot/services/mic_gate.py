"""Mic gate: mute control and (optional) speech-time gating.

Always provides the panel's mute toggle. With ECHO_MODE=gate it also gates
mic audio while the robot is speaking (+ a short tail) as a fallback when
PipeWire echo cancellation isn't available — sacrificing barge-in for
guaranteed no self-hearing.

Gating means SILENCE, never dropped frames: it writes the shared GATE dict
that local_audio's capture callback uses to zero audio before the transport
VAD (no self-interruption), and zeroes any frame that slips through here.
Dropping frames instead starves Riva's streaming ASR sequence — after a
long mute the server expires the sequence ("must specify the START flag")
and the session is permanently deaf from then on.
"""

import os
import time

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from .local_audio import GATE

SPEECH_TAIL_SECS = 0.3
# safety valve: no robot utterance lasts this long — a BotStoppedSpeaking
# that never arrives must not leave the mic gated forever
MAX_SPEECH_GATE_SECS = 60.0


class MicGateProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.muted = False
        self._gate_while_speaking = os.getenv("ECHO_MODE", "aec").strip().lower() == "gate"
        self._bot_speaking = False
        self._bot_started_at = 0.0
        self._bot_stopped_at = 0.0
        if self._gate_while_speaking:
            logger.info("MicGate: ECHO_MODE=gate — mic silenced while the robot speaks")

    def set_muted(self, muted: bool):
        self.muted = muted
        GATE["muted"] = muted
        logger.info(f"MicGate: {'muted' if muted else 'unmuted'}")

    def _set_bot_speaking(self, speaking: bool):
        self._bot_speaking = speaking
        now = time.monotonic()
        if speaking:
            self._bot_started_at = now
            if self._gate_while_speaking:
                GATE["bot_speaking"] = True
        else:
            self._bot_stopped_at = now
            GATE["bot_speaking"] = False
            GATE["tail_until"] = now + SPEECH_TAIL_SECS

    def _in_speech_window(self) -> bool:
        if self._bot_speaking:
            return True
        return (time.monotonic() - self._bot_stopped_at) < SPEECH_TAIL_SECS

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._set_bot_speaking(True)
        elif isinstance(frame, (BotStoppedSpeakingFrame, InterruptionFrame)):
            self._set_bot_speaking(False)

        if isinstance(frame, InputAudioRawFrame):
            if self._bot_speaking and (
                time.monotonic() - self._bot_started_at > MAX_SPEECH_GATE_SECS
            ):
                logger.warning("MicGate: speech gate stuck open >60s — clearing")
                self._set_bot_speaking(False)
            if self.muted or (self._gate_while_speaking and self._in_speech_window()):
                # silence, not a gap — the ASR stream must keep its cadence
                frame.audio = b"\x00" * len(frame.audio)

        await self.push_frame(frame, direction)
