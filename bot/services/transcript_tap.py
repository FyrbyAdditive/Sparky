"""Transcript taps for the control panel.

Two thin processors forward conversation text to a callback as it flows
through the pipeline: user turns (TranscriptionFrame, after STT) and
assistant speech (TTSTextFrame, after TTS). The callback receives
{"role": ..., "text": ...} and must be thread/loop-safe.
"""

from typing import Callable

from pipecat.frames.frames import AudioRawFrame, Frame, TranscriptionFrame, TTSTextFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


class TranscriptTap(FrameProcessor):
    def __init__(self, role: str, callback: Callable[[dict], None]):
        super().__init__()
        self._role = role
        self._callback = callback

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # fast path: audio frames dominate traffic and are never tapped
        if isinstance(frame, AudioRawFrame):
            await self.push_frame(frame, direction)
            return

        if self._role == "user" and isinstance(frame, TranscriptionFrame) and frame.text:
            item = {"role": "user", "text": frame.text}
            # SpeakerLabelerProcessor stores the display label in user_id
            if frame.user_id:
                item["speaker"] = frame.user_id
            self._callback(item)
        elif self._role == "assistant" and isinstance(frame, TTSTextFrame) and frame.text:
            self._callback({"role": "assistant", "text": frame.text})

        await self.push_frame(frame, direction)
