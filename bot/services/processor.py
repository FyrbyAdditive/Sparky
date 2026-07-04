from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    UserStartedSpeakingFrame
)
from .reachy_service import ReachyService

class ReachyWobblerProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.service = ReachyService.get_instance()
        # Attempt connection on initialization
        if not self.service.connected:
            self.service.connect()

        # Track bot speaking state
        self.bot_is_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Track bot speaking state
        if isinstance(frame, BotStartedSpeakingFrame):
            self.bot_is_speaking = True

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.bot_is_speaking = False
            self.service.set_listening_pose()

        elif isinstance(frame, UserStartedSpeakingFrame):
            self.bot_is_speaking = False
            self.service.set_listening_pose()

        # Only feed audio if bot is actively speaking. (This used to md5 every
        # PCM chunk to dedup frames — pipecat does not duplicate frames on
        # this path, and hashing 24kHz audio per-frame is pure CPU waste.)
        elif isinstance(frame, AudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            if self.bot_is_speaking:
                self.service.feed_audio(frame.audio)

        await self.push_frame(frame, direction)
