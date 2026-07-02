"""Vision from the robot's own camera.

Upstream used the browser webcam (via WebRTC) as the vision source. With the
bot co-located with the robot, the robot's USB camera is the natural eye:
this processor intercepts the UserImageRequestFrame that NATVisionLLMService
sends upstream and answers it with a frame captured from the robot camera,
so the browser never needs camera permission (VISION_SOURCE=robot).
"""

import asyncio
import os

from loguru import logger

from pipecat.frames.frames import Frame, UserImageRawFrame, UserImageRequestFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


class RobotCameraResponder(FrameProcessor):
    def __init__(self, device_index: int | None = None):
        super().__init__()
        self._device_index = device_index if device_index is not None else int(os.getenv("ROBOT_CAMERA_INDEX", "0"))
        self._capture = None

    def _open(self):
        import cv2

        if self._capture is not None and self._capture.isOpened():
            return True
        self._capture = cv2.VideoCapture(self._device_index)
        if not self._capture.isOpened():
            logger.warning(f"RobotCameraResponder: cannot open camera {self._device_index}")
            self._capture = None
            return False
        return True

    def _grab(self):
        import cv2

        if not self._open():
            return None
        # Drain a couple of stale frames so the answer reflects "now"
        for _ in range(2):
            self._capture.grab()
        ok, frame_bgr = self._capture.read()
        if not ok or frame_bgr is None:
            logger.warning("RobotCameraResponder: capture failed, reopening next time")
            self._capture.release()
            self._capture = None
            return None
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        return frame_rgb.tobytes(), (w, h)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserImageRequestFrame) and direction == FrameDirection.UPSTREAM:
            result = await asyncio.to_thread(self._grab)
            if result is not None:
                image_bytes, size = result
                logger.info(f"RobotCameraResponder: captured {size[0]}x{size[1]} robot camera frame")
                await self.push_frame(
                    UserImageRawFrame(
                        image=image_bytes,
                        size=size,
                        format="RGB",
                        user_id=frame.user_id,
                        text=frame.text,
                        append_to_context=False,
                    ),
                    FrameDirection.DOWNSTREAM,
                )
            else:
                logger.warning("RobotCameraResponder: no frame available for image request")
            # Consume the request either way; don't ask the browser too.
            return

        await self.push_frame(frame, direction)
