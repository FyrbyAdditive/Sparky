"""Vision from the robot's own camera.

Upstream used the browser webcam (via WebRTC) as the vision source. With the
bot co-located with the robot, the robot's USB camera is the natural eye:
this processor intercepts the UserImageRequestFrame that NATVisionLLMService
sends upstream and answers it with a frame captured from the robot camera,
so the browser never needs camera permission (VISION_SOURCE=robot).

The camera itself is owned by services/camera_service.py (shared with the
panel's live MJPEG stream behind one lock); this processor only turns a
grab into a pipecat frame.
"""

import asyncio

from loguru import logger

from pipecat.frames.frames import Frame, UserImageRawFrame, UserImageRequestFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from . import camera_service

# Back-compat re-exports (robot_api and env docs referenced these here)
CAMERA = camera_service.CAMERA
CAMERA_RESOLUTIONS = camera_service.CAMERA_RESOLUTIONS


class RobotCameraResponder(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserImageRequestFrame) and direction == FrameDirection.UPSTREAM:
            result = await asyncio.to_thread(camera_service.grab_rgb)
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
