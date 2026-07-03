"""
Custom LLM service that automatically fetches user images for vision queries.

This integrates with the NAT router to automatically capture images when
the router determines a query should go to the vision model.
"""

import asyncio
import base64
import io
from typing import Optional

from loguru import logger
from PIL import Image
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    UserImageRequestFrame,
    UserImageRawFrame,
    InterruptionFrame,
    InputAudioRawFrame,
    UserSpeakingFrame,
    BotSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.llm import NvidiaLLMService


class NATVisionLLMService(NvidiaLLMService):
    """
    Custom NVIDIA LLM service that automatically fetches user images.
    
    When it receives an LLMMessagesFrame, it:
    1. Automatically requests the user's camera image (once per turn)
    2. Waits for the image to arrive
    3. Manually adds the image to the context in OpenAI's multimodal format
    4. Sends the messages + image to the NAT router
    
    The NAT router will then decide whether to use the vision model or not.
    """

    def __init__(self, *args, user_id: Optional[str] = None, max_image_dimension: int = 256, image_quality: int = 40, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(f"NATVisionLLMService: Initialized with max_dimension={max_image_dimension}, quality={image_quality} (streaming disabled for NAT)")
        self._user_id = user_id
        self._pending_image_future: Optional[asyncio.Future] = None
        self._last_image: Optional[UserImageRawFrame] = None
        self._current_turn_has_image = False  # Track if we've fetched image for current turn
        self._max_image_dimension = max_image_dimension
        self._image_quality = image_quality
        self._pending_messages_frame: Optional[Frame] = None  # Store the frame to get context

    def set_user_id(self, user_id: str):
        """Set the user ID for image requests."""
        self._user_id = user_id
        logger.debug(f"NATVisionLLMService: User ID set to {user_id}")

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to stay within max dimension while preserving aspect ratio."""
        width, height = image.size
        max_dim = self._max_image_dimension

        if width <= max_dim and height <= max_dim:
            return image

        if width > height:
            new_width = max_dim
            new_height = int((max_dim / width) * height)
        else:
            new_height = max_dim
            new_width = int((max_dim / height) * width)

        resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        return resized

    def _encode_image_to_base64(self, frame: UserImageRawFrame) -> Optional[str]:
        """
        Encode UserImageRawFrame to base64 JPEG data URL.
        
        Returns:
            Data URL string like "data:image/jpeg;base64,..." or None on error
        """
        try:
            # Convert frame to PIL Image
            image_format = getattr(frame, 'format', 'RGB')
            image_size = getattr(frame, 'size', (640, 360))
            
            image = Image.frombytes(image_format, image_size, frame.image)

            # Resize if needed
            image = self._resize_image(image)

            # Convert to RGB if needed
            if image.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(
                    image,
                    mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None
                )
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')

            # Compress to JPEG
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=self._image_quality)
            image_bytes = buffer.getvalue()

            # Base64 encode
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            return f"data:image/jpeg;base64,{image_b64}"

        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None

    def _scrub_old_images(self, messages):
        """Remove image parts from earlier turns, keeping their text.

        Only the current frame should travel with the request: history images
        grow the prompt every turn and vLLM rejects prompts over its
        images-per-prompt limit (the robot then goes silent).

        Only messages that actually contain an image are rewritten — a
        blanket rewrite of every list-content message is O(history) work per
        turn and would destroy future non-image content types.
        """
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            if not any(isinstance(p, dict) and p.get("type") == "image_url"
                       for p in content):
                continue
            texts = [p.get("text", "") for p in content
                     if isinstance(p, dict) and p.get("type") == "text"]
            msg["content"] = " ".join(t for t in texts if t)

    def _add_image_to_context(self, context, image_data_url: str, user_message: str):
        """
        Manually add image to the last user message in context using OpenAI's multimodal format.
        
        Args:
            context: LLMContext object from the LLMMessagesFrame
            image_data_url: Data URL string like "data:image/jpeg;base64,..."
            user_message: The user's text question about the image
        """
        if not context:
            logger.warning("No context provided, cannot add image")
            return

        messages = context.messages
        if not messages:
            logger.warning("No messages in context, cannot add image")
            return

        # Keep exactly one image in flight: the frame for this turn
        self._scrub_old_images(messages)

        # Find the last user message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                # Convert content to multimodal format
                current_content = messages[i].get("content", "")
                
                # If it's already a list (multimodal), append to it
                if isinstance(current_content, list):
                    messages[i]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": image_data_url, "detail": "auto"}
                    })
                    logger.debug("Appended image to existing multimodal user message")
                else:
                    # Convert string to multimodal format
                    messages[i]["content"] = [
                        {"type": "text", "text": current_content or user_message},
                        {"type": "image_url", "image_url": {"url": image_data_url, "detail": "auto"}}
                    ]
                    logger.debug("Converted user message to multimodal format with image")
                return

        logger.warning("No user message found in context to add image to")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Intercept frames to handle automatic image fetching.
        """
        # Frame-level tracing only at TRACE: this fires dozens of times per
        # turn even after excluding audio/image/speaking frames
        if not isinstance(frame, (InputAudioRawFrame, UserImageRawFrame, UserSpeakingFrame, BotSpeakingFrame)):
            logger.trace(f"NATVisionLLMService.process_frame called with {type(frame).__name__}, direction={direction}")
        
        # Reset turn state on interruption (new user input)
        if isinstance(frame, InterruptionFrame):
            self._current_turn_has_image = False
        
        # Capture incoming images for later use
        if isinstance(frame, UserImageRawFrame):
            self._last_image = frame
            
            # If we're waiting for an image, resolve the future
            if self._pending_image_future and not self._pending_image_future.done():
                logger.debug("NATVisionLLMService: Image captured for pending request")
                self._pending_image_future.set_result(frame)
            
            # Don't pass the raw image frame downstream
            return

        # For LLMContextFrame (not LLMMessagesFrame!), fetch image first (only once per turn)
        if isinstance(frame, LLMContextFrame):
            # Check if there's a user message in the context
            has_user_message = any(msg.get("role") == "user" for msg in frame.context.messages)
            
            if not has_user_message:
                # No user message yet, skip image fetch
                logger.debug("NATVisionLLMService: No user message in context, skipping image fetch")
                await super().process_frame(frame, direction)
                return
            
            if not self._current_turn_has_image:
                logger.debug("NATVisionLLMService: Intercepting LLMMessagesFrame to add image")

                # Fetch the image first
                await self._fetch_and_wait_for_image(frame)

                # Now add it to this frame's context
                if self._last_image:
                    logger.debug("NATVisionLLMService: Encoding and adding image to context")
                    image_data_url = self._encode_image_to_base64(self._last_image)
                    
                    if image_data_url:
                        context = getattr(frame, 'context', None)
                        messages = frame.context.messages if context else []
                        
                        # Extract user message text
                        user_message = ""
                        for msg in reversed(messages):
                            if msg.get("role") == "user":
                                content = msg.get("content", "")
                                user_message = content if isinstance(content, str) else ""
                                break
                        
                        self._add_image_to_context(context, image_data_url, user_message)
                        self._current_turn_has_image = True
                        logger.info("NATVisionLLMService: image attached to this turn")

                        # Per-message context dump only when debugging
                        for i, msg in enumerate(messages):
                            role = msg.get("role", "unknown")
                            content = msg.get("content", "")
                            if isinstance(content, list):
                                logger.debug(f"  Message {i} ({role}): multimodal with {len(content)} items")
                            else:
                                logger.debug(f"  Message {i} ({role}): text only, length {len(content) if content else 0}")
                else:
                    logger.warning("NATVisionLLMService: No image received, sending frame without image")
            else:
                logger.debug("NATVisionLLMService: Already have image this turn, skipping fetch")

        # Continue with normal processing
        await super().process_frame(frame, direction)

    async def _fetch_and_wait_for_image(self, context_frame: LLMContextFrame):
        """
        Request a user image and wait for it to arrive.
        """
        if not self._user_id:
            logger.warning("NATVisionLLMService: No user_id set, cannot fetch image")
            return

        # Extract the user's question from the messages
        user_message = ""
        for msg in reversed(context_frame.context.messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_message = content
                    break

        # Create a future to wait for the image
        self._pending_image_future = asyncio.Future()

        # Request the image (don't use append_to_context since we're adding manually)
        await self.push_frame(
            UserImageRequestFrame(
                user_id=self._user_id,
                text=user_message,
                append_to_context=False,  # We'll add it manually
            ),
            FrameDirection.UPSTREAM,
        )

        # Wait for the image with a timeout
        try:
            await asyncio.wait_for(self._pending_image_future, timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("NATVisionLLMService: Timeout waiting for user image")
        finally:
            self._pending_image_future = None

