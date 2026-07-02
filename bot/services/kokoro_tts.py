"""Kokoro-FastAPI TTS with true server-side streaming.

pipecat's OpenAITTSService streams the HTTP response, but Kokoro-FastAPI
synthesizes the whole utterance before sending unless the request carries
``stream: true`` — costing ~0.8s of silence per sentence. This subclass
injects that flag (chunks arrive as they're generated, first audio in
~200-300ms) and registers the configured Kokoro voice with pipecat's
OpenAI voice validation.
"""

from typing import AsyncGenerator

from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.openai import tts as openai_tts
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.utils.tracing.service_decorators import traced_tts


class KokoroTTSService(OpenAITTSService):
    def __init__(self, *, voice: str, **kwargs):
        # Let the non-OpenAI voice name pass client-side validation untouched.
        openai_tts.VALID_VOICES[voice] = voice
        super().__init__(voice=voice, **kwargs)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating streaming TTS [{text}]")
        try:
            create_params = {
                "input": text,
                "model": self._settings.model,
                "voice": self._settings.voice,
                "response_format": "pcm",
                "extra_body": {"stream": True},
            }
            if self._settings.speed:
                create_params["speed"] = self._settings.speed

            async with self._client.audio.speech.with_streaming_response.create(
                **create_params
            ) as r:
                if r.status_code != 200:
                    error = await r.text()
                    logger.error(f"{self} error getting audio (status: {r.status_code}, error: {error})")
                    yield ErrorFrame(error=f"Error getting audio (status: {r.status_code}, error: {error})")
                    return

                await self.start_tts_usage_metrics(text)

                async for chunk in r.iter_bytes(self.chunk_size):
                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(chunk, self.sample_rate, 1, context_id=context_id)
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"Kokoro TTS error: {e}")
