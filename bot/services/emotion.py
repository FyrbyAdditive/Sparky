"""Emotion reactions — lightweight ONNX sentiment classifier driving animations.

Adapted from NVIDIA-AI-IOT/reachy-mini-jetson-assistant (Apache-2.0),
app/emotion.py + app/movements.py: a quantized DistilBERT sentiment model
runs on CPU (~5-15ms) over user transcripts, mapped to richer emotion
categories with text heuristics, then to expressive animation clips with
confidence gating and per-emotion cooldown so the robot reacts while the
LLM is still thinking.

Model files download once into the Hugging Face cache; offline afterwards.
"""

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from .reachy_service import ReachyService

logger = logging.getLogger(__name__)

MODEL_REPO = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

MIN_CONFIDENCE = 0.75
COOLDOWN_SECS = 8.0


class Emotion(Enum):
    HAPPY = "happy"
    SAD = "sad"
    CURIOUS = "curious"
    EXCITED = "excited"
    GREETING = "greeting"
    FAREWELL = "farewell"
    GRATEFUL = "grateful"
    NEUTRAL = "neutral"


# Emotion -> candidate animation clips from bot/animations; one is chosen
# at random per reaction so repeated emotions don't replay one gesture.
# Names that aren't in the loaded library are dropped at reactor startup
# (the Pollen imports are optional), falling back to the always-present
# photo-booth clips listed last.
EMOTION_ANIMATIONS = {
    Emotion.HAPPY: ["cheerful1", "laughing1", "success1", "antennaSmallWiggle"],
    Emotion.EXCITED: ["enthusiastic1", "enthusiastic2", "amazed1", "antennaLargeWiggle"],
    Emotion.SAD: ["sad1", "sad2", "attentive"],  # attentive = sympathetic lean-in
    Emotion.CURIOUS: ["curious1", "inquiring1", "inquiring2", "intrigued5"],
    Emotion.GREETING: ["welcoming1", "welcoming2", "antennaLargeWiggle"],
    Emotion.FAREWELL: ["nod", "yes1"],
    Emotion.GRATEFUL: ["grateful1", "proud1", "nod"],
    # NEUTRAL: no reaction
}


@dataclass
class EmotionResult:
    emotion: Emotion
    confidence: float
    sentiment: str
    sentiment_score: float
    inference_ms: float


_GREETING_RE = re.compile(
    r"\b(hi|hello|hey|howdy|good\s+(morning|afternoon|evening)|what'?s\s+up|yo)\b", re.I
)
_FAREWELL_RE = re.compile(
    r"\b(bye|goodbye|see\s+you|later|good\s*night|take\s+care)\b", re.I
)
_GRATEFUL_RE = re.compile(
    r"\b(thanks?|thank\s+you|appreciate|grateful)\b", re.I
)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class EmotionDetector:
    """CPU ONNX sentiment classifier with heuristic emotion mapping."""

    def __init__(self):
        self._session = None
        self._tokenizer = None
        self._labels = ["NEGATIVE", "POSITIVE"]

    def load(self) -> bool:
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            model_path = hf_hub_download(MODEL_REPO, "onnx/model.onnx")
            tokenizer_path = hf_hub_download(MODEL_REPO, "onnx/tokenizer.json")

            self._session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            self._tokenizer.enable_truncation(max_length=128)
            self._tokenizer.enable_padding(length=128)
            logger.info("Emotion detector loaded")
            return True
        except Exception as e:
            logger.warning(f"Emotion detector unavailable: {e}")
            return False

    @property
    def loaded(self) -> bool:
        return self._session is not None and self._tokenizer is not None

    def detect(self, text: str) -> EmotionResult:
        if not text.strip():
            return EmotionResult(Emotion.NEUTRAL, 0.0, "NEUTRAL", 0.5, 0.0)

        t0 = time.perf_counter()
        sentiment, score = self._classify_sentiment(text)
        emotion, confidence = self._map_emotion(text, sentiment, score)
        dt = (time.perf_counter() - t0) * 1000
        return EmotionResult(emotion, confidence, sentiment, score, dt)

    def _classify_sentiment(self, text: str) -> tuple[str, float]:
        if not self.loaded:
            return "NEUTRAL", 0.5

        encoded = self._tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        outputs = self._session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })

        probs = _softmax(outputs[0][0])
        idx = int(np.argmax(probs))
        return self._labels[idx], float(probs[idx])

    def _map_emotion(self, text: str, sentiment: str, score: float) -> tuple[Emotion, float]:
        """Priority: greeting/farewell/thanks > question > strong sentiment > weak sentiment.

        Questions outrank sentiment (unlike the upstream ordering) because
        SST-2 reads most factual questions as strongly positive, and CURIOUS
        is the right listening reaction for an assistant.
        """
        if _GREETING_RE.search(text):
            return Emotion.GREETING, 0.95
        if _FAREWELL_RE.search(text):
            return Emotion.FAREWELL, 0.95
        if _GRATEFUL_RE.search(text):
            return Emotion.GRATEFUL, 0.90

        if text.rstrip().endswith("?"):
            return Emotion.CURIOUS, 0.80

        if sentiment == "POSITIVE" and score > 0.85:
            return (Emotion.EXCITED, score) if "!" in text else (Emotion.HAPPY, score)
        if sentiment == "NEGATIVE" and score > 0.85:
            return Emotion.SAD, score

        if sentiment == "POSITIVE" and score > 0.6:
            return Emotion.HAPPY, score
        if sentiment == "NEGATIVE" and score > 0.6:
            return Emotion.SAD, score

        return Emotion.NEUTRAL, 0.5


class EmotionReactor:
    """Maps detected emotions to animation clips with gating and cooldown."""

    def __init__(self, service: ReachyService | None = None):
        self.service = service or ReachyService.get_instance()
        self.detector = EmotionDetector()
        self._last_reaction: dict[Emotion, float] = {}
        self._clips_cache: dict[Emotion, list[str]] = {}

    def _available_clips(self, emotion: Emotion) -> list[str]:
        """Candidates that actually exist in the loaded library (cached).
        Keeps the map forward-compatible: unimported Pollen names drop out
        with a log line instead of failing at play time."""
        cached = self._clips_cache.get(emotion)
        if cached is not None:
            return cached
        candidates = EMOTION_ANIMATIONS.get(emotion, [])
        available = [c for c in candidates if self.service.animations.get(c)]
        missing = sorted(set(candidates) - set(available))
        if missing:
            logger.info(f"Emotion {emotion.value}: clips not in library, "
                        f"skipping: {', '.join(missing)}")
        self._clips_cache[emotion] = available
        return available

    def load(self) -> bool:
        return self.detector.load()

    def react_to_text(self, text: str):
        """Classify text and, if warranted, queue a matching animation."""
        result = self.detector.detect(text)
        if result.emotion is Emotion.NEUTRAL or result.confidence < MIN_CONFIDENCE:
            return

        clips = self._available_clips(result.emotion)
        if not clips:
            return
        clip = random.choice(clips)

        now = time.monotonic()
        if now - self._last_reaction.get(result.emotion, 0.0) < COOLDOWN_SECS:
            logger.debug(f"Emotion {result.emotion.value} in cooldown, skipping")
            return
        self._last_reaction[result.emotion] = now

        logger.info(
            f"Emotion: {result.emotion.value} ({result.confidence:.2f}, "
            f"{result.inference_ms:.0f}ms) -> animation '{clip}'"
        )
        self.service.play_animation(clip)


class EmotionReactorProcessor(FrameProcessor):
    """Pipecat processor: reacts to final user transcripts as they flow by.

    Sits after STT in the pipeline. The model preloads in a background thread
    at startup, and detection runs in a worker thread so the (~5-15ms CPU)
    ONNX call never blocks the audio path.
    """

    def __init__(self, reactor: EmotionReactor | None = None):
        super().__init__()
        self.reactor = reactor or EmotionReactor()
        self._loaded = False

        import threading

        def _preload():
            self._loaded = self.reactor.load()

        threading.Thread(target=_preload, daemon=True, name="emotion-preload").start()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._loaded and isinstance(frame, TranscriptionFrame) and frame.text:
            await asyncio.to_thread(self.reactor.react_to_text, frame.text)

        await self.push_frame(frame, direction)
