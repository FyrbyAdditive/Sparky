"""Bake diarization speaker labels into user transcripts.

The Nemotron ASR NIM's streaming sortformer tags every word of a final
result with a speaker_tag (0-based, stable for the life of the gRPC
stream). pipecat's aggregator keeps only frame.text, so the label must be
baked into the text here — "Speaker 2: ..." or a registered name — for it
to reach the panel, the router, and every LLM prompt. Names come from the
registry in robot_api (POST /speakers via the panel or the agent's
remember-speaker tool); history is never rewritten, so naming someone only
affects new lines (keeps the LLM prefix cache intact).

Tags are only trusted on finals (interims revise them); a final containing
several speakers is split into one labeled frame per speaker run, the same
way pipecat's Speechmatics service handles multi-speaker segments.
"""

import re

from loguru import logger
from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from .robot_api import get_speaker_name, set_speaker_name

# Deterministic self-introduction fallback: the router LLM cannot be relied
# on to send "my name is X" to the tool-using agent route, so obvious
# introductions register directly here. Requires a capitalized name (the
# ASR capitalizes proper nouns), which keeps "i am hungry" out.
# phrase is case-insensitive by explicit classes; the NAME stays strictly
# capitalized (a blanket IGNORECASE would let "i am hungry" register "hungry").
# ASR punctuation lands mid-phrase ("My name. Is Daniel?"), so allow ., ,
# between the words.
_INTRO_RE = re.compile(
    r"\b(?:[Mm]y name[.,]?\s*(?:[Ii]s|'s)|[Cc]all me|[Ii] am|[Ii]'m)[.,]?\s+([A-Z][a-zA-Z'\-]{1,20})")


class SpeakerLabelerProcessor(FrameProcessor):
    def _label(self, tag: int) -> str:
        return get_speaker_name(tag) or f"Speaker {tag + 1}"

    def _maybe_register_introduction(self, tag: int, text: str):
        if get_speaker_name(tag) is not None:
            return
        m = _INTRO_RE.search(text)
        if m:
            name = m.group(1).strip(".,!?'\"")
            set_speaker_name(tag, name)
            logger.info(f"SpeakerLabeler: Speaker {tag + 1} introduced themselves as {name}")

    def _speaker_runs(self, frame) -> list[tuple[int, list[str]]] | None:
        """Consecutive same-speaker word runs from the Riva result, or None
        when there is no usable word/tag data (diarization off or degraded
        result) — in which case the frame passes through unlabeled."""
        try:
            words = frame.result.alternatives[0].words
        except (AttributeError, IndexError, TypeError):
            return None
        if not words:
            return None
        runs: list[tuple[int, list[str]]] = []
        for w in words:
            tag = getattr(w, "speaker_tag", None)
            word = getattr(w, "word", "")
            if tag is None or not word:
                continue
            if runs and runs[-1][0] == tag:
                runs[-1][1].append(word)
            else:
                runs.append((tag, [word]))
        return runs or None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, TranscriptionFrame) or isinstance(frame, InterimTranscriptionFrame):
            await self.push_frame(frame, direction)
            return

        runs = self._speaker_runs(frame)
        if runs is None:
            await self.push_frame(frame, direction)
            return

        if len(runs) == 1:
            tag = runs[0][0]
            self._maybe_register_introduction(tag, frame.text)
            label = self._label(tag)
            frame.user_id = label
            frame.text = f"{label}: {frame.text}"
            await self.push_frame(frame, direction)
            return

        # several speakers inside one final: one labeled frame per run so
        # the aggregator keeps them distinguishable
        logger.debug(f"SpeakerLabeler: splitting final across {len(runs)} speaker runs")
        for tag, words in runs:
            self._maybe_register_introduction(tag, " ".join(words))
            label = self._label(tag)
            await self.push_frame(
                TranscriptionFrame(
                    text=f"{label}: {' '.join(words)}",
                    user_id=label,
                    timestamp=frame.timestamp,
                    language=frame.language,
                    result=frame.result,
                    finalized=True,
                ),
                direction,
            )
