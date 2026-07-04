"""Moves head given audio samples."""

import time
import queue
import logging
import threading
from typing import Tuple
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .speech_tapper import HOP_MS, SwayRollRT


SAMPLE_RATE = 24000
# Downsample to reduce load on simulator
DOWNSAMPLE_RATE = 16000
# Limit queue to prevent overwhelming simulator
MAX_QUEUE_SIZE = 50
# seconds between audio and robot movement
MOVEMENT_LATENCY_S = 0.08
logger = logging.getLogger(__name__)


class HeadWobbler:
    """Converts raw s16 TTS audio into head movement offsets."""

    def __init__(
        self,
        set_speech_offsets: Callable[
            [Tuple[float, float, float, float, float, float]], None
        ]
    ) -> None:
        """Initialize the head wobbler."""
        self._apply_offsets = set_speech_offsets
        self._base_ts: float | None = None
        self._hops_done: int = 0
        self._was_swaying: bool = False

        self.audio_queue: (
            "queue.Queue[Tuple[int, int, NDArray[np.int16]]]"
        ) = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.sway = SwayRollRT()

        # Synchronization primitives
        self._state_lock = threading.Lock()
        self._sway_lock = threading.Lock()
        self._generation = 0

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Track dropped frames for monitoring
        self._dropped_chunks = 0

    def feed(self, pcm: bytes) -> None:
        """Thread-safe: push raw s16 PCM into the consumer queue.

        Takes bytes directly — the old path base64-encoded every TTS chunk
        on the pipeline thread only to decode it again here. Downsampling
        is a real linear resample now: the previous integer decimation was
        a silent no-op (24000 // 16000 == 1), which fed the sway engine
        1.5x the intended samples at a mislabeled rate and stretched its
        oscillator timing.
        """
        samples = np.frombuffer(pcm, dtype=np.int16)

        if SAMPLE_RATE != DOWNSAMPLE_RATE and len(samples) > 1:
            n_out = int(len(samples) * DOWNSAMPLE_RATE / SAMPLE_RATE)
            x_old = np.linspace(0.0, 1.0, len(samples), endpoint=False)
            x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
            samples = np.interp(x_new, x_old, samples).astype(np.int16)
        buf = samples.reshape(1, -1)

        with self._state_lock:
            generation = self._generation

        # Try to add to queue, but don't block if full (drop oldest/skip)
        try:
            self.audio_queue.put_nowait(
                (generation, DOWNSAMPLE_RATE, buf)
            )
        except queue.Full:
            # Queue is full - simulator is overwhelmed
            # Drop this chunk to prevent freezing
            self._dropped_chunks += 1
            if self._dropped_chunks % 10 == 1:  # Log every 10th drop
                logger.warning(
                    "Audio queue full, dropped %d chunks total",
                    self._dropped_chunks
                )

    def start(self) -> None:
        """Start the head wobbler loop in a thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()
        logger.debug("Head wobbler started")

    def stop(self) -> None:
        """Stop the head wobbler loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        logger.debug("Head wobbler stopped")

    def working_loop(self) -> None:
        """Convert audio deltas into head movement offsets."""
        hop_dt = HOP_MS / 1000.0

        logger.debug("Head wobbler thread started")
        while not self._stop_event.is_set():
            queue_ref = self.audio_queue
            try:
                chunk_generation, sr, chunk = queue_ref.get_nowait()
            except queue.Empty:
                # Speech has drained: return the head to neutral once.
                # Nothing else ever zeroes the sway offsets, so without this
                # the head holds the last mid-sway offset forever (and the
                # movement loop keeps composing a non-zero offset every tick).
                if self._was_swaying:
                    self._was_swaying = False
                    self._apply_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                # avoid while to never exit
                time.sleep(MOVEMENT_LATENCY_S)
                continue

            self._was_swaying = True

            try:
                with self._state_lock:
                    current_generation = self._generation
                if chunk_generation != current_generation:
                    continue

                if self._base_ts is None:
                    with self._state_lock:
                        if self._base_ts is None:
                            self._base_ts = time.monotonic()

                pcm = np.asarray(chunk).squeeze(0)
                with self._sway_lock:
                    results = self.sway.feed(pcm, sr)

                i = 0
                while i < len(results):
                    with self._state_lock:
                        if self._generation != current_generation:
                            break
                        base_ts = self._base_ts
                        hops_done = self._hops_done

                    if base_ts is None:
                        base_ts = time.monotonic()
                        with self._state_lock:
                            if self._base_ts is None:
                                self._base_ts = base_ts
                                hops_done = self._hops_done

                    target = base_ts + MOVEMENT_LATENCY_S + hops_done * hop_dt
                    now = time.monotonic()

                    if now - target >= hop_dt:
                        lag_hops = int((now - target) / hop_dt)
                        drop = min(lag_hops, len(results) - i - 1)
                        if drop > 0:
                            with self._state_lock:
                                self._hops_done += drop
                                hops_done = self._hops_done
                            i += drop
                            continue

                    if target > now:
                        time.sleep(target - now)
                        with self._state_lock:
                            if self._generation != current_generation:
                                break

                    r = results[i]
                    offsets = (
                        r["x_mm"] / 1000.0,
                        r["y_mm"] / 1000.0,
                        r["z_mm"] / 1000.0,
                        r["roll_rad"],
                        r["pitch_rad"],
                        r["yaw_rad"],
                    )

                    with self._state_lock:
                        if self._generation != current_generation:
                            break

                    self._apply_offsets(offsets)

                    with self._state_lock:
                        self._hops_done += 1
                    i += 1
            finally:
                queue_ref.task_done()
        logger.debug("Head wobbler thread exited")
