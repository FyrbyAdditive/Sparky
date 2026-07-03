"""Self-healing local audio transport.

pipecat's LocalAudio transport never recovers a dead PortAudio stream: one
host error ([Errno -9999]) permanently closes the output ([Errno -9988]
"Stream closed" on every subsequent write) and a dead input stream fails
silently — both observed on the Reachy/PipeWire path. These subclasses
reopen streams on failure and watchdog the mic, and export health counters
for the control panel's /status.
"""

import asyncio
import time

from loguru import logger

from pipecat.frames.frames import OutputAudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.local.audio import (
    LocalAudioInputTransport,
    LocalAudioOutputTransport,
    LocalAudioTransport,
    LocalAudioTransportParams,
)

REOPEN_COOLDOWN_SECS = 2.0
MIC_STALL_SECS = 8.0
MIC_CHECK_INTERVAL_SECS = 5.0

# Read by the control panel's /status
AUDIO_STATS = {
    "mic_last_frame_ts": 0.0,
    "mic_reopens": 0,
    "out_reopens": 0,
    "out_write_errors": 0,
}

# Shared gate state, written by MicGateProcessor. The capture callback zeroes
# audio while gated so the robot's own speech never reaches the transport's
# VAD (frame-dropping later in the pipeline is too late — VAD runs in the
# transport and its UserStartedSpeaking would interrupt the reply mid-word).
GATE = {"bot_speaking": False, "tail_until": 0.0, "muted": False}


def gate_active() -> bool:
    return GATE["muted"] or GATE["bot_speaking"] or time.monotonic() < GATE["tail_until"]


class ResilientAudioInput(LocalAudioInputTransport):
    """Input with deep buffering, its own PortAudio instance, and a watchdog.

    The stock 20ms buffers through the pipewire ALSA plugin stalled the
    capture stream every ~10s on the Reachy's 16kHz device; 100ms buffers
    and an unshared PyAudio instance keep it fed.
    """

    def __init__(self, py_audio, params):
        super().__init__(py_audio, params)
        self._watchdog_task = None

    def _audio_in_callback(self, in_data, frame_count, time_info, status):
        AUDIO_STATS["mic_last_frame_ts"] = time.time()
        if gate_active():
            in_data = b"\x00" * len(in_data)
        return super()._audio_in_callback(in_data, frame_count, time_info, status)

    async def start(self, frame: StartFrame):
        # Reimplemented (skipping LocalAudioInputTransport.start) to control
        # frames_per_buffer; grandparent handles the base lifecycle.
        await super(LocalAudioInputTransport, self).start(frame)
        if self._in_stream:
            return
        self._sample_rate = self._params.audio_in_sample_rate or frame.audio_in_sample_rate
        await asyncio.get_running_loop().run_in_executor(None, self._open)
        AUDIO_STATS["mic_last_frame_ts"] = time.time()
        await self.set_transport_ready(frame)
        if self._watchdog_task is None:
            self._watchdog_task = self.create_task(self._watchdog())

    def _open(self):
        num_frames = int(self._sample_rate / 10)  # 100ms buffers
        self._in_stream = self._py_audio.open(
            format=self._py_audio.get_format_from_width(2),
            channels=self._params.audio_in_channels,
            rate=self._sample_rate,
            frames_per_buffer=num_frames,
            stream_callback=self._audio_in_callback,
            input=True,
            input_device_index=self._params.input_device_index,
        )
        self._in_stream.start_stream()

    async def _watchdog(self):
        while True:
            await asyncio.sleep(MIC_CHECK_INTERVAL_SECS)
            age = time.time() - AUDIO_STATS["mic_last_frame_ts"]
            if age > MIC_STALL_SECS:
                logger.error(f"ResilientAudioInput: mic stalled ({age:.0f}s without frames), reopening stream")
                AUDIO_STATS["mic_reopens"] += 1
                try:
                    await asyncio.get_running_loop().run_in_executor(None, self._reopen)
                    AUDIO_STATS["mic_last_frame_ts"] = time.time()
                except Exception as e:
                    logger.error(f"ResilientAudioInput: reopen failed: {e}")

    def _reopen(self):
        old = self._in_stream
        self._in_stream = None
        try:
            if old:
                old.close()
        except Exception:
            pass
        self._open()
        logger.info("ResilientAudioInput: mic stream reopened")


class ResilientAudioOutput(LocalAudioOutputTransport):
    def __init__(self, py_audio, params):
        super().__init__(py_audio, params)
        self._last_reopen = 0.0
        self._last_write = 0.0
        self._silence_task = None

    async def start(self, frame: StartFrame):
        # Reimplemented to deep-buffer the output: the default tiny buffer
        # underruns audibly (rapid stutter) at the start of each utterance.
        await super(LocalAudioOutputTransport, self).start(frame)
        if self._out_stream:
            return
        self._sample_rate = self._params.audio_out_sample_rate or frame.audio_out_sample_rate
        await asyncio.get_running_loop().run_in_executor(None, self._open)
        await self.set_transport_ready(frame)
        if self._silence_task is None:
            # An open-but-idle stream underruns; through the pipewire ALSA
            # plugin that replays stale buffer fragments as periodic noise
            # bursts. Keep the stream fed with silence between utterances.
            self._silence_task = self.create_task(self._silence_feeder())

    async def _silence_feeder(self):
        chunk_secs = 0.08
        silence = b"\x00" * int(2 * chunk_secs * (self._sample_rate or 24000))
        while True:
            await asyncio.sleep(chunk_secs / 2)
            if self._out_stream and (time.monotonic() - self._last_write) > chunk_secs:
                self._last_write = time.monotonic()
                try:
                    await self.get_event_loop().run_in_executor(
                        self._executor, self._out_stream.write, silence
                    )
                except Exception:
                    pass  # real writes handle reopen

    def _open(self):
        self._out_stream = self._py_audio.open(
            format=self._py_audio.get_format_from_width(2),
            channels=self._params.audio_out_channels,
            rate=self._sample_rate,
            frames_per_buffer=int(self._sample_rate / 10),  # 100ms
            output=True,
            output_device_index=self._params.output_device_index,
        )
        self._out_stream.start_stream()

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        self._last_write = time.monotonic()
        try:
            return await super().write_audio_frame(frame)
        except Exception as e:
            AUDIO_STATS["out_write_errors"] += 1
            now = time.monotonic()
            if now - self._last_reopen > REOPEN_COOLDOWN_SECS:
                self._last_reopen = now
                logger.warning(f"ResilientAudioOutput: write failed ({e}), reopening stream")
                AUDIO_STATS["out_reopens"] += 1
                try:
                    await asyncio.get_running_loop().run_in_executor(None, self._reopen)
                    return await super().write_audio_frame(frame)
                except Exception as e2:
                    logger.error(f"ResilientAudioOutput: reopen failed: {e2}")
            return False

    def _reopen(self):
        old = self._out_stream
        self._out_stream = None
        try:
            if old:
                old.close()
        except Exception:
            pass
        self._open()
        logger.info("ResilientAudioOutput: output stream reopened")


class DeepBufferedOutput(LocalAudioOutputTransport):
    """Deep buffers + utterance pre-roll.

    Two distinct start-of-speech stutter causes, both fixed here:
    - tiny device buffers underrun on write jitter -> 100ms frames_per_buffer
    - TTS chunks arrive at synthesis cadence at utterance start while the
      device drains in exact realtime, so every arrival gap is an audible
      underrun until cushion builds -> hold PREROLL_MS of audio before
      starting playback of each utterance (flushed early if TTS pauses).
    """

    def __init__(self, py_audio, params):
        super().__init__(py_audio, params)
        import os

        self._preroll_secs = float(os.getenv("PREROLL_MS", "300")) / 1000.0
        self._pending: list[bytes] = []
        self._pending_bytes = 0
        self._last_frame_at = 0.0
        self._last_device_write = 0.0
        self._flusher_task = None

    async def start(self, frame: StartFrame):
        await super(LocalAudioOutputTransport, self).start(frame)
        if self._out_stream:
            return
        self._sample_rate = self._params.audio_out_sample_rate or frame.audio_out_sample_rate
        self._out_stream = self._py_audio.open(
            format=self._py_audio.get_format_from_width(2),
            channels=self._params.audio_out_channels,
            rate=self._sample_rate,
            frames_per_buffer=int(self._sample_rate / 10),  # 100ms
            output=True,
            output_device_index=self._params.output_device_index,
        )
        self._out_stream.start_stream()
        await self.set_transport_ready(frame)
        if self._flusher_task is None:
            self._flusher_task = self.create_task(self._pending_flusher())

    def _preroll_bytes(self) -> int:
        return int(2 * self._preroll_secs * (self._sample_rate or 24000))

    async def _device_write(self, data: bytes) -> bool:
        if not self._out_stream:
            return False
        self._last_device_write = time.monotonic()
        await self.get_event_loop().run_in_executor(self._executor, self._out_stream.write, data)
        return True

    async def _flush_pending(self) -> bool:
        if not self._pending:
            return True
        data = b"".join(self._pending)
        self._pending.clear()
        self._pending_bytes = 0
        return await self._device_write(data)

    async def _pending_flusher(self):
        # An utterance shorter than the pre-roll (or a synthesis stall) must
        # still play: flush whatever is held once frames stop arriving.
        while True:
            await asyncio.sleep(0.06)
            if self._pending and (time.monotonic() - self._last_frame_at) > 0.15:
                await self._flush_pending()

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        now = time.monotonic()
        new_utterance = (now - self._last_device_write) > 0.5 and not self._pending
        self._last_frame_at = now

        if new_utterance or self._pending:
            self._pending.append(frame.audio)
            self._pending_bytes += len(frame.audio)
            if self._pending_bytes >= self._preroll_bytes():
                return await self._flush_pending()
            return True

        return await self._device_write(frame.audio)


class DeepBufferedLocalAudioTransport(LocalAudioTransport):
    """Stage-2 combination: deep-buffered output + the resilient input.

    The stock blocking input path hangs on the raw Reachy device while
    PipeWire drives the card's output; the callback-driven input with its
    own PortAudio instance and 100ms buffers is the configuration that
    demonstrably delivers frames (and brings the watchdog + telemetry).
    """

    def input(self):
        if not self._input:
            import pyaudio

            self._input = ResilientAudioInput(pyaudio.PyAudio(), self._params)
        return self._input

    def output(self):
        if not self._output:
            self._output = DeepBufferedOutput(self._pyaudio, self._params)
        return self._output


class ResilientLocalAudioTransport(LocalAudioTransport):
    def input(self) -> FrameProcessor:
        if not self._input:
            import pyaudio

            # Own PortAudio instance: sharing one across duplex streams via
            # the pipewire plugin contributed to capture stalls.
            self._input = ResilientAudioInput(pyaudio.PyAudio(), self._params)
        return self._input

    def output(self) -> FrameProcessor:
        if not self._output:
            self._output = ResilientAudioOutput(self._pyaudio, self._params)
        return self._output
