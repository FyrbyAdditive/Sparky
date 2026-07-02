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

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
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
        self._out_stream = self._py_audio.open(
            format=self._py_audio.get_format_from_width(2),
            channels=self._params.audio_out_channels,
            rate=self._sample_rate,
            output=True,
            output_device_index=self._params.output_device_index,
        )
        self._out_stream.start_stream()
        logger.info("ResilientAudioOutput: output stream reopened")


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
