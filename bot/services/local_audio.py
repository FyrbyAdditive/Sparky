"""Self-healing local audio transport.

pipecat's LocalAudio transport never recovers a dead PortAudio stream: one
host error ([Errno -9999]) permanently closes the output ([Errno -9988]
"Stream closed" on every subsequent write) and a dead input stream fails
silently — both observed on the Reachy/PipeWire path. These subclasses
reopen streams on failure and watchdog the mic, and export health counters
for the control panel's /status.
"""

import asyncio
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from loguru import logger

from pipecat.frames.frames import OutputAudioRawFrame, StartFrame
from pipecat.transports.local.audio import (
    LocalAudioInputTransport,
    LocalAudioOutputTransport,
    LocalAudioTransport,
    LocalAudioTransportParams,
)

REOPEN_COOLDOWN_SECS = 2.0
MIC_STALL_SECS = 8.0
MIC_CHECK_INTERVAL_SECS = 2.0  # heartbeat-based death detection needs tighter checks

# Read by the control panel's /status
AUDIO_STATS = {
    "mic_last_frame_ts": 0.0,
    "mic_reopens": 0,
    "out_reopens": 0,
    "out_write_errors": 0,
    # capture-continuity telemetry. NB in capture-process mode the gap
    # numbers measure bot-side pipe-read jitter (lossless — the pipe
    # buffers); REAL capture loss is the helper-side overflow counter.
    "mic_overflows": 0,       # in-process fallback path only
    "mic_gap_events": 0,      # >150ms between 100ms reads (jitter, not loss)
    "mic_max_gap_ms": 0,
    "mic_helper_chunks": 0,       # helper heartbeat: chunks captured
    "mic_helper_overflows": 0,    # helper heartbeat: REAL lost buffers
    "mic_helper_hb_age_secs": 0,  # computed by /status readers
    "mic_helper_last_hb_ts": 0.0,
}

# Shared gate state, written by MicGateProcessor. The capture callback zeroes
# audio while gated so the robot's own speech never reaches the transport's
# VAD (frame-dropping later in the pipeline is too late — VAD runs in the
# transport and its UserStartedSpeaking would interrupt the reply mid-word).
GATE = {"bot_speaking": False, "tail_until": 0.0, "muted": False}

# Speaker volume as software gain applied to every outgoing sample —
# works identically on macOS and Linux (pactl only existed on the Sparks
# and the panel slider vanished whenever the bot ran on a Mac).
VOLUME = {"percent": max(0, min(120, int(os.getenv("SPEAKER_VOLUME", "100"))))}


def gate_active() -> bool:
    return GATE["muted"] or GATE["bot_speaking"] or time.monotonic() < GATE["tail_until"]


def _apply_gain(data: bytes) -> bytes:
    pct = VOLUME["percent"]
    if pct == 100 or not data:
        return data
    import numpy as np

    samples = np.frombuffer(data, dtype=np.int16).astype(np.int32)
    samples = (samples * pct) // 100
    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()


class ResilientAudioInput(LocalAudioInputTransport):
    """Input from a dedicated capture process, with a stall watchdog.

    Capture cannot share the bot process: GIL contention (100Hz motion
    thread, pipeline churn) starves an in-process PortAudio callback for
    150-200ms every couple of seconds — measured 23 gaps/40s — losing audio
    slices that shred ASR finals mid-utterance. A helper process captures
    cleanly and the pipe absorbs bot-side scheduling jitter losslessly.
    AUDIO_CAPTURE_PROCESS=0 falls back to the old in-process callback.
    (That path keeps its 100ms buffers + unshared PyAudio instance: stock
    20ms buffers through the pipewire ALSA plugin stalled every ~10s.)
    """

    def __init__(self, py_audio, params):
        super().__init__(py_audio, params)
        self._watchdog_task = None
        self._use_capture_process = os.getenv("AUDIO_CAPTURE_PROCESS", "1").strip() != "0"
        self._capture_proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._reader_generation = 0

    # --- shared telemetry + gating, both capture paths ---

    def _note_frame(self, data: bytes) -> bytes:
        now = time.time()
        last = AUDIO_STATS["mic_last_frame_ts"]
        if last > 0:
            gap_ms = int((now - last) * 1000)
            if gap_ms > 150:  # cadence is 100ms; larger means a hole/jitter
                AUDIO_STATS["mic_gap_events"] += 1
                if gap_ms > AUDIO_STATS["mic_max_gap_ms"]:
                    AUDIO_STATS["mic_max_gap_ms"] = gap_ms
        AUDIO_STATS["mic_last_frame_ts"] = now
        if gate_active():
            data = b"\x00" * len(data)
        return data

    def _audio_in_callback(self, in_data, frame_count, time_info, status):
        if status:  # PortAudio overflow/underflow flags
            AUDIO_STATS["mic_overflows"] += 1
        in_data = self._note_frame(in_data)
        return super()._audio_in_callback(in_data, frame_count, time_info, status)

    async def start(self, frame: StartFrame):
        # Reimplemented (skipping LocalAudioInputTransport.start) to control
        # frames_per_buffer; grandparent handles the base lifecycle.
        await super(LocalAudioInputTransport, self).start(frame)
        if self._in_stream or self._capture_proc:
            return
        self._sample_rate = self._params.audio_in_sample_rate or frame.audio_in_sample_rate
        await asyncio.get_running_loop().run_in_executor(None, self._open)
        AUDIO_STATS["mic_last_frame_ts"] = time.time()
        await self.set_transport_ready(frame)
        if self._watchdog_task is None:
            self._watchdog_task = self.create_task(self._watchdog())

    def _open(self):
        if self._use_capture_process:
            try:
                self._open_capture_process()
                return
            except Exception as e:
                logger.error(f"ResilientAudioInput: capture process failed ({e}); "
                             "falling back to in-process capture")
                self._use_capture_process = False
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

    def _open_capture_process(self):
        helper = Path(__file__).resolve().parent / "capture_helper.py"
        device_name = os.getenv("AUDIO_IN_DEVICE", "Reachy Mini").strip().strip('"')
        self._capture_proc = subprocess.Popen(
            [sys.executable, "-u", str(helper), device_name, str(self._sample_rate),
             str(self._params.audio_in_channels)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )
        self._reader_generation += 1
        AUDIO_STATS["mic_helper_last_hb_ts"] = time.time()  # grace period
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(self._capture_proc, self._reader_generation),
            daemon=True, name="mic-capture-reader",
        )
        self._reader_thread.start()
        threading.Thread(
            target=self._stderr_loop,
            args=(self._capture_proc, self._reader_generation),
            daemon=True, name="mic-capture-heartbeat",
        ).start()
        logger.info(f"ResilientAudioInput: capture process started (pid {self._capture_proc.pid}, "
                    f"device '{device_name}', {self._sample_rate}Hz)")

    def _reader_loop(self, proc: subprocess.Popen, generation: int):
        from pipecat.frames.frames import InputAudioRawFrame

        chunk = int(self._sample_rate / 10) * 2 * self._params.audio_in_channels
        loop = self.get_event_loop()
        stdout = proc.stdout
        buf = b""
        while generation == self._reader_generation:
            try:
                data = stdout.read(chunk - len(buf))
            except Exception:
                break
            if not data:
                break  # helper exited; watchdog respawns via stall detection
            buf += data
            if len(buf) < chunk:
                continue
            frame_bytes = self._note_frame(buf)
            buf = b""
            frame = InputAudioRawFrame(
                audio=frame_bytes,
                sample_rate=self._sample_rate,
                num_channels=self._params.audio_in_channels,
            )
            try:
                asyncio.run_coroutine_threadsafe(self.push_audio_frame(frame), loop)
            except RuntimeError:
                break  # loop closed — shutting down
        if generation == self._reader_generation:
            logger.warning("ResilientAudioInput: capture reader ended")

    def _stderr_loop(self, proc: subprocess.Popen, generation: int):
        """Parse helper heartbeats into AUDIO_STATS (real-loss telemetry)."""
        stderr = proc.stderr
        while generation == self._reader_generation:
            try:
                line = stderr.readline()
            except Exception:
                break
            if not line:
                break
            try:
                parts = line.decode(errors="replace").split()
                if parts and parts[0] == "hb" and len(parts) >= 3:
                    AUDIO_STATS["mic_helper_last_hb_ts"] = time.time()
                    AUDIO_STATS["mic_helper_chunks"] = int(parts[1])
                    AUDIO_STATS["mic_helper_overflows"] = int(parts[2])
                else:
                    logger.warning(f"capture helper: {line.decode(errors='replace').rstrip()}")
            except Exception:
                pass

    async def _watchdog(self):
        while True:
            await asyncio.sleep(MIC_CHECK_INTERVAL_SECS)
            # heartbeat check first: detects helper death in seconds rather
            # than waiting out the frame-stall threshold
            if self._use_capture_process and self._capture_proc is not None:
                hb_age = time.time() - AUDIO_STATS["mic_helper_last_hb_ts"]
                if hb_age > 3.5:
                    logger.error(f"ResilientAudioInput: capture helper heartbeat lost "
                                 f"({hb_age:.0f}s), respawning")
                    AUDIO_STATS["mic_reopens"] += 1
                    try:
                        await asyncio.get_running_loop().run_in_executor(None, self._reopen)
                        AUDIO_STATS["mic_last_frame_ts"] = time.time()
                    except Exception as e:
                        logger.error(f"ResilientAudioInput: helper respawn failed: {e}")
                    continue
            age = time.time() - AUDIO_STATS["mic_last_frame_ts"]
            if age > MIC_STALL_SECS:
                logger.error(f"ResilientAudioInput: mic stalled ({age:.0f}s without frames), reopening stream")
                AUDIO_STATS["mic_reopens"] += 1
                try:
                    await asyncio.get_running_loop().run_in_executor(None, self._reopen)
                    AUDIO_STATS["mic_last_frame_ts"] = time.time()
                except Exception as e:
                    logger.error(f"ResilientAudioInput: reopen failed: {e}")

    def _kill_capture_process(self):
        proc = self._capture_proc
        self._capture_proc = None
        self._reader_generation += 1  # detach any live reader thread
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _reopen(self):
        self._kill_capture_process()
        old = self._in_stream
        self._in_stream = None
        try:
            if old:
                old.close()
        except Exception:
            pass
        self._open()
        logger.info("ResilientAudioInput: mic capture reopened")

    async def cleanup(self):
        self._kill_capture_process()
        await super().cleanup()



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
        self._idle_pauser_task = None
        # serializes stream mutation (writes, stop, reopen) across the three
        # coroutines that touch _out_stream; the pending-append path in
        # write_audio_frame deliberately stays lock-free so frames keep
        # buffering during an in-flight flush
        self._stream_lock: asyncio.Lock | None = None

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
        if self._stream_lock is None:
            self._stream_lock = asyncio.Lock()
        if self._flusher_task is None:
            self._flusher_task = self.create_task(self._pending_flusher())
        if self._idle_pauser_task is None:
            self._idle_pauser_task = self.create_task(self._idle_pauser())

    async def cleanup(self):
        # FrameProcessor.cleanup does not cancel arbitrary created tasks;
        # without this, restarts accumulate orphaned flusher/pauser loops
        # poking a reopened stream (observed: 2 idle-pausers after one
        # restart — the pauser handle used to be discarded)
        for attr in ("_flusher_task", "_idle_pauser_task"):
            task = getattr(self, attr)
            if task is not None:
                await self.cancel_task(task)
                setattr(self, attr, None)
        await super().cleanup()

    def _preroll_bytes(self) -> int:
        return int(2 * self._preroll_secs * (self._sample_rate or 24000))

    async def _device_write(self, data: bytes) -> bool:
        if self._stream_lock is None:
            self._stream_lock = asyncio.Lock()
        # one writer at a time: the pending flusher, write_audio_frame and
        # the idle pauser otherwise interleave stream restarts/writes at
        # utterance boundaries (out-of-order audio) and can stop a stream
        # mid-reopen
        async with self._stream_lock:
            if not self._out_stream:
                return False
            data = _apply_gain(data)
            self._last_device_write = time.monotonic()
            try:
                if not self._out_stream.is_active():
                    await self.get_event_loop().run_in_executor(self._executor, self._out_stream.start_stream)
                await self.get_event_loop().run_in_executor(self._executor, self._out_stream.write, data)
                return True
            except Exception as e:
                AUDIO_STATS["out_write_errors"] += 1
                logger.warning(f"DeepBufferedOutput: write failed ({e}), reopening stream")
                AUDIO_STATS["out_reopens"] += 1
                try:
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
                        frames_per_buffer=int(self._sample_rate / 10),
                        output=True,
                        output_device_index=self._params.output_device_index,
                    )
                    self._out_stream.start_stream()
                    await self.get_event_loop().run_in_executor(self._executor, self._out_stream.write, data)
                    return True
                except Exception as e2:
                    logger.error(f"DeepBufferedOutput: reopen failed: {e2}")
                    return False

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

    async def _idle_pauser(self):
        """Stop the stream when no audio has flowed for a while: a stopped
        stream can neither underrun (periodic noise bursts) nor need silence
        injection (which stuttered speech when injected mid-utterance).
        _device_write restarts it, and the pre-roll cushions the resume."""
        while True:
            await asyncio.sleep(0.25)
            if (self._out_stream and self._out_stream.is_active()
                    and not self._pending
                    and (time.monotonic() - self._last_device_write) > 1.0):
                if self._stream_lock is None:
                    continue
                async with self._stream_lock:
                    # re-check under the lock: a write may have restarted
                    # the stream while we waited
                    if not (self._out_stream and self._out_stream.is_active()
                            and not self._pending
                            and (time.monotonic() - self._last_device_write) > 1.0):
                        continue
                    try:
                        await self.get_event_loop().run_in_executor(
                            self._executor, self._out_stream.stop_stream)
                    except Exception:
                        pass


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


