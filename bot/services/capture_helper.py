"""Standalone mic-capture helper.

Runs as a child process of the bot and writes raw PCM to stdout. Capture
must live outside the bot process: the bot's GIL contention (100Hz motion
thread, pipeline churn, logging) starves an in-process PortAudio callback
for 150-200ms slices every couple of seconds, punching holes in the audio
that shred ASR finals mid-utterance. A dedicated process captures cleanly
(verified on the same device while the bot ran), and the OS pipe buffers
~2s of 16kHz audio, so bot-side scheduling jitter can no longer lose sound.

Usage: capture_helper.py <device_name_substring> <sample_rate> [channels]
Writes 100ms chunks of s16le PCM to stdout. Exits nonzero on device loss
(the bot's watchdog respawns it).
"""

import sys

import pyaudio


def find_device(pa: pyaudio.PyAudio, name_substring: str) -> int | None:
    name_substring = name_substring.lower()
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if name_substring in str(info.get("name", "")).lower() and info.get("maxInputChannels", 0) > 0:
            return i
    return None


def main() -> int:
    name = sys.argv[1]
    rate = int(sys.argv[2])
    channels = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    pa = pyaudio.PyAudio()
    idx = find_device(pa, name)
    if idx is None:
        print(f"capture_helper: no input device matching {name!r}", file=sys.stderr)
        return 2

    frames = int(rate / 10)  # 100ms
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=rate,
        input=True,
        input_device_index=idx,
        frames_per_buffer=frames,
    )
    out = sys.stdout.buffer
    import time

    chunks = 0
    overflows = 0
    last_hb = time.monotonic()
    try:
        while True:
            try:
                # overflow must raise so real capture loss is COUNTED —
                # this is the only place loss can actually happen
                data = stream.read(frames, exception_on_overflow=True)
            except OSError:
                overflows += 1
                continue
            out.write(data)
            out.flush()
            chunks += 1
            now = time.monotonic()
            if now - last_hb >= 1.0:
                # heartbeat on stderr: parsed by the parent into AUDIO_STATS,
                # and its absence detects helper death within seconds
                print(f"hb {chunks} {overflows}", file=sys.stderr, flush=True)
                last_hb = now
    except BrokenPipeError:
        return 0  # parent went away
    except OSError:
        return 1  # device vanished; parent respawns
    finally:
        try:
            stream.stop_stream()
            stream.close()
            pa.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main() or 0)
