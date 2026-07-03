#!/usr/bin/env python3
"""Benchmark the local inference endpoints a profile provides.

Measures, per LLM role: time-to-first-token and decode tok/s (streaming);
for TTS: time-to-first-audio-byte and total synthesis time; for STT
(optional, needs nvidia-riva-client + a 16kHz mono wav): transcription
latency and real-time factor.

Run from the bot env so httpx (and optionally riva) are available:

    cd bot && uv run --env-file ../.env python ../scripts/bench.py [--wav sample.wav] [--md]
"""

import argparse
import json
import os
import statistics
import sys
import time

import httpx

PROMPT = "Give me three fun facts about Jupiter, in complete sentences."
TTS_TEXT = "Hello! I am Sparky, a fully local robot assistant running on a DGX Spark."
RUNS = 3


def bench_llm(name: str, base_url: str, model: str) -> dict | None:
    url = f"{base_url.rstrip('/')}/chat/completions"
    ttfts, rates = [], []
    for _ in range(RUNS):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 256,
            "stream": True,
            # under MTP/spec-decode one SSE chunk carries several tokens, so
            # chunk counting undercounts — ask the server for real usage
            "stream_options": {"include_usage": True},
        }
        chunks = 0
        usage_tokens = None
        t0 = time.perf_counter()
        ttft = None
        try:
            with httpx.stream("POST", url, json=body, timeout=120.0,
                              headers={"Authorization": "Bearer EMPTY"}) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.startswith("data:") or line.strip() == "data: [DONE]":
                        continue
                    chunk = json.loads(line[5:])
                    # the final usage chunk has an EMPTY choices list
                    delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                    if delta.get("content"):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        chunks += 1
                    if chunk.get("usage"):
                        usage_tokens = chunk["usage"].get("completion_tokens")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            return None
        total = time.perf_counter() - t0
        tokens = usage_tokens if usage_tokens else chunks
        if ttft is None or tokens < 2:
            print(f"  {name}: no streamed tokens returned")
            return None
        ttfts.append(ttft)
        rates.append((tokens - 1) / max(total - ttft, 1e-6))
    return {
        "name": name, "model": model,
        "ttft_s": statistics.median(ttfts),
        "tok_s": statistics.median(rates),
    }


def bench_tts(base_url: str, voice: str, model: str) -> dict | None:
    url = f"{base_url.rstrip('/')}/audio/speech"
    firsts, totals = [], []
    for _ in range(RUNS):
        body = {"model": model, "voice": voice, "input": TTS_TEXT, "response_format": "pcm"}
        t0 = time.perf_counter()
        first = None
        n = 0
        try:
            with httpx.stream("POST", url, json=body, timeout=60.0,
                              headers={"Authorization": "Bearer EMPTY"}) as r:
                r.raise_for_status()
                for chunk in r.iter_bytes():
                    if chunk and first is None:
                        first = time.perf_counter() - t0
                    n += len(chunk)
        except Exception as e:
            print(f"  tts: FAILED ({e})")
            return None
        totals.append(time.perf_counter() - t0)
        firsts.append(first or totals[-1])
    return {"name": "tts", "model": model,
            "ttfa_s": statistics.median(firsts),
            "total_s": statistics.median(totals),
            "bytes": n}


def bench_stt(server: str, wav_path: str) -> dict | None:
    """Streaming benchmark against the Nemotron ASR NIM: feed the wav at
    realtime pace and measure speech-end -> final-transcript latency (the
    number that matters for voice-turn feel). The NIM is streaming-only, so
    the old offline_recognize path does not apply."""
    try:
        import riva.client
        import wave
    except ImportError:
        print("  stt: nvidia-riva-client not installed, skipping")
        return None
    try:
        with wave.open(wav_path, "rb") as w:
            rate = w.getframerate()
            pcm = w.readframes(w.getnframes())
        duration = len(pcm) / (2 * rate)
        auth = riva.client.Auth(uri=server, use_ssl=False)
        asr = riva.client.ASRService(auth)
        cfg = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hertz=rate, language_code="en-US",
                max_alternatives=1, enable_automatic_punctuation=True,
                audio_channel_count=1),
            interim_results=True)
        chunk = int(rate * 0.1) * 2  # 100ms
        lats, text = [], ""
        for _ in range(RUNS):
            audio_done_at = None

            def chunks():
                nonlocal audio_done_at
                for i in range(0, len(pcm), chunk):
                    yield pcm[i:i + chunk]
                    time.sleep(0.1)
                audio_done_at = time.perf_counter()
                for _ in range(30):  # trailing silence until the final lands
                    yield b"\x00" * chunk
                    time.sleep(0.1)

            final_at = None
            for resp in asr.streaming_response_generator(
                    audio_chunks=chunks(), streaming_config=cfg):
                for r in resp.results:
                    if r.is_final and r.alternatives:
                        text = r.alternatives[0].transcript
                        final_at = time.perf_counter()
                if final_at:
                    break
            if final_at is None or audio_done_at is None:
                print("  stt: no final transcript returned")
                return None
            lats.append(max(0.0, final_at - audio_done_at))
        lat = statistics.median(lats)
        return {"name": "stt", "latency_s": lat, "rtf_x": duration / max(lat, 1e-6),
                "text": text[:60]}
    except Exception as e:
        print(f"  stt: FAILED ({e})")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", help="16kHz mono wav for the STT benchmark")
    parser.add_argument("--md", action="store_true", help="print a markdown row block for BENCHMARKS.md")
    args = parser.parse_args()

    roles = [
        ("agent", os.getenv("AGENT_LLM_BASE_URL", "http://localhost:8010/v1"),
         os.getenv("AGENT_LLM_MODEL", "RedHatAI/Qwen3.6-35B-A3B-NVFP4")),
        ("chitchat", os.getenv("CHITCHAT_LLM_BASE_URL", "http://localhost:8010/v1"),
         os.getenv("CHITCHAT_LLM_MODEL", "RedHatAI/Qwen3.6-35B-A3B-NVFP4")),
        ("router", os.getenv("ROUTER_LLM_BASE_URL", "http://localhost:8030/v1"),
         os.getenv("ROUTER_LLM_MODEL", "microsoft/Phi-3-mini-128k-instruct")),
        ("vision", os.getenv("VISION_LLM_BASE_URL", "http://localhost:8010/v1"),
         os.getenv("VISION_LLM_MODEL", "RedHatAI/Qwen3.6-35B-A3B-NVFP4")),
    ]
    # skip duplicate endpoints (agent/chitchat/vision share the one shodan engine)
    seen, results = set(), []
    print(f"Benchmarking ({RUNS} runs each, median reported)...")
    for name, base, model in roles:
        key = (base, model)
        if key in seen:
            print(f"  {name}: same endpoint as a previous role, skipping")
            continue
        seen.add(key)
        r = bench_llm(name, base, model)
        if r:
            results.append(r)
            print(f"  {name:9s} TTFT {r['ttft_s']*1000:7.0f}ms   {r['tok_s']:7.1f} tok/s   {model}")

    tts = bench_tts(os.getenv("KOKORO_BASE_URL", "http://localhost:8880/v1"),
                    os.getenv("KOKORO_VOICE", "af_heart"),
                    os.getenv("KOKORO_MODEL", "kokoro"))
    if tts:
        print(f"  {'tts':9s} TTFA {tts['ttfa_s']*1000:7.0f}ms   total {tts['total_s']:.2f}s")

    stt = bench_stt(os.getenv("RIVA_SERVER", "localhost:50051"), args.wav) if args.wav else None
    if stt:
        print(f"  {'stt':9s} {stt['latency_s']*1000:7.0f}ms   {stt['rtf_x']:.0f}x realtime   \"{stt['text']}\"")

    # rough voice-turn estimate: STT finalize + router + agent TTFT + TTS TTFA
    agent = next((r for r in results if r["name"] == "agent"), None)
    router = next((r for r in results if r["name"] == "router"), results[0] if results else None)
    if agent and tts:
        est = (stt["latency_s"] if stt else 0.15) + router["ttft_s"] + agent["ttft_s"] + tts["ttfa_s"]
        print(f"\nEstimated voice-turn latency (speech end -> first audio): ~{est:.2f}s")

    if args.md:
        print("\n| role | model | TTFT/TTFA | rate |")
        print("|---|---|---|---|")
        for r in results:
            print(f"| {r['name']} | {r['model']} | {r['ttft_s']*1000:.0f}ms | {r['tok_s']:.1f} tok/s |")
        if tts:
            print(f"| tts | {tts['model']} | {tts['ttfa_s']*1000:.0f}ms | total {tts['total_s']:.2f}s |")
        if stt:
            print(f"| stt | riva-asr | {stt['latency_s']*1000:.0f}ms | {stt['rtf_x']:.0f}x realtime |")

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
