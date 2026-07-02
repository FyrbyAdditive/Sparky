# Benchmark results

## magi — parity-1spark (2026-07-02)

DGX Spark GB10, all services co-resident, `scripts/bench.py` (3 runs, median).

| role | model | TTFT/TTFA | rate |
|---|---|---|---|
| agent + chitchat | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 (thinking off) | 177ms | 40.9 tok/s |
| router | microsoft/Phi-3-mini-128k-instruct | 100ms | 22.8 tok/s |
| vision | nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8 | 211ms | 12.4 tok/s |
| tts | Kokoro-82M (kokoro-fastapi GPU) | 788ms first audio | 0.79s total/sentence |
| stt | Riva Parakeet 1.1B CTC NIM | streaming (interim results) | — |

Estimated voice-turn latency (speech end → first audio): **~1.2s**.
40.9 tok/s decode is ~10x faster than speech rate — no TTS starvation.

## magi — after optimization round (2026-07-02, late)

| what | before | after |
|---|---|---|
| chitchat: first byte back from NAT | 4.5s (non-streaming) | **0.79s** (token streaming through the router) |
| Kokoro TTS first audio | 788ms/sentence | **~5ms** (server-side stream=true) |
| wiki tool query | 6-14s (9GB index vs page cache) | **0.17s** (slim index, cache-resident) |
| quiet session | dead after ~5 min (idle timeout) | stays alive (BOT_IDLE_TIMEOUT_SECS) |

GPU audit: LLMs, Riva STT and Kokoro TTS all confirmed on GPU; VAD/turn/
emotion deliberately CPU (tiny ONNX models). Perceived voice-turn latency
now dominated by STT finalize + router hop (~1s total to first audio).

Remaining candidates: vision decode via NVFP4-QAD variant, unified-1spark
profile (Qwen3.6-35B-A3B + MTP), NAT step_adaptor filtering to trim SSE
intermediate_data noise.
