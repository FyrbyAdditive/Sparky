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

Remaining candidates: vision decode via NVFP4-QAD variant, NAT step_adaptor
filtering to trim SSE intermediate_data noise.

## magi — unified-1spark: Qwen3.6-35B-A3B NVFP4 + MTP (2026-07-02, night)

Checkpoint: RedHatAI/Qwen3.6-35B-A3B-NVFP4 (compressed-tensors). The
nvidia/ModelOpt NVFP4 checkpoint crashes vLLM 26.05's MTP weight loader
(KeyError on quantized expert scales) — use RedHatAI's on this image.

- **MTP verified active**: drafter loaded ("Detected MTP model", shared
  embeddings) and accepting — mean acceptance length 2.6-3.1 of 3 drafted.
- Decode: ~14 SSE chunks/s x ~3 accepted tokens/chunk ≈ **~43 tok/s
  effective** (bench.py counts chunks, which undercounts under spec-decode).
- TTFT 430ms (vs 177ms parity — multimodal prefill). Warm chitchat first
  chunk via NAT: 1.33s (router role now runs on the same 35B instead of a
  dedicated phi-3; parity was 0.79s).
- All three routes verified on the ONE model: chitchat, vision (described
  a drawn test image correctly), agent + robot tool.
- Memory: single engine at 0.50 fraction frees ~25GB vs parity's three
  engines. Disk note: / hit 100% during the double download (killed NAT's
  tmpdir); freed by pruning docker build cache + the dead checkpoint.

Tuning option: run the phi-3 router container alongside unified
(COMPOSE_PROFILES=unified + start vllm-router) to get routing back to
~100ms and first-chunk under ~1s.
