> NOTE: sections before "duo-2spark split live" describe retired layouts
> (single-Spark parity/unified, alt two-Spark strategies) kept as history.
> The platform is now fixed at two Sparks: magi (audio) + shodan (inference).

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

## magi — final unified+router config (2026-07-02, late night)

Qwen3.6-35B-A3B NVFP4 (RedHatAI) + MTP on eugr/spark-vllm-docker's
vllm-node image with recipe tuning (FP8 KV, async sched, prefix caching),
plus the dedicated phi-3 router re-enabled alongside (routing 100ms vs
~600ms on the 35B):

| metric | unified alone | unified + router |
|---|---|---|
| NAT first chunk (warm) | 1.33-1.42s | **0.91-0.94s** |
| effective decode | ~43-45 tok/s (MTP acc 2.5-3.1/3) | same |
| memory free | ~35GB | ~26GB |

Recipe tuning verdict: no single-stream gain (build auto-selects
FLASHINFER_CUTLASS for NVFP4 either way); prefix caching still helps long
conversations. The router split is what moves perceived latency.

## Robot-native audio findings (2026-07-03)

- Speech cutouts root cause 1: pipecat never reopens a dead PortAudio
  stream (one host error -> permanent "Stream closed"). Fixed with the
  self-healing transport (bot/services/local_audio.py).
- Root cause 2: PipeWire's *capture* of the Reachy 16kHz USB device stalls
  every ~10s (any buffering, EC or raw node), while raw ALSA capture never
  failed. Final architecture: mic direct via ALSA/PortAudio, output via
  PipeWire (volume control), ECHO_MODE=gate. PipeWire WebRTC AEC worked in
  principle (barge-in) but rides the unstable capture path — revisit if
  the mic array exposes a hardware-AEC channel or PipeWire fixes the stall.
- Mic health after fix: frame age steady at 0.1s, zero watchdog reopens.
- Panel gains volume slider (pactl on Reachy sink) + mic-flow telemetry.

## duo-2spark split live (2026-07-03 morning)

Audio on magi (robot host), inference on shodan, ConnectX-7 link
(192.168.100.0/24, 70.9 Gbit/s TCP measured; both machines' default route
is WiFi — the link keeps inference off the air).

| metric | all-on-magi | duo split |
|---|---|---|
| agent TTFT (warm, direct) | 430ms | **20-60ms** (prefix cache + dedicated box) |
| effective decode | ~43 tok/s | **47-72 tok/s** |
| NAT chitchat first chunk | 0.91-0.94s | 1.08s (router local to NAT on magi) |
| wiki query (FULL 9GB index) | 6-14s | **0.4-0.6s** (RAM-cached on shodan) |
| magi memory free | ~26GB | **~91GB** (stutter-contention hypothesis test) |

Lessons: engine carries ~30GB host-side overhead beyond its GPU fraction
(shodan OOM'd userspace at fraction 0.65 + simultaneous bring-up; budget
everything to fit boot storms). Router placed WITH NAT on the robot host
(tiny + fastest routing hop). Rollback: flip .env + docker start the
stopped magi containers.

## Final audio architecture (2026-07-03, v0.4-duo-smooth-voice — Tim: "this is good")

Output path (bot/services/local_audio.py DeepBufferedOutput):
- 100ms device buffers (write-jitter tolerance)
- 300ms utterance pre-roll (PREROLL_MS) — absorbs TTS synthesis-cadence gaps
- idle stream PAUSE after 1s (never inject silence: a feeder that filled
  gaps also filled mid-utterance gaps = constant stutter; a stopped stream
  can neither underrun nor stutter)
- reopen-on-failure (PortAudio host errors no longer mute the robot)
Input: ResilientAudioInput — direct hw mic, own PortAudio instance, 100ms
buffers, 8s stall watchdog, capture-callback mute gate.
GPU note: Kokoro synthesis bursts magi's GPU to 96% during speech; the
buffering rides it out. If variable stutter ever returns, move Kokoro to
the inference host (costs ~1ms over the link).
Process lesson: hash-verify every deploy (a --relative rsync silently
left main.py stale; three "staged tests" ran phantom configurations).

## 2026-07-03 — perf batches 1-3 (remote-client Mac + duo split), post speaker-diarization

Measured from the Mac bot host over LAN (magi=ASR/TTS/router, shodan=agent+wiki),
with diarization enabled and the robot session live.

| role | model | TTFT/TTFA | rate |
|---|---|---|---|
| agent | RedHatAI/Qwen3.6-35B-A3B-NVFP4 | 172ms | 41.4 tok/s (usage-corrected under MTP) |
| router | microsoft/Phi-3-mini-128k-instruct | 130ms | 17.1 tok/s |
| tts | kokoro | TTFA 402ms | total 0.43s |

Estimated voice-turn latency (speech end -> first audio): ~0.85s.
Bot idle CPU on the Mac: ~10.8% -> ~8.5% after the motion-loop idle fast path
(secondary-pose short-circuit + listening-idle pose cache + snapshot throttle).
Note: tok/s now counted from usage.completion_tokens (chunk counting undercounts
under MTP), so rates are not directly comparable to pre-July-3 rows.
Movement requests now bypass the router LLM entirely (deterministic pre-route),
removing ~130ms+ from action-turn latency.

## 2026-07-03 — router tier eliminated; shodan memory relief

Routing battery (20 prompts, accept-sets; median full-response latency):
| candidate | score | median |
|---|---|---|
| phi-3-mini (dedicated magi container) | 14/20 | 659ms |
| Nemotron-3-Nano-30B FP8 (dedicated magi container) | 17/20 | 532ms |
| **Qwen3.6-35B NVFP4 on shodan's existing engine** | **19/20** | **303ms (over WiFi)** |

Routing now runs on shodan's engine (prefix caching absorbs the constant
router prompt); the dedicated router container is gone and magi runs no
LLM at all (101GB available). Shodan engine trimmed: fraction 0.38,
seqs 3, batched-tokens 4096 → engine 100GB→80GB, swap zero, and agent
TTFT improved to 150ms / 61.6 tok/s.

## 2026-07-03 — speculative execution + prefix warmup

Chitchat first-chunk from NAT (the dominant turn type): **1.08s → 0.41s
median** (speculative chitchat starts concurrently with the router call;
loser cancelled; prefix caching absorbs repeated prefill). Vision turns
prefetch the camera at speech onset, removing the grab+encode from the
post-speech critical path. Boot warmup pre-caches the persona prefill.

## 2026-07-03 — multilingual ASR: Nemotron 3.5 multi profile + Swedish (Phase A)

ASR switched to the NIM's `type=multi` profile (same 1.2.0 image — the
"incompatible" marking on multi profiles was only NIM_TAGS_SELECTOR
filtering; loaded profile `multi-bs64-26.06.1`, sortformer diarizer
intact): 40 locales, `language_code=auto` = automatic language ID.
The server registers all 40 locale codes plus `auto` on one model;
"multi" is not a valid code and comma lists are rejected.

Direct-stream checks against magi:50051 (bypassing the robot mic):

| input | config | result |
|---|---|---|
| FLEURS sv-SE (real human speech) | auto | near-word-perfect Swedish; LID picks Swedish |
| FLEURS sv-SE (real human speech) | sv-SE pinned | near-word-perfect (broad-coverage tier: ~2 minor word errors/sentence) |
| say -v Daniel (synthetic en-GB) | auto | word-perfect |
| say -v Alva (synthetic sv) | auto | UNUSABLE — LID misfires (Korean/Norwegian fragments); pinned sv-SE often empty. Synthetic macOS Swedish is not a valid test signal; the model is fine on real speech. |

Word boosting at pipecat's default score 4.0 corrupts the multi profile's
decode outright ("Hello" -> "Sparky nodly"); at 1.0 it still fixes
"Sparty" -> "Sparky" and leaves Swedish decodes untouched. New env:
RIVA_BOOST_SCORE (default 1.0), ASR_LANGUAGE (default auto).

Live E2E (roaming rig: robot on the Mac, Mac speakers -> robot mic,
ASR/TTS on magi, LLM on shodan): English regression clean (word-perfect
transcript, ask-name behavior fires); real Swedish audio -> Swedish
transcript, distinct diarization tag, spoken reply IN SWEDISH; typed
"Vad är huvudstaden i Sverige?" -> "Huvudstaden i Sverige är Stockholm.";
"Nicka med huvudet." -> deterministic action route + physical nod.
Kokoro has no Swedish voice, so Swedish replies speak with English
phonology until a Swedish TTS lands (Phase B — on hold pending a
fidelity comparison of options).

Addendum (same night, after live Swedish testing): the model conditions its
OUTPUT language per stream (`target_lang` in the HF card; the NIM maps
`language_code` onto it — verified: adding `custom_configuration
target_lang:auto` changes nothing when language_code=auto, and pinning
en-US + target_lang:auto is worse). Auto mode with diarization+boost still
decodes Swedish as Swedish on long clean utterances, but live short
utterances through the robot mic bias toward English. Fix shipped: ASR
language switch in the panel top bar (Auto / English / Svenska) → POST
/language re-conditions the stream via reconnect. Pinned sv-SE E2E through
speakers→robot mic: correct Swedish transcript, Swedish reply. NB speaker
tags restart at 1 after a language switch (new gRPC stream).
