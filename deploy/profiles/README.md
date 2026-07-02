# Model profiles

A profile is two small env files — nothing else changes between model setups:

- `<name>.env` — **Spark-side**: consumed by `docker compose --env-file`.
  Sets `COMPOSE_PROFILES` (which services run) and the model/memory knobs the
  compose files parameterize (`AGENT_MODEL`, `*_GPU_FRACTION`, `*_EXTRA_ARGS`, …).
- `<name>.bot.env` — **bot-host-side**: copied/merged into the repo-root `.env`.
  Points each logical role (agent, chitchat, vision, router, STT, TTS, wiki) at
  the right host:port and model name. `${SPARK_A_HOST}`-style references expand
  via python-dotenv on the bot host.

## Shipped profiles

| Profile | Sparks | Text model | Vision | Notes |
|---|---|---|---|---|
| `parity-1spark` (default) | 1 | Nemotron-3-Nano-30B-A3B-FP8 | Nemotron-Nano-12B-v2-VL-FP8 | Same models as the upstream cloud demo |
| `unified-1spark` | 1 | Qwen3.6-35B-A3B-NVFP4 (MTP) | same model (multimodal) | Fastest; one endpoint serves every role |
| `quality-2spark-split` | 2 | Nemotron-3-Super-120B-A12B-NVFP4 (MTP) on A | 12B-v2-VL on B | Best quality with predictable latency |
| `max-2spark-tp2` | 2 | Qwen3-235B-A22B-FP4, TP=2 over RoCE | 12B-v2-VL on B | Maximum quality; see `deploy/tp2/` + `interconnect.md` |

## Usage

```bash
# Spark A
cd deploy/spark-a && docker compose --env-file ../profiles/parity-1spark.env up -d --build
# Spark B (2-Spark profiles only)
cd deploy/spark-b && docker compose --env-file ../profiles/quality-2spark-split.env up -d --build
# Bot host
cp deploy/profiles/parity-1spark.bot.env .env   # then set SPARK_A_HOST
```

## Adding a profile (e.g. a faster or newer model)

1. Copy the closest `.env` pair under a new name.
2. Spark side: change `AGENT_MODEL`/`VISION_MODEL`/fractions; put model-specific
   vLLM flags (speculative/MTP config, mamba cache dtype, quantization) in
   `*_EXTRA_ARGS`. `COMPOSE_PROFILES` picks which containers run: `parity`
   (3 LLMs + speech), `unified` (1 LLM + speech), `split` (agent LLM only).
3. Bot side: update the `*_LLM_MODEL` names and hosts to match.
4. Benchmark it: `python scripts/bench.py` (see repo README) and record results
   in `deploy/BENCHMARKS.md`.

Candidate models worth trying as they land in the Spark vLLM builds:
`Qwen3.5-122B-A10B` (NVFP4 + MTP single-Spark recipe exists), `GLM-4.7-Flash`,
`Nemotron-3-Nano-Omni-30B-A3B` (audio+vision+text in one model).
