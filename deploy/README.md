# Deploying the Sparky inference stack — two DGX Sparks

The platform is exactly two Sparks, linked by a direct ConnectX-7 200GbE
cable ([interconnect.md](interconnect.md)):

- **magi** — the audio host and the robot's home: `nemotron-asr` (streaming
  STT + speaker diarization), `kokoro-tts`, `vllm-router` (phi-3 intent
  router, co-located with NAT for the fastest routing hop), `tls-proxy`
  (HTTPS front for the control panel).
- **shodan** — inference: `vllm-agent` (Qwen3.6-35B-A3B NVFP4 with MTP —
  one multimodal engine serving the agent, chitchat and vision roles) and
  `wiki-offline` (full ~9GB txtai Wikipedia index).

The bot trio (reachy daemon + NAT + bot) runs wherever the robot is plugged
in: on magi under systemd (`systemd/`), or on a roaming Mac/Linux machine
via `app/launcher.py`. Everything the bot consumes is an HTTP/gRPC endpoint.

## Prerequisites

- DGX Spark OS with Docker + NVIDIA Container Toolkit (stock image has both)
- One-time online setup: `NGC_API_KEY` (NGC personal key with *NGC Catalog*
  scope; `docker login nvcr.io` with user `$oauthtoken` / password = the key)
  and optionally `HF_TOKEN`. After models are cached, runtime is offline.

## Bring-up (per host)

```bash
cd deploy/stack
cp ../.env.example .env        # once per machine: NGC key, HF cache path
docker login nvcr.io           # user: $oauthtoken, password: <NGC_API_KEY>
cat .env ../profiles/magi.env > envfile-merged     # (shodan.env on shodan)
docker compose --env-file envfile-merged up -d --build
watch docker compose ps        # wait for all services to report healthy
```

Regenerate `envfile-merged` after any env edit — it goes stale silently.
First start downloads the models (Qwen 35B NVFP4 ≈ 20GB, ASR NIM image
37GB, phi-3 ≈ 8GB, Kokoro, wiki index ≈ 9GB); later starts load from cache.

## Memory budget (121GB usable unified memory per Spark)

Measured steady-state (2026-07-03). GPU allocations and process RAM share
the same unified pool; engines carry large host-side overhead on top of
their `--gpu-memory-utilization` fraction.

**magi** (~42GB services, ~75GB free):

| Service | CPU RAM | GPU alloc | Notes |
|---|---|---|---|
| nemotron-asr | ~9GB | ~14GB | Triton + ASR + sortformer diarizer |
| vllm-router | ~2GB | ~15GB | phi-3, fraction 0.14 (0.08 starves the KV cache) |
| kokoro-tts | ~0.6GB | ~1GB | |
| tls-proxy | ~6MB | — | |

**shodan** (~107GB services — tight by design; add nothing here):

| Service | CPU RAM | GPU alloc | Notes |
|---|---|---|---|
| vllm-agent | ~7GB | ~94GB | fraction 0.50 ≈ 60GB + ~34GB CUDA-graph/MTP/multimodal overhead |
| wiki-offline | ~6GB (16GB cap) | — | full index also sits in page cache |

Boot storms restart all containers simultaneously (restart policies ignore
`depends_on`) — everything must fit at once. If an engine dies at first
boot with "No available memory for the cache blocks", restart it after the
others load.

## Health checks

```bash
# magi
curl http://magi:9000/v1/health/ready              # ASR NIM (gRPC on :50051)
curl http://magi:8880/v1/models                    # Kokoro TTS
curl http://magi:8030/health                       # phi-3 router
# shodan
curl http://shodan:8010/health                     # Qwen agent engine
curl "http://shodan:8040/search?q=jupiter&n=1"     # offline wiki

# Smoke-test a completion:
curl http://shodan:8010/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "RedHatAI/Qwen3.6-35B-A3B-NVFP4",
  "messages": [{"role": "user", "content": "Say hi in five words."}]}'
```

## Replicating on fresh machines

1. **Sync the repo**: `rsync -az --exclude .venv --exclude __pycache__ Sparky/ user@spark:~/Sparky/`
   (always hash-verify what you deployed).
2. **NGC auth** (once per machine) + put the key in `deploy/stack/.env`;
   set `HF_CACHE_DIR=/home/<user>/.cache/huggingface` there too.
3. **Bring up each host** as above (magi.env / shodan.env).
4. **Interconnect**: [interconnect.md](interconnect.md) — the bot reaches
   shodan via `SPARK_B_IB=192.168.100.2` (`profiles/magi.bot.env`).
5. **Bot host prep** (machine with the robot): `./deploy/bot-host-setup.sh`
   on Linux, or `app/install.sh` for the roaming client. Robot on magi:
   `cp deploy/profiles/magi.bot.env .env` and install `systemd/`.

Pitfalls these steps encode (all hit on first deploy):
- NIM cache volume: fresh named volumes are root-owned; the NIM dies with
  "manifest download: Permission denied" — `init-volumes` handles this.
- Unified memory: a model's weights must fit *inside* its fraction with room
  for KV cache; simultaneous engine starts race each other's memory
  measurements.
- The SDK client must not grab the robot camera/mic;
  `REACHY_MEDIA_BACKEND=no_media` is the code default.
- `pkill -f reachy` from a remote ssh command kills your own ssh wrapper —
  bracket the pattern (`pkill -f "[r]eachy..."`).

## Notes

- `VLLM_IMAGE` default is NVIDIA's NGC build; shodan runs the community
  Spark build `vllm-node:latest` (set in `shodan.env`) for MTP support.
- Benchmarks and tuning history: [BENCHMARKS.md](BENCHMARKS.md).
