# Deploying the Sparky inference stack on DGX Spark

Everything the assistant consumes is an HTTP/gRPC endpoint, so the robot/bot host
(Spark, Mac, or Linux) is interchangeable — it just needs LAN access to these services.

## Prerequisites (Spark A)

- DGX Spark OS with Docker + NVIDIA Container Toolkit (stock DGX Spark image has both)
- One-time online setup:
  - `NGC_API_KEY` — NGC personal key with *NGC Catalog* scope, used once to pull and
    initialize the Riva Parakeet NIM (`nvcr.io/nim/...` requires `docker login nvcr.io`
    with user `$oauthtoken` / password = the key)
  - `HF_TOKEN` — optional, speeds up / gates some Hugging Face model downloads
- After models and containers are cached, the stack runs with no internet access.

## Bring-up

```bash
cd deploy/spark-a
cp ../.env.example .env        # then edit
docker login nvcr.io           # user: $oauthtoken, password: <NGC_API_KEY>
docker compose up -d --build
watch docker compose ps        # wait for all services to report healthy
```

First start downloads ~65GB of models (Nemotron 30B FP8 ≈ 40GB, VL 12B FP8 ≈ 13GB,
phi-3-mini ≈ 8GB, Riva NIM, Kokoro, txtai wiki index ≈ 9GB). Subsequent starts load
from the `hf-cache` / `nim-cache` volumes.

## Memory budget (128GB unified, shared with OS)

| Service | Model | Approx. usage |
|---|---|---|
| vllm-agent | Nemotron-3-Nano-30B-A3B-FP8 | ~51GB (0.40 fraction incl. KV) |
| vllm-vision | Nemotron-Nano-12B-v2-VL-FP8 | ~26GB (0.20) |
| vllm-router | Phi-3-mini-128k-instruct | ~10GB (0.08) |
| riva-stt | Parakeet 1.1B CTC NIM | ~6GB |
| kokoro-tts | Kokoro-82M | ~2GB |
| wiki-offline | txtai index (CPU) | ~10GB RAM |
| **Total** | | **~105GB** — leaves headroom for OS |

Tune `*_GPU_FRACTION` in `.env` if you change models; keep the three vLLM fractions ≤ 0.70 combined.

## Health checks

```bash
SPARK_A=<spark-a-ip>
curl http://$SPARK_A:8010/health                      # vLLM agent
curl http://$SPARK_A:8020/health                      # vLLM vision
curl http://$SPARK_A:8030/health                      # vLLM router
curl http://$SPARK_A:9000/v1/health/ready             # Riva STT (gRPC on :50051)
curl http://$SPARK_A:8880/v1/models                   # Kokoro TTS
curl "http://$SPARK_A:8040/search?q=jupiter&n=1"      # offline wiki

# Smoke-test a completion:
curl http://$SPARK_A:8010/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
  "messages": [{"role": "user", "content": "Say hi in five words."}]}'

# Smoke-test TTS (writes a wav):
curl http://$SPARK_A:8880/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "kokoro", "voice": "af_heart", "input": "Hello from the Spark", "response_format": "wav"}' \
  -o /tmp/tts.wav
```

## Replicating on a new machine (checklist from the magi/shodan bring-up)

1. **Sync the repo**: `rsync -az --exclude .venv --exclude __pycache__ Sparky/ user@spark:~/Sparky/`
2. **NGC auth** (once per machine): `docker login nvcr.io` with user `$oauthtoken`,
   password = NGC API key. Put the same key in `deploy/spark-a/.env`
   (`NGC_API_KEY=...`) — the Riva NIM also needs it at first start to fetch its
   model profile. Set `HF_CACHE_DIR=/home/<user>/.cache/huggingface` there too.
3. **Bring up the stack**: `cd deploy/spark-a && cp .env envfile-merged && cat
   ../profiles/<profile>.env >> envfile-merged && docker compose --env-file
   envfile-merged up -d --build`. First start downloads ~65GB; watch with
   `docker ps` until all services are `(healthy)`.
4. **Bot host prep** (the machine with the robot): `./deploy/bot-host-setup.sh`
   (installs libportaudio2, adds dialout/video/audio groups, installs uv, syncs
   envs). Re-login afterwards.
5. **Env**: `cp deploy/profiles/<profile>.bot.env .env`, set `SPARK_A_HOST`.

Pitfalls these steps encode (all hit on first deploy):
- NIM cache volume: fresh named volumes are root-owned; the NIM runs non-root
  and dies with "manifest download: Permission denied" — the `init-volumes`
  service now handles this automatically.
- Unified memory: vLLM's `--gpu-memory-utilization` is a fraction of the whole
  128GB shared pool; a model's weights must fit *inside* its fraction with room
  for KV cache (phi-3 needed 0.14, not 0.08), and simultaneous engine starts
  race each other's memory measurements — if a service dies at first boot with
  "No available memory for the cache blocks", restart it after the others load.
- Reasoning models: Nemotron-3's chat template defaults `enable_thinking=True`,
  which leaks chain-of-thought into replies; the parity profile passes
  `--default-chat-template-kwargs '{"enable_thinking":false}'` (right for a
  voice assistant).
- The SDK client must not grab the robot camera/mic (bot uses WebRTC media);
  `REACHY_MEDIA_BACKEND=no_media` is the code default.
- `pkill -f reachy` from a remote ssh command kills your own ssh wrapper —
  bracket the pattern (`pkill -f "[r]eachy..."`).

## Notes

- vLLM image default is NVIDIA's Spark-tuned NGC build (`nvcr.io/nvidia/vllm`);
  override `VLLM_IMAGE` in `.env` to use upstream `vllm/vllm-openai` cu130 builds.
- `--mamba_ssm_cache_dtype float32` is required for the hybrid-Mamba Nemotron models;
  check the model card if you swap models.
- Dual-Spark profiles (service split across A/B, and TP=2 over the ConnectX-7 RoCE
  link) live in `deploy/profiles/` — see `interconnect.md` (Phase 3).
