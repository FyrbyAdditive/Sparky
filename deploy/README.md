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

## Notes

- vLLM image default is NVIDIA's Spark-tuned NGC build (`nvcr.io/nvidia/vllm`);
  override `VLLM_IMAGE` in `.env` to use upstream `vllm/vllm-openai` cu130 builds.
- `--mamba_ssm_cache_dtype float32` is required for the hybrid-Mamba Nemotron models;
  check the model card if you swap models.
- Dual-Spark profiles (service split across A/B, and TP=2 over the ConnectX-7 RoCE
  link) live in `deploy/profiles/` — see `interconnect.md` (Phase 3).
