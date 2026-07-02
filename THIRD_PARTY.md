# Third-Party Code and Assets

This fork of [brevdev/reachy-personal-assistant](https://github.com/brevdev/reachy-personal-assistant)
(BSD 2-Clause / Apache-2.0 components, see LICENSE) incorporates code and assets from the
following Apache-2.0 licensed projects:

## NVIDIA spark-reachy-photo-booth

https://github.com/NVIDIA/spark-reachy-photo-booth (Apache-2.0)

- `bot/animations/` — Reachy Mini animation clip library (JSON keyframes + audio),
  originally `animation-database-service/assets/animLibrary/`
- Deployment patterns in `deploy/` (Riva Parakeet NIM, TensorRT-LLM, Kokoro serving on
  DGX Spark) adapted from its Docker Compose configuration

## NVIDIA-AI-IOT reachy-mini-jetson-assistant

https://github.com/NVIDIA-AI-IOT/reachy-mini-jetson-assistant (Apache-2.0)

- `bot/services/emotion.py` — ONNX sentiment-based emotion detection, adapted from `app/emotion.py`
- Reachy daemon retry/cleanup logic in `bot/services/reachy_service.py`, adapted from `app/reachy.py`

## Models and containers (downloaded at setup time, run locally)

| Component | Source | License |
|---|---|---|
| NVIDIA-Nemotron-3-Nano-30B-A3B (FP8) | Hugging Face `nvidia/` | NVIDIA Open Model License |
| NVIDIA-Nemotron-Nano-12B-v2-VL (FP8) | Hugging Face `nvidia/` | NVIDIA Open Model License |
| Qwen3.6-35B-A3B (NVFP4) | Hugging Face `Qwen/` + `nvidia/` | Apache-2.0 |
| Parakeet 1.1B CTC en-US ASR NIM | `nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us` | NVIDIA AI Product License |
| Kokoro-82M (Kokoro-FastAPI) | `ghcr.io/remsky/kokoro-fastapi-gpu` | Apache-2.0 |
| txtai-wikipedia index | Hugging Face `NeuML/txtai-wikipedia` | CC-BY-SA (Wikipedia content) |
