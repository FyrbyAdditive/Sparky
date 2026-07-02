# Sparky — Fully-Offline Reachy Mini Personal Assistant on DGX Spark

A real-time voice + vision AI assistant controlling a **Reachy Mini** robot, running
**entirely on local hardware** — no cloud APIs, no API keys at runtime. Forked from
[brevdev/reachy-personal-assistant](https://github.com/brevdev/reachy-personal-assistant)
with every cloud service replaced by a local equivalent on one or two **NVIDIA DGX Sparks**:

| Capability | Original (cloud) | This fork (local) |
|---|---|---|
| Speech-to-text | ElevenLabs API | Riva Parakeet NIM (streaming, on Spark) |
| Text-to-speech | ElevenLabs API | Kokoro-82M (Kokoro-FastAPI, on Spark) |
| Agent + chat LLM | NVIDIA cloud (`nemotron-3-nano-30b-a3b`) | Same model, vLLM FP8 on Spark |
| Vision LLM | NVIDIA cloud (`nemotron-nano-12b-v2-vl`) | Same model, vLLM FP8 on Spark |
| Intent router | NVIDIA cloud (`phi-3-mini`) | Same model, vLLM on Spark |
| Wikipedia tool | wikipedia.org | Offline txtai semantic index (~9GB) |
| WebRTC transport | Daily (optional cloud) | Local small-webrtc only |

The agent uses an intelligent LLM router to dynamically route between a chat model,
a vision-language model, and a ReAct agent with tools. Model choices are **profile-based
and swappable** (see `deploy/`) — including modern MoE/MTP upgrades and dual-Spark
configurations over the 200GbE ConnectX-7 interconnect.

## Architecture

Three components run in parallel on the *bot host* (a Spark with the robot on USB, or a
Mac/Linux machine on the same LAN):

1. **Reachy Mini Daemon** — controls the robot hardware (or MuJoCo simulation)
2. **Bot Service** (pipecat) — WebRTC UI, VAD/turn-taking (local ONNX), streams STT/TTS,
   drives robot motion (breathing, sway, speech wobble, animations)
3. **NeMo Agent Service** (NAT) — router + ReAct agent, fans out to local vLLM endpoints

All inference services run on the Spark(s) — see [deploy/README.md](deploy/README.md).

## Prerequisites

- [uv](https://github.com/astral-sh/uv) package manager (bot host)
- Python 3.13 (bot), 3.12+ (nat) — uv installs these automatically
- One or two DGX Sparks running the inference stack (`deploy/`), or any
  OpenAI-compatible + Riva endpoints on your LAN
- One-time online setup to download models/containers; **runtime is fully offline**

## Setup

### 1. Bring up the inference stack on the Spark

See [deploy/README.md](deploy/README.md). Verify health checks pass.

### 2. Create the environment file

```bash
cp .env.template .env
# Edit: point RIVA_SERVER, KOKORO_BASE_URL, *_LLM_BASE_URL at your Spark's IP.
# Defaults assume everything runs on this machine.
```

### 3. Install services

```bash
(cd bot && uv sync)
(cd nat && uv sync)
```

## Running the System

Three terminals on the bot host:

### Terminal 1: Reachy Mini Daemon

```bash
cd bot
# macOS simulation:
uv run mjpython -m reachy_mini.daemon.app.main --sim --no-localhost-only
# Linux simulation:
uv run -m reachy_mini.daemon.app.main --sim --no-localhost-only
# Real robot (USB): drop --sim, and set REACHY_USE_SIM=false in .env
```

### Terminal 2: Bot Service

```bash
cd bot
uv run --env-file ../.env python main.py
```

### Terminal 3: NeMo Agent Service

```bash
cd nat
uv run --env-file ../.env nat serve --config_file src/ces_tutorial/config.yml --port 8001
```

Then open the WebRTC UI printed by the bot service (default `http://localhost:7860`),
allow mic/camera, and talk to the robot.

## How It Works

1. **Vision & Audio Input**: the bot captures camera frames and streams mic audio to
   the local Riva Parakeet server for transcription
2. **Agent Processing**: the NAT router selects the model per turn —
   chit-chat → chat LLM · visual queries → vision LLM · actions/knowledge → ReAct agent
   (with the offline Wikipedia tool)
3. **Robot Actions**: responses are spoken through Kokoro TTS while the robot wobbles,
   breathes, and plays expressive animations

## Project Structure

```
├── bot/                    # pipecat bot: WebRTC, speech, robot control
│   ├── main.py             # pipeline wiring (STT → LLM → TTS → robot)
│   ├── nat_vision_llm.py   # vision-aware LLM client for the NAT router
│   └── services/           # robot motion, wobble, animations, daemon handling
├── nat/                    # NeMo Agent Toolkit workflow
│   └── src/ces_tutorial/
│       ├── config.yml      # router/agent config; all LLM endpoints env-driven
│       └── functions/      # router, router_agent, offline wiki tool
├── deploy/                 # DGX Spark inference stack (compose, profiles)
└── .env.template           # all endpoints/settings (no API keys)
```

## Troubleshooting

- **Service health**: run the curl checks in [deploy/README.md](deploy/README.md)
- **Robot connection**: start the daemon before the bot service; the bot retries and
  will run without the robot if unavailable. `REACHY_USE_SIM` must match how the
  daemon was started
- **Port conflicts**: NAT uses 8001; the WebRTC UI uses 7860

## Upstream & Attribution

- Original tutorial: [brevdev/reachy-personal-assistant](https://github.com/brevdev/reachy-personal-assistant)
- Animation assets and deployment patterns: see [THIRD_PARTY.md](THIRD_PARTY.md)
