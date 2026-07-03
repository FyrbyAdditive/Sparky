# Sparky — Fully-Offline Reachy Mini Personal Assistant on DGX Spark

A real-time voice + vision AI assistant controlling a **Reachy Mini** robot, running
**entirely on local hardware** — no cloud APIs, no API keys at runtime. Forked from
[brevdev/reachy-personal-assistant](https://github.com/brevdev/reachy-personal-assistant)
with every cloud service replaced by a local equivalent on **two NVIDIA DGX Sparks**
("magi" = speech + routing, "shodan" = inference), linked by a 200GbE ConnectX-7
interconnect:

| Capability | Original (cloud) | This platform (local) |
|---|---|---|
| Speech-to-text + diarization | ElevenLabs API | Nemotron 3.5 streaming ASR NIM + sortformer speaker diarization (magi) |
| Text-to-speech | ElevenLabs API | Kokoro-82M (Kokoro-FastAPI, magi) |
| Agent / chat / vision LLM | NVIDIA cloud (three models) | One multimodal Qwen3.6-35B-A3B NVFP4 + MTP, vLLM (shodan) |
| Intent router | NVIDIA cloud (`phi-3-mini`) | Same model, vLLM (magi, plus deterministic pre-routes) |
| Wikipedia tool | wikipedia.org | Offline txtai semantic index (~9GB, shodan) |
| Web UI | Daily WebRTC playground | Robot-native audio + local control panel (live camera, transcript, controls) |

The agent routes each turn between chat, vision and a tool-using ReAct agent
(movement, offline Wikipedia, speaker naming). The robot tells speakers apart
and learns their names.

## Architecture

Three components run in parallel on the *bot host* (a Spark with the robot on USB, or a
Mac/Linux machine on the same LAN):

1. **Reachy Mini Daemon** — controls the robot hardware (or MuJoCo simulation for dev)
2. **Bot Service** (pipecat) — robot-native audio, VAD/turn-taking (local ONNX),
   streaming STT/TTS, speaker labeling, control panel, robot motion
3. **NeMo Agent Service** (NAT) — router + ReAct agent, fans out to the Spark endpoints

All inference services run on the two Sparks — see [deploy/README.md](deploy/README.md).

## Prerequisites

- [uv](https://github.com/astral-sh/uv) package manager (bot host)
- Python 3.13 (bot), 3.12+ (nat) — uv installs these automatically
- Two DGX Sparks running the inference stack (`deploy/`)
- One-time online setup to download models/containers; **runtime is fully offline**

## Setup

### 1. Bring up the inference stack on the Spark

See [deploy/README.md](deploy/README.md). Verify health checks pass.

### 2. Create the environment file

```bash
# Robot on magi:      cp deploy/profiles/magi.bot.env .env
# Roaming Mac/Linux:  handled by app/launcher.py (writes ~/.sparky/remote.env)
# Custom setups:      cp .env.template .env and point the URLs at magi/shodan
```

### 3. Install services

```bash
(cd bot && uv sync)
(cd nat && uv sync)
```

## Running the System

On a Spark robot host the trio runs under systemd (`deploy/systemd/`):
`reachy-daemon`, `sparky-nat`, `sparky-bot` — the robot greets on start and
the control panel is at `https://<host>/` (typed chat, transcript,
animations, volume, status). On a remote Mac/Linux machine, use the
launcher app below. Manual equivalent (any host): start the daemon, NAT
and bot exactly as `app/launcher.py` does.

## How It Works

1. **Vision & Audio Input**: the bot captures camera frames and streams mic audio to
   the local Nemotron ASR NIM for transcription and speaker diarization
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
├── deploy/                 # two-Spark inference stack (compose + host envs)
└── .env.template           # all endpoints/settings (no API keys)
```

## Remote client (macOS / Linux)

Plug the robot into any Mac or Linux machine on the same network as the
Sparks — the machine relays robot I/O while the Sparks do all inference:

```bash
./app/install.sh          # one time: deps + envs + click launcher
# then double-click Sparky.app (macOS) or the Sparky desktop entry (Linux)
```

First launch runs a short wizard (which Spark hosts speech / LLM), probes
the endpoints, and offers to pause the Spark-side bot over SSH (restored
on quit). The control panel opens at `http://localhost:7861`. Logs live in
`~/.sparky/`. Notes: macOS asks for microphone permission on first run;
streaming STT prefers wired or strong WiFi.

## Troubleshooting

- **macOS sim: `mjpython` fails with `Library not loaded: libpython3.13.dylib`**
  (uv-managed Pythons don't place the dylib where MuJoCo's launcher looks):
  ```bash
  ln -sf ~/.local/share/uv/python/cpython-3.13*/lib/libpython3.13.dylib bot/.venv/lib/
  ```
- **Service health**: run the curl checks in [deploy/README.md](deploy/README.md)
- **Robot connection**: start the daemon before the bot service; the bot retries and
  will run without the robot if unavailable. `REACHY_USE_SIM` must match how the
  daemon was started
- **Port conflicts**: NAT uses 8001; the panel/robot API uses 7861; daemon 8000

## Upstream & Attribution

- Original tutorial: [brevdev/reachy-personal-assistant](https://github.com/brevdev/reachy-personal-assistant)
- Animation assets and deployment patterns: see [THIRD_PARTY.md](THIRD_PARTY.md)
