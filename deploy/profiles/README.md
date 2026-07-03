# Host configuration

The platform is exactly two Sparks plus (optionally) a roaming bot host.
Four files configure everything:

| File | Applies to | Purpose |
|---|---|---|
| `magi.env` | magi (Spark) | `COMPOSE_PROFILES=magi` + router knobs — brings up nemotron-asr, kokoro-tts, vllm-router, tls-proxy |
| `shodan.env` | shodan (Spark) | `COMPOSE_PROFILES=shodan` + agent/wiki knobs — brings up vllm-agent (Qwen3.6-35B) and wiki-offline |
| `magi.bot.env` | magi (bot host) | Bot trio env when the robot lives on magi; copy to the repo root as `.env` (systemd units read it) |
| `remote-client.bot.env` | Mac/Linux | Template consumed by `app/launcher.py` — `@AUDIO_HOST@`/`@LLM_HOST@` filled in by the first-run wizard |

## Spark bring-up (per host)

```bash
cd deploy/stack
cp ../.env.example .env            # once: NGC key etc. (gitignored)
cat .env ../profiles/<host>.env > envfile-merged
docker compose --env-file envfile-merged up -d --build
```

Regenerate `envfile-merged` after any env edit — it goes stale silently.

Model/memory knobs live in the host env files; the measured budgets and the
"engines carry ~30GB host-side overhead" rule are documented in
`deploy/README.md` and `deploy/BENCHMARKS.md`.
