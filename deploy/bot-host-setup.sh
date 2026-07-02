#!/usr/bin/env bash
# One-time setup for a machine that runs the robot daemon + bot (Linux).
# Discovered the hard way on a DGX Spark — see deploy/README.md "Replicating
# on a new machine". Requires sudo. Re-login (or reconnect SSH) afterwards
# so the new group memberships apply.
set -euo pipefail

echo "== Installing system libraries (PortAudio for the robot's audio stack)"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q libportaudio2

echo "== Adding $USER to hardware access groups"
sudo usermod -a -G dialout "$USER"   # robot motor serial (/dev/ttyACM*)
sudo usermod -a -G video "$USER"     # robot camera (/dev/video*)
sudo usermod -a -G audio "$USER"     # robot speaker/mic

echo "== Installing Reachy Mini udev rules (device access for systemd services)"
sudo cp "$(dirname "$0")/udev/99-reachy-mini.rules" /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger

echo "== Installing uv (Python manager) if missing"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

echo "== Syncing bot and nat environments"
cd "$(dirname "$0")/.."
(cd bot && ~/.local/bin/uv python pin 3.13 && ~/.local/bin/uv sync)
(cd nat && ~/.local/bin/uv sync)

echo
echo "Done. Re-login for group changes, then:"
echo "  1. start the robot daemon:  cd bot && uv run -m reachy_mini.daemon.app.main --no-localhost-only"
echo "     (add --sim for simulation; on macOS use mjpython for sim)"
echo "  2. cp deploy/profiles/<profile>.bot.env .env   # set SPARK_A_HOST"
echo "  3. start NAT:  cd nat && uv run --env-file ../.env nat serve --config_file src/ces_tutorial/config.yml --port 8001"
echo "  4. start bot:  cd bot && uv run --env-file ../.env python main.py"
