#!/usr/bin/env bash
# Install the Sparky user services (daemon, NAT, bot) with auto-restart.
# Run on the bot host. Requires sudo once for login lingering (services
# keep running without an active SSH session).
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p ~/.config/systemd/user
cp reachy-daemon.service sparky-nat.service sparky-bot.service ~/.config/systemd/user/

sudo loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemctl --user enable --now reachy-daemon sparky-nat sparky-bot

echo "Installed. Status:"
systemctl --user --no-pager status reachy-daemon sparky-nat sparky-bot | grep -E "●|Active:"
echo
echo "Logs: journalctl --user -u sparky-bot -f"
