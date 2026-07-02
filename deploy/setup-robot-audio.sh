#!/usr/bin/env bash
# One-time robot-audio setup on the bot host (idempotent).
# Installs the pyaudio build dep, discovers the Reachy Mini audio device,
# installs a PipeWire echo-cancel module targeting it, and makes the
# echo-cancelled nodes the default source/sink for applications.
set -euo pipefail

cd "$(dirname "$0")"

echo "== pyaudio build dependency"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q portaudio19-dev

echo "== discovering Reachy Mini audio nodes"
SINK=$(pactl list short sinks | grep -i reachy | awk '{print $2}' | head -1 || true)
SOURCE=$(pactl list short sources | grep -i reachy | grep -v monitor | awk '{print $2}' | head -1 || true)
if [ -z "$SINK" ] || [ -z "$SOURCE" ]; then
  echo "!! Reachy Mini audio device not found (is the robot plugged in?)"
  pactl list short sinks; pactl list short sources
  exit 1
fi
echo "   sink:   $SINK"
echo "   source: $SOURCE"

echo "== installing echo-cancel config"
mkdir -p ~/.config/pipewire/pipewire.conf.d
sed -e "s|@REACHY_SOURCE@|$SOURCE|" -e "s|@REACHY_SINK@|$SINK|" \
  pipewire/99-sparky-echo-cancel.conf.template \
  > ~/.config/pipewire/pipewire.conf.d/99-sparky-echo-cancel.conf

echo "== restarting pipewire"
systemctl --user restart pipewire pipewire-pulse wireplumber
sleep 3

echo "== setting echo-cancelled nodes as defaults"
EC_SINK_ID=$(wpctl status | grep -i "Sparky Echo-Cancelled Out" | grep -oE "^[ │*]*[0-9]+" | grep -oE "[0-9]+" | head -1 || true)
EC_SOURCE_ID=$(wpctl status | grep -i "Sparky Echo-Cancelled Mic" | grep -oE "^[ │*]*[0-9]+" | grep -oE "[0-9]+" | head -1 || true)
if [ -z "$EC_SINK_ID" ] || [ -z "$EC_SOURCE_ID" ]; then
  echo "!! echo-cancel nodes did not appear; check: pw-cli ls Node | grep -i sparky"
  exit 1
fi
wpctl set-default "$EC_SINK_ID"
wpctl set-default "$EC_SOURCE_ID"
echo "   defaults set (sink $EC_SINK_ID, source $EC_SOURCE_ID)"

echo
echo "Done. Verify: pw-cli ls Node | grep -i sparky_ec"
echo "The bot's pyaudio uses the default device, which now routes through AEC."
