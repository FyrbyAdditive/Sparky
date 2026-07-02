#!/usr/bin/env bash
# One-time robot-audio setup on the bot host (idempotent).
#
# Final architecture (learned the hard way — see deploy/BENCHMARKS.md):
#   INPUT:  the bot captures the Reachy mic DIRECTLY via ALSA/PortAudio.
#           PipeWire's capture of this 16kHz USB device stalls every ~10s
#           regardless of buffering, while raw ALSA never failed once.
#           The card is therefore set to an output-only PipeWire profile.
#   OUTPUT: through PipeWire (stable; enables pactl volume control).
#   ECHO:   ECHO_MODE=gate (mic dropped while the robot speaks). PipeWire
#           WebRTC AEC was tried and works in principle but rides on the
#           unstable capture path; revisit if barge-in becomes a must.
set -euo pipefail

echo "== dependencies"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q portaudio19-dev pulseaudio-utils

echo "== removing any previous echo-cancel config"
rm -f ~/.config/pipewire/pipewire.conf.d/99-sparky-echo-cancel.conf

echo "== discovering Reachy Mini audio card"
CARD=$(pactl list short cards 2>/dev/null | grep -i reachy | awk '{print $2}' | head -1 || true)
SINK=$(pactl list short sinks | grep -i reachy | awk '{print $2}' | head -1 || true)
if [ -z "$CARD" ]; then
  echo "!! Reachy Mini audio card not found (is the robot plugged in?)"; exit 1
fi
echo "   card: $CARD"

echo "== pipewire: output-only profile (capture belongs to the bot, via raw ALSA)"
pactl set-card-profile "$CARD" output:analog-stereo
systemctl --user restart pipewire pipewire-pulse wireplumber
sleep 3
SINK=$(pactl list short sinks | grep -i reachy | awk '{print $2}' | head -1)
pactl set-default-sink "$SINK"
pactl set-sink-volume "$SINK" 100%
pactl set-sink-mute "$SINK" 0
echo "   default sink: $SINK @100%"

echo
echo "Done. Ensure .env has:"
echo "  AUDIO_IN_DEVICE=Reachy Mini"
echo "  AUDIO_OUT_DEVICE=pipewire"
echo "  ECHO_MODE=gate"
