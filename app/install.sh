#!/usr/bin/env bash
# One-time setup for the Sparky remote client (macOS or Linux).
# After this, launch via Sparky.app (macOS) / the Sparky desktop entry
# (Linux), or: uv run --project bot python app/launcher.py
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

OS="$(uname -s)"
echo "== Sparky remote client setup ($OS)"

if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  echo "== installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [ "$OS" = "Darwin" ]; then
  command -v brew >/dev/null || { echo "!! Homebrew required (https://brew.sh)"; exit 1; }
  brew list portaudio >/dev/null 2>&1 || brew install portaudio
else
  echo "== Linux system setup (sudo needed once)"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q libportaudio2 portaudio19-dev
  sudo usermod -a -G dialout,video,audio "$USER"
  sudo cp deploy/udev/99-reachy-mini.rules /etc/udev/rules.d/
  sudo udevadm control --reload && sudo udevadm trigger
  echo "   (re-login after first install so group changes apply)"
fi

echo "== syncing Python environments (bot + nat)"
(cd bot && uv python pin 3.13 >/dev/null 2>&1; uv sync)
(cd nat && uv sync)

echo "== installing the launcher"
if [ "$OS" = "Darwin" ]; then
  APP_SRC="$REPO/app/macos/Sparky.app"
  # bake the repo path into the bundle's executable
  sed -e "s|@REPO@|$REPO|" "$REPO/app/macos/sparky-exec.template" > "$APP_SRC/Contents/MacOS/sparky"
  chmod +x "$APP_SRC/Contents/MacOS/sparky"
  ln -sfn "$APP_SRC" /Applications/Sparky.app
  echo "   Sparky.app -> /Applications  (first run: macOS will ask for microphone permission)"
else
  mkdir -p ~/.local/share/applications
  sed -e "s|@REPO@|$REPO|" "$REPO/app/linux/sparky.desktop.template" > ~/.local/share/applications/sparky.desktop
  echo "   Sparky desktop entry installed"
fi

echo
echo "Done. Plug in the Reachy Mini and launch Sparky."
