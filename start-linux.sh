#!/bin/bash
# Destiny Bot launcher for Linux: run ./start-linux.sh
# First run: creates a private Python environment in this folder (.venv), installs
# Destiny Bot, and downloads a natural offline voice (~60 MB). All books, settings
# and audio stay in the "data" folder next to this file.
cd "$(dirname "$0")" || exit 1
export DESTINY_HOME="$PWD/data"
VENV="$PWD/.venv"

find_python() {
  for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

if [ ! -f "$VENV/destiny-installed.txt" ]; then
  echo
  echo "  Setting up Destiny Bot for the first time. This takes a few minutes..."
  echo
  PY="$(find_python)" || {
    echo "  Destiny Bot needs Python 3.10 or newer."
    echo "  Install it with your package manager (for example: sudo apt install python3 python3-venv),"
    echo "  then run this script again."
    read -r -p "  Press Return to close."
    exit 1
  }
  rm -rf "$VENV"
  "$PY" -m venv "$VENV" &&
    "$VENV/bin/python" -m pip install --upgrade pip &&
    "$VENV/bin/python" -m pip install -e ".[ai,edge,ocr,system]" || {
      echo "  Setup didn't finish. Check your internet connection and try again."
      echo "  To start over, delete the .venv folder in this folder."
      rm -rf "$VENV"
      read -r -p "  Press Return to close."
      exit 1
    }
  # The natural offline voices are optional: keep going if this computer can't install them.
  "$VENV/bin/python" -m pip install "piper-tts>=1.3" ||
    echo "  Note: natural offline voices aren't available on this computer; the system voice will be used."
  echo installed > "$VENV/destiny-installed.txt"
fi

if ! ls "$DESTINY_HOME"/voices/piper/*.onnx >/dev/null 2>&1; then
  echo "  Downloading a natural-sounding offline voice, about 60 MB, one time only..."
  "$VENV/bin/python" -m destiny_bot voices download en_US-lessac-medium
fi

exec "$VENV/bin/python" -m destiny_bot serve
