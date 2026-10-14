#!/bin/bash
# Athena launcher for macOS: double-click to start.
# First run: creates a private Python environment in this folder (.venv), installs
# Athena, downloads natural offline voices (~180 MB) and adds an Athena shortcut
# (once). Books, settings and audio are kept privately in your own user folder;
# "Erase all my data" in Settings removes them.
cd "$(dirname "$0")" || exit 1
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
  echo "  Setting up Athena for the first time. This takes a few minutes..."
  echo
  PY="$(find_python)" || {
    echo "  Athena needs Python 3.10 or newer."
    echo "  Download it from https://www.python.org/downloads/ , install it,"
    echo "  then double-click this file again."
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
  # The natural offline voices are optional: keep going if this computer can't install them
  # (they need onnxruntime, which can lag behind the newest Python for a while).
  "$VENV/bin/python" -m pip install "piper-tts>=1.3" ||
    echo "  Note: natural offline voices aren't available on this computer; the system voice will be used."
  "$VENV/bin/python" -m pip install -e ".[kokoro]" ||
    echo "  Note: the Kokoro voices aren't available on this computer."
  # The app window is optional too: without it Athena opens in a browser window.
  "$VENV/bin/python" -m pip install "pywebview>=5.0" || true
  echo installed > "$VENV/destiny-installed.txt"
fi

if [ ! -f "$VENV/destiny-voices.txt" ]; then
  echo "  Downloading natural-sounding offline voices, about 180 MB, one time only..."
  "$VENV/bin/python" -m destiny_bot voices download en_US-lessac-medium &&
    echo done > "$VENV/destiny-voices.txt"
  "$VENV/bin/python" -m destiny_bot voices download kokoro-v1.0
fi

if [ ! -f "$VENV/destiny-shortcut.txt" ]; then
  echo "  Adding an Athena shortcut to your Applications folder and desktop..."
  "$VENV/bin/python" -m destiny_bot shortcut && echo created > "$VENV/destiny-shortcut.txt"
fi

# Athena opens in its own window; this Terminal window can be closed.
nohup "$VENV/bin/python" -m destiny_bot app >/dev/null 2>&1 &
disown
echo "  Athena is opening in its own window. You can close this Terminal window."
