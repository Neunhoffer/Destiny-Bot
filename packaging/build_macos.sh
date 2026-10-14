#!/bin/bash
# Builds Athena for macOS: dist/Athena.app and dist/Athena-<version>.dmg
#
#   bash packaging/build_macos.sh [--no-kokoro] [--skip-installer]
#
# Needs Python 3.10+ (3.12 recommended; python.org or Homebrew). Uses a private build
# environment in packaging/build/venv, downloads the bundled voices, builds Athena.app
# with PyInstaller, runs its self-test, then packs it into a .dmg with hdiutil.
# The app is not code-signed: see packaging/README.md for signing and notarizing.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for candidate in python3.12 python3.13 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
  done
fi
VENV="packaging/build/venv"
if [ ! -x "$VENV/bin/python" ]; then
  echo "Creating the build environment in $VENV"
  "$PYTHON" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -e ".[build]"
"$VENV/bin/python" packaging/build.py "$@"
