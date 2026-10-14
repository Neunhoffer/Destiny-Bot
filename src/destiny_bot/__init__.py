"""Athena: a private, offline-first study reader that reads textbooks aloud."""

import os

# onnxruntime (used by the Kokoro and Piper voices) phones home to Microsoft as soon
# as it is imported. Only this variable, set before that import, turns it off; its
# disable_telemetry_events() API does not. This runs before any submodule loads.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

__version__ = "0.1.0"
