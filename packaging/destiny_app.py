"""Entry point of the packaged Athena app (``Athena.exe`` / ``Athena.app``).

    Athena                      open the reader in its own window
    Athena --self-test          check that everything works (report in <data>/logs/self-test.*)
    Athena --self-test-quick    the same, skipping the slow parts (used as the build's smoke test)

The app is built without a console window, so output goes to log files in the
data folder (see destiny_bot.desktop.setup_logging).
"""

from __future__ import annotations

import multiprocessing
import os
import sys

# Before anything can import onnxruntime: stop its telemetry to Microsoft (see destiny_bot/__init__.py).
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")


def _port(args: list[str]) -> int:
    for index, arg in enumerate(args):
        value = arg.split("=", 1)[1] if arg.startswith("--port=") else None
        if arg == "--port" and index + 1 < len(args):
            value = args[index + 1]
        if value and value.isdigit() and 0 < int(value) < 65536:
            return int(value)
    return 8765


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()
    args = list(sys.argv[1:] if argv is None else argv)

    from destiny_bot import desktop

    if "--self-test" in args or "--self-test-quick" in args:
        desktop.setup_logging(name="self-test.log")
        desktop.use_bundled_voices()
        from destiny_bot.selftest import run_self_test

        return run_self_test(quick="--self-test-quick" in args or "--quick" in args)
    return desktop.run_app(port=_port(args))


if __name__ == "__main__":
    code = main()
    if getattr(sys, "frozen", False):
        # Closing the window quits Athena even if a background task (an export, a
        # voice warming up) is still running in another thread.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (AttributeError, OSError, ValueError):
                pass
        os._exit(code)
    sys.exit(code)
