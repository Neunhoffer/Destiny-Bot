"""``python -m destiny_bot``: the same as the ``destiny`` command."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
