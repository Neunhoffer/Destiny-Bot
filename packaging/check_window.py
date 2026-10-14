"""CI check of the real app window on Windows.

Starts the installed Athena.exe the way a double-click does, then checks that:
the reader's window ("Athena") appears and WebView2 starts; the server answers;
a second launch opens a second window onto the same reader (no second server);
closing the windows quits the app cleanly and removes its lock file. Saves a
screenshot of the window.

Usage: python packaging/check_window.py "C:\\...\\Athena.exe" [--screenshot athena-window.png]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from ctypes import wintypes
from pathlib import Path

WM_CLOSE = 0x0010


def windows_of(pid: int) -> list[tuple[int, str]]:
    """Visible top-level windows of process ``pid``: (handle, title)."""
    user32 = ctypes.windll.user32
    found: list[tuple[int, str]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid:
            length = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            found.append((hwnd, buf.value))
        return True

    user32.EnumWindows(visit, 0)
    return found


def wait_for(what: str, check, timeout: float):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.5)
    raise SystemExit(f"FAILED: {what} (waited {timeout:.0f} s)")


def read_lock(lock: Path) -> dict | None:
    """The running copy's lock info once its server is up."""
    try:
        info = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return info if isinstance(info, dict) and info.get("url") else None


def page_ok(url: str) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=5) as response:
            return b"destiny-token" in response.read()
    except OSError:
        return False


def screenshot(hwnd: int, out: Path, *, print_base64: bool = False) -> None:
    import base64
    import io

    from PIL import ImageGrab

    rect = wintypes.RECT()
    ctypes.windll.user32.SetForegroundWindow(hwnd)
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
    time.sleep(1.0)
    image = ImageGrab.grab(bbox=(rect.left, rect.top, rect.right, rect.bottom), all_screens=True).convert("RGB")
    image.save(out)
    # How much of the window shows the reader's background colour (light or dark theme)?
    raw = image.resize((160, 110)).tobytes()
    pixels = [raw[i : i + 3] for i in range(0, len(raw), 3)]
    themed = sum(
        1
        for r, g, b in pixels
        if (abs(r - 245) < 12 and abs(g - 243) < 12 and abs(b - 238) < 12)
        or (abs(r - 20) < 12 and abs(g - 23) < 12 and abs(b - 27) < 12)
    )
    print(
        f"Screenshot saved to {out} ({image.width}x{image.height}); {100 * themed / len(pixels):.0f}% reader background"
    )
    if print_base64:  # for reading the screenshot from the job log when artifacts can't be fetched
        small = image.copy()
        small.thumbnail((900, 900))
        buf = io.BytesIO()
        small.save(buf, format="JPEG", quality=70)
        print("SCREENSHOT_JPEG_BASE64 " + base64.b64encode(buf.getvalue()).decode("ascii"))


def close(proc: subprocess.Popen, hwnd: int, what: str) -> None:
    ctypes.windll.user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
    try:
        code = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise SystemExit(f"FAILED: {what} didn't quit after its window was closed") from None
    if code != 0:
        raise SystemExit(f"FAILED: {what} exited with code {code}")
    print(f"{what} quit cleanly after its window was closed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("exe", type=Path)
    parser.add_argument("--screenshot", type=Path, default=Path("athena-window.png"))
    parser.add_argument("--home", type=Path, help="data folder to use (default: a new temp folder)")
    parser.add_argument("--print-screenshot", action="store_true", help="also print it as base64 JPEG")
    args = parser.parse_args()
    if sys.platform != "win32":
        print("This check only runs on Windows.")
        return 0
    home = args.home or Path(tempfile.mkdtemp(prefix="athena-window-"))
    env = {**os.environ, "DESTINY_HOME": str(home)}
    lock = home / "app.lock"

    first = subprocess.Popen([str(args.exe)], env=env, cwd=str(args.exe.parent))
    print(f"Started {args.exe} (pid {first.pid}), data folder {home}")
    try:
        info = wait_for("the reader's server to start", lambda: read_lock(lock), 90)
        url = info["url"]
        if info["pid"] != first.pid:
            raise SystemExit(f"FAILED: the lock belongs to pid {info['pid']}, not {first.pid}")
        if not page_ok(url):
            raise SystemExit(f"FAILED: {url} doesn't serve the reader")
        print(f"Reader server at {url}")
        hwnd, title = wait_for("the Athena window to appear", lambda: windows_of(first.pid), 60)[0]
        print(f"Window {hwnd:#x}: {title!r}")
        if "Athena" not in title:
            raise SystemExit(f"FAILED: the window title is {title!r}")
        wait_for("WebView2 to start", lambda: (home / "webview" / "EBWebView").is_dir(), 60)
        print("WebView2 started (webview/EBWebView exists)")
        time.sleep(6)  # let the reader render
        screenshot(hwnd, args.screenshot, print_base64=args.print_screenshot)

        # Opening Athena again while it runs: a second window onto the same reader.
        time.sleep(max(0.0, 17 - (time.time() - float(info.get("started") or 0))))
        second = subprocess.Popen([str(args.exe)], env=env, cwd=str(args.exe.parent))
        hwnd2, title2 = wait_for("a second window from a second launch", lambda: windows_of(second.pid), 60)[0]
        print(f"Second launch (pid {second.pid}) opened window {title2!r}")
        if (read_lock(lock) or {}).get("pid") != first.pid:
            raise SystemExit("FAILED: the second launch took over the lock (it started its own server)")
        close(second, hwnd2, "The second window")
        if first.poll() is not None:
            raise SystemExit("FAILED: closing the second window quit the first one")

        close(first, hwnd, "Athena")
        if lock.exists():
            raise SystemExit("FAILED: the lock file was left behind")
    finally:
        if first.poll() is None:
            first.kill()
        log = home / "logs" / "app.log"
        if log.is_file():
            print("--- app.log")
            print(log.read_text(encoding="utf-8", errors="replace"))
    print("Window check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
