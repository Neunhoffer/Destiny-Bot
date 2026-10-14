"""Athena as a desktop app: the reader in its own window, plus desktop shortcuts.

:func:`run_app` starts the local server in a background thread and shows the
reader in a native window (pywebview: WebView2 on Windows, WebKit on macOS, GTK
or Qt on Linux). Closing the window stops the server and quits. When a native
window isn't possible it falls back to Microsoft Edge or Google Chrome in "app
mode" (a window with no tabs or address bar), and finally to the default browser.

Only one copy runs at a time: a lock file in the data folder records the running
copy's process id and address, and starting Athena again opens a window onto
that copy instead of starting a second server.

:func:`create_shortcuts` adds Desktop / Start menu (Windows), ~/Applications
(macOS) or application-menu (Linux) shortcuts for people who run Athena from a
source folder with the start-* launcher scripts. The Windows installer makes its
own shortcuts.
"""

from __future__ import annotations

import faulthandler
import inspect
import io
import json
import logging
import os
import plistlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import IO, Any, Callable

from . import __version__, config

log = logging.getLogger(__name__)

APP_NAME = "Athena"
APP_ID = "Athena.Reader"  # Windows AppUserModelID; the installer gives its shortcuts the same id
APP_MUTEX = "AthenaReaderAppMutex"  # the installer's AppMutex: it asks the user to close Athena first
MAC_SHORTCUT_BUNDLE_ID = "com.athena.reader.source"
WINDOW_SIZE = (1280, 860)
MIN_WINDOW_SIZE = (420, 600)
LIGHT_BACKGROUND = "#f5f3ee"  # the reader's --bg colours, shown while the page loads
DARK_BACKGROUND = "#14171b"

LOCK_FILE = "app.lock"
LOG_FILE = "app.log"
CRASH_FILE = "crash.log"
LOG_MAX_BYTES = 2 * 1024 * 1024
LOG_BACKUPS = 3
STARTING_GRACE_S = 30.0  # how long another copy may take to start before its lock counts as stale
JUST_STARTED_S = 15.0  # a copy younger than this is still opening its own window
EXTRA_VOICE_DIRS_ENV = "DESTINY_EXTRA_VOICE_DIRS"
WEBVIEW2_CLIENT_KEY = r"Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
CREATE_NO_WINDOW = 0x08000000

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # the source checkout (src/destiny_bot/desktop.py)
_log_stream: _RotatingLog | None = None
_crash_file: IO[str] | None = None
_mutex_handle: Any = None


class ShortcutError(OSError):
    """Shortcuts couldn't be created; the message is safe to show to the user."""


# -- paths and resources


def is_frozen() -> bool:
    """True inside the packaged app (PyInstaller)."""
    return bool(getattr(sys, "frozen", False))


def _bundle_root() -> Path | None:
    root = getattr(sys, "_MEIPASS", None)
    return Path(root) if is_frozen() and root else None


def logs_dir() -> Path:
    path = config.data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_file() -> Path:
    return logs_dir() / LOG_FILE


def bundled_voices_dir() -> Path | None:
    """The voices shipped with the app: ``<bundle>/voices``, or ``packaging/build/voices`` in a checkout."""
    bundle = _bundle_root()
    candidates = [bundle / "voices"] if bundle else [_PROJECT_ROOT / "packaging" / "build" / "voices"]
    for path in candidates:
        if path.is_dir():
            return path
    return None


def use_bundled_voices() -> Path | None:
    """Add the bundled voices folder to ``DESTINY_EXTRA_VOICE_DIRS`` (searched read-only for voices)."""
    folder = bundled_voices_dir()
    if folder is None:
        return None
    current = [p for p in os.environ.get(EXTRA_VOICE_DIRS_ENV, "").split(os.pathsep) if p.strip()]
    if str(folder) not in current:
        os.environ[EXTRA_VOICE_DIRS_ENV] = os.pathsep.join([str(folder), *current])
    return folder


def icon_path(ext: str | None = None) -> Path | None:
    """The app icon for this platform (``.ico`` on Windows, ``.icns`` on macOS, ``.png`` elsewhere)."""
    if ext is None:
        ext = {"win32": ".ico", "darwin": ".icns"}.get(_platform(), ".png")
    name = "icon-512.png" if ext == ".png" else f"icon{ext}"
    bundle = _bundle_root()
    folders = [bundle / "icons"] if bundle else []
    folders.append(_PROJECT_ROOT / "packaging" / "icons")
    for folder in folders:
        if (folder / name).is_file():
            return folder / name
    return None


def _platform() -> str:
    if sys.platform == "win32":
        return "win32"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


# -- logging


class _RotatingLog(io.TextIOBase):
    """A text stream that appends to a log file and rotates it when it grows too big.

    Used for ``sys.stdout`` / ``sys.stderr`` in the windowed app, where they would
    otherwise be ``None`` and any ``print`` would crash.
    """

    def __init__(self, path: Path, max_bytes: int = LOG_MAX_BYTES, backups: int = LOG_BACKUPS) -> None:
        super().__init__()
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.backups = backups
        self._lock = threading.RLock()
        self._fh: IO[str] | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return "utf-8"

    @property
    def errors(self) -> str:  # type: ignore[override]
        return "replace"

    def _file(self) -> IO[str]:
        if self._fh is None:
            self._fh = open(self.path, "a", encoding="utf-8", errors="replace")  # noqa: SIM115
        return self._fh

    def _rotate(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        for index in range(self.backups - 1, 0, -1):
            older = self.path.with_name(f"{self.path.name}.{index}")
            if older.exists():
                try:
                    os.replace(older, self.path.with_name(f"{self.path.name}.{index + 1}"))
                except OSError:
                    pass
        try:
            os.replace(self.path, self.path.with_name(f"{self.path.name}.1"))
        except OSError:
            pass

    def write(self, text: str) -> int:  # type: ignore[override]
        if not text:
            return 0
        with self._lock:
            try:
                fh = self._file()
                fh.write(text)
                fh.flush()
                if fh.tell() > self.max_bytes:
                    self._rotate()
            except (OSError, ValueError):
                pass  # logging must never take the app down
        return len(text)

    def flush(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.flush()
                except (OSError, ValueError):
                    pass

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def setup_logging(*, redirect: bool | None = None, name: str = LOG_FILE) -> Path:
    """Log to ``<data>/logs/<name>``; with ``redirect`` also send stdout/stderr there.

    ``redirect`` defaults to True in the packaged app and whenever there is no
    console (``pythonw``), where ``sys.stdout`` is ``None``. Safe to call twice.
    """
    global _log_stream, _crash_file
    path = logs_dir() / name
    if redirect is None:
        redirect = is_frozen() or sys.stdout is None or sys.stderr is None
    if _log_stream is None or _log_stream.path != path:
        _log_stream = _RotatingLog(path)
    stream = _log_stream
    if redirect:
        sys.stdout = stream  # type: ignore[assignment]
        sys.stderr = stream  # type: ignore[assignment]
        if _crash_file is None:
            try:
                # Native crashes (e.g. in a voice library) leave a Python traceback here.
                _crash_file = open(logs_dir() / CRASH_FILE, "w", encoding="utf-8")  # noqa: SIM115
                faulthandler.enable(file=_crash_file, all_threads=True)
            except (OSError, ValueError, RuntimeError):
                _crash_file = None

    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_athena", False):
            root.removeHandler(handler)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler._athena = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    if (
        not redirect
        and sys.stderr is not None
        and not any(
            isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stderr for h in root.handlers
        )
    ):
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(logging.WARNING)
        console._athena = True  # type: ignore[attr-defined]
        root.addHandler(console)
    return path


# -- small platform helpers


def _hidden_flags() -> int:
    return CREATE_NO_WINDOW if os.name == "nt" else 0


def _run_hidden(
    cmd: list[str], *, env: dict[str, str] | None = None, timeout: float = 60
) -> subprocess.CompletedProcess:
    """Run a helper program without flashing a console window; never raises for a non-zero exit."""
    return subprocess.run(
        cmd,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=timeout,
        creationflags=_hidden_flags(),
        check=False,
    )


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: it exists but belongs to someone else
        try:
            code = wintypes.DWORD()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == 259  # STILL_ACTIVE
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _url_answers(url: str, timeout: float = 2.0) -> bool:
    """True if ``url`` is a running Athena reader (it serves the page with the API token)."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route 127.0.0.1 via a proxy
    try:
        with opener.open(urllib.request.Request(url, headers={"Accept": "text/html"}), timeout=timeout) as response:
            if response.status != 200:
                return False
            return b"destiny-token" in response.read(256 * 1024)
    except (OSError, ValueError):
        return False


def _webview2_version() -> str | None:
    """The installed Microsoft Edge WebView2 runtime version (Windows), or None."""
    if os.name != "nt":
        return None
    import winreg

    places = (
        (winreg.HKEY_LOCAL_MACHINE, "SOFTWARE\\WOW6432Node\\" + WEBVIEW2_CLIENT_KEY),
        (winreg.HKEY_LOCAL_MACHINE, "SOFTWARE\\" + WEBVIEW2_CLIENT_KEY),
        (winreg.HKEY_CURRENT_USER, "Software\\" + WEBVIEW2_CLIENT_KEY),
    )
    for root, key_path in places:
        try:
            with winreg.OpenKey(root, key_path) as key:
                version = str(winreg.QueryValueEx(key, "pv")[0] or "")
        except OSError:
            continue
        if version and version != "0.0.0.0":
            return version
    return None


def _prefers_dark() -> bool:
    if os.name != "nt":
        return False
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
        ) as key:
            return winreg.QueryValueEx(key, "AppsUseLightTheme")[0] == 0
    except OSError:
        return False


def _windows_app_identity() -> None:
    """Group the window under Athena's own taskbar icon and tell the installer that Athena is running."""
    global _mutex_handle
    if os.name != "nt":
        return
    import ctypes

    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(ctypes.c_wchar_p(APP_ID))
    except (AttributeError, OSError):
        pass
    if _mutex_handle is None:
        try:
            _mutex_handle = ctypes.windll.kernel32.CreateMutexW(None, False, APP_MUTEX)
        except (AttributeError, OSError):
            _mutex_handle = None


def _show_error(message: str) -> None:
    """Show a native error dialog (best effort) and log the message."""
    log.error("%s", message)
    platform = _platform()
    try:
        if platform == "win32":
            import ctypes

            # MB_ICONERROR | MB_SETFOREGROUND | MB_TOPMOST
            ctypes.windll.user32.MessageBoxW(None, message, APP_NAME, 0x10 | 0x10000 | 0x40000)
            return
        if platform == "darwin":

            def applescript(text: str) -> str:
                return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'

            script = (
                f"display dialog {applescript(message)} with title {applescript(APP_NAME)} "
                'buttons {"OK"} default button "OK" with icon stop'
            )
            _run_hidden(["osascript", "-e", script], timeout=600)
            return
        for tool in (["zenity", "--error", "--title", APP_NAME, "--text"], ["kdialog", "--title", APP_NAME, "--error"]):
            if shutil.which(tool[0]):
                _run_hidden([*tool, message], timeout=600)
                return
    except Exception as exc:  # the dialog is a courtesy; the log has the details either way
        log.warning("Couldn't show the error dialog: %s", exc)
    if sys.stderr is not None and sys.stderr is not _log_stream:  # the log already has it
        print(message, file=sys.stderr)


# -- single instance


def _lock_path() -> Path:
    return config.data_dir() / LOCK_FILE


def read_lock() -> dict[str, Any] | None:
    """The running copy's ``{"pid", "url", "started"}``, or None if there is no readable lock."""
    try:
        data = json.loads(_lock_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("pid"), int):
        return None
    return data


def _write_lock(url: str | None, started: float) -> None:
    path = _lock_path()
    payload = json.dumps({"pid": os.getpid(), "url": url, "started": started, "version": __version__})
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".app-lock-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _create_lock(started: float) -> bool:
    """Atomically create the lock file; False if it already exists."""
    try:
        fd = os.open(_lock_path(), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"pid": os.getpid(), "url": None, "started": started, "version": __version__}))
    return True


def _remove_lock_file() -> None:
    try:
        _lock_path().unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        log.warning("Couldn't remove the lock file: %s", exc)


def release_lock() -> None:
    """Remove the lock file if this process owns it."""
    info = read_lock()
    if info is not None and info.get("pid") == os.getpid():
        _remove_lock_file()


def acquire_instance(*, wait_s: float = STARTING_GRACE_S) -> dict[str, Any] | None:
    """Claim the single-instance lock.

    Returns None when this process now owns the lock, or the lock info of another
    running copy (its ``url`` answers). A copy that is still starting is waited for
    up to ``wait_s`` seconds; stale locks (dead process, server gone) are replaced.
    """
    deadline = time.monotonic() + wait_s
    started = time.time()
    for _ in range(10_000):
        if _create_lock(started):
            return None
        info = read_lock()
        if info is None:
            # Unreadable: another copy may be writing it right now, or it is corrupt.
            try:
                age = time.time() - _lock_path().stat().st_mtime
            except OSError:
                continue
            if age > 5:
                _remove_lock_file()
            else:
                time.sleep(0.1)
            continue
        pid, url = int(info["pid"]), info.get("url")
        if pid == os.getpid():
            _write_lock(None, started)
            return None
        if not _pid_alive(pid):
            log.info("Removing a stale lock from process %s", pid)
            _remove_lock_file()
            continue
        if url:
            if _url_answers(str(url)):
                return info
            log.info("The copy in process %s stopped answering; starting a new one", pid)
            _remove_lock_file()
            continue
        lock_age = time.time() - float(info.get("started") or 0)
        if lock_age < STARTING_GRACE_S and time.monotonic() < deadline:
            time.sleep(0.25)  # the other copy is still starting its server
            continue
        log.info("The copy in process %s never finished starting; starting a new one", pid)
        _remove_lock_file()
    raise RuntimeError("Couldn't tell whether Athena is already running.")


# -- windows


def _webview_module() -> Any | None:
    """pywebview, if it is installed and can show a modern window on this computer."""
    try:
        import webview
    except Exception as exc:  # ImportError, or a broken install
        log.info("pywebview isn't available: %s", exc)
        return None
    platform = _platform()
    if platform == "win32" and _webview2_version() is None:
        # Without WebView2 pywebview would fall back to Internet Explorer, which can't run the reader.
        log.info("The Microsoft Edge WebView2 runtime isn't installed")
        return None
    if platform == "linux" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        log.info("No graphical display")
        return None
    return webview


def _supported(func: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Only the keyword arguments ``func`` accepts (pywebview's options differ between versions)."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _run_webview(webview: Any, url: str) -> None:
    """Show ``url`` in a native window; returns when the window is closed."""
    try:
        webview.settings["ALLOW_DOWNLOADS"] = True  # exported MP3s get a normal Save dialog
    except (AttributeError, KeyError, TypeError):
        pass
    storage = config.data_dir() / "webview"
    storage.mkdir(parents=True, exist_ok=True)
    width, height = WINDOW_SIZE
    window_options = {
        "width": width,
        "height": height,
        "min_size": MIN_WINDOW_SIZE,
        "text_select": True,  # readers select passages to ask about
        "zoomable": True,
        "background_color": DARK_BACKGROUND if _prefers_dark() else LIGHT_BACKGROUND,
    }
    webview.create_window(APP_NAME, url, **_supported(webview.create_window, window_options))
    start_options: dict[str, Any] = {"private_mode": False, "storage_path": str(storage)}
    icon = icon_path(".png" if _platform() == "linux" else None)
    if icon is not None and _platform() != "darwin":
        start_options["icon"] = str(icon)
    webview.start(**_supported(webview.start, start_options))


def _app_browser_candidates() -> list[Path]:
    platform = _platform()
    found: list[Path] = []
    if platform == "win32":
        bases = [os.environ.get(v) for v in ("ProgramFiles(x86)", "ProgramFiles", "ProgramW6432", "LOCALAPPDATA")]
        for rel in (
            ("Microsoft", "Edge", "Application", "msedge.exe"),
            ("Google", "Chrome", "Application", "chrome.exe"),
        ):
            found.extend(Path(base, *rel) for base in bases if base)
        found.extend(_app_paths_from_registry())
    elif platform == "darwin":
        for app in (
            "Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "Google Chrome.app/Contents/MacOS/Google Chrome",
            "Chromium.app/Contents/MacOS/Chromium",
        ):
            found.extend([Path("/Applications") / app, Path.home() / "Applications" / app])
    else:
        for name in (
            "microsoft-edge",
            "microsoft-edge-stable",
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "chromium-browser",
        ):
            exe = shutil.which(name)
            if exe:
                found.append(Path(exe))
    return found


def _app_paths_from_registry() -> list[Path]:
    """Edge / Chrome locations registered under "App Paths" (custom install folders)."""
    import winreg

    paths: list[Path] = []
    for exe in ("msedge.exe", "chrome.exe"):
        for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                with winreg.OpenKey(root, rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe}") as key:
                    value = winreg.QueryValueEx(key, "")[0]
            except OSError:
                continue
            if value:
                paths.append(Path(str(value).strip('"')))
    return paths


def find_app_browser() -> Path | None:
    """Microsoft Edge or Google Chrome, which can show the reader as a standalone app window."""
    for candidate in _app_browser_candidates():
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _browser_profile() -> Path:
    path = config.data_dir() / "browser-profile"
    path.mkdir(parents=True, exist_ok=True)
    return path


def app_browser_command(browser: Path, url: str, profile: Path) -> list[str]:
    """The command that opens ``url`` as an app window with its own browser profile."""
    width, height = WINDOW_SIZE
    return [
        str(browser),
        f"--app={url}",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-mode",
        f"--window-size={width},{height}",
    ]


def _launch_app_browser(command: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=_hidden_flags(),
    )


def _profile_in_use(profile: Path) -> bool:
    """True while a browser still runs with ``profile`` (it may have taken over our window)."""
    if os.name == "nt":
        return (profile / "lockfile").exists()  # deleted by Windows when the browser exits
    lock = profile / "SingletonLock"  # a symlink to "<host>-<pid>"
    try:
        target = os.readlink(lock)
    except OSError:
        return False
    try:
        return _pid_alive(int(target.rsplit("-", 1)[-1]))
    except ValueError:
        return True


def _wait_for_browser(proc: subprocess.Popen, profile: Path, poll_s: float = 1.0) -> None:
    proc.wait()
    while _profile_in_use(profile):
        time.sleep(poll_s)


def _open_default_browser(url: str) -> bool:
    try:
        return bool(webbrowser.open(url))
    except Exception as exc:
        log.warning("Couldn't open the default browser: %s", exc)
        return False


def _wait_until_interrupted(stop: threading.Event | None = None) -> None:
    """Block until Ctrl+C (or ``stop`` is set)."""
    stop = stop or threading.Event()
    try:
        while not stop.wait(1.0):
            pass
    except KeyboardInterrupt:
        pass


def show_window(url: str, *, wait: bool = True) -> str:
    """Show the reader at ``url``; returns how: "webview", "app-browser" or "browser".

    With ``wait`` this blocks until the user closes it (for the default browser,
    until Ctrl+C, since there is no way to tell when a tab closes).
    """
    webview = _webview_module()
    if webview is not None:
        try:
            _run_webview(webview, url)
            log.info("Showed the reader in a native window")
            return "webview"
        except Exception:
            log.exception("The native window failed; trying a browser app window instead")

    browser = find_app_browser()
    if browser is not None:
        profile = _browser_profile()
        try:
            proc = _launch_app_browser(app_browser_command(browser, url, profile))
        except OSError:
            log.exception("Couldn't start %s", browser)
        else:
            log.info("Showed the reader in an app window of %s", browser)
            if wait:
                _wait_for_browser(proc, profile)
            return "app-browser"

    if not _open_default_browser(url):
        raise RuntimeError(
            "Athena couldn't open its window because no web browser was found. "
            "Install Microsoft Edge or Google Chrome, then open Athena again."
        )
    log.info("Showed the reader in the default browser")
    if wait:
        print(f"Athena is running at {url} (press Ctrl+C to stop)", flush=True)
        _wait_until_interrupted()
    return "browser"


# -- the app


def _start_server(port: int) -> tuple[str, Callable[[], None]]:
    from .server import start_server_in_thread

    return start_server_in_thread(host="127.0.0.1", port=port)


def _startup_problem(exc: BaseException) -> str:
    detail = str(exc).strip()
    if isinstance(exc, (OSError, RuntimeError, ValueError)) and detail and not isinstance(exc, ImportError):
        reason = detail
    elif isinstance(exc, ImportError):
        reason = "A part of Athena is missing. Please reinstall Athena."
    else:
        reason = f"Something unexpected went wrong ({type(exc).__name__}: {detail or 'no details'})."
    return reason


def _fatal(exc: BaseException, log_path: Path | None) -> int:
    log.error("Athena couldn't start", exc_info=exc)
    where = f"\n\nDetails were saved in:\n{log_path}" if log_path else ""
    _show_error(
        f"Athena couldn't start.\n\n{_startup_problem(exc)}\n\n"
        f"Try again in a moment. If it keeps happening, restart your computer.{where}"
    )
    return 1


def run_app(*, port: int = 8765) -> int:
    """Run Athena in its own window until the window is closed. Returns an exit code."""
    log_path: Path | None = None
    try:
        log_path = setup_logging()
    except Exception as exc:  # a read-only or full disk must not stop the app
        if sys.stderr is not None:
            print(f"Couldn't open the log file: {exc}", file=sys.stderr)
    log.info(
        "Starting Athena %s (frozen=%s, python %s, %s)", __version__, is_frozen(), sys.version.split()[0], sys.platform
    )

    try:
        voices = use_bundled_voices()
        if voices is not None:
            log.info("Bundled voices: %s", voices)
        _windows_app_identity()
        other = acquire_instance()
    except Exception as exc:
        return _fatal(exc, log_path)

    if other is not None:
        url = str(other["url"])
        age = time.time() - float(other.get("started") or 0)
        if age < JUST_STARTED_S:
            log.info("Athena was started moments ago (process %s); its window is opening", other["pid"])
            return 0
        log.info("Athena is already running at %s; opening another window onto it", url)
        try:
            show_window(url, wait=False)  # the other copy owns the server; a native window still blocks
        except Exception as exc:
            return _fatal(exc, log_path)
        return 0

    stop: Callable[[], None] | None = None
    try:
        url, stop = _start_server(port)
        _write_lock(url, time.time())
        log.info("Reader server running at %s", url)
        how = show_window(url, wait=True)
        log.info("The %s window was closed; shutting down", how)
        return 0
    except Exception as exc:
        return _fatal(exc, log_path)
    finally:
        if stop is not None:
            try:
                stop()
            except Exception:
                log.exception("Stopping the server failed")
        release_lock()


# -- shortcuts (for source installs)


def _gui_python() -> Path:
    """The Python that runs Athena without a console window (``pythonw.exe`` on Windows)."""
    exe = Path(sys.executable)
    if _platform() == "win32" and exe.name.lower() == "python.exe":
        windowless = exe.with_name("pythonw.exe")
        if windowless.is_file():
            return windowless
    return exe


def _launch_target() -> tuple[Path, list[str], dict[str, str]]:
    """(program, arguments, extra environment) that open Athena from this installation."""
    if is_frozen():
        return Path(sys.executable), [], {}
    env: dict[str, str] = {}
    home = os.environ.get("DESTINY_HOME")
    if home:
        env["DESTINY_HOME"] = str(Path(home).expanduser().resolve())
    return _gui_python(), ["-m", "destiny_bot", "app"], env


def _working_dir() -> Path:
    if (_PROJECT_ROOT / "pyproject.toml").is_file():
        return _PROJECT_ROOT
    return Path.home()


def _windows_arguments(args: list[str], env: dict[str, str]) -> str:
    """Shortcut arguments; a .lnk can't set environment variables, so DESTINY_HOME goes through ``-c``."""
    if env.get("DESTINY_HOME") and args[:2] == ["-m", "destiny_bot"]:
        code = (
            "import os,sys;os.environ['DESTINY_HOME']=sys.argv[1];"
            "from destiny_bot.cli import main;sys.exit(main(sys.argv[2:]))"
        )
        return subprocess.list2cmdline(["-c", code, env["DESTINY_HOME"], *args[2:]])
    return subprocess.list2cmdline(args)


_PS_SHORTCUTS = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$shell = New-Object -ComObject WScript.Shell
$folders = @()
if ($env:ATHENA_SC_DESKTOP -eq '1') { $folders += [Environment]::GetFolderPath('Desktop') }
if ($env:ATHENA_SC_START -eq '1') { $folders += [Environment]::GetFolderPath('Programs') }
foreach ($folder in $folders) {
  if (-not $folder) { continue }
  if (-not (Test-Path -LiteralPath $folder)) { New-Item -ItemType Directory -Path $folder -Force | Out-Null }
  $path = [System.IO.Path]::Combine($folder, $env:ATHENA_SC_NAME + '.lnk')
  $link = $shell.CreateShortcut($path)
  $link.TargetPath = $env:ATHENA_SC_TARGET
  $link.Arguments = $env:ATHENA_SC_ARGS
  $link.WorkingDirectory = $env:ATHENA_SC_WORKDIR
  if ($env:ATHENA_SC_ICON) { $link.IconLocation = $env:ATHENA_SC_ICON + ',0' }
  $link.Description = $env:ATHENA_SC_DESCRIPTION
  $link.Save()
  Write-Output $path
}
"""


def _powershell() -> str:
    exe = shutil.which("powershell") or shutil.which("powershell.exe")
    if exe:
        return exe
    return str(
        Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    )


def windows_shortcut_job(*, desktop: bool, start_menu: bool) -> tuple[list[str], dict[str, str], str]:
    """(command, environment, script) that create the Windows shortcuts with WScript.Shell."""
    target, args, env = _launch_target()
    icon = icon_path(".ico")
    if icon is None and is_frozen():
        icon = Path(sys.executable)
    values = {
        "ATHENA_SC_DESKTOP": "1" if desktop else "0",
        "ATHENA_SC_START": "1" if start_menu else "0",
        "ATHENA_SC_NAME": APP_NAME,
        "ATHENA_SC_TARGET": str(target),
        "ATHENA_SC_ARGS": _windows_arguments(args, env),
        "ATHENA_SC_WORKDIR": str(_working_dir()),
        "ATHENA_SC_ICON": str(icon) if icon else "",
        "ATHENA_SC_DESCRIPTION": "Athena reads your textbooks aloud",
    }
    command = [_powershell(), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File"]
    return command, {**os.environ, **values}, _PS_SHORTCUTS


def _create_windows_shortcuts(*, desktop: bool, start_menu: bool) -> list[Path]:
    command, env, script = windows_shortcut_job(desktop=desktop, start_menu=start_menu)
    with tempfile.TemporaryDirectory(prefix="athena-shortcut-") as tmp:
        ps1 = Path(tmp) / "make-shortcuts.ps1"
        ps1.write_text(script, encoding="utf-8-sig")  # the BOM makes Windows PowerShell read it as UTF-8
        try:
            proc = _run_hidden([*command, str(ps1)], env=env, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ShortcutError(f"Couldn't create the shortcuts: Windows PowerShell didn't run ({exc}).") from exc
    output = (proc.stdout or b"").decode("utf-8", "replace")
    if proc.returncode != 0:
        detail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        raise ShortcutError("Couldn't create the shortcuts" + (f": {detail[-1]}" if detail else "."))
    return [Path(line.strip().lstrip("\ufeff")) for line in output.splitlines() if line.strip().endswith(".lnk")]


def _mac_launcher_script() -> str:
    target, args, env = _launch_target()
    lines = ["#!/bin/bash", f"# Opens {APP_NAME}. Created by 'destiny shortcut'; safe to delete."]
    for key, value in env.items():
        lines.append(f"export {key}={shlex.quote(value)}")
    lines.append(f"cd {shlex.quote(str(_working_dir()))} 2>/dev/null")
    lines.append("exec " + " ".join(shlex.quote(part) for part in [str(target), *args]))
    return "\n".join(lines) + "\n"


def build_mac_app(bundle: Path) -> Path:
    """Write a small ``Athena.app`` that starts Athena from this installation."""
    plist_path = bundle / "Contents" / "Info.plist"
    if bundle.exists():
        try:
            existing = plistlib.loads(plist_path.read_bytes())
        except (OSError, plistlib.InvalidFileException, ValueError):
            existing = {}
        if existing.get("CFBundleIdentifier") != MAC_SHORTCUT_BUNDLE_ID:
            raise ShortcutError(f"There is already an app at {bundle}. Remove it, then try again.")
        shutil.rmtree(bundle)
    macos = bundle / "Contents" / "MacOS"
    resources = bundle / "Contents" / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)
    launcher = macos / APP_NAME
    launcher.write_text(_mac_launcher_script(), encoding="utf-8")
    launcher.chmod(0o755)
    info = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": MAC_SHORTCUT_BUNDLE_ID,
        "CFBundleExecutable": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": __version__,
        "CFBundleVersion": __version__,
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "10.13",
    }
    icon = icon_path(".icns")
    if icon is not None:
        shutil.copyfile(icon, resources / "icon.icns")
        info["CFBundleIconFile"] = "icon"
    plist_path.write_bytes(plistlib.dumps(info))
    return bundle


def _create_mac_shortcuts(*, desktop: bool, start_menu: bool) -> list[Path]:
    created: list[Path] = []
    desktop_dir = Path.home() / "Desktop"
    if start_menu:
        apps = Path.home() / "Applications"
        apps.mkdir(parents=True, exist_ok=True)
        bundle = build_mac_app(apps / f"{APP_NAME}.app")
        created.append(bundle)
        if desktop and desktop_dir.is_dir():
            alias = desktop_dir / APP_NAME
            if alias.is_symlink():
                alias.unlink()
            if alias.exists():
                log.warning("Not replacing %s: it isn't a shortcut made by Athena", alias)
            else:
                alias.symlink_to(bundle, target_is_directory=True)
                created.append(alias)
    elif desktop and desktop_dir.is_dir():
        created.append(build_mac_app(desktop_dir / f"{APP_NAME}.app"))
    return created


def _desktop_entry_arg(arg: str) -> str:
    """Quote one Exec= argument as the freedesktop Desktop Entry spec requires."""
    reserved = set(" \t\n\"'\\><~|&;$*?#()`")
    if arg and not (set(arg) & reserved):
        return arg.replace("%", "%%")
    quoted = '"' + "".join("\\" + ch if ch in '"`$\\' else ch for ch in arg) + '"'
    # The string-value escape rule applies on top of the quoting rule.
    return quoted.replace("\\", "\\\\").replace("%", "%%")


def linux_desktop_entry() -> str:
    target, args, env = _launch_target()
    command = [str(target), *args]
    if env:
        command = ["env", *(f"{key}={value}" for key, value in env.items()), *command]
    icon = icon_path(".png")
    lines = [
        "[Desktop Entry]",
        "Type=Application",
        "Version=1.0",
        f"Name={APP_NAME}",
        "Comment=Reads your textbooks aloud",
        "Exec=" + " ".join(_desktop_entry_arg(part) for part in command),
        f"Path={_working_dir()}",
        f"Icon={icon if icon else 'accessories-text-editor'}",
        "Terminal=false",
        "Categories=Education;Office;",
        "StartupNotify=true",
    ]
    return "\n".join(lines) + "\n"


def _linux_desktop_dir() -> Path | None:
    if shutil.which("xdg-user-dir"):
        try:
            proc = _run_hidden(["xdg-user-dir", "DESKTOP"], timeout=10)
            value = (proc.stdout or b"").decode("utf-8", "replace").strip()
            if value and Path(value) != Path.home() and Path(value).is_dir():
                return Path(value)
        except (OSError, subprocess.TimeoutExpired):
            pass
    fallback = Path.home() / "Desktop"
    return fallback if fallback.is_dir() else None


def _create_linux_shortcuts(*, desktop: bool, start_menu: bool) -> list[Path]:
    entry = linux_desktop_entry()
    created: list[Path] = []
    folders: list[Path] = []
    if start_menu:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
        folders.append(base / "applications")
    if desktop:
        desktop_dir = _linux_desktop_dir()
        if desktop_dir is not None:
            folders.append(desktop_dir)
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "athena.desktop"
        path.write_text(entry, encoding="utf-8")
        path.chmod(0o755)
        if shutil.which("gio"):
            try:  # GNOME only launches desktop files that are marked as trusted
                _run_hidden(["gio", "set", str(path), "metadata::trusted", "true"], timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
        created.append(path)
    return created


def create_shortcuts(*, desktop: bool = True, start_menu: bool = True) -> list[Path]:
    """Create shortcuts that open Athena in its own window; returns the paths created.

    Windows: ``Athena.lnk`` on the Desktop (OneDrive-redirected Desktops included)
    and in the Start menu. macOS: ``~/Applications/Athena.app`` plus a Desktop alias.
    Linux: ``athena.desktop`` in the applications menu and on the Desktop.
    Raises :class:`ShortcutError` with a friendly message.
    """
    platform = _platform()
    try:
        if platform == "win32":
            return _create_windows_shortcuts(desktop=desktop, start_menu=start_menu)
        if platform == "darwin":
            return _create_mac_shortcuts(desktop=desktop, start_menu=start_menu)
        return _create_linux_shortcuts(desktop=desktop, start_menu=start_menu)
    except ShortcutError:
        raise
    except OSError as exc:
        raise ShortcutError(f"Couldn't create the shortcuts: {exc.strerror or exc}") from exc


__all__ = [
    "APP_NAME",
    "ShortcutError",
    "acquire_instance",
    "app_browser_command",
    "bundled_voices_dir",
    "create_shortcuts",
    "find_app_browser",
    "icon_path",
    "is_frozen",
    "log_file",
    "read_lock",
    "release_lock",
    "run_app",
    "setup_logging",
    "show_window",
    "use_bundled_voices",
]
