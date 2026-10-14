"""Tests for the desktop app wrapper (destiny_bot.desktop).

A native window can't be shown here, so pywebview, the browsers and the server are
replaced with fakes; the real Windows/macOS behaviour is exercised by the CI builds.
"""

from __future__ import annotations

import http.server
import json
import logging
import os
import plistlib
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

import destiny_bot
from destiny_bot import config, desktop

# tests/test_cli.py swaps these modules out through sys.modules. A submodule imported here
# would also become an attribute of the destiny_bot package, which `from . import desktop`
# prefers over sys.modules, so drop the attributes again.
for _name in ("desktop", "selftest"):
    vars(destiny_bot).pop(_name, None)


@pytest.fixture(autouse=True)
def clean_desktop_state(monkeypatch):
    """Undo logging handlers and globals that desktop functions install."""
    monkeypatch.setattr(desktop.faulthandler, "enable", lambda *a, **k: None)
    monkeypatch.setattr(desktop, "_log_stream", None)
    monkeypatch.setattr(desktop, "_crash_file", None)
    monkeypatch.setattr(desktop, "_windows_app_identity", lambda: None)
    monkeypatch.delenv(desktop.EXTRA_VOICE_DIRS_ENV, raising=False)
    monkeypatch.setenv(desktop.EXTRA_VOICE_DIRS_ENV, "")
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_athena", False):
            root.removeHandler(handler)
    if desktop._log_stream is not None:
        desktop._log_stream.close()


def split_windows_command_line(line: str) -> list[str]:
    """Split a command line the way the Microsoft C runtime does (what pythonw.exe sees)."""
    args: list[str] = []
    buf: list[str] = []
    in_quotes = has_arg = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "\\":
            j = i
            while j < len(line) and line[j] == "\\":
                j += 1
            count = j - i
            if j < len(line) and line[j] == '"':
                buf.append("\\" * (count // 2))
                if count % 2:
                    buf.append('"')
                    j += 1
            else:
                buf.append("\\" * count)
            i = j
            has_arg = True
        elif ch == '"':
            in_quotes = not in_quotes
            has_arg = True
            i += 1
        elif ch in " \t" and not in_quotes:
            if has_arg:
                args.append("".join(buf))
                buf, has_arg = [], False
            i += 1
        else:
            buf.append(ch)
            has_arg = True
            i += 1
    if has_arg:
        args.append("".join(buf))
    return args


# -- bundled resources


def test_use_bundled_voices_from_checkout(tmp_path, monkeypatch):
    voices = tmp_path / "packaging" / "build" / "voices"
    (voices / "piper").mkdir(parents=True)
    monkeypatch.setattr(desktop, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setenv(desktop.EXTRA_VOICE_DIRS_ENV, "/somewhere/else")

    assert desktop.use_bundled_voices() == voices
    assert os.environ[desktop.EXTRA_VOICE_DIRS_ENV] == os.pathsep.join([str(voices), "/somewhere/else"])
    desktop.use_bundled_voices()  # idempotent
    assert os.environ[desktop.EXTRA_VOICE_DIRS_ENV].count(str(voices)) == 1


def test_no_bundled_voices_leaves_environment_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop, "_PROJECT_ROOT", tmp_path)
    assert desktop.use_bundled_voices() is None
    assert os.environ[desktop.EXTRA_VOICE_DIRS_ENV] == ""


def test_frozen_app_uses_its_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "_internal"
    (bundle / "voices" / "kokoro").mkdir(parents=True)
    (bundle / "icons").mkdir()
    (bundle / "icons" / "icon.ico").write_bytes(b"ico")
    (bundle / "icons" / "icon-512.png").write_bytes(b"png")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(desktop, "_PROJECT_ROOT", tmp_path / "nowhere")

    assert desktop.is_frozen()
    assert desktop.bundled_voices_dir() == bundle / "voices"
    assert desktop.icon_path(".ico") == bundle / "icons" / "icon.ico"
    assert desktop.icon_path(".png") == bundle / "icons" / "icon-512.png"
    assert desktop.icon_path(".icns") is None


def test_repository_icons_exist():
    for ext in (".ico", ".icns", ".png"):
        path = desktop.icon_path(ext)
        assert path is not None and path.stat().st_size > 1000, ext


# -- logging


def test_frozen_app_redirects_output_to_a_log_file(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    path = desktop.setup_logging()

    assert path == config.data_dir() / "logs" / "app.log"
    assert isinstance(sys.stdout, desktop._RotatingLog) and sys.stderr is sys.stdout
    print("hello from print")
    sys.stderr.write("an error line\n")
    logging.getLogger("destiny_bot.test").warning("a warning record")
    assert not sys.stdout.isatty()
    text = path.read_text(encoding="utf-8")
    assert "hello from print" in text
    assert "an error line" in text
    assert "a warning record" in text


def test_pythonw_without_console_is_redirected_too(monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    desktop.setup_logging()
    assert isinstance(sys.stdout, desktop._RotatingLog)


def test_console_runs_keep_their_output(monkeypatch, capsys):
    before = sys.stdout
    path = desktop.setup_logging()
    assert sys.stdout is before
    logging.getLogger("destiny_bot.test").info("info goes to the file")
    assert "info goes to the file" in path.read_text(encoding="utf-8")


def test_log_rotation(tmp_path):
    stream = desktop._RotatingLog(tmp_path / "app.log", max_bytes=200, backups=2)
    for i in range(60):
        stream.write(f"line {i:03d} with some padding to fill the file\n")
    stream.close()
    assert (tmp_path / "app.log.1").is_file()
    assert (tmp_path / "app.log.2").is_file()
    assert not (tmp_path / "app.log.3").exists()
    text = "".join(p.read_text() for p in sorted(tmp_path.glob("app.log*")))
    assert "line 059" in text
    stream.write("after rotation\n")
    assert "after rotation" in (tmp_path / "app.log").read_text()
    stream.close()


# -- single instance


def _lock_file() -> Path:
    return config.data_dir() / desktop.LOCK_FILE


def _write_other_lock(pid: int, url: str | None, started: float) -> None:
    _lock_file().write_text(json.dumps({"pid": pid, "url": url, "started": started}), encoding="utf-8")


def test_first_copy_claims_the_lock():
    assert desktop.acquire_instance() is None
    assert desktop.read_lock()["pid"] == os.getpid()
    desktop.release_lock()
    assert not _lock_file().exists()


def test_release_lock_leaves_other_copies_alone():
    _write_other_lock(os.getpid() + 1, "http://127.0.0.1:8765/", time.time())
    desktop.release_lock()
    assert _lock_file().exists()


def test_lock_of_a_dead_process_is_replaced(monkeypatch):
    _write_other_lock(999_999, "http://127.0.0.1:8765/", time.time())
    monkeypatch.setattr(desktop, "_pid_alive", lambda pid: False)
    assert desktop.acquire_instance() is None
    assert desktop.read_lock()["pid"] == os.getpid()


def test_running_copy_is_found(monkeypatch):
    _write_other_lock(4242, "http://127.0.0.1:8765/", time.time() - 600)
    monkeypatch.setattr(desktop, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(desktop, "_url_answers", lambda url, timeout=2.0: True)
    info = desktop.acquire_instance()
    assert info is not None and info["pid"] == 4242 and info["url"] == "http://127.0.0.1:8765/"
    assert desktop.read_lock()["pid"] == 4242  # untouched


def test_copy_whose_server_is_gone_is_stale(monkeypatch):
    _write_other_lock(4242, "http://127.0.0.1:8765/", time.time() - 600)
    monkeypatch.setattr(desktop, "_pid_alive", lambda pid: True)  # e.g. the pid was reused after a reboot
    monkeypatch.setattr(desktop, "_url_answers", lambda url, timeout=2.0: False)
    assert desktop.acquire_instance() is None


def test_waits_for_a_copy_that_is_still_starting(monkeypatch):
    _write_other_lock(4242, None, time.time())
    monkeypatch.setattr(desktop, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(desktop, "_url_answers", lambda url, timeout=2.0: True)

    def finish_starting() -> None:
        time.sleep(0.4)
        _write_other_lock(4242, "http://127.0.0.1:8766/", time.time())

    thread = threading.Thread(target=finish_starting)
    thread.start()
    begin = time.monotonic()
    info = desktop.acquire_instance(wait_s=10)
    thread.join()
    assert info is not None and info["url"] == "http://127.0.0.1:8766/"
    assert time.monotonic() - begin >= 0.3


def test_copy_stuck_starting_is_stale(monkeypatch):
    _write_other_lock(4242, None, time.time() - desktop.STARTING_GRACE_S - 5)
    monkeypatch.setattr(desktop, "_pid_alive", lambda pid: True)
    assert desktop.acquire_instance() is None


def test_corrupt_lock_is_replaced():
    _lock_file().write_text("{not json", encoding="utf-8")
    old = time.time() - 60
    os.utime(_lock_file(), (old, old))
    assert desktop.acquire_instance() is None


def test_pid_alive():
    assert desktop._pid_alive(os.getpid())
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert not desktop._pid_alive(proc.pid)
    assert not desktop._pid_alive(0)


class _PageHandler(http.server.BaseHTTPRequestHandler):
    body = b""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args):
        pass


@pytest.mark.parametrize(
    ("body", "expected"),
    [(b'<meta name="destiny-token" content="abc">', True), (b"<html>another program</html>", False)],
)
def test_url_answers_only_for_the_reader(body, expected):
    handler = type("Handler", (_PageHandler,), {"body": body})
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert desktop._url_answers(f"http://127.0.0.1:{server.server_port}/") is expected
    finally:
        server.shutdown()
        server.server_close()
    assert desktop._url_answers(f"http://127.0.0.1:{server.server_port}/", timeout=0.5) is False


# -- run_app


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple] = []


def test_run_app_serves_shows_the_window_and_stops(monkeypatch):
    rec = _Recorder()
    url = "http://127.0.0.1:8765/"
    monkeypatch.setattr(
        desktop,
        "_start_server",
        lambda port: (rec.calls.append(("start", port)), (url, lambda: rec.calls.append(("stop",))))[1],
    )

    def fake_show(shown_url, *, wait=True):
        lock = desktop.read_lock()
        rec.calls.append(("show", shown_url, wait, lock["url"], lock["pid"]))
        return "webview"

    monkeypatch.setattr(desktop, "show_window", fake_show)
    assert desktop.run_app(port=9100) == 0
    assert rec.calls == [("start", 9100), ("show", url, True, url, os.getpid()), ("stop",)]
    assert not _lock_file().exists()


def test_second_copy_opens_a_window_onto_the_first(monkeypatch):
    _write_other_lock(4242, "http://127.0.0.1:8770/", time.time() - 600)
    monkeypatch.setattr(desktop, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(desktop, "_url_answers", lambda url, timeout=2.0: True)
    monkeypatch.setattr(desktop, "_start_server", lambda port: pytest.fail("must not start a second server"))
    shown: list[tuple[str, bool]] = []
    monkeypatch.setattr(desktop, "show_window", lambda url, *, wait=True: shown.append((url, wait)) or "webview")
    assert desktop.run_app() == 0
    assert shown == [("http://127.0.0.1:8770/", False)]
    assert desktop.read_lock()["pid"] == 4242


def test_double_click_while_starting_opens_only_one_window(monkeypatch):
    _write_other_lock(4242, "http://127.0.0.1:8770/", time.time() - 1)
    monkeypatch.setattr(desktop, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(desktop, "_url_answers", lambda url, timeout=2.0: True)
    monkeypatch.setattr(desktop, "_start_server", lambda port: pytest.fail("must not start a second server"))
    monkeypatch.setattr(desktop, "show_window", lambda url, *, wait=True: pytest.fail("no second window"))
    assert desktop.run_app() == 0


def test_startup_failure_shows_a_friendly_message(monkeypatch):
    def broken(port):
        raise OSError("Couldn't start: ports 8765 to 8774 are all in use.")

    messages: list[str] = []
    monkeypatch.setattr(desktop, "_start_server", broken)
    monkeypatch.setattr(desktop, "_show_error", messages.append)
    assert desktop.run_app() == 1
    assert len(messages) == 1
    assert messages[0].startswith("Athena couldn't start.")
    assert "ports 8765 to 8774 are all in use" in messages[0]
    assert str(config.data_dir() / "logs" / "app.log") in messages[0]
    assert not _lock_file().exists()


def test_window_failure_stops_the_server(monkeypatch):
    stopped: list[bool] = []
    monkeypatch.setattr(desktop, "_start_server", lambda port: ("http://127.0.0.1:8765/", lambda: stopped.append(True)))
    monkeypatch.setattr(
        desktop, "show_window", lambda url, *, wait=True: (_ for _ in ()).throw(RuntimeError("no window"))
    )
    monkeypatch.setattr(desktop, "_show_error", lambda message: None)
    assert desktop.run_app() == 1
    assert stopped == [True]


def test_import_errors_ask_for_a_reinstall():
    assert "reinstall" in desktop._startup_problem(ImportError("No module named 'fastapi'"))
    assert "KeyError" in desktop._startup_problem(KeyError("x"))


def test_show_error_falls_back_to_stderr(monkeypatch, capsys):
    monkeypatch.setattr(desktop, "_platform", lambda: "linux")
    monkeypatch.setattr(desktop.shutil, "which", lambda name: None)
    desktop._show_error("Something broke.")
    assert "Something broke." in capsys.readouterr().err


def test_show_error_uses_zenity_when_available(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(desktop, "_platform", lambda: "linux")
    monkeypatch.setattr(desktop.shutil, "which", lambda name: "/usr/bin/zenity" if name == "zenity" else None)
    monkeypatch.setattr(desktop, "_run_hidden", lambda cmd, **kw: calls.append(cmd))
    desktop._show_error("Oops")
    assert calls == [["zenity", "--error", "--title", "Athena", "--text", "Oops"]]


# -- choosing a window


class FakeWebview(types.SimpleNamespace):
    def __init__(self, fail: bool = False) -> None:
        super().__init__(settings={"ALLOW_DOWNLOADS": False}, created=[], started=[], fail=fail)

    def create_window(
        self,
        title,
        url=None,
        *,
        width=800,
        height=600,
        min_size=(200, 100),
        text_select=False,
        zoomable=False,
        background_color="#FFFFFF",
    ):
        self.created.append(
            {
                "title": title,
                "url": url,
                "width": width,
                "height": height,
                "min_size": min_size,
                "text_select": text_select,
                "zoomable": zoomable,
                "background_color": background_color,
            }
        )

    def start(self, *, private_mode=True, storage_path=None, icon=None, gui=None):
        if self.fail:
            raise RuntimeError("WebView2 failed")
        self.started.append({"private_mode": private_mode, "storage_path": storage_path, "icon": icon})


def test_native_window_settings(monkeypatch):
    fake = FakeWebview()
    monkeypatch.setattr(desktop, "_webview_module", lambda: fake)
    monkeypatch.setattr(desktop, "find_app_browser", lambda: pytest.fail("no fallback needed"))
    assert desktop.show_window("http://127.0.0.1:8765/") == "webview"
    assert fake.settings["ALLOW_DOWNLOADS"] is True
    (window,) = fake.created
    assert window["title"] == "Athena"
    assert window["url"] == "http://127.0.0.1:8765/"
    assert (window["width"], window["height"]) == (1280, 860)
    assert window["min_size"] == (420, 600)
    assert window["text_select"] is True
    (started,) = fake.started
    assert started["private_mode"] is False
    assert started["storage_path"] == str(config.data_dir() / "webview")
    assert Path(started["storage_path"]).is_dir()


def test_older_pywebview_without_newer_options(monkeypatch):
    calls: list[dict] = []
    old = types.SimpleNamespace(
        settings={},
        create_window=lambda title, url=None, width=800, height=600, min_size=(200, 100): calls.append(
            {"title": title, "width": width}
        ),
        start=lambda private_mode=True, storage_path=None: calls.append({"private_mode": private_mode}),
    )
    desktop._run_webview(old, "http://127.0.0.1:8765/")
    assert calls == [{"title": "Athena", "width": 1280}, {"private_mode": False}]


class FakeProc:
    def __init__(self) -> None:
        self.waited = False

    def wait(self) -> int:
        self.waited = True
        return 0


def test_falls_back_to_an_app_browser_window(monkeypatch, tmp_path):
    fake = FakeWebview(fail=True)
    edge = tmp_path / "msedge.exe"
    launched: list[list[str]] = []
    proc = FakeProc()
    monkeypatch.setattr(desktop, "_webview_module", lambda: fake)
    monkeypatch.setattr(desktop, "find_app_browser", lambda: edge)
    monkeypatch.setattr(desktop, "_launch_app_browser", lambda cmd: launched.append(cmd) or proc)
    monkeypatch.setattr(desktop, "_profile_in_use", lambda profile: False)
    monkeypatch.setattr(desktop, "_open_default_browser", lambda url: pytest.fail("no default browser needed"))

    assert desktop.show_window("http://127.0.0.1:8765/") == "app-browser"
    (cmd,) = launched
    profile = config.data_dir() / "browser-profile"
    assert cmd[0] == str(edge)
    assert "--app=http://127.0.0.1:8765/" in cmd
    assert f"--user-data-dir={profile}" in cmd
    assert "--no-first-run" in cmd and "--no-default-browser-check" in cmd
    assert proc.waited


def test_app_browser_window_without_waiting(monkeypatch, tmp_path):
    proc = FakeProc()
    monkeypatch.setattr(desktop, "_webview_module", lambda: None)
    monkeypatch.setattr(desktop, "find_app_browser", lambda: tmp_path / "chrome")
    monkeypatch.setattr(desktop, "_launch_app_browser", lambda cmd: proc)
    assert desktop.show_window("http://127.0.0.1:8765/", wait=False) == "app-browser"
    assert not proc.waited


def test_falls_back_to_the_default_browser(monkeypatch):
    opened: list[str] = []
    waited: list[bool] = []
    monkeypatch.setattr(desktop, "_webview_module", lambda: None)
    monkeypatch.setattr(desktop, "find_app_browser", lambda: None)
    monkeypatch.setattr(desktop, "_open_default_browser", lambda url: opened.append(url) or True)
    monkeypatch.setattr(desktop, "_wait_until_interrupted", lambda: waited.append(True))
    assert desktop.show_window("http://127.0.0.1:8765/") == "browser"
    assert opened == ["http://127.0.0.1:8765/"] and waited == [True]


def test_browser_that_fails_to_launch_falls_through(monkeypatch, tmp_path):
    def cannot_launch(cmd):
        raise OSError("not executable")

    monkeypatch.setattr(desktop, "_webview_module", lambda: None)
    monkeypatch.setattr(desktop, "find_app_browser", lambda: tmp_path / "chrome")
    monkeypatch.setattr(desktop, "_launch_app_browser", cannot_launch)
    monkeypatch.setattr(desktop, "_open_default_browser", lambda url: False)
    with pytest.raises(RuntimeError, match="no web browser was found"):
        desktop.show_window("http://127.0.0.1:8765/")


def test_wait_until_interrupted_returns_on_stop():
    stop = threading.Event()
    stop.set()
    desktop._wait_until_interrupted(stop)


def test_webview_needs_webview2_on_windows(monkeypatch):
    fake = types.ModuleType("webview")
    monkeypatch.setitem(sys.modules, "webview", fake)
    monkeypatch.setattr(desktop, "_platform", lambda: "win32")
    monkeypatch.setattr(desktop, "_webview2_version", lambda: None)
    assert desktop._webview_module() is None
    monkeypatch.setattr(desktop, "_webview2_version", lambda: "129.0.2792.65")
    assert desktop._webview_module() is fake


def test_webview_needs_a_display_on_linux(monkeypatch):
    fake = types.ModuleType("webview")
    monkeypatch.setitem(sys.modules, "webview", fake)
    monkeypatch.setattr(desktop, "_platform", lambda: "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert desktop._webview_module() is None
    monkeypatch.setenv("DISPLAY", ":0")
    assert desktop._webview_module() is fake


def test_missing_pywebview(monkeypatch):
    monkeypatch.setitem(sys.modules, "webview", None)  # makes `import webview` fail
    assert desktop._webview_module() is None


def test_finds_edge_on_windows(monkeypatch, tmp_path):
    edge = tmp_path / "x86" / "Microsoft" / "Edge" / "Application" / "msedge.exe"
    edge.parent.mkdir(parents=True)
    edge.write_bytes(b"")
    monkeypatch.setattr(desktop, "_platform", lambda: "win32")
    monkeypatch.setattr(desktop, "_app_paths_from_registry", list)
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "x86"))
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "pf"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert desktop.find_app_browser() == edge


def test_finds_chrome_on_macos(monkeypatch, tmp_path):
    chrome = tmp_path / "Applications" / "Google Chrome.app" / "Contents" / "MacOS" / "Google Chrome"
    chrome.parent.mkdir(parents=True)
    chrome.write_bytes(b"")
    monkeypatch.setattr(desktop, "_platform", lambda: "darwin")
    monkeypatch.setattr(desktop.Path, "home", lambda: tmp_path)
    found = desktop.find_app_browser()
    assert found == chrome or (found is not None and found.is_file())


def test_finds_chromium_on_linux(monkeypatch):
    monkeypatch.setattr(desktop, "_platform", lambda: "linux")
    monkeypatch.setattr(desktop.shutil, "which", lambda name: sys.executable if name == "chromium" else None)
    assert desktop.find_app_browser() == Path(sys.executable)


@pytest.mark.skipif(os.name == "nt", reason="POSIX browser profile lock")
def test_browser_profile_lock(tmp_path):
    assert not desktop._profile_in_use(tmp_path)
    (tmp_path / "SingletonLock").symlink_to(f"myhost-{os.getpid()}")
    assert desktop._profile_in_use(tmp_path)
    (tmp_path / "SingletonLock").unlink()
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    (tmp_path / "SingletonLock").symlink_to(f"myhost-{proc.pid}")
    assert not desktop._profile_in_use(tmp_path)  # left behind by a crashed browser


def test_waits_while_the_browser_profile_is_in_use(monkeypatch, tmp_path):
    states = iter([True, True, False])
    monkeypatch.setattr(desktop, "_profile_in_use", lambda profile: next(states))
    proc = FakeProc()
    desktop._wait_for_browser(proc, tmp_path, poll_s=0.01)
    assert proc.waited


def test_browser_launch_hides_console_windows(monkeypatch):
    seen: dict = {}

    def fake_popen(cmd, **kwargs):
        seen.update(kwargs, cmd=cmd)
        return FakeProc()

    monkeypatch.setattr(desktop.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(desktop, "_hidden_flags", lambda: desktop.CREATE_NO_WINDOW)
    desktop._launch_app_browser(["msedge.exe", "--app=x"])
    assert seen["creationflags"] == desktop.CREATE_NO_WINDOW
    assert seen["stdin"] == subprocess.DEVNULL


# -- shortcuts


def _fake_venv(tmp_path: Path, windows: bool) -> Path:
    scripts = tmp_path / ".venv" / ("Scripts" if windows else "bin")
    scripts.mkdir(parents=True)
    python = scripts / ("python.exe" if windows else "python")
    python.write_bytes(b"")
    if windows:
        (scripts / "pythonw.exe").write_bytes(b"")
    return python


def test_windows_shortcut_job(monkeypatch, tmp_path):
    python = _fake_venv(tmp_path, windows=True)
    monkeypatch.setattr(desktop, "_platform", lambda: "win32")
    monkeypatch.setattr(desktop.sys, "executable", str(python))
    monkeypatch.delenv("DESTINY_HOME")

    command, env, script = desktop.windows_shortcut_job(desktop=True, start_menu=False)

    assert command[1:] == ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File"]
    assert env["ATHENA_SC_TARGET"] == str(python.with_name("pythonw.exe"))
    assert env["ATHENA_SC_ARGS"] == "-m destiny_bot app"
    assert env["ATHENA_SC_NAME"] == "Athena"
    assert env["ATHENA_SC_DESKTOP"] == "1" and env["ATHENA_SC_START"] == "0"
    assert env["ATHENA_SC_ICON"].endswith("icon.ico")
    assert "[Environment]::GetFolderPath('Desktop')" in script  # OneDrive-redirected Desktops
    assert "[Environment]::GetFolderPath('Programs')" in script
    assert "WScript.Shell" in script


def test_windows_shortcut_keeps_the_launchers_data_folder(monkeypatch, tmp_path):
    python = _fake_venv(tmp_path, windows=True)
    home = r"C:\Users\Zoë O'Brien\Athena stuff\data"
    monkeypatch.setattr(desktop, "_platform", lambda: "win32")
    monkeypatch.setattr(
        desktop,
        "_launch_target",
        lambda: (python.with_name("pythonw.exe"), ["-m", "destiny_bot", "app"], {"DESTINY_HOME": home}),
    )

    _, env, _ = desktop.windows_shortcut_job(desktop=True, start_menu=True)

    argv = split_windows_command_line(env["ATHENA_SC_ARGS"])
    assert argv[0] == "-c" and argv[2:] == [home, "app"]
    code = argv[1]
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "home from shortcut"), "--version"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "0." in result.stdout  # `destiny --version` ran with the given data folder


def test_create_windows_shortcuts_runs_powershell_hidden(monkeypatch, tmp_path):
    python = _fake_venv(tmp_path, windows=True)
    monkeypatch.setattr(desktop, "_platform", lambda: "win32")
    monkeypatch.setattr(desktop.sys, "executable", str(python))
    seen: dict = {}

    def fake_run(cmd, *, env=None, timeout=60):
        ps1 = Path(cmd[-1])
        seen["script_bytes"] = ps1.read_bytes()
        seen["cmd"] = cmd
        seen["env"] = env
        out = "C:\\Users\\zoe\\OneDrive\\Desktop\\Athena.lnk\r\nC:\\Users\\zoe\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Athena.lnk\r\n"
        return subprocess.CompletedProcess(cmd, 0, out.encode("utf-8"), b"")

    monkeypatch.setattr(desktop, "_run_hidden", fake_run)
    paths = desktop.create_shortcuts()
    assert len(paths) == 2 and all(str(p).endswith("\\Athena.lnk") for p in paths)
    assert "OneDrive" in str(paths[0])
    assert seen["script_bytes"].startswith(b"\xef\xbb\xbf")  # UTF-8 BOM for Windows PowerShell 5
    assert seen["cmd"][-2] == "-File" and seen["cmd"][-1].endswith(".ps1")
    assert seen["env"]["ATHENA_SC_START"] == "1"


def test_create_windows_shortcuts_reports_failures(monkeypatch, tmp_path):
    python = _fake_venv(tmp_path, windows=True)
    monkeypatch.setattr(desktop, "_platform", lambda: "win32")
    monkeypatch.setattr(desktop.sys, "executable", str(python))
    monkeypatch.setattr(
        desktop,
        "_run_hidden",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, b"", b"Access to the path is denied.\r\n"),
    )
    with pytest.raises(desktop.ShortcutError, match="Access to the path is denied"):
        desktop.create_shortcuts()


def test_frozen_shortcut_targets_the_app(monkeypatch, tmp_path):
    exe = tmp_path / "Athena.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(desktop.sys, "executable", str(exe))
    assert desktop._launch_target() == (exe, [], {})


@pytest.mark.skipif(os.name == "nt", reason="builds a macOS bundle with POSIX permissions and symlinks")
def test_macos_shortcuts(monkeypatch, tmp_path):
    python = _fake_venv(tmp_path, windows=False)
    home = tmp_path / "home"
    (home / "Desktop").mkdir(parents=True)
    monkeypatch.setattr(desktop, "_platform", lambda: "darwin")
    monkeypatch.setattr(desktop.sys, "executable", str(python))
    monkeypatch.setattr(desktop.Path, "home", lambda: home)
    monkeypatch.setenv("DESTINY_HOME", str(tmp_path / "my data"))

    paths = desktop.create_shortcuts()

    bundle = home / "Applications" / "Athena.app"
    assert paths == [bundle, home / "Desktop" / "Athena"]
    assert (home / "Desktop" / "Athena").resolve() == bundle.resolve()
    info = plistlib.loads((bundle / "Contents" / "Info.plist").read_bytes())
    assert info["CFBundleExecutable"] == "Athena" and info["CFBundleName"] == "Athena"
    assert info["CFBundleIconFile"] == "icon"
    assert (bundle / "Contents" / "Resources" / "icon.icns").is_file()
    launcher = bundle / "Contents" / "MacOS" / "Athena"
    assert os.access(launcher, os.X_OK)
    script = launcher.read_text()
    assert script.startswith("#!/bin/bash")
    assert f"export DESTINY_HOME='{tmp_path / 'my data'}'" in script
    assert f"exec {python} -m destiny_bot app" in script

    assert desktop.create_shortcuts() == paths  # running it again replaces them

    other = home / "Applications" / "Athena.app"
    plistlib.dump({"CFBundleIdentifier": "com.someone.else"}, (other / "Contents" / "Info.plist").open("wb"))
    with pytest.raises(desktop.ShortcutError, match="already an app"):
        desktop.create_shortcuts()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_macos_desktop_only(monkeypatch, tmp_path):
    python = _fake_venv(tmp_path, windows=False)
    home = tmp_path / "home"
    (home / "Desktop").mkdir(parents=True)
    monkeypatch.setattr(desktop, "_platform", lambda: "darwin")
    monkeypatch.setattr(desktop.sys, "executable", str(python))
    monkeypatch.setattr(desktop.Path, "home", lambda: home)
    assert desktop.create_shortcuts(start_menu=False) == [home / "Desktop" / "Athena.app"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_linux_shortcuts(monkeypatch, tmp_path):
    python = _fake_venv(tmp_path, windows=False)
    desktop_dir = tmp_path / "Desktop"
    desktop_dir.mkdir()
    monkeypatch.setattr(desktop, "_platform", lambda: "linux")
    monkeypatch.setattr(desktop.sys, "executable", str(python))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    monkeypatch.setenv("DESTINY_HOME", str(tmp_path / "my data"))
    monkeypatch.setattr(desktop, "_linux_desktop_dir", lambda: desktop_dir)
    monkeypatch.setattr(desktop.shutil, "which", lambda name: None)

    paths = desktop.create_shortcuts()

    assert paths == [tmp_path / "share" / "applications" / "athena.desktop", desktop_dir / "athena.desktop"]
    text = paths[0].read_text()
    assert "Name=Athena" in text and "Terminal=false" in text
    exec_line = next(line for line in text.splitlines() if line.startswith("Exec="))
    assert exec_line == f'Exec=env "DESTINY_HOME={tmp_path / "my data"}" {python} -m destiny_bot app'
    assert os.access(paths[1], os.X_OK)
    assert desktop.create_shortcuts(desktop=False) == paths[:1]


def test_desktop_entry_quoting():
    assert desktop._desktop_entry_arg("/usr/bin/python3") == "/usr/bin/python3"
    assert desktop._desktop_entry_arg("50%") == "50%%"
    assert desktop._desktop_entry_arg("/a b/c") == '"/a b/c"'
    assert desktop._desktop_entry_arg('say "hi"') == '"say \\\\"hi\\\\""'
    assert desktop._desktop_entry_arg("$HOME") == '"\\\\$HOME"'


def test_shortcut_os_errors_are_friendly(monkeypatch):
    def broken(**kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(desktop, "_platform", lambda: "linux")
    monkeypatch.setattr(desktop, "_create_linux_shortcuts", broken)
    with pytest.raises(desktop.ShortcutError, match="Couldn't create the shortcuts: Permission denied"):
        desktop.create_shortcuts()


def test_cli_wires_to_these_functions():
    from destiny_bot import cli

    parser = cli.build_parser()
    assert parser.parse_args(["app", "--port", "9000"]).port == 9000
    assert parser.parse_args(["shortcut", "--no-start-menu"]).no_start_menu is True
