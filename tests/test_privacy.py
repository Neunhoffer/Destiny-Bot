"""Privacy guarantees: nothing leaves the computer, nothing private lingers, the API key stays secret."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from destiny_bot import config
from destiny_bot.library import wipe_user_data

SRC = Path(__file__).resolve().parents[1] / "src"


def _run_python(code: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env={**env, "PYTHONPATH": str(SRC)},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_onnxruntime_telemetry_is_off_before_any_voice_loads(tmp_path):
    """onnxruntime's built-in 1DS telemetry phones home to mobile.events.data.microsoft.com as
    soon as it is imported, and keeps a device id under ~/.cache/Microsoft/DeveloperTools.
    ``disable_telemetry_events()`` is too late for that; only ``ORT_DISABLE_TELEMETRY=1`` set
    before the import (done in ``destiny_bot/__init__.py``) stops it."""
    pytest.importorskip("onnxruntime")
    home = tmp_path / "user"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "ORT_DISABLE_TELEMETRY"}
    env.update(HOME=str(home), XDG_CACHE_HOME=str(home / ".cache"), LOCALAPPDATA=str(home / "AppData" / "Local"),
               APPDATA=str(home / "AppData" / "Roaming"), USERPROFILE=str(home), DESTINY_HOME=str(tmp_path / "data"))
    code = """
        import time
        import destiny_bot.tts.kokoro_engine, onnxruntime
        time.sleep(2)  # the uploader starts in the background right after the import
        """
    strace = shutil.which("strace") if sys.platform.startswith("linux") else None
    trace = tmp_path / "connect.log"
    if strace is not None:
        command = [strace, "-f", "-qq", "-e", "trace=connect", "-o", str(trace), sys.executable, "-c", textwrap.dedent(code)]
        result = subprocess.run(command, env={**env, "PYTHONPATH": str(SRC)}, capture_output=True, text=True,
                                timeout=120, check=False)
        if result.returncode != 0 and "ptrace" in result.stderr.lower():
            strace = None  # not allowed to trace here: fall back to the file check alone
            result = _run_python(code, env)
    else:
        result = _run_python(code, env)
    assert result.returncode == 0, result.stderr
    leftovers = [str(p) for p in home.rglob("*") if "DeveloperTools" in str(p) or "onnxruntime" in p.name]
    assert leftovers == []
    if strace is not None:
        # The process needs no network at all: any IPv4/IPv6 connect (DNS included) is a leak.
        connects = [line for line in trace.read_text().splitlines() if "AF_INET" in line]
        assert connects == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions (on Windows %LOCALAPPDATA% is per-user)")
def test_data_folder_is_private_to_the_user(tmp_path, monkeypatch):
    old_umask = os.umask(0o022)
    try:
        fresh = tmp_path / "fresh" / "Athena"
        monkeypatch.setenv("DESTINY_HOME", str(fresh))
        assert stat.S_IMODE(config.data_dir().stat().st_mode) == 0o700

        existing = tmp_path / "existing"
        existing.mkdir(mode=0o755)
        existing.chmod(0o755)
        monkeypatch.setenv("DESTINY_HOME", str(existing))
        config.data_dir()
        assert stat.S_IMODE(existing.stat().st_mode) == 0o700
    finally:
        os.umask(old_umask)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_settings_file_is_owner_only():
    config.update_settings({"anthropic_api_key": "sk-ant-api03-SECRET-0123456789"})
    assert stat.S_IMODE(config.settings_path().stat().st_mode) == 0o600


def test_api_key_never_shows_in_settings_repr():
    settings = config.Settings(ai_enabled=True, anthropic_api_key="sk-ant-api03-SECRET-0123456789")
    assert "SECRET" not in repr(settings)
    assert "SECRET" not in str(settings)
    assert "SECRET" not in str(settings.public_dict())
    assert settings.anthropic_api_key == "sk-ant-api03-SECRET-0123456789"


@pytest.mark.parametrize("key", ["sk-ant-api03-AB CD", "sk-ant-api03-AB​CD", "sk-ant-api03-AB\nCD", "sk-ant\x00x"])
def test_api_key_with_spaces_or_invisible_characters_is_refused(key):
    with pytest.raises(ValueError, match="spaces or invisible characters"):
        config.update_settings({"anthropic_api_key": key})
    assert config.load_settings().anthropic_api_key is None
    # Leading/trailing whitespace from copy-paste is still simply trimmed.
    assert config.update_settings({"anthropic_api_key": "  sk-ant-api03-OK \n"}).anthropic_api_key == "sk-ant-api03-OK"


def test_wipe_also_empties_the_log_files():
    """The desktop app logs to <data>/logs; failed imports log the book's file name there."""
    logs = config.data_dir() / "logs"
    logs.mkdir()
    (logs / "app.log").write_text("INFO import job j_1 failed: 'Therapy notes.pdf' is damaged\n", encoding="utf-8")
    (logs / "app.log.1").write_text("older: 'Diary.docx'\n", encoding="utf-8")
    (logs / "crash.log").write_text("Fatal Python error\n", encoding="utf-8")
    with open(logs / "app.log", "a", encoding="utf-8") as live:  # the app keeps its log open
        wipe_user_data()
        live.write("after wipe\n")
    remaining = "".join(p.read_text(encoding="utf-8") for p in logs.iterdir() if p.is_file())
    assert "Therapy" not in remaining and "Diary" not in remaining and "Fatal" not in remaining


def _browser_storage(name: str) -> Path:
    storage = config.data_dir() / name / "EBWebView" / "Default" / "Local Storage" / "leveldb"
    storage.mkdir(parents=True)
    (storage / "000003.log").write_bytes(b"destiny.lastBook=3fa2")
    (storage / "LOCK").write_bytes(b"")
    return storage


def test_wipe_erases_the_reader_window_browser_data():
    for name in ("webview", "browser-profile"):
        _browser_storage(name)
    wipe_user_data()
    assert not (config.data_dir() / "webview").exists()
    assert not (config.data_dir() / "browser-profile").exists()


def test_browser_data_the_open_window_holds_is_erased_at_next_start(monkeypatch):
    """On Windows the open window locks some of its files; the wipe must still succeed."""
    from destiny_bot import library

    storage = _browser_storage("webview")
    real_unlink = os.unlink

    def locked(path, *args, **kwargs):
        if os.fspath(path).endswith("LOCK"):
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        return real_unlink(path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(os, "unlink", locked)
        m.setattr(os, "remove", locked)
        wipe_user_data()  # doesn't raise
    assert (storage / "LOCK").exists() and not (storage / "000003.log").exists()

    library.finish_pending_erase()  # next start, before the window opens
    assert not (config.data_dir() / "webview").exists()
    library.finish_pending_erase()  # nothing left to do
