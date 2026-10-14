"""Tests for the ``destiny`` command (destiny_bot.cli)."""

from __future__ import annotations

import io
import struct
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from destiny_bot import __version__, cli, config, ingest
from destiny_bot.ingest import IngestError
from destiny_bot.ingest.common import doc_id_for_file
from destiny_bot.library import Library
from destiny_bot.models import Block, BlockKind, Document, Section, Utterance
from destiny_bot.tts.base import AudioClip, TTSEngine, Voice


def tiny_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<80h", *([500] * 80)))
    return buf.getvalue()


class FakeEngine(TTSEngine):
    def __init__(self, name: str, *, online: bool = False, available: bool = True, voices=("v1",)) -> None:
        self.name = name
        self.label = f"Fake {name}"
        self.online = online
        self.available = available
        self.voices = list(voices)
        self.calls: list[tuple[str, str | None, float]] = []

    def is_available(self):
        return (self.available, "" if self.available else f"{self.name} needs installing")

    def list_voices(self):
        return [Voice(id=v, name=v.upper(), engine=self.name, language="en-US") for v in self.voices]

    def synthesize(self, text, voice=None, rate=1.0):
        self.calls.append((text, voice, rate))
        return AudioClip(tiny_wav(), "audio/wav", 16000)


class FakeTTS:
    ENGINE_NAMES = ("piper", "system", "edge", "browser")

    def __init__(self) -> None:
        self.engines = {
            "piper": FakeEngine("piper"),
            "system": FakeEngine("system", voices=("sys1",)),
            "edge": FakeEngine("edge", online=True),
            "browser": FakeEngine("browser", voices=()),
        }

    def get_engine(self, name):
        return self.engines[name]

    def all_engines(self):
        return list(self.engines.values())

    def resolve_engine(self, preferred, *, online_allowed):
        for name in ("piper", "system", "browser"):
            if self.engines[name].available:
                return self.engines[name]
        return self.engines["browser"]


class FakeCache:
    def __init__(self, root) -> None:
        self.store: dict = {}

    def get_or_create(self, key_parts, make):
        if key_parts not in self.store:
            self.store[key_parts] = make()
        return self.store[key_parts]


def fake_plan(section: Section, options=None) -> list[Utterance]:
    plan = []
    for i, block in enumerate(section.blocks):
        if block.kind == BlockKind.TABLE:
            plan.append(Utterance(index=len(plan), block=i, start=0, end=len(block.text), kind=block.kind,
                                  speech="Table with 2 rows and 2 columns", pause_after_ms=0))
            continue
        for start, end in _sentences(block.text):
            plan.append(Utterance(index=len(plan), block=i, start=start, end=end, kind=block.kind,
                                  speech=block.text[start:end].replace("e.g.", "for example"), pause_after_ms=0))
    return plan


def _sentences(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for i, ch in enumerate(text):
        if ch == "." and (i + 1 == len(text) or text[i + 1] == " ") and not text[max(0, i - 3):i + 1].endswith("e.g."):
            spans.append((start, i + 1))
            start = i + 2
    if start < len(text):
        spans.append((start, len(text)))
    return spans


class CatalogVoice(BaseModel):
    id: str
    name: str
    language: str
    quality: str
    size_mb: float
    description: str


def make_doc(path: Path, name: str) -> Document:
    title = Path(name).stem.replace("_", " ").title()
    return Document(
        id=doc_id_for_file(path),
        title=title,
        source_name=name,
        source_type=ingest.source_type_for(name) or "txt",
        warnings=["Pages 3, 9 had no text layer and were read with OCR"] if "scan" in name else [],
        sections=[
            Section(id="s0", title="Cells", blocks=[
                Block(kind=BlockKind.HEADING, text="Cells", level=1),
                Block(kind=BlockKind.PARAGRAPH, text="Cells are small. They divide, e.g. by mitosis."),
                Block(kind=BlockKind.TABLE, text="A B 1 2", rows=[["A", "B"], ["1", "2"]]),
            ]),
            Section(id="s1", title="Membranes", level=2, page_start=4, page_end=6, blocks=[
                Block(kind=BlockKind.HEADING, text="Membranes", level=2),
                Block(kind=BlockKind.LIST_ITEM, text="Lipids form a bilayer."),
                Block(kind=BlockKind.PARAGRAPH, text="Read the notes.", meta={"speaker_notes": True}),
            ]),
        ],
    )


def fake_load_document(path, *, source_name=None, ocr=True, progress=None):
    path = Path(path)
    name = source_name or path.name
    if ingest.source_type_for(name) is None:
        raise IngestError(f"Destiny Bot can't open '{Path(name).suffix}' files yet.")
    if path.read_bytes().startswith(b"BROKEN"):
        raise IngestError("This file is damaged.")
    if progress:
        progress(0.5, "Reading")
    return make_doc(path, name)


@pytest.fixture
def fakes(monkeypatch):
    tts = FakeTTS()
    played: list[AudioClip] = []
    audio = SimpleNamespace(play_clip=played.append, trim_silence=lambda clip: clip)
    exports: list[dict] = []
    catalog = SimpleNamespace(
        RECOMMENDED_VOICE="en_US-lessac-medium",
        CATALOG=[CatalogVoice(id="en_US-lessac-medium", name="Lessac", language="en-US", quality="medium",
                              size_mb=63.2, description="Clear US English")],
        downloads=[],
    )

    def installed(voices_dir=None):
        folder = config.voices_dir() / "piper"
        return sorted(folder.glob("*.onnx")) if folder.is_dir() else []

    def download(voice_id, *, voices_dir=None, progress=None, base_url=None, cancel=None):
        catalog.downloads.append(voice_id)
        if progress:
            progress(1.0, "done")
        path = config.voices_dir() / "piper" / f"{voice_id}.onnx"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"onnx")
        return path

    catalog.installed_piper_voices = installed
    catalog.download_piper_voice = download

    def export(doc, *, engine, voice, options, out_dir, section_ids=None, fmt="mp3", rate=1.0, progress=None,
               cancel=None):
        exports.append(dict(doc=doc, engine=engine, voice=voice, out_dir=out_dir, section_ids=section_ids,
                            fmt=fmt, rate=rate))
        out = []
        for i, section in enumerate(doc.sections):
            if section_ids is None or section.id in section_ids:
                path = Path(out_dir) / f"{i + 1:02d} - {section.title}.{fmt}"
                path.write_bytes(b"audio")
                out.append(path)
        return out

    monkeypatch.setattr(cli, "_tts", lambda: tts)
    monkeypatch.setattr(cli, "_speech", lambda: SimpleNamespace(plan_section=fake_plan))
    monkeypatch.setattr(cli, "_audio", lambda: audio)
    monkeypatch.setattr(cli, "_audio_cache_class", lambda: FakeCache)
    monkeypatch.setattr(cli, "_export_audiobook", lambda: export)
    monkeypatch.setattr(cli, "_catalog", lambda: catalog)
    monkeypatch.setattr(cli, "_ocr_status", lambda: (True, "Tesseract 5 is installed."))
    monkeypatch.setattr(ingest, "load_document", fake_load_document)
    return SimpleNamespace(tts=tts, played=played, audio=audio, exports=exports, catalog=catalog)


def write(tmp_path: Path, name: str, text: str = "# Notes\nSome text.\n") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def add_book(tmp_path: Path, name: str, text: str | None = None) -> Document:
    return Library().import_file(write(tmp_path, name, text or f"content of {name}"))


# -- library commands


def test_list_empty(capsys, fakes):
    assert cli.main(["list"]) == 0
    assert "Your library is empty" in capsys.readouterr().out


def test_import_and_list(tmp_path, capsys, fakes):
    a = write(tmp_path, "cell_biology.md")
    b = write(tmp_path, "scanned_handout.png", "image-bytes")
    assert cli.main(["import", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "Imported: Cell Biology (2 sections)" in out
    assert "Scanned Handout" in out
    assert "Note: Pages 3, 9 had no text layer and were read with OCR" in out

    Library().set_progress(doc_id_for_file(a), "s1", 2)
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert doc_id_for_file(a) in out and "Cell Biology" in out and "Scanned Handout" in out
    assert "at s1, sentence 3" in out


def test_import_errors(tmp_path, capsys, fakes):
    good = write(tmp_path, "good.md")
    bad = write(tmp_path, "bad.md", "BROKEN")
    code = cli.main(["import", str(good), str(bad), str(tmp_path / "missing.pdf"), str(write(tmp_path, "x.exe"))])
    captured = capsys.readouterr()
    assert code == 1
    assert "Imported: Good" in captured.out
    assert "Couldn't import 'bad.md': This file is damaged." in captured.err
    assert "missing.pdf': there's no such file" in captured.err
    assert "can't open '.exe'" in captured.err
    assert "Traceback" not in captured.err
    assert len(Library().list()) == 1


def test_toc(tmp_path, capsys, fakes):
    doc = add_book(tmp_path, "biology.md")
    assert cli.main(["toc", doc.id]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "Biology (2 sections)"
    assert out[1].split() == ["s0", "Cells"]
    assert out[2].startswith("  s1    Membranes") and out[2].endswith("(pp. 4-6)")


def test_text(tmp_path, capsys, fakes):
    doc = add_book(tmp_path, "biology.md")
    assert cli.main(["text", "biol", "--section", "s1"]) == 0
    out = capsys.readouterr().out
    assert "## Membranes" in out and "- Lipids form a bilayer." in out
    assert "[Speaker notes] Read the notes." in out
    assert "Cells are small" not in out

    assert cli.main(["text", doc.id]) == 0
    out = capsys.readouterr().out
    assert "=== Cells [s0] ===" in out and "=== Membranes [s1] ===" in out
    assert "A | B\n1 | 2" in out

    assert cli.main(["text", doc.id, "--section", "s0", "--speech"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["Cells", "Cells are small.", "They divide, for example by mitosis.",
                     "Table with 2 rows and 2 columns"]


def test_text_unknown_section(tmp_path, capsys, fakes):
    doc = add_book(tmp_path, "biology.md")
    assert cli.main(["text", doc.id, "--section", "s9"]) == 1
    err = capsys.readouterr().err
    assert "has no section 's9'" in err and "destiny toc" in err and "Traceback" not in err


def test_text_of_a_file_imports_it(tmp_path, capsys, fakes):
    path = write(tmp_path, "lecture_notes.txt")
    assert cli.main(["text", str(path), "--section", "s0"]) == 0
    out = capsys.readouterr().out
    assert "Using: Lecture Notes" in out and "# Cells" in out
    assert [b.id for b in Library().list()] == [doc_id_for_file(path)]


# -- book resolution


def test_book_resolution(tmp_path, capsys, fakes):
    intro = add_book(tmp_path, "intro_biology.md")
    advanced = add_book(tmp_path, "advanced_biology.md")
    chem = add_book(tmp_path, "chemistry.md")
    lib = Library()
    assert cli.resolve_book(lib, intro.id).id == intro.id
    assert cli.resolve_book(lib, chem.id[:4]).id == chem.id
    assert cli.resolve_book(lib, chem.id[:6].upper()).id == chem.id
    assert cli.resolve_book(lib, "CHEM").id == chem.id
    assert cli.resolve_book(lib, "intro biology").id == intro.id
    assert cli.resolve_book(lib, "Advanced").id == advanced.id

    with pytest.raises(cli.CLIError) as info:
        cli.resolve_book(lib, "biology")
    message = str(info.value)
    assert "matches more than one book" in message
    assert intro.id in message and advanced.id in message and chem.id not in message

    with pytest.raises(cli.CLIError, match="No book in your library matches"):
        cli.resolve_book(lib, "physics")
    with pytest.raises(cli.CLIError, match="no file"):
        cli.resolve_book(lib, "physics.pdf")


def test_ambiguous_book_exit_code(tmp_path, capsys, fakes):
    add_book(tmp_path, "intro_biology.md")
    add_book(tmp_path, "advanced_biology.md")
    assert cli.main(["toc", "biology"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: 'biology' matches more than one book.")
    assert "Intro Biology" in err and "Advanced Biology" in err


def test_ambiguous_id_prefix(tmp_path):
    from destiny_bot.library import BookSummary

    def summary(book_id: str, title: str) -> BookSummary:
        return BookSummary(id=book_id, title=title, source_name="x.md", source_type="md",
                           added_at="2026-10-14T02:00:00.000Z", section_count=1)

    books = [summary("abcd111111111111", "Physics"), summary("abcd222222222222", "Algebra")]
    lib = SimpleNamespace(list=lambda: books, get=lambda book_id: SimpleNamespace(id=book_id))
    with pytest.raises(cli.CLIError, match="more than one") as info:
        cli.resolve_book(lib, "abcd")
    assert "abcd111111111111  Physics" in str(info.value)
    assert cli.resolve_book(lib, "abcd2").id == "abcd222222222222"
    # Fewer than 4 characters is a title search, not an id prefix.
    with pytest.raises(cli.CLIError, match="No book"):
        cli.resolve_book(lib, "abc")


# -- reading aloud


def test_read_plays_and_saves_progress(tmp_path, capsys, fakes):
    doc = add_book(tmp_path, "biology.md")
    assert cli.main(["read", doc.id, "--rate", "1.5"]) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line]
    assert "with Fake piper" in lines[0]
    assert lines[1:] [:5] == ["== Cells ==", "Cells", "Cells are small.", "They divide, e.g. by mitosis.",
                              "Table with 2 rows and 2 columns"]
    assert "Lipids form a bilayer." in out and "end of the book" in out
    assert len(fakes.played) == 7
    piper = fakes.tts.engines["piper"]
    assert piper.calls[2] == ("They divide, for example by mitosis.", "v1", 1.5)
    progress = Library().get_progress(doc.id)
    assert (progress.section_id, progress.utterance) == ("s1", 2)


def test_read_resumes_and_stops_cleanly(tmp_path, capsys, fakes):
    doc = add_book(tmp_path, "biology.md")
    Library().set_progress(doc.id, "s0", 2)
    calls = {"n": 0}

    def play(clip):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt

    fakes.audio.play_clip = play
    assert cli.main(["read", "biology"]) == 0
    out = capsys.readouterr().out
    assert "Continuing where you left off." in out
    assert "Cells are small." not in out and "They divide, e.g. by mitosis." in out
    assert "Stopped. Your place is saved" in out
    progress = Library().get_progress(doc.id)
    # The interrupted sentence is where reading picks up next time.
    assert (progress.section_id, progress.utterance) == ("s1", 0)


def test_read_section_option_overrides_progress(tmp_path, capsys, fakes):
    doc = add_book(tmp_path, "biology.md")
    Library().set_progress(doc.id, "s0", 1)
    assert cli.main(["read", doc.id, "--section", "s1", "--engine", "system", "--voice", "sys1"]) == 0
    out = capsys.readouterr().out
    assert "== Cells ==" not in out and "== Membranes ==" in out
    assert fakes.tts.engines["system"].calls[0] == ("Membranes", "sys1", 1.0)


def test_read_voice_choice(tmp_path, capsys, fakes):
    doc = add_book(tmp_path, "biology.md")
    piper = fakes.tts.engines["piper"]
    assert cli.main(["read", doc.id, "--voice", "x9"]) == 1
    err = capsys.readouterr().err
    assert "The voice 'x9' isn't installed" in err and "Installed voices: v1" in err
    assert piper.calls == []

    config.update_settings({"voices": {"piper": "uninstalled-voice"}})
    assert cli.main(["read", doc.id, "--section", "s1"]) == 0
    assert {call[1] for call in piper.calls} == {"v1"}  # a saved voice that's gone falls back to the default

    piper.voices = []
    assert cli.main(["read", doc.id]) == 1
    assert "has no voices installed" in capsys.readouterr().err


def test_read_engine_errors(tmp_path, capsys, fakes):
    doc = add_book(tmp_path, "biology.md")
    fakes.tts.engines["piper"].available = False
    fakes.tts.engines["system"].available = False
    assert cli.main(["read", doc.id]) == 1
    err = capsys.readouterr().err
    assert "destiny voices download en_US-lessac-medium" in err

    assert cli.main(["read", doc.id, "--engine", "edge"]) == 1
    assert "Online voices are turned off" in capsys.readouterr().err
    assert fakes.tts.engines["edge"].calls == []

    assert cli.main(["read", doc.id, "--engine", "robot"]) == 1
    assert "no voice engine called 'robot'" in capsys.readouterr().err
    assert cli.main(["read", doc.id, "--engine", "system"]) == 1
    assert "system needs installing" in capsys.readouterr().err
    assert fakes.played == []


# -- export


def test_export(tmp_path, capsys, fakes, monkeypatch):
    doc = add_book(tmp_path, "biology.md")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["export", doc.id, "--section", "s1", "--format", "wav", "--rate", "1.25"]) == 0
    out = capsys.readouterr().out
    call = fakes.exports[0]
    assert call["section_ids"] == ["s1"] and call["fmt"] == "wav" and call["rate"] == 1.25
    assert call["out_dir"] == tmp_path / "Biology"
    assert "Saved 1 file in" in out and "02 - Membranes.wav" in out

    out_dir = tmp_path / "custom out"
    assert cli.main(["export", doc.id, "--out", str(out_dir)]) == 0
    assert sorted(p.name for p in out_dir.iterdir()) == ["01 - Cells.mp3", "02 - Membranes.mp3"]
    assert cli.main(["export", doc.id, "--section", "s7"]) == 1
    assert cli.main(["export", doc.id, "--format", "ogg"]) == 2


# -- voices


def test_voices(capsys, fakes):
    assert cli.main(["voices"]) == 0
    out = capsys.readouterr().out
    assert "piper: Fake piper  (ready)" in out and "v1  V1  (en-US)" in out
    assert "edge: Fake edge (online)\n  Online voices are turned off" in out
    assert cli.main(["voices", "--engine", "system"]) == 0
    out = capsys.readouterr().out
    assert "sys1" in out and "piper" not in out
    assert cli.main(["voices", "--engine", "robot"]) == 1


def test_voices_catalog_and_download(capsys, fakes):
    assert cli.main(["voices", "catalog"]) == 0
    out = capsys.readouterr().out
    assert "en_US-lessac-medium" in out and "63 MB" in out and "(recommended)" in out
    assert cli.main(["voices", "download", "en_US-lessac-medium"]) == 0
    assert fakes.catalog.downloads == ["en_US-lessac-medium"]
    assert "Done." in capsys.readouterr().out
    assert cli.main(["voices", "catalog"]) == 0
    assert "installed" in capsys.readouterr().out
    assert cli.main(["voices", "download", "../evil"]) == 1
    assert "isn't a voice you can download" in capsys.readouterr().err
    fakes.catalog.is_valid_voice_id = lambda voice_id: voice_id == "de_DE-thorsten-medium"
    assert cli.main(["voices", "download", "de_DE-thorsten-medium"]) == 0  # not curated, but a real Piper voice
    assert fakes.catalog.downloads[-1] == "de_DE-thorsten-medium"


# -- settings, doctor, wipe


def test_ai_key(capsys, fakes, monkeypatch):
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt="": "  sk-ant-super-secret  ")
    assert cli.main(["ai-key"]) == 0
    out = capsys.readouterr().out
    assert "sk-ant-super-secret" not in out
    assert "still turned off" in out
    assert config.load_settings().anthropic_api_key == "sk-ant-super-secret"

    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt="": "")
    assert cli.main(["ai-key"]) == 1
    assert config.load_settings().anthropic_api_key == "sk-ant-super-secret"

    assert cli.main(["ai-key", "--remove"]) == 0
    assert config.load_settings().anthropic_api_key is None


def test_doctor(tmp_path, capsys, fakes):
    add_book(tmp_path, "biology.md")
    config.update_settings({"anthropic_api_key": "sk-ant-never-print-me"})
    fakes.tts.engines["system"].available = False
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert f"Athena {__version__}" in out
    assert f"Python {sys.version_info.major}.{sys.version_info.minor}" in out
    assert f"Data folder: {config.data_dir()}" in out
    assert "Library: 1 book" in out
    assert "piper (Fake piper): ready" in out
    assert "system (Fake system): not available: system needs installing" in out
    assert "online voices are turned off" in out
    assert "OCR): ready" in out
    assert "AI study helpers: off; API key set" in out
    assert "sk-ant-never-print-me" not in out


def test_wipe(tmp_path, capsys, fakes):
    add_book(tmp_path, "biology.md")
    config.update_settings({"anthropic_api_key": "sk-x"})
    voice = config.voices_dir() / "piper" / "keep.onnx"
    voice.parent.mkdir(parents=True)
    voice.write_bytes(b"v")
    assert cli.main(["wipe"]) == 1  # stdin isn't a terminal here: --yes is required
    assert "--yes" in capsys.readouterr().err
    assert len(Library().list()) == 1
    assert cli.main(["wipe", "--yes"]) == 0
    assert "erased" in capsys.readouterr().out
    assert Library().list() == []
    assert config.load_settings().anthropic_api_key is None
    assert voice.exists()


def test_wipe_interactive(tmp_path, capsys, fakes, monkeypatch):
    add_book(tmp_path, "biology.md")
    monkeypatch.setattr(cli, "_isatty", lambda stream: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "nope")
    assert cli.main(["wipe"]) == 1
    assert len(Library().list()) == 1
    monkeypatch.setattr("builtins.input", lambda prompt="": "DELETE")
    assert cli.main(["wipe"]) == 0
    assert Library().list() == []


# -- serve and entry points


def test_serve_defaults(monkeypatch):
    from destiny_bot import server

    calls = []
    monkeypatch.setattr(server, "run", lambda **kwargs: calls.append(kwargs))
    assert cli.main([]) == 0
    assert cli.main(["serve", "--port", "9000", "--no-browser"]) == 0
    assert cli.main(["serve", "--host", "0.0.0.0"]) == 0
    assert calls == [
        {"host": "127.0.0.1", "port": 8765, "open_browser": True, "allow_remote": False},
        {"host": "127.0.0.1", "port": 9000, "open_browser": False, "allow_remote": False},
        {"host": "0.0.0.0", "port": 8765, "open_browser": True, "allow_remote": True},
    ]
    assert cli.main(["serve", "--port", "70000"]) == 2


def _fake_module(name: str, **attrs):
    import types

    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def test_app_shortcut_and_self_test(monkeypatch, capsys, tmp_path):
    calls = []

    def run_app(*, port):
        calls.append(("app", port))
        return 0

    def create_shortcuts(*, desktop, start_menu):
        calls.append(("shortcut", desktop, start_menu))
        return [tmp_path / "Destiny Bot.lnk"] + ([tmp_path / "Start Menu" / "Destiny Bot.lnk"] if start_menu else [])

    def run_self_test(*, quick):
        calls.append(("self-test", quick))
        return 3

    monkeypatch.setitem(sys.modules, "destiny_bot.desktop",
                        _fake_module("destiny_bot.desktop", run_app=run_app, create_shortcuts=create_shortcuts))
    monkeypatch.setitem(sys.modules, "destiny_bot.selftest",
                        _fake_module("destiny_bot.selftest", run_self_test=run_self_test))

    assert cli.main(["app"]) == 0
    assert cli.main(["app", "--port", "9100"]) == 0
    assert cli.main(["shortcut"]) == 0
    out = capsys.readouterr().out
    assert f"Created {tmp_path / 'Destiny Bot.lnk'}" in out and "Start Menu" in out
    assert cli.main(["shortcut", "--no-start-menu"]) == 0
    assert "Start Menu" not in capsys.readouterr().out
    assert cli.main(["self-test"]) == 3
    assert cli.main(["self-test", "--quick"]) == 3
    assert calls == [("app", 8765), ("app", 9100), ("shortcut", True, True), ("shortcut", True, False),
                     ("self-test", False), ("self-test", True)]


def test_app_commands_without_their_modules(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "destiny_bot.desktop", None)
    monkeypatch.setitem(sys.modules, "destiny_bot.selftest", None)
    for argv in (["app"], ["shortcut"], ["self-test", "--quick"]):
        assert cli.main(argv) == 1, argv
        err = capsys.readouterr().err
        assert err.startswith("Error: ") and "isn't available" in err and "Traceback" not in err


def test_server_refuses_remote_without_permission():
    from destiny_bot import server

    with pytest.raises(ValueError, match="Refusing"):
        server.run(host="0.0.0.0", open_browser=False)


def test_unexpected_errors_show_details(tmp_path, capsys, fakes, monkeypatch):
    def boom(args):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli, "_cmd_list", boom)
    assert cli.main(["list"]) == 1
    err = capsys.readouterr().err
    assert "Something went wrong: kaboom" in err and "Traceback" in err


def test_usage_errors(capsys):
    assert cli.main(["frobnicate"]) == 2
    assert cli.main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_python_dash_m():
    result = subprocess.run([sys.executable, "-m", "destiny_bot", "--version"], capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 0
    assert result.stdout.strip() == f"Athena {__version__}"


# -- integration with the real modules (skipped while they aren't available)


def _ready(name: str, *attrs: str) -> bool:
    try:
        module = __import__(name, fromlist=["_"])
    except Exception:
        return False
    return all(hasattr(module, a) for a in attrs)


@pytest.mark.skipif(not (_ready("destiny_bot.speech", "plan_section") and _ready("destiny_bot.ingest.text", "load_text_file")),
                    reason="the real speech planner or text loader isn't available yet")
def test_integration_text_speech(tmp_path, capsys):
    path = write(tmp_path, "notes.md", "# Cells\n\nCells divide, e.g. by mitosis. See Fig. 2.\n")
    assert cli.main(["text", str(path), "--speech"]) == 0
    out = capsys.readouterr().out
    spoken = out.split("\n", 1)[1]  # after the "Using: ..." line
    assert "Cells" in spoken and "for example" in spoken and "e.g." not in spoken


@pytest.mark.skipif(not _ready("destiny_bot.tts", "all_engines"), reason="destiny_bot.tts isn't available yet")
def test_integration_doctor(capsys):
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "Voices:" in out and "piper" in out and "browser" in out


def _system_voice_ready() -> bool:
    if not _ready("destiny_bot.tts", "get_engine"):
        return False
    try:
        from destiny_bot.tts import get_engine

        return bool(get_engine("system").is_available()[0])
    except Exception:
        return False


@pytest.mark.skipif(not (_ready("destiny_bot.speech", "plan_section") and _system_voice_ready()),
                    reason="the real speech planner or a system voice isn't available")
def test_integration_read_with_real_voice(tmp_path, capsys, monkeypatch):
    from destiny_bot import audio

    played: list[AudioClip] = []
    monkeypatch.setattr(audio, "play_clip", played.append)
    monkeypatch.setattr(ingest, "load_document", fake_load_document)
    doc = add_book(tmp_path, "biology.md")
    assert cli.main(["read", doc.id, "--engine", "system", "--section", "s1"]) == 0
    out = capsys.readouterr().out
    assert "Lipids form a bilayer." in out
    assert played and all(clip.data for clip in played)
    assert Library().get_progress(doc.id).section_id == "s1"
