"""Tests for the local library (athena.library)."""

from __future__ import annotations

import errno
import json
import os
import re
import threading
import types
from pathlib import Path

import pytest

from athena import config, ingest, library as library_module
from athena.ingest import IngestError
from athena.ingest.common import doc_id_for_bytes, doc_id_for_file
from athena.library import Library, is_valid_doc_id, utc_now, wipe_user_data
from athena.models import Block, BlockKind, Document, Section

ISO_Z = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z")


def _fake_document(text: str, title: str, doc_id: str, source_name: str, source_type: str) -> Document:
    """A tiny Markdown-ish parser: '# ' lines start sections."""
    sections: list[Section] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            sections.append(
                Section(id="", title=heading, blocks=[Block(kind=BlockKind.HEADING, text=heading, level=1)])
            )
        else:
            if not sections:
                sections.append(Section(id="", title=title, blocks=[]))
            sections[-1].blocks.append(Block(kind=BlockKind.PARAGRAPH, text=line))
    if not sections:
        sections = [Section(id="", title=title, blocks=[])]
    for i, section in enumerate(sections):
        section.id = f"s{i}"
    return Document(
        id=doc_id,
        title=sections[0].title if sections[0].blocks and sections[0].blocks[0].kind == BlockKind.HEADING else title,
        source_name=source_name,
        source_type=source_type,
        sections=sections,
        warnings=["Fake warning"] if "WARN" in text else [],
    )


class FakeLoader:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []

    def load_document(self, path, *, source_name=None, ocr=True, progress=None):
        path = Path(path)
        name = source_name or path.name
        if ingest.source_type_for(name) is None:
            raise IngestError(f"Athena can't open '{Path(name).suffix}' files yet.")
        self.calls.append((path, name))
        if progress:
            progress(0.5, "Reading")
        text = path.read_text(encoding="utf-8")
        return _fake_document(text, Path(name).stem, doc_id_for_file(path), name, ingest.source_type_for(name))

    def document_from_text(self, text, title, *, source_type="paste"):
        return _fake_document(text, title, doc_id_for_bytes(text.encode("utf-8")), title, source_type)


@pytest.fixture
def loader(monkeypatch) -> FakeLoader:
    fake = FakeLoader()
    monkeypatch.setattr(ingest, "load_document", fake.load_document)
    monkeypatch.setattr(ingest, "document_from_text", fake.document_from_text)
    return fake


@pytest.fixture
def clock(monkeypatch):
    """Deterministic, strictly increasing timestamps."""
    state = {"n": 0}

    def fake_now() -> str:
        state["n"] += 1
        return f"2026-10-14T02:00:{state['n']:02d}.000Z"

    monkeypatch.setattr(library_module, "utc_now", fake_now)
    return state


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


BIO = "# Cells\nCells are the unit of life.\n# Membranes\nMembranes are lipid bilayers.\n"


def test_import_stores_document_meta_and_source(tmp_path, loader):
    lib = Library()
    src = write(tmp_path, "bio notes.md", BIO)
    doc = lib.import_file(src)

    assert doc.id == doc_id_for_file(src)
    assert is_valid_doc_id(doc.id)
    book_dir = config.library_dir() / doc.id
    assert (book_dir / "document.json").is_file()
    meta = json.loads((book_dir / "meta.json").read_text(encoding="utf-8"))
    assert ISO_Z.fullmatch(meta["added_at"])
    assert meta["last_opened"] == meta["added_at"]
    assert meta["progress"] is None and meta["bookmarks"] == []
    source = lib.source_file(doc.id)
    assert source is not None and source.name == "source.md"
    assert source.read_bytes() == src.read_bytes()
    assert [p.name for p in book_dir.iterdir() if p.name.startswith(".tmp-")] == []

    again = Library().get(doc.id)
    assert again == doc
    summary = lib.summary(doc.id)
    assert summary.model_dump() == {
        "id": doc.id,
        "title": "Cells",
        "author": None,
        "source_name": "bio notes.md",
        "source_type": "md",
        "added_at": meta["added_at"],
        "section_count": 2,
        "page_count": None,
        "progress": None,
    }


def test_import_uses_original_name_for_type_and_title(tmp_path, loader):
    upload = write(tmp_path, "upload-tmp", "Just a paragraph.\n")
    doc = Library().import_file(upload, original_name="Lecture 3.TXT")
    assert doc.source_name == "Lecture 3.TXT"
    assert Library().source_file(doc.id).name == "source.txt"


def test_import_reports_progress(tmp_path, loader):
    seen: list[tuple[float, str]] = []
    Library().import_file(write(tmp_path, "a.md", BIO), progress=lambda f, m: seen.append((f, m)))
    assert seen == [(0.5, "Reading"), (1.0, "Saving to your library")]


def test_import_cancelled_after_reading_saves_nothing(tmp_path, loader):
    """The progress callback is the job's cancel checkpoint: "Erase all my data" may have run meanwhile."""

    class Cancelled(Exception):
        pass

    def progress(fraction: float, message: str) -> None:
        if message.startswith("Saving"):
            raise Cancelled()

    lib = Library()
    with pytest.raises(Cancelled):
        lib.import_file(write(tmp_path, "a.md", BIO), progress=progress)
    assert lib.list() == [] and list(lib.root.iterdir()) == []


def test_import_dedups_by_content_and_refreshes_last_opened(tmp_path, loader, clock):
    lib = Library()
    first = lib.import_file(write(tmp_path, "a.md", BIO))
    other = lib.import_file(write(tmp_path, "b.md", "# Other\nText.\n"))
    assert [b.id for b in lib.list()] == [other.id, first.id]

    copy = write(tmp_path, "copy of a.md", BIO)
    messages: list[str] = []
    again = lib.import_file(copy, progress=lambda f, m: messages.append(m))
    assert again.id == first.id
    assert len(loader.calls) == 2  # the duplicate wasn't parsed again
    assert messages == ["This file is already in your library"]
    assert [b.id for b in lib.list()] == [first.id, other.id]
    assert len(lib.list()) == 2


def test_list_order_most_recent_first(tmp_path, loader, clock):
    lib = Library()
    ids = [lib.import_file(write(tmp_path, f"{n}.md", f"# Book {n}\nText {n}.\n")).id for n in "abc"]
    assert [b.id for b in lib.list()] == ids[::-1]
    lib.mark_opened(ids[0])
    assert [b.id for b in lib.list()] == [ids[0], ids[2], ids[1]]


def test_list_uses_cached_summary_without_parsing_documents(tmp_path, loader, monkeypatch):
    lib = Library()
    doc = lib.import_file(write(tmp_path, "a.md", BIO))

    def boom(*args, **kwargs):
        raise AssertionError("list() parsed document.json")

    monkeypatch.setattr(library_module.Document, "model_validate_json", boom)
    books = Library().list()  # a fresh instance has no in-memory cache
    assert [b.id for b in books] == [doc.id]
    assert books[0].title == "Cells"


def test_list_notices_a_changed_document(tmp_path, loader):
    lib = Library()
    doc = lib.import_file(write(tmp_path, "a.md", BIO))
    path = config.library_dir() / doc.id / "document.json"
    changed = doc.model_copy(update={"title": "Renamed Book Title"})
    path.write_text(changed.model_dump_json(), encoding="utf-8")
    assert Library().list()[0].title == "Renamed Book Title"
    assert lib.get(doc.id).title == "Renamed Book Title"


def test_progress_roundtrip(tmp_path, loader):
    lib = Library()
    doc = lib.import_file(write(tmp_path, "a.md", BIO))
    assert lib.get_progress(doc.id) is None
    progress = lib.set_progress(doc.id, "s1", 7)
    assert progress.section_id == "s1" and progress.utterance == 7
    assert ISO_Z.fullmatch(progress.updated_at)
    assert lib.get_progress(doc.id) == progress
    assert lib.summary(doc.id).progress == progress
    assert lib.list()[0].progress == progress

    with pytest.raises(ValueError):
        lib.set_progress(doc.id, "s9", 0)
    with pytest.raises(ValueError):
        lib.set_progress(doc.id, "s0", -1)
    with pytest.raises(KeyError):
        lib.set_progress("0123456789abcdef", "s0", 0)
    with pytest.raises(KeyError):
        lib.get_progress("0123456789abcdef")


def test_bookmarks(tmp_path, loader, clock):
    lib = Library()
    doc = lib.import_file(write(tmp_path, "a.md", BIO))
    first = lib.add_bookmark(doc.id, "s0", 2, "  Important   part ")
    second = lib.add_bookmark(doc.id, "s1", 0, "")
    assert re.fullmatch(r"[0-9a-f]{16}", first.id) and first.id != second.id
    assert first.label == "Important part"
    assert second.label == "Membranes"  # blank labels become the section title
    assert ISO_Z.fullmatch(first.created_at)
    assert lib.list_bookmarks(doc.id) == [first, second]

    lib.delete_bookmark(doc.id, first.id)
    assert lib.list_bookmarks(doc.id) == [second]
    with pytest.raises(KeyError):
        lib.delete_bookmark(doc.id, first.id)
    with pytest.raises(KeyError):
        lib.delete_bookmark(doc.id, "../meta")
    with pytest.raises(ValueError):
        lib.add_bookmark(doc.id, "nope", 0, "x")
    long = lib.add_bookmark(doc.id, "s0", 0, "x" * 500)
    assert len(long.label) == library_module.MAX_LABEL_CHARS


def test_concurrent_meta_updates_are_not_lost(tmp_path, loader):
    lib = Library()
    doc = lib.import_file(write(tmp_path, "a.md", BIO))

    def worker(n: int) -> None:
        for i in range(5):
            lib.add_bookmark(doc.id, "s0", n * 10 + i, f"mark {n}-{i}")
            lib.set_progress(doc.id, "s1", n)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(lib.list_bookmarks(doc.id)) == 30
    assert lib.get_progress(doc.id).section_id == "s1"


def test_delete(tmp_path, loader):
    lib = Library()
    doc = lib.import_file(write(tmp_path, "a.md", BIO))
    keep = lib.import_file(write(tmp_path, "b.md", "# B\nText.\n"))
    lib.delete(doc.id)
    assert not (config.library_dir() / doc.id).exists()
    assert [b.id for b in lib.list()] == [keep.id]
    with pytest.raises(KeyError):
        lib.get(doc.id)
    with pytest.raises(KeyError):
        lib.delete(doc.id)
    assert not [p for p in config.library_dir().iterdir() if p.name.startswith(".deleted-")]
    # Importing it again works after a delete.
    assert lib.import_file(write(tmp_path, "a2.md", BIO)).id == doc.id


def test_corrupt_document_is_skipped(tmp_path, loader):
    lib = Library()
    good = lib.import_file(write(tmp_path, "a.md", BIO))
    bad = lib.import_file(write(tmp_path, "b.md", "# Broken\nText.\n"))
    (config.library_dir() / bad.id / "document.json").write_text("{not json", encoding="utf-8")
    fresh = Library()
    assert [b.id for b in fresh.list()] == [good.id]
    with pytest.raises(KeyError):
        fresh.get(bad.id)
    with pytest.raises(KeyError):
        fresh.summary(bad.id)
    # Valid JSON that isn't a Document is skipped too.
    (config.library_dir() / bad.id / "document.json").write_text('{"id": 3}', encoding="utf-8")
    assert [b.id for b in Library().list()] == [good.id]
    # Re-importing the same file repairs it.
    assert fresh.import_file(write(tmp_path, "b2.md", "# Broken\nText.\n")).id == bad.id
    assert {b.id for b in fresh.list()} == {good.id, bad.id}


def test_corrupt_meta_resets_but_keeps_the_book(tmp_path, loader):
    lib = Library()
    doc = lib.import_file(write(tmp_path, "a.md", BIO))
    lib.set_progress(doc.id, "s1", 3)
    (config.library_dir() / doc.id / "meta.json").write_text("[[[", encoding="utf-8")
    fresh = Library()
    books = fresh.list()
    assert [b.id for b in books] == [doc.id]
    assert books[0].progress is None
    assert ISO_Z.fullmatch(books[0].added_at)
    assert fresh.get_progress(doc.id) is None
    assert fresh.list_bookmarks(doc.id) == []
    fresh.set_progress(doc.id, "s0", 1)
    assert fresh.get_progress(doc.id).utterance == 1


def test_stray_entries_are_ignored(tmp_path, loader):
    lib = Library()
    doc = lib.import_file(write(tmp_path, "a.md", BIO))
    root = config.library_dir()
    (root / "notes.txt").write_text("hello", encoding="utf-8")
    (root / "not-a-book").mkdir()
    (root / "0123456789abcdef").mkdir()  # valid id, but no document
    (root / "ABCDEF0123456789").mkdir()  # uppercase isn't a valid id
    assert [b.id for b in lib.list()] == [doc.id]


@pytest.mark.parametrize(
    "bad_id",
    ["..", "../..", "../../etc/passwd", "0123456789abcde", "0123456789abcdef0", "0123456789ABCDEF",
     "0123456789abcdeg", "0123456789abcdef/..", "/tmp", "", "0123456789abcde\n"],
)
def test_invalid_ids_are_rejected_everywhere(tmp_path, loader, bad_id):
    lib = Library()
    lib.import_file(write(tmp_path, "a.md", BIO))
    for call in (
        lambda: lib.get(bad_id),
        lambda: lib.summary(bad_id),
        lambda: lib.source_file(bad_id),
        lambda: lib.delete(bad_id),
        lambda: lib.get_progress(bad_id),
        lambda: lib.set_progress(bad_id, "s0", 0),
        lambda: lib.list_bookmarks(bad_id),
        lambda: lib.add_bookmark(bad_id, "s0", 0, "x"),
        lambda: lib.delete_bookmark(bad_id, "0123456789abcdef"),
        lambda: lib.mark_opened(bad_id),
    ):
        with pytest.raises(KeyError):
            call()
    assert len(lib.list()) == 1


def test_unsupported_and_missing_files(tmp_path, loader):
    lib = Library()
    with pytest.raises(IngestError, match="can't open"):
        lib.import_file(write(tmp_path, "virus.exe", "MZ"))
    with pytest.raises(IngestError, match="not found"):
        lib.import_file(tmp_path / "missing.pdf")
    assert lib.list() == []


def test_loader_errors_leave_no_book(tmp_path, monkeypatch):
    def failing(path, **kwargs):
        raise IngestError("This PDF is password-protected.")

    monkeypatch.setattr(ingest, "load_document", failing)
    lib = Library()
    with pytest.raises(IngestError, match="password"):
        lib.import_file(write(tmp_path, "a.pdf", "%PDF-1.7"))
    assert lib.list() == []


def test_import_text(loader):
    lib = Library()
    doc = lib.import_text("  My   notes ", "# Topic\nSome pasted text.\n")
    assert doc.source_type == "paste"
    assert lib.source_file(doc.id).read_text(encoding="utf-8") == "# Topic\nSome pasted text.\n"
    assert lib.list()[0].id == doc.id
    assert lib.import_text("Again", "# Topic\nSome pasted text.\n").id == doc.id
    assert len(lib.list()) == 1
    with pytest.raises(IngestError):
        lib.import_text("Empty", "   \n ")
    untitled = lib.import_text("", "Plain words.")
    assert untitled.title == "Pasted text"


def test_import_text_fixes_invalid_ids(monkeypatch):
    def bad_id(text, title, *, source_type="paste"):
        return _fake_document(text, title, "not-an-id", title, source_type)

    monkeypatch.setattr(ingest, "document_from_text", bad_id)
    doc = Library().import_text("T", "Hello there.")
    assert doc.id == doc_id_for_bytes("Hello there.".encode("utf-8"))
    assert Library().get(doc.id).id == doc.id


def test_custom_root_and_wipe(tmp_path, loader):
    root = tmp_path / "custom-library"
    lib = Library(root)
    doc = lib.import_file(write(tmp_path, "a.md", BIO))
    assert (root / doc.id / "document.json").is_file()
    lib.wipe()
    assert root.is_dir() and list(root.iterdir()) == []
    assert lib.list() == []


def test_wipe_user_data_keeps_voices(tmp_path, loader):
    lib = Library()
    lib.import_file(write(tmp_path, "a.md", BIO))
    config.update_settings({"anthropic_api_key": "sk-secret", "speed": 1.5})
    (config.cache_dir() / "clip.wav").write_bytes(b"RIFF")
    (config.exports_dir() / "j_1" / "x.mp3").parent.mkdir(parents=True)
    (config.exports_dir() / "j_1" / "x.mp3").write_bytes(b"ID3")
    library_module.temp_dir().joinpath("upload-x").mkdir()
    voice = config.voices_dir() / "piper" / "en_US-test.onnx"
    voice.parent.mkdir(parents=True)
    voice.write_bytes(b"onnx")

    wipe_user_data()

    assert not config.settings_path().exists()
    assert config.load_settings().anthropic_api_key is None
    for folder in (config.library_dir(), config.cache_dir(), config.exports_dir()):
        assert folder.is_dir() and list(folder.iterdir()) == []
    assert not (config.data_dir() / "tmp").exists()
    assert voice.read_bytes() == b"onnx"
    lib.clear_cache()
    assert lib.list() == []


def test_utc_now_format():
    assert ISO_Z.fullmatch(utc_now())


# -- integration with the real text loader (skipped while it isn't available)


def _real_text_loader_works() -> bool:
    try:
        from athena.ingest import text  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _real_text_loader_works(), reason="athena.ingest.text isn't available yet")
def test_real_markdown_import(tmp_path):
    lib = Library()
    src = write(tmp_path, "biology.md", "# Cells\n\nCells are small.\n\n## Membranes\n\nMembranes are lipid bilayers.\n")
    doc = lib.import_file(src)
    assert doc.id == doc_id_for_file(src)
    assert doc.source_type == "md"
    assert len(doc.sections) >= 1
    assert lib.list()[0].id == doc.id
    assert lib.get(doc.id) == doc
    pasted = lib.import_text("Pasted", "# Title\n\nSome text to read.\n")
    assert lib.summary(pasted.id).source_type == "paste"


# -- robustness: full disks, Windows file locks, concurrent imports


def _disk_full(*args, **kwargs):
    raise OSError(errno.ENOSPC, "No space left on device")


def test_disk_full_during_import_leaves_no_half_written_book(tmp_path, loader, monkeypatch):
    lib = Library()
    real_write = library_module._atomic_write

    def write_or_fail(path: Path, data: bytes) -> None:
        if path.name == "document.json":
            _disk_full()
        real_write(path, data)

    monkeypatch.setattr(library_module, "_atomic_write", write_or_fail)
    with pytest.raises(OSError) as info:
        lib.import_file(write(tmp_path, "big.md", BIO))
    assert info.value.errno == errno.ENOSPC
    assert list(lib.root.iterdir()) == []  # no hidden folder holding a copy of the book
    monkeypatch.setattr(library_module, "_atomic_write", real_write)
    doc = lib.import_file(write(tmp_path, "big.md", BIO))  # works once there is room again
    assert [b.id for b in lib.list()] == [doc.id]


def test_disk_full_while_saving_progress_keeps_the_old_progress(tmp_path, loader, monkeypatch):
    lib = Library()
    doc = lib.import_file(write(tmp_path, "a.md", BIO))
    lib.set_progress(doc.id, "s0", 4)
    with monkeypatch.context() as m, pytest.raises(OSError):
        m.setattr(library_module.os, "fsync", _disk_full)
        lib.set_progress(doc.id, "s1", 9)
    assert lib.get_progress(doc.id).utterance == 4
    assert [p.name for p in (lib.root / doc.id).iterdir() if p.name.startswith(".tmp-")] == []


def test_meta_writes_wait_for_a_briefly_locked_file(tmp_path, loader, monkeypatch):
    """Windows refuses to replace a file another program (a virus scanner) has open for a moment."""
    lib = Library()
    doc = lib.import_file(write(tmp_path, "a.md", BIO))
    real_replace, attempts = os.replace, []

    def replace(src, dst):
        attempts.append(Path(dst).name)
        if len(attempts) <= 2:
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        real_replace(src, dst)

    monkeypatch.setattr(config, "sys", types.SimpleNamespace(platform="win32"))
    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(config.time, "sleep", lambda seconds: None)
    lib.set_progress(doc.id, "s1", 2)
    assert attempts == ["meta.json"] * 3
    assert lib.get_progress(doc.id).utterance == 2


def test_concurrent_imports_of_the_same_file(tmp_path, loader):
    lib = Library()
    paths = [write(tmp_path, f"copy{i}.md", BIO) for i in range(8)]
    results: list[str] = []
    errors: list[BaseException] = []

    def run(path: Path) -> None:
        try:
            results.append(lib.import_file(path).id)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(p,)) for p in paths]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and len(set(results)) == 1
    assert [b.id for b in lib.list()] == results[:1]
    assert sorted(p.name for p in (lib.root / results[0]).iterdir()) == ["document.json", "meta.json", "source.md"]


def test_delete_while_the_book_is_being_read(tmp_path, loader):
    lib = Library()
    doc = lib.import_file(write(tmp_path, "a.md", BIO))
    opened = lib.get(doc.id)  # e.g. an export job holding the document
    lib.delete(doc.id)
    assert opened.sections[0].title == "Cells"  # the reader's copy still works
    with pytest.raises(KeyError):
        lib.set_progress(doc.id, "s0", 1)
    with pytest.raises(KeyError):
        lib.add_bookmark(doc.id, "s0", 1, "x")
    assert lib.list() == []


def test_mislabeled_pdf_is_stored_as_a_pdf(tmp_path):
    import pymupdf

    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "Chapter 1. Cells are the basic unit of life.", fontsize=12)
    path = tmp_path / "notes.docx"  # really a PDF
    pdf.save(path)
    pdf.close()
    lib = Library()
    doc = lib.import_file(path)
    assert doc.source_type == "pdf"
    assert lib.source_file(doc.id).suffix == ".pdf"
