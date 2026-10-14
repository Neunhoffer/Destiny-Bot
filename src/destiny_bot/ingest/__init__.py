"""Turn files (PDF, EPUB, Word, PowerPoint, HTML, text, images) into a :class:`Document`."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Callable

from ..models import Document

# progress(fraction 0..1, message)
ProgressFn = Callable[[float, str], None]


class IngestError(RuntimeError):
    """The file could not be read; the message is safe to show to the user."""


SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".epub": "epub",
    ".docx": "docx",
    ".pptx": "pptx",
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    ".txt": "txt",
    ".md": "md",
    ".markdown": "md",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tif": "image",
    ".tiff": "image",
    ".bmp": "image",
    ".webp": "image",
}


_KIND_LABELS = {
    "pdf": "a PDF",
    "epub": "an EPUB e-book",
    "docx": "a Word document",
    "pptx": "a PowerPoint presentation",
    "html": "a web page",
    "txt": "a text file",
    "md": "a Markdown file",
    "image": "a picture",
}


def source_type_for(name: str) -> str | None:
    return SUPPORTED_EXTENSIONS.get(Path(name).suffix.lower())


def sniff_kind(path: Path) -> str | None:
    """What the file really is, from its first bytes: a loader kind, "ole" (an older or
    password-protected Office file), or None when unknown (plain text, say)."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(1024)
    except OSError:
        return None
    if head.lstrip(b"\x00\t\r\n ").startswith(b"%PDF-"):
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as zf:
                names = set(zf.namelist())
        except Exception:
            return None
        if "word/document.xml" in names:
            return "docx"
        if "ppt/presentation.xml" in names:
            return "pptx"
        if "META-INF/container.xml" in names:
            return "epub"
        return None
    if head.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")) or (head[:4] == b"RIFF" and head[8:12] == b"WEBP"):
        return "image"
    if head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return "ole"
    start = head.lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    if start.startswith((b"<!doctype html", b"<html")):
        return "html"
    return None


def _real_kind(path: Path, name: str, kind: str) -> str:
    """The loader to use: the file's real type when its name says something else."""
    real = sniff_kind(path)
    if real is None or real == kind:
        return kind
    label = _KIND_LABELS.get(kind, "a file")
    if real == "ole":
        if kind in ("docx", "pptx"):
            return kind  # the Word / PowerPoint loader explains older and password-protected files
        raise IngestError(
            f"'{name}' isn't really {label}: it is an older Word or PowerPoint file (.doc, .ppt) or a "
            "password-protected one. Open it in Word or PowerPoint and use Save As to make a .docx or .pptx copy."
        )
    if real == "html":
        if kind in ("txt", "md"):
            return kind
        raise IngestError(
            f"'{name}' isn't really {label}: it is a web page, maybe a download or sign-in page that was saved "
            "instead of the real file. Open the link in your web browser, download the file again and import that."
        )
    return real


def load_document(
    path: Path,
    *,
    source_name: str | None = None,
    ocr: bool = True,
    progress: ProgressFn | None = None,
) -> Document:
    """Load any supported file. ``source_name`` is the original file name (uploads
    are saved to temp files with different names). Raises :class:`IngestError`."""
    path = Path(path)
    name = source_name or path.name
    kind = source_type_for(name) or source_type_for(path.name)
    if kind is None:
        raise IngestError(
            f"Athena can't open '{Path(name).suffix or name}' files yet. "
            "Supported: " + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )
    if not path.is_file():
        raise IngestError(f"File not found: {path}")
    named_kind = kind
    kind = _real_kind(path, name, kind)
    doc = _load(path, name, kind, ocr, progress)
    if kind != named_kind:
        doc.warnings.insert(
            0,
            f"This file's name says it is {_KIND_LABELS.get(named_kind, 'a file')}, but it is really "
            f"{_KIND_LABELS.get(kind, 'a file')}, so it was read as one.",
        )
    return doc


def _load(path: Path, name: str, kind: str, ocr: bool, progress: ProgressFn | None) -> Document:
    try:
        if kind == "pdf":
            from .pdf import load_pdf

            return load_pdf(path, source_name=name, ocr=ocr, progress=progress)
        if kind == "epub":
            from .epub import load_epub

            return load_epub(path, source_name=name, progress=progress)
        if kind == "docx":
            from .docx import load_docx

            return load_docx(path, source_name=name, progress=progress)
        if kind == "pptx":
            from .pptx import load_pptx

            return load_pptx(path, source_name=name, progress=progress)
        if kind == "html":
            from .html import load_html

            return load_html(path, source_name=name, progress=progress)
        if kind in ("txt", "md"):
            from .text import load_text_file

            return load_text_file(path, source_name=name, progress=progress)
        if kind == "image":
            from .image import load_image

            return load_image(path, source_name=name, progress=progress)
    except IngestError:
        raise
    except Exception as exc:  # a broken file must produce a friendly error, not a crash
        raise IngestError(f"Couldn't read '{name}': {exc}") from exc
    raise IngestError(f"Unsupported file type: {kind}")


def document_from_text(text: str, title: str, *, source_type: str = "paste") -> Document:
    """Build a document from pasted text (Markdown headings are recognised)."""
    from .text import document_from_text as _from_text

    return _from_text(text, title, source_type=source_type)
