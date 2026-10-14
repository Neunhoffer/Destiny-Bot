"""Turn files (PDF, EPUB, Word, PowerPoint, HTML, text, images) into a :class:`Document`."""

from __future__ import annotations

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


def source_type_for(name: str) -> str | None:
    return SUPPORTED_EXTENSIONS.get(Path(name).suffix.lower())


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
