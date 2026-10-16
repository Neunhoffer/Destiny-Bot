"""Photos and scans of handouts (PNG, JPEG, TIFF, BMP, WebP), read with OCR.

The recognised text is split into headings, paragraphs and bullets with the
same rules as plain-text files. Each page of a multi-page TIFF becomes its own
section, "Page N".
"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import Block, Document, Section
from . import IngestError, ProgressFn
from .common import doc_id_for_file, finalize_sections, make_section, sections_from_blocks, title_from_filename
from .ocr import ocr_available, ocr_image
from .text import plain_text_blocks

_HAS_WORD_CHAR = re.compile(r"[^\W_]")


def _clean_ocr_text(text: str) -> str:
    """Drop OCR noise lines (stray rules, specks) that hold no letters or digits."""
    lines = [line if _HAS_WORD_CHAR.search(line) else "" for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines)


def _page_list(numbers: list[int]) -> str:
    return ", ".join(str(n) for n in numbers)


def load_image(path: Path, *, source_name: str, progress: ProgressFn | None = None) -> Document:
    """OCR a picture of text. Raises :class:`IngestError` when Tesseract isn't installed."""
    path = Path(path)
    ok, reason = ocr_available()
    if not ok:
        raise IngestError(f"Reading text from pictures needs OCR. {reason}")

    from PIL import Image, ImageSequence, UnidentifiedImageError

    pages: list[list[Block]] = []
    try:
        with Image.open(path) as image:
            multi_page = (image.format or "").upper() == "TIFF" and getattr(image, "n_frames", 1) > 1
            total = image.n_frames if multi_page else 1
            frames = ImageSequence.Iterator(image) if multi_page else [image]
            for number, frame in enumerate(frames, start=1):
                if progress:
                    progress(0.9 * (number - 1) / total, f"Recognising text on page {number} of {total}")
                frame.load()
                text = ocr_image(frame.copy())
                pages.append(plain_text_blocks(_clean_ocr_text(text)))
    except UnidentifiedImageError as exc:
        raise IngestError("This picture couldn't be opened. It may be damaged or in an unsupported format.") from exc
    except Image.DecompressionBombError as exc:
        raise IngestError("This picture is too large to read. Try a smaller copy of it.") from exc
    except OSError as exc:
        raise IngestError(f"This picture couldn't be read ({exc}). It may be damaged or incomplete.") from exc

    title = title_from_filename(source_name)
    sections: list[Section]
    if len(pages) == 1:
        sections = sections_from_blocks(pages[0], title)
    else:
        sections = [make_section(f"Page {number}", 1, blocks) for number, blocks in enumerate(pages, start=1)]

    warnings: list[str] = []
    empty = [number for number, blocks in enumerate(pages, start=1) if not blocks]
    if len(empty) == len(pages):
        warnings.append(
            "No text was found in this picture. Make sure the photo is sharp, well lit and taken straight on, "
            "then try again."
        )
    elif empty:
        warnings.append(f"No text was found on page{'s' if len(empty) > 1 else ''} {_page_list(empty)}.")
    if progress:
        progress(1.0, "Done")
    return Document(
        id=doc_id_for_file(path),
        title=title,
        source_name=source_name,
        source_type="image",
        page_count=len(pages),
        sections=finalize_sections(sections, title),
        warnings=warnings,
    )
