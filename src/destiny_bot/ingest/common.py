"""Helpers shared by every file loader."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

from ..models import Block, BlockKind, Section

# Characters PDF/EPUB extraction commonly leaves behind.
_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}
_INVISIBLE = dict.fromkeys(map(ord, "­​‌‍⁠﻿"), None)  # soft hyphen, zero-widths, BOM
_SPACES = re.compile(r"[ \t  -   　]+")
# "infor-\nmation" -> "information" (only when the next line continues a lowercase word)
_HYPHEN_BREAK = re.compile(r"(\w)[-‐‑]\s*\n\s*([a-z])")
_NEWLINES = re.compile(r"\s*\n\s*")

# A section longer than this many characters is split into parts (~40 minutes of speech).
MAX_SECTION_CHARS = 40_000


def doc_id_for_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def doc_id_for_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def fix_characters(text: str) -> str:
    """Repair ligatures and drop invisible characters; keeps line breaks."""
    for lig, repl in _LIGATURES.items():
        text = text.replace(lig, repl)
    text = text.translate(_INVISIBLE)
    # Compose accents ("e" + combining acute -> "é") without NFKC's lossy mappings (keeps "²", "½", ...).
    return unicodedata.normalize("NFC", text)


def clean_display_text(text: str) -> str:
    """Turn raw extracted text into a single clean line for display and reading.

    Repairs ligatures, removes soft hyphens / zero-width characters, rejoins words
    hyphenated across line breaks, turns remaining line breaks into spaces and
    collapses runs of whitespace.
    """
    text = fix_characters(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _NEWLINES.sub(" ", text)
    text = _SPACES.sub(" ", text)
    return text.strip()


def title_from_filename(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[_]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem or "Untitled"


def _page_range(blocks: list[Block]) -> tuple[int | None, int | None]:
    pages = [b.page for b in blocks if b.page is not None]
    return (min(pages), max(pages)) if pages else (None, None)


def make_section(title: str, level: int, blocks: list[Block]) -> Section:
    start, end = _page_range(blocks)
    return Section(id="", title=title, level=level, blocks=blocks, page_start=start, page_end=end)


def sections_from_blocks(blocks: list[Block], fallback_title: str, split_level: int = 2) -> list[Section]:
    """Split a flat block list into sections at headings of level <= ``split_level``.

    Each heading stays as the first block of its section. Content before the first
    heading becomes a section titled ``fallback_title``.
    """
    sections: list[Section] = []
    current_title, current_level, current = fallback_title, 1, []
    for block in blocks:
        if block.kind == BlockKind.HEADING and (block.level or 1) <= split_level and block.text.strip():
            if current:
                sections.append(make_section(current_title, current_level, current))
            current_title, current_level, current = block.text.strip(), block.level or 1, [block]
        else:
            current.append(block)
    if current:
        sections.append(make_section(current_title, current_level, current))
    return sections


def split_long_sections(sections: list[Section], max_chars: int = MAX_SECTION_CHARS) -> list[Section]:
    """Break very long sections into "Title (part N)" pieces at block boundaries."""
    result: list[Section] = []
    for sec in sections:
        total = sum(len(b.text) for b in sec.blocks)
        if total <= max_chars:
            result.append(sec)
            continue
        parts: list[list[Block]] = [[]]
        size = 0
        for block in sec.blocks:
            if size + len(block.text) > max_chars and parts[-1]:
                parts.append([])
                size = 0
            parts[-1].append(block)
            size += len(block.text)
        for i, chunk in enumerate(parts, start=1):
            title = sec.title if i == 1 else f"{sec.title} (part {i})"
            level = sec.level if i == 1 else sec.level + 1
            result.append(make_section(title, level, chunk))
    return result


def finalize_sections(sections: list[Section], fallback_title: str) -> list[Section]:
    """Drop empty blocks/sections, split huge sections, and assign ids s0, s1, ...

    Always returns at least one section.
    """
    cleaned: list[Section] = []
    for sec in sections:
        blocks = [b for b in sec.blocks if b.text.strip() or b.rows]
        if not blocks:
            continue
        start, end = _page_range(blocks)
        cleaned.append(
            sec.model_copy(
                update={
                    "blocks": blocks,
                    "title": sec.title.strip() or fallback_title,
                    "page_start": sec.page_start if sec.page_start is not None else start,
                    "page_end": sec.page_end if sec.page_end is not None else end,
                }
            )
        )
    cleaned = split_long_sections(cleaned)
    if not cleaned:
        cleaned = [Section(id="", title=fallback_title, level=1, blocks=[])]
    for i, sec in enumerate(cleaned):
        sec.id = f"s{i}"
    return cleaned
