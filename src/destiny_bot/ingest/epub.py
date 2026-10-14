"""EPUB e-books (EPUB 2 and EPUB 3).

Chapters are read in spine order. Section titles come from the book's own table
of contents (EPUB 3 nav or EPUB 2 NCX), including links to anchors inside a
chapter file; files the table of contents doesn't mention are split at their
headings. DRM-protected books are detected and refused with a clear message.
"""

from __future__ import annotations

import posixpath
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

from lxml import etree

from ..models import Block, BlockKind, Document, Section
from . import IngestError, ProgressFn
from .common import (
    clean_display_text,
    doc_id_for_file,
    finalize_sections,
    make_section,
    sections_from_blocks,
    title_from_filename,
)
from .html import HtmlContent, parse_html

# Font obfuscation is not DRM: it only scrambles embedded fonts.
_FONT_OBFUSCATION = frozenset({"http://www.idpf.org/2008/embedding", "http://ns.adobe.com/pdf/enc#RC"})
_DRM_MESSAGE = (
    "This e-book is DRM-protected, so it can't be opened. Destiny Bot can only read DRM-free books. "
    "Check whether the store or your school offers a DRM-free EPUB or a PDF of this textbook."
)
_BROKEN_MESSAGE = "This EPUB file couldn't be opened. It may be damaged or not really an EPUB e-book."


def _local(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _check_drm(path: Path) -> None:
    """Raise :class:`IngestError` when the book's content is encrypted."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = {name.lower(): name for name in zf.namelist()}
            if "meta-inf/sinf.xml" in names:  # Apple FairPlay
                raise IngestError(_DRM_MESSAGE)
            name = names.get("meta-inf/encryption.xml")
            if name is None:
                return
            data = zf.read(name)
    except zipfile.BadZipFile as exc:
        raise IngestError(_BROKEN_MESSAGE) from exc
    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False)
    try:
        root = etree.fromstring(data, parser)
    except etree.XMLSyntaxError:
        if b"EncryptedData" in data:
            raise IngestError(_DRM_MESSAGE) from None
        return
    encrypted = [el for el in root.iter() if _local(el.tag) == "EncryptedData"]
    algorithms = [el.get("Algorithm", "") for el in root.iter() if _local(el.tag) == "EncryptionMethod"]
    if any(algorithm not in _FONT_OBFUSCATION for algorithm in algorithms) or (encrypted and not algorithms):
        raise IngestError(_DRM_MESSAGE)


def _ebooklib():
    import ebooklib
    from ebooklib import epub

    class TolerantReader(epub.EpubReader):
        """ebooklib's reader, but a missing file or a broken table of contents isn't fatal."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.missing: list[str] = []

        def read_file(self, name):
            try:
                return super().read_file(name)
            except KeyError:
                normalized = posixpath.normpath(name)
                if normalized in ("META-INF/container.xml", posixpath.normpath(self.opf_file or "")):
                    raise
                self.missing.append(normalized)
                return b""

        def _load_spine(self):
            try:
                super()._load_spine()
            except Exception:
                if not getattr(self.book, "spine", None):
                    raise
                self.book.toc = []

        def _parse_ncx(self, data):
            try:
                super()._parse_ncx(data)
            except Exception:
                self.book.toc = []

        def _parse_nav(self, data, base_path, navtype="toc"):
            try:
                super()._parse_nav(data, base_path, navtype=navtype)
            except Exception:
                if navtype == "toc" and not isinstance(getattr(self.book, "toc", None), (list, tuple)):
                    self.book.toc = []

    return ebooklib, epub, TolerantReader


@dataclass
class _TocEntry:
    order: int
    title: str
    level: int
    fragment: str


@dataclass
class _Builder:
    sections: list[Section] = field(default_factory=list)

    @property
    def current(self) -> Section | None:
        return self.sections[-1] if self.sections else None

    def start(self, title: str, level: int, blocks: list[Block]) -> None:
        self.sections.append(make_section(title, level, list(blocks)))

    def extend(self, blocks: list[Block], fallback_title: str) -> None:
        if not blocks:
            return
        if self.current is None:
            self.start(fallback_title, 1, blocks)
        else:
            self.current.blocks.extend(blocks)


def _flatten_toc(entries, level: int = 1) -> list[tuple[str, str, int]]:
    """``[(title, href, level)]`` from ebooklib's nested ``book.toc``."""
    flat: list[tuple[str, str, int]] = []
    for entry in entries or []:
        if isinstance(entry, tuple) and len(entry) == 2:
            head, children = entry
            href = getattr(head, "href", None)
            if href:
                flat.append((getattr(head, "title", "") or "", href, level))
            flat.extend(_flatten_toc(children, level + 1))
        elif isinstance(entry, (list, tuple)):
            flat.extend(_flatten_toc(entry, level))
        elif getattr(entry, "href", None):
            flat.append((getattr(entry, "title", "") or "", entry.href, level))
    return flat


def _split_href(href: str, base: str) -> tuple[str, str]:
    path, _, fragment = href.partition("#")
    path = unquote(path)
    if path:
        path = posixpath.normpath(posixpath.join(base, path) if base else path)
    return path, unquote(fragment)


def _metadata(book, name: str) -> list[str]:
    try:
        values = book.get_metadata("DC", name)
    except Exception:
        return []
    out: list[str] = []
    for value, _attrs in values:
        text = clean_display_text(value or "")
        if text and text not in out:
            out.append(text)
    return out


def _has_text(content: HtmlContent) -> bool:
    return any(block.kind != BlockKind.FIGURE for block in content.blocks)


def load_epub(path: Path, *, source_name: str, progress: ProgressFn | None = None) -> Document:
    """Load an EPUB e-book. Raises :class:`IngestError` for DRM-protected or broken files."""
    path = Path(path)
    _check_drm(path)
    ebooklib, epub, TolerantReader = _ebooklib()
    if progress:
        progress(0.0, "Opening e-book")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ebooklib's option / deprecation chatter
            reader = TolerantReader(str(path), {"ignore_ncx": True})
            book = reader.load()
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(_BROKEN_MESSAGE) from exc

    titles = _metadata(book, "title")
    title = titles[0] if titles else title_from_filename(source_name)
    creators = _metadata(book, "creator")
    author = ", ".join(creators[:3]) + (" and others" if len(creators) > 3 else "") if creators else None

    # Spine documents in reading order (the navigation document itself is skipped).
    documents = []
    for entry in getattr(book, "spine", None) or []:
        idref = entry[0] if isinstance(entry, (tuple, list)) else entry
        item = book.get_item_with_id(idref)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT or isinstance(item, epub.EpubNav):
            continue
        documents.append(item)
    if not documents:
        documents = [
            item
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
            if not isinstance(item, epub.EpubNav)
        ]

    # Table of contents, grouped by the document each entry points at.
    nav_item = next((item for item in book.get_items() if isinstance(item, epub.EpubNav)), None)
    ncx_item = next((item for item in book.get_items() if isinstance(item, epub.EpubNcx)), None)
    toc_base = "" if nav_item is not None or ncx_item is None else posixpath.dirname(ncx_item.file_name)
    toc_by_doc: dict[str, list[_TocEntry]] = {}
    for order, (entry_title, href, level) in enumerate(_flatten_toc(getattr(book, "toc", []))):
        doc_path, fragment = _split_href(href, toc_base)
        entry_title = clean_display_text(entry_title)
        if doc_path and entry_title:
            toc_by_doc.setdefault(doc_path, []).append(_TocEntry(order, entry_title, level, fragment))

    guide_skip = set()
    for ref in getattr(book, "guide", None) or []:
        if (ref.get("type") or "").lower() == "toc" and ref.get("href"):
            guide_skip.add(_split_href(ref["href"], "")[0])

    builder = _Builder()
    warnings_out: list[str] = []
    unreadable = 0
    total = len(documents)
    for index, item in enumerate(documents):
        if progress:
            progress(0.05 + 0.9 * index / max(1, total), f"Reading part {index + 1} of {total}")
        name = posixpath.normpath(item.file_name)
        if name in guide_skip:
            continue  # an HTML table-of-contents page: our sections replace it
        try:
            content = parse_html(item.content or b"")
        except Exception:
            unreadable += 1
            continue
        if not _has_text(content):
            label = f"{name} {item.get_id() or ''}".lower()
            if not content.blocks or index == 0 or "cover" in label:
                continue  # cover / image-only page
        blocks = content.blocks
        entries = toc_by_doc.get(name, [])
        fallback = content.title or title
        points: list[tuple[int, _TocEntry]] = []
        for entry in entries:
            position = content.anchors.get(entry.fragment) if entry.fragment else 0
            if position is None:
                position = 0 if not points else None
            if position is not None:
                points.append((position, entry))
        if points:
            points.sort(key=lambda point: (point[0], point[1].order))
            builder.extend(blocks[: points[0][0]], fallback)
            for i, (position, entry) in enumerate(points):
                end = points[i + 1][0] if i + 1 < len(points) else len(blocks)
                builder.start(entry.title, entry.level, blocks[position:end])
            continue
        # Not in the table of contents: continue the current chapter, splitting at headings.
        for section in sections_from_blocks(blocks, fallback):
            first = section.blocks[0] if section.blocks else None
            starts_with_heading = (
                first is not None and first.kind == BlockKind.HEADING and first.text.strip() == section.title
            )
            if starts_with_heading:
                builder.start(section.title, section.level, section.blocks)
            else:
                builder.extend(section.blocks, fallback)

    if any(name.lower().endswith((".xhtml", ".html", ".htm", ".xml")) for name in reader.missing):
        warnings_out.append("Some chapters listed inside this e-book were missing and were skipped.")
    if unreadable:
        warnings_out.append(f"{unreadable} part(s) of this e-book couldn't be read and were skipped.")
    if not any(block.kind != BlockKind.FIGURE for sec in builder.sections for block in sec.blocks):
        warnings_out.append(
            "No readable text was found in this e-book. It may contain only pictures of pages."
        )
    if progress:
        progress(1.0, "Done")
    return Document(
        id=doc_id_for_file(path),
        title=title,
        author=author,
        source_name=source_name,
        source_type="epub",
        sections=finalize_sections(builder.sections, title),
        warnings=warnings_out,
    )
