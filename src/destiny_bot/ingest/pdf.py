"""Load PDF files (textbooks, articles, handouts) into a :class:`Document`.

The pipeline:

1. Extract every text line with its spans (font size, weight, position), find
   ruled tables and framed boxes, and OCR pages that are only scanned images.
2. Drop running headers, footers and page numbers: lines in the top/bottom
   margin bands whose normalized text repeats on nearby pages.
3. Per page: detect columns, put content in reading order, merge fragments that
   share a row, and group lines into chunks (paragraph candidates).
4. Classify chunks (heading, list item, caption, footnote, equation, code, ...),
   rank heading styles into levels, and stitch paragraphs that continue across
   column/page breaks. Floats (footnotes, captions, tables, boxes) that
   interrupt a paragraph are moved after it.
5. Build sections from the PDF outline, else from headings, else page ranges.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Callable

import pymupdf

from ..models import Block, BlockKind, Document, Section
from . import IngestError, ProgressFn
from .common import (
    clean_display_text,
    doc_id_for_file,
    finalize_sections,
    fix_characters,
    make_section,
    sections_from_blocks,
    title_from_filename,
)

if hasattr(pymupdf, "no_recommend_layout"):  # silence an advert printed to stdout by find_tables()
    pymupdf.no_recommend_layout()

_TEXT_FLAGS = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_PRESERVE_LIGATURES | pymupdf.TEXT_MEDIABOX_CLIP

_BAND = 0.12  # top/bottom share of the page where running headers and footers live
_OCR_DPI = 300
_OCR_MAX_PIXELS = 40_000_000  # cap for huge pages (posters) so rendering stays within memory
_PAGES_PER_PART = 10  # page-range sections for long documents without headings
_MAX_TABLE_DRAWINGS = 5000  # pages with more vector paths than this are figures, not tables

# --------------------------------------------------------------------------- patterns

_BULLETS = frozenset("•●○◦▪▫■□◆◇►▸▹▶‣⁃∙·✓✔✗➢➤➔→❖♦❑❒⦿⦾")
_NUMBERED_RE = re.compile(
    r"^(?:\d{1,3}[.)]|\(\d{1,3}\)|[a-z][.)]|\([a-zA-Z]\)|[A-Z]\)"
    r"|(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii)[.)]|\((?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii)\))\s+\S"
)
_DASH_BULLET_RE = re.compile(r"^[-–—*]\s+\S")
_CAPTION_RE = re.compile(
    r"^(?:fig(?:ure)?s?\.?|tables?|tab\.|exhibit|chart|graph|diagram|illustration|plate|map|photo|scheme|listing)"
    r"\s*(?:[A-Z]?\d+(?:[.\-–]\d+)*[a-z]?|[IVXLC]+\b)\s*(?P<rest>.*)$",
    re.I | re.S,
)
_FOOTNOTE_START_RE = re.compile(r"^(?:\d{1,3}|[*†‡§¶]{1,3})(?:[.)]?\s+|(?=[A-Z\"“(]))")
_REF_MARK_RE = re.compile(r"^[\d,\-–*†‡§¶]+$")
_ROMAN_RE = re.compile(r"^(?=[ivxlcdm]+$)m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})$", re.I)
_PAGE_NUM_RE = re.compile(
    r"^(?:page|pg\.?|p\.)?\s*[-–—]?\s*(?P<n>\d{1,4}|[ivxlcdm]{1,8})\s*[-–—]?\s*(?:(?:of|/)\s*\d{1,4})?$", re.I
)
_PAGE_NUM_SIDE_RE = re.compile(
    r"^(?P<a>\d{1,4}|[ivxlcdm]{1,8})\s*[|•·]\s*\S.{0,120}$|^.{0,120}\S\s*[|•·]\s*(?P<b>\d{1,4}|[ivxlcdm]{1,8})$", re.I
)
_CHAPTER_LABEL_RE = re.compile(
    r"^(?:chapter|unit|part|module|lesson|book|section|appendix|topic|week)\s+"
    r"(?:\d{1,3}|[ivxlc]{1,7}|[a-z]|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\.?$",
    re.I,
)
_CHAPTERISH_RE = re.compile(
    r"^(?:chapter|unit|part|module|lesson|book|appendix)\s+(?:\d{1,3}|[ivxlc]{1,7}|[a-z]\b|one|two|three|four|five|"
    r"six|seven|eight|nine|ten|eleven|twelve)\b",
    re.I,
)
_SECTION_NUM_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){0,4})\.?\s+\S")
_LEADER_RE = re.compile(r"(?:[.·…_]\s?){4,}\s*\d{1,4}$")
_TERMINAL = ".!?:;…"
_CLOSERS = " \"'”’)»]"
_FUNCTION_WORDS = frozenset(
    "a an the of to in on at by for with from and or but nor as than that which who whom whose is are was were "
    "be been being this these those its their his her our your into onto upon between among about such".split()
)
_MATH_CHARS = frozenset(
    "=+<>−×÷±∓∑∏∫∮√∞≤≥≠≈≡∝∂∇∆∈∉⊂⊃⊆⊇∪∩∀∃→←↔⇒⇔^·∘⋅′″"
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ₀₁₂₃₄₅₆₇₈₉αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ"
)
_OPERATOR_CHARS = frozenset("=<>≤≥≠≈≡∝∑∏∫√→⇒⇔±×÷+−^⁰¹²³⁴⁵⁶⁷⁸⁹")
_SUPERSCRIPT = str.maketrans("0123456789+-−=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁼⁽⁾ⁿ")
_SUPERSCRIPTABLE = frozenset("0123456789+-−=()n")

_JUNK_TITLE_RE = re.compile(
    r"^(?:microsoft\s+(?:word|powerpoint|excel)\b|untitled\b|title$|document\s*\d*$|slide\s*\d+$|"
    r"powerpoint presentation$|presentation\s*\d*$|layout\s*\d+$|unknown$|none$|null$|n/?a$|\d+$|[a-z]:\\|/)"
    r"|\.(?:docx?|pptx?|xlsx?|pdf|indd|tex|dvi|rtf|odt|odp|qxd|txt|html?|ps|eps|ai|pages|key|md)$",
    re.I,
)
_JUNK_AUTHOR_RE = re.compile(
    r"^(?:administrator|admin|user|owner|unknown|windows user|microsoft office user|default|author|"
    r"the author|none|null|n/?a|\W*)$",
    re.I,
)


# --------------------------------------------------------------------------- data


@dataclass(slots=True)
class _Line:
    page: int  # 0-based
    x0: float
    y0: float
    x1: float
    y1: float
    spans: list[tuple]  # (text, size, flags, font, x0, y0, x1, y1, origin_y)
    pblock: int = 0  # PyMuPDF block number (for line-pitch statistics)
    text: str = ""
    size: float = 0.0
    baseline: float = 0.0
    bold: bool = False
    italic: bool = False
    mono: bool = False
    caps: bool = False
    chars: int = 0
    math: float = 0.0
    operators: int = 0
    symbol_lead: bool = False
    lead_marker: bool = False
    role: str = "body"  # "body" | "footnote"
    drop: bool = False


@dataclass(slots=True)
class _Table:
    bbox: tuple[float, float, float, float]
    rows: list[list[str]]


@dataclass(slots=True)
class _Elem:
    kind: str  # "line" | "box" | "table"
    x0: float
    y0: float
    x1: float
    y1: float
    line: _Line | None = None
    lines: list[_Line] = field(default_factory=list)
    table: _Table | None = None


@dataclass
class _Page:
    index: int
    width: float
    height: float
    lines: list[_Line] = field(default_factory=list)
    tables: list[_Table] = field(default_factory=list)
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    image_rects: list[tuple[float, float, float, float]] = field(default_factory=list)
    ocr_paragraphs: list[str] | None = None


@dataclass
class _Chunk:
    kind: BlockKind
    page: int  # 0-based page where the chunk starts
    col: tuple  # column key where it starts
    end_col: tuple  # column key where it ends (changes when merged across breaks)
    y0: float
    size: float
    lines: list[_Line] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)
    is_float: bool = False
    in_box: bool = False
    after_break: bool = False  # something (a float, an image gap) interrupted the flow before it
    bold: bool = False
    level: int | None = None
    rows: list[list[str]] | None = None
    style: tuple | None = None
    quote: bool = False  # indented on both sides of the column (block quotation)
    text: str = ""


@dataclass
class _Stats:
    body_size: float = 10.0
    pitch: float = 1.2  # baseline-to-baseline distance / font size for body text


# --------------------------------------------------------------------------- public API


def load_pdf(
    path: Path,
    *,
    source_name: str,
    ocr: bool = True,
    progress: ProgressFn | None = None,
) -> Document:
    """Read a PDF into a :class:`Document`. Raises :class:`IngestError` with a friendly message."""
    path = Path(path)
    display_name = Path(source_name).name or path.name
    doc = _open_pdf(path, display_name)
    try:
        return _load(doc, path, source_name, ocr, progress)
    except IngestError:
        raise
    except MemoryError as exc:
        raise IngestError(f"'{display_name}' is too large to open on this computer.") from exc
    finally:
        doc.close()


def _open_pdf(path: Path, display_name: str) -> pymupdf.Document:
    try:
        doc = pymupdf.open(path, filetype="pdf")
    except FileNotFoundError as exc:
        raise IngestError(f"File not found: {path}") from exc
    except Exception as exc:  # FileDataError, RuntimeError, ...
        raise IngestError(
            f"'{display_name}' doesn't look like a valid PDF file, or it is damaged. "
            "Try opening it in a PDF reader and saving a new copy."
        ) from exc
    if doc.needs_pass and not doc.authenticate(""):
        doc.close()
        raise IngestError(
            f"'{display_name}' is password-protected. Open it in a PDF reader with the password, "
            "save an unprotected copy (for example with Print > Save as PDF), and import that copy."
        )
    if doc.page_count == 0:
        doc.close()
        raise IngestError(f"'{display_name}' has no pages, or it is too damaged to read.")
    return doc


def _load(doc: pymupdf.Document, path: Path, source_name: str, ocr: bool, progress: ProgressFn | None) -> Document:
    n = doc.page_count
    warnings: list[str] = []
    if getattr(doc, "is_repaired", False):
        warnings.append("This PDF looks damaged. Destiny Bot repaired it, but some text may be missing.")

    pages: list[_Page] = []
    size_weights: Counter[float] = Counter()
    pitch_samples: list[float] = []
    failed: list[int] = []
    scanned: list[int] = []  # 0-based indices of pages without a text layer
    ocr_done: list[int] = []
    ocr_empty: list[int] = []
    ocr_failed: list[int] = []
    ocr_state: dict[str, object] = {}

    for i in range(n):
        if progress:
            progress(0.9 * i / n, f"Reading page {i + 1} of {n}")
        try:
            page = doc[i]
            pg = _extract_page(page, i, size_weights, pitch_samples)
        except Exception:
            failed.append(i)
            pages.append(_Page(index=i, width=612, height=792))
            continue
        if _needs_ocr(page, pg):
            scanned.append(i)
            pg.lines = []
            if ocr:
                if progress:
                    progress(0.9 * i / n, f"Reading scanned page {i + 1} of {n} (OCR)")
                result = _ocr_page(page, ocr_state)
                if result is None:
                    pass  # OCR unavailable; reported below
                elif isinstance(result, Exception):
                    ocr_failed.append(i)
                elif result:
                    pg.ocr_paragraphs = result
                    ocr_done.append(i)
                else:
                    ocr_empty.append(i)
        pages.append(pg)

    if progress:
        progress(0.92, "Organizing the text")

    stats = _Stats()
    if size_weights:
        stats.body_size = max(size_weights.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    if len(pitch_samples) >= 3:
        pitch_samples.sort()
        stats.pitch = min(max(pitch_samples[len(pitch_samples) // 2], 1.0), 3.0)

    _mark_headers_footers(pages, stats)
    chunks: list[_Chunk] = []
    for pg in pages:
        if pg.ocr_paragraphs:
            chunks.extend(_ocr_chunks(pg, stats))
        else:
            chunks.extend(_page_chunks(pg, stats))
    _drop_repeated_ocr_margins(chunks)
    _classify(chunks, stats)
    _assign_heading_levels(chunks)
    chunks = _join_heading_labels(chunks)
    chunks = _assemble(chunks)
    blocks, positions = _to_blocks(chunks)

    # ---- warnings
    unread = [i for i in scanned if i not in ocr_done and i not in ocr_empty and i not in ocr_failed]
    if failed:
        warnings.append(f"{_pages_phrase(failed)} couldn't be read and {_was_were(failed)} skipped.")
    if ocr_done:
        warnings.append(
            f"{_pages_phrase(ocr_done)} had no text layer and {_was_were(ocr_done)} read with OCR. "
            "Recognition mistakes are possible."
        )
    if unread:
        if not ocr:
            warnings.append(
                f"{_pages_phrase(unread)} {_is_are(unread)} scanned images with no text and "
                f"{_was_were(unread)} skipped because OCR is turned off."
            )
        else:
            reason = str(ocr_state.get("reason") or "")
            msg = (
                f"{_pages_phrase(unread)} {_is_are(unread)} scanned images with no text. "
                "Install Tesseract OCR to have them read aloud."
            )
            if reason and "tesseract" not in reason.lower():
                msg += f" ({reason})"
            warnings.append(msg)
    if ocr_failed:
        warnings.append(f"OCR failed on {_pages_phrase(ocr_failed).lower()}; {_is_are(ocr_failed, 'it is', 'they are')} skipped.")
    if ocr_empty:
        warnings.append(
            f"No text was found on scanned {_pages_phrase(ocr_empty).lower()} "
            "(they may be pictures or blank)."
        )
    if not blocks and not scanned and not failed:
        warnings.append("No readable text was found in this PDF.")

    # ---- metadata and sections
    meta = doc.metadata or {}
    title = _metadata_title(meta.get("title")) or _heading_title(blocks) or title_from_filename(source_name)
    author = _metadata_author(meta.get("author"))
    sections = _sections_from_outline(doc, blocks, positions, title)
    if sections is None:
        if any(b.kind == BlockKind.HEADING for b in blocks):
            sections = sections_from_blocks(blocks, title, split_level=2)
        elif n > _PAGES_PER_PART + 2:
            sections = _page_range_sections(blocks)
        else:
            sections = [make_section(title, 1, blocks)]
    sections = finalize_sections(sections, title)
    if progress:
        progress(1.0, f"Finished reading {n} page{'s' if n != 1 else ''}")
    return Document(
        id=doc_id_for_file(path),
        title=title,
        author=author,
        source_name=source_name,
        source_type="pdf",
        page_count=n,
        sections=sections,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- extraction


@lru_cache(maxsize=512)
def _font_info(font: str) -> tuple[bool, bool, bool, bool, bool]:
    """(bold, italic, mono, math, symbol) guessed from a font name."""
    name = font.split("+", 1)[-1]
    low = name.lower()
    bold = bool(re.search(r"bold|black|heavy|semibold|demi|extrabold|ultrabold|[-,_]bd\b|cmbx", low))
    italic = bool(re.search(r"italic|oblique|[-,_]it\b|[-,_]ital|cmti|cmmi", low))
    mono = bool(re.search(r"courier|mono|consolas|menlo|inconsolata|typewriter|cmtt|lucidaconsole|code", low))
    math = bool(
        re.match(r"(cmmi|cmsy|cmex|msam|msbm|rsfs|eufm|stix|esint|mtsy|mtex|rtxmi|txsy|txex|pxsy|pxex|lms|lmm)", low)
        or "math" in low
    )
    symbol = bool(re.search(r"symbol|dingbat|wingding|webding|zapf", low))
    return bold, italic, mono, math, symbol


def _page_dims(page: pymupdf.Page) -> tuple[float, float]:
    rect = page.rect
    if page.rotation in (90, 270):
        return rect.height, rect.width
    return rect.width, rect.height


def _rotator(direction: tuple[int, int], w: float, h: float) -> Callable[[float, float], tuple[float, float]] | None:
    """Map unrotated page coordinates to 'reading' coordinates where text runs left to right."""
    if direction == (1, 0):
        return None
    if direction == (0, -1):  # bottom-to-top text
        return lambda x, y: (h - y, x)
    if direction == (0, 1):  # top-to-bottom text
        return lambda x, y: (y, w - x)
    return lambda x, y: (w - x, h - y)  # upside down


def _extract_page(page: pymupdf.Page, index: int, size_weights: Counter, pitch_samples: list[float]) -> _Page:
    w, h = _page_dims(page)
    pg = _Page(index=index, width=w, height=h)
    data = page.get_text("dict", flags=_TEXT_FLAGS)

    raw_lines: list[tuple[tuple[int, int], dict, int]] = []
    dir_weight: Counter[tuple[int, int]] = Counter()
    for block in data.get("blocks", ()):
        if block.get("type", 0) != 0:
            continue
        for ln in block.get("lines", ()):
            dx, dy = ln.get("dir", (1.0, 0.0))
            direction = (round(dx), round(dy)) if abs(abs(dx) - 1) < 0.05 or abs(abs(dy) - 1) < 0.05 else (9, 9)
            nchars = sum(len(s["text"].strip()) for s in ln["spans"])
            if not nchars:
                continue
            dir_weight[direction] += nchars
            raw_lines.append((direction, ln, block.get("number", 0)))
    if not raw_lines:
        pg.image_rects = _image_rects(page)
        return pg

    main_dir = dir_weight.most_common(1)[0][0]
    if main_dir == (9, 9):
        main_dir = (1, 0)
    rot = _rotator(main_dir, w, h)
    if rot is not None and main_dir[0] == 0:
        pg.width, pg.height = h, w

    prev_by_block: dict[int, _Line] = {}
    for direction, ln, bno in raw_lines:
        if direction != main_dir:
            continue  # rotated margin notes, sideways labels in figures, ...
        spans = []
        for s in ln["spans"]:
            if not s["text"]:
                continue
            x0, y0, x1, y1 = s["bbox"]
            ox, oy = s["origin"]
            if rot is not None:
                ax, ay = rot(x0, y0)
                bx, by = rot(x1, y1)
                x0, x1 = min(ax, bx), max(ax, bx)
                y0, y1 = min(ay, by), max(ay, by)
                _, oy = rot(ox, oy)
            spans.append((s["text"], round(s["size"], 2), s["flags"], s["font"], x0, y0, x1, y1, oy))
        if not spans:
            continue
        line = _Line(
            page=index,
            x0=min(s[4] for s in spans),
            y0=min(s[5] for s in spans),
            x1=max(s[6] for s in spans),
            y1=max(s[7] for s in spans),
            spans=spans,
            pblock=bno,
        )
        _finish_line(line)
        if not line.text:
            continue
        size_weights[round(line.size * 2) / 2] += line.chars
        prev = prev_by_block.get(bno)
        if prev is not None and abs(prev.size - line.size) < 0.5 and line.size > 0:
            dy = line.baseline - prev.baseline
            if 0 < dy < 3 * line.size and min(prev.x1, line.x1) > max(prev.x0, line.x0):
                pitch_samples.append(dy / line.size)
        prev_by_block[bno] = line
        pg.lines.append(line)

    pg.image_rects = _image_rects(page)
    if rot is None:
        _find_tables_and_boxes(page, pg)
    return pg


def _finish_line(line: _Line) -> None:
    """Compute the display text and style attributes of a line from its spans."""
    spans = sorted(line.spans, key=lambda s: s[4]) if len(line.spans) > 1 else line.spans
    weights: Counter[float] = Counter()
    for s in spans:
        n = len(s[0].strip())
        if n:
            weights[s[1]] += n
    if not weights:
        line.text = ""
        return
    main = max(weights.items(), key=lambda kv: (kv[1], kv[0]))[0]
    bases = sorted(s[8] for s in spans if abs(s[1] - main) < 0.2 and s[0].strip())
    baseline = bases[len(bases) // 2] if bases else spans[0][8]

    pieces: list[str] = []
    prev_x1: float | None = None
    total = bold = italic = mono = mathc = ops = 0
    lead_marker = False
    symbol_lead = False
    first = True
    for text, size, flags, font, x0, _y0, x1, _y1, oy in spans:
        stripped = text.strip()
        if not stripped:
            if pieces and not pieces[-1].endswith(" "):
                pieces.append(" ")
            prev_x1 = x1
            continue
        f_bold, f_italic, f_mono, f_math, f_symbol = _font_info(font)
        n = len(stripped.replace(" ", ""))
        if first:
            first = False
            c = stripped[0]
            if len(stripped) <= 2 and (f_symbol or c in _BULLETS or 0xE000 <= ord(c) <= 0xF8FF):
                symbol_lead = True
        raised = size < main * 0.9 and (bool(flags & 1) or oy < baseline - 0.2 * main)
        if raised:
            before = "".join(pieces)
            if not before.strip():
                pieces.append(stripped + " ")  # a footnote's own number at the start of the line
                lead_marker = True
                prev_x1 = x1
                continue
            if _REF_MARK_RE.match(stripped) and _is_reference_context(before, stripped):
                prev_x1 = x1  # footnote reference marker: not part of the display text
                continue
            sup = stripped.translate(_SUPERSCRIPT) if set(stripped) <= _SUPERSCRIPTABLE else stripped
            pieces.append(sup)
            total += n
            mathc += n
            ops += n
            prev_x1 = x1
            continue
        if (
            pieces
            and prev_x1 is not None
            and not pieces[-1].endswith((" ", "\t"))
            and not text.startswith((" ", "\t"))
            and x0 - prev_x1 > 0.2 * size
        ):
            pieces.append(" ")
        pieces.append(text)
        prev_x1 = x1
        total += n
        if f_bold or flags & 16:
            bold += n
        if f_italic or flags & 2:
            italic += n
        if f_mono or flags & 8:
            mono += n
        if f_math:
            mathc += n
        else:
            mathc += sum(1 for ch in stripped if ch in _MATH_CHARS)
        ops += sum(1 for ch in stripped if ch in _OPERATOR_CHARS)

    text = re.sub(r"[ \t]+", " ", "".join(pieces)).strip()
    line.text = text
    line.size = main
    line.baseline = baseline
    line.chars = max(total, 1)
    line.bold = bold >= 0.8 * total if total else False
    line.italic = italic >= 0.8 * total if total else False
    line.mono = mono >= 0.8 * total if total else False
    line.math = mathc / total if total else 0.0
    line.operators = ops
    letters = [c for c in text if c.isalpha()]
    line.caps = len(letters) >= 2 and all(c.isupper() for c in letters)
    line.symbol_lead = symbol_lead
    line.lead_marker = lead_marker


def _is_reference_context(before: str, marker: str) -> bool:
    """Is a superscript after ``before`` a footnote reference (vs. an exponent like x², 10³)?"""
    if any(ch in "*†‡§¶" for ch in marker):
        return True
    tail = before.rstrip()
    if not tail:
        return False
    if before != tail:  # "word ¹": a space before the marker
        return True
    c = tail[-1]
    if c in ".,;:!?\"'”’»":
        return True
    if c.isalpha():
        m = re.search(r"[^\W\d_]+$", tail)
        return bool(m) and len(m.group(0)) >= 3
    return False


def _image_rects(page: pymupdf.Page) -> list[tuple[float, float, float, float]]:
    try:
        infos = page.get_image_info()
    except Exception:
        return []
    rects = []
    for info in infos:
        x0, y0, x1, y1 = info.get("bbox", (0, 0, 0, 0))
        if x1 - x0 > 2 and y1 - y0 > 2:
            rects.append((x0, y0, x1, y1))
    return rects


def _find_tables_and_boxes(page: pymupdf.Page, pg: _Page) -> None:
    """Find ruled tables (PyMuPDF's table finder) and framed / shaded boxes around text."""
    try:
        paths = page.get_cdrawings()
    except Exception:
        return
    if not paths:
        return
    area = pg.width * pg.height
    horiz = vert = rects = 0
    box_candidates: list[tuple[float, float, float, float]] = []
    for p in paths[:_MAX_TABLE_DRAWINGS]:
        for item in p.get("items", ()):
            op = item[0]
            if op == "l":
                (ax, ay), (bx, by) = item[1], item[2]
                if abs(ay - by) < 1 and abs(ax - bx) > 10:
                    horiz += 1
                elif abs(ax - bx) < 1 and abs(ay - by) > 10:
                    vert += 1
            elif op == "re":
                rx0, ry0, rx1, ry1 = item[1]
                rw, rh = abs(rx1 - rx0), abs(ry1 - ry0)
                if rh < 2 and rw > 10:
                    horiz += 1
                elif rw < 2 and rh > 10:
                    vert += 1
                else:
                    rects += 1
                    filled = "f" in p.get("type", "") and p.get("fill") not in (None, (1.0, 1.0, 1.0), [1.0, 1.0, 1.0])
                    stroked = "s" in p.get("type", "")
                    if (filled or stroked) and rw >= 0.2 * pg.width and rh >= 18 and rw * rh <= 0.8 * area:
                        box_candidates.append((min(rx0, rx1), min(ry0, ry1), max(rx0, rx1), max(ry0, ry1)))

    if len(paths) <= _MAX_TABLE_DRAWINGS and (rects >= 2 or (horiz >= 2 and vert >= 2)):
        try:
            found = page.find_tables()
            tables = list(found.tables)
        except Exception:
            tables = []
        for tab in tables:
            try:
                rows = tab.extract()
                header = getattr(tab, "header", None)
                bbox = tuple(tab.bbox)
                if header is not None and getattr(header, "external", False):
                    rows = [list(header.names)] + rows
                    hb = tuple(header.bbox)
                    bbox = (min(bbox[0], hb[0]), min(bbox[1], hb[1]), max(bbox[2], hb[2]), max(bbox[3], hb[3]))
            except Exception:
                continue
            cleaned = _clean_table(rows)
            if cleaned is not None:
                pg.tables.append(_Table(bbox=bbox, rows=cleaned))

    for b in box_candidates:
        if any(_overlap_frac(b, t.bbox) > 0.5 for t in pg.tables):
            continue
        if b not in pg.boxes:
            pg.boxes.append(b)


def _clean_table(rows: list[list[str | None]]) -> list[list[str]] | None:
    """Clean cell text, drop empty rows/columns; None if it isn't really a table."""
    grid = [[clean_display_text(c or "") for c in row] for row in rows if row]
    grid = [row for row in grid if any(row)]
    if not grid:
        return None
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]
    keep = [j for j in range(width) if any(r[j] for r in grid)]
    grid = [[r[j] for j in keep] for r in grid]
    if len(grid) < 2 or len(keep) < 2:
        return None
    filled = sum(1 for r in grid for c in r if c)
    if filled < 0.4 * len(grid) * len(keep):
        return None
    return grid


def _overlap_frac(a: tuple, b: tuple) -> float:
    """Share of rectangle ``a`` covered by ``b``."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    return w * h / area


def _inside(line: _Line, rect: tuple, tol: float = 2.0) -> bool:
    cx = (line.x0 + line.x1) / 2
    cy = (line.y0 + line.y1) / 2
    return rect[0] - tol <= cx <= rect[2] + tol and rect[1] - tol <= cy <= rect[3] + tol


def _needs_ocr(page: pymupdf.Page, pg: _Page) -> bool:
    """A page with no usable text layer that shows something (a scan, or text drawn as outlines)."""
    chars = sum(ln.chars for ln in pg.lines)
    page_area = max(pg.width * pg.height, 1.0)
    coverage = sum((r[2] - r[0]) * (r[3] - r[1]) for r in pg.image_rects) / page_area
    if chars == 0:
        if coverage >= 0.1:
            return True
        try:
            return len(page.get_cdrawings()) >= 200  # text converted to vector outlines
        except Exception:
            return False
    return chars < 20 and coverage >= 0.8  # a scan with a tiny stamp ("Scanned by ...")


# --------------------------------------------------------------------------- OCR


def _ocr_page(page: pymupdf.Page, state: dict) -> list[str] | Exception | None:
    """OCR one page. Returns paragraphs, an exception if OCR failed, or None if OCR is unavailable."""
    if "fn" not in state:
        fn, reason = _ocr_function()
        state["fn"], state["reason"] = fn, reason
    fn = state["fn"]
    if fn is None:
        return None
    try:
        w_in = page.rect.width / 72
        h_in = page.rect.height / 72
        dpi = _OCR_DPI
        if w_in * h_in * dpi * dpi > _OCR_MAX_PIXELS:
            dpi = max(72, int(math.sqrt(_OCR_MAX_PIXELS / max(w_in * h_in, 1e-6))))
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
        text = fn(pix.tobytes("png"))
    except Exception as exc:
        return exc
    return _ocr_paragraphs(text or "")


def _ocr_function() -> tuple[Callable[[bytes], str] | None, str]:
    try:
        from . import ocr as ocr_mod
    except ImportError:
        return None, "OCR support isn't installed."
    try:
        ok, reason = ocr_mod.ocr_available()
    except Exception as exc:
        return None, str(exc)
    if not ok:
        return None, reason
    return ocr_mod.ocr_image_bytes, ""


def _ocr_paragraphs(text: str) -> list[str]:
    """Split OCR output into paragraphs (blank-line separated); keep bullet lines separate."""
    text = text.replace("\f", "\n").replace("\r\n", "\n").replace("\r", "\n")
    paras: list[str] = []
    for part in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in part.split("\n") if ln.strip()]
        current: list[str] = []
        for ln in lines:
            if current and (ln[0] in _BULLETS or _DASH_BULLET_RE.match(ln)):
                paras.append("\n".join(current))
                current = []
            current.append(ln)
        if current:
            paras.append("\n".join(current))
    return [p for p in paras if any(c.isalnum() for c in p)]


def _ocr_chunks(pg: _Page, stats: _Stats) -> list[_Chunk]:
    out = []
    for k, para in enumerate(pg.ocr_paragraphs or []):
        first = para.split("\n", 1)[0]
        if _is_page_number(first.strip()) and "\n" not in para:
            continue
        ch = _Chunk(
            kind=BlockKind.PARAGRAPH,
            page=pg.index,
            col=("ocr", pg.index),
            end_col=("ocr", pg.index),
            y0=float(k),
            size=stats.body_size,
            raw=[para],
        )
        ch.text = clean_display_text(para)
        if ch.text and ch.text[0] in _BULLETS:
            ch.kind = BlockKind.LIST_ITEM
            ch.text = ch.text[1:].strip()
        ch.after_break = k == 0
        out.append(ch)
    return out


def _drop_repeated_ocr_margins(chunks: list[_Chunk]) -> None:
    """Remove running headers/footers that OCR picked up (first/last short paragraph of each page)."""
    by_page: dict[int, list[_Chunk]] = defaultdict(list)
    for ch in chunks:
        if ch.col[0] == "ocr":
            by_page[ch.page].append(ch)
    if len(by_page) < 2:
        return
    seen: dict[str, set[int]] = defaultdict(set)
    for page, items in by_page.items():
        for ch in {id(items[0]): items[0], id(items[-1]): items[-1]}.values():
            if len(ch.text) <= 80:
                seen[_margin_key(ch.text)].add(page)
    threshold = 2 if len(by_page) <= 6 else 3
    bad = {k for k, v in seen.items() if len(v) >= threshold and k}
    if bad:
        chunks[:] = [
            ch
            for ch in chunks
            if not (ch.col[0] == "ocr" and len(ch.text) <= 80 and _margin_key(ch.text) in bad)
        ]


# --------------------------------------------------------------------------- headers / footers


def _margin_key(text: str) -> str:
    t = re.sub(r"\d+", "#", text.lower())
    t = re.sub(r"[^\w#]+", " ", t)
    return " ".join("#" if _ROMAN_RE.match(w) else w for w in t.split())


def _is_page_number(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 140:
        return False
    m = _PAGE_NUM_RE.match(t)
    if m:
        n = m.group("n")
        return n.isdigit() or bool(_ROMAN_RE.match(n))
    m = _PAGE_NUM_SIDE_RE.match(t)
    if m:
        n = m.group("a") or m.group("b") or ""
        return n.isdigit() or bool(_ROMAN_RE.match(n))
    return False


def _isolated(ln: _Line, lines: list[_Line], zone: str, stats: _Stats) -> bool:
    """Is a margin line set apart from the body (extra space towards the page's middle)?"""
    best: tuple[float, float] | None = None
    for other in lines:
        if other is ln or min(other.y1, ln.y1) - max(other.y0, ln.y0) > 0:
            continue  # itself, or on the same row
        if zone == "top" and other.y0 > ln.y0:
            dist = other.baseline - ln.baseline
        elif zone == "bottom" and other.y1 < ln.y1:
            dist = ln.baseline - other.baseline
        else:
            continue
        if best is None or dist < best[0]:
            best = (dist, other.size)
    if best is None:
        return True
    return best[0] > 1.45 * stats.pitch * max(ln.size, best[1])


def _mark_headers_footers(pages: list[_Page], stats: _Stats) -> None:
    """Drop page numbers, and lines in the margin bands that repeat at the same spot on nearby pages."""
    n_pages = sum(1 for p in pages if p.lines)
    occurrences: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
    candidates: list[tuple[tuple, _Line]] = []
    for pg in pages:
        if not pg.lines:
            continue
        top_band = pg.height * _BAND
        bottom_band = pg.height * (1 - _BAND)
        for ln in pg.lines:
            if ln.y1 <= top_band:
                zone = "top"
            elif ln.y0 >= bottom_band:
                zone = "bottom"
            else:
                continue
            if any(_inside(ln, t.bbox) for t in pg.tables):
                continue
            if not _isolated(ln, pg.lines, zone, stats):
                continue  # sits in the flow of the body text
            if _is_page_number(ln.text):
                ln.drop = True
                continue
            if ln.size > stats.body_size * 1.6:
                continue  # big display text (a chapter title) is never a running header
            key = (zone, _margin_key(ln.text), round(ln.size))
            if not key[1]:
                continue
            occurrences[key].append((pg.index, ln.y0 if zone == "top" else pg.height - ln.y1))
            candidates.append((key, ln))

    threshold = 2 if n_pages <= 6 else 3
    repeated = set()
    for key, occ in occurrences.items():
        if len({p for p, _ in occ}) < threshold:
            continue
        # Running headers sit at a fixed distance from the page edge.
        ys = sorted(y for _, y in occ)
        median = ys[len(ys) // 2]
        plist = sorted({p for p, y in occ if abs(y - median) <= 4})
        if len(plist) >= threshold and any(b - a <= 2 for a, b in zip(plist, plist[1:])):
            repeated.add(key)
    for key, ln in candidates:
        if key in repeated:
            ln.drop = True


# --------------------------------------------------------------------------- page layout


def _page_chunks(pg: _Page, stats: _Stats) -> list[_Chunk]:
    lines = [ln for ln in pg.lines if not ln.drop]
    if not lines and not pg.tables:
        return []
    lines = [ln for ln in lines if not any(_inside(ln, t.bbox) for t in pg.tables)]
    lines = _drop_figure_labels(lines, pg)

    box_lines: dict[int, list[_Line]] = defaultdict(list)
    free: list[_Line] = []
    boxes = sorted(pg.boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))  # smallest first
    for ln in lines:
        for bi, b in enumerate(boxes):
            if b[0] - 2 <= ln.x0 and ln.x1 <= b[2] + 2 and b[1] - 2 <= ln.y0 and ln.y1 <= b[3] + 2:
                box_lines[bi].append(ln)
                break
        else:
            free.append(ln)
    if box_lines and not free and len(box_lines) == 1:
        free = next(iter(box_lines.values()))  # the "box" is just a frame around the whole page text
        box_lines = {}

    elems = [_Elem("line", ln.x0, ln.y0, ln.x1, ln.y1, line=ln) for ln in free]
    for bi, blines in box_lines.items():
        b = boxes[bi]
        elems.append(_Elem("box", b[0], b[1], b[2], b[3], lines=blines))
    for t in pg.tables:
        elems.append(_Elem("table", *t.bbox, table=t))

    groups = _reading_order(elems, pg)
    _mark_footnotes(groups, stats)

    chunks: list[_Chunk] = []
    interrupted = False
    for key, items in groups:
        pending_lines: list[_Line] = []
        for el in items:
            if el.kind == "line":
                pending_lines.append(el.line)
                continue
            if pending_lines:
                chunks.extend(_group_lines(pending_lines, (pg.index,) + key, stats, interrupted))
                interrupted = False
                pending_lines = []
            if el.kind == "table":
                t = el.table
                chunks.append(
                    _Chunk(
                        kind=BlockKind.TABLE,
                        page=pg.index,
                        col=(pg.index,) + key,
                        end_col=(pg.index,) + key,
                        y0=t.bbox[1],
                        size=stats.body_size,
                        is_float=True,
                        rows=t.rows,
                        text=" / ".join(" | ".join(c for c in row) for row in t.rows),
                    )
                )
            else:
                rows = _merge_rows(el.lines)
                box_key = (pg.index, "box", round(el.y0), round(el.x0))
                for ch in _group_lines(rows, box_key, stats, False):
                    ch.is_float = True
                    ch.in_box = True
                    chunks.append(ch)
            interrupted = True
        if pending_lines:
            chunks.extend(_group_lines(pending_lines, (pg.index,) + key, stats, interrupted))
            interrupted = False
    return chunks


def _drop_figure_labels(lines: list[_Line], pg: _Page) -> list[_Line]:
    """Short text sitting on top of a (non-background) picture is a diagram label, not prose."""
    page_area = pg.width * pg.height
    figs = [r for r in pg.image_rects if (r[2] - r[0]) * (r[3] - r[1]) < 0.6 * page_area]
    if not figs:
        return lines
    out = []
    for ln in lines:
        if len(ln.text) <= 40 and len(ln.text.split()) <= 5 and any(_inside(ln, r, tol=0) for r in figs):
            continue
        out.append(ln)
    return out


def _reading_order(elems: list[_Elem], pg: _Page) -> list[tuple[tuple, list[_Elem]]]:
    """Order page elements by column. Returns groups (column key, elements in order)."""
    if not elems:
        return []
    gutters = _detect_gutters(elems)
    if not gutters:
        return [((0, 0), _order_group(elems))]
    bounds = [min(e.x0 for e in elems) - 1]
    for g0, g1 in gutters:
        bounds.extend([g0, g1])
    bounds.append(max(e.x1 for e in elems) + 1)
    columns = [(bounds[2 * k], bounds[2 * k + 1]) for k in range(len(gutters) + 1)]

    def col_of(e: _Elem) -> int:
        for j, (c0, c1) in enumerate(columns):
            if e.x0 >= c0 - 2 and e.x1 <= c1 + 2:
                return j
        return -1

    spanning = sorted((e for e in elems if col_of(e) == -1), key=lambda e: e.y0)
    span_mid = [(s.y0 + s.y1) / 2 for s in spanning]
    segs: dict[tuple[int, int], list[_Elem]] = defaultdict(list)
    for e in elems:
        j = col_of(e)
        if j == -1:
            continue
        mid = (e.y0 + e.y1) / 2
        seg = sum(1 for m in span_mid if m < mid)
        segs[(seg, j)].append(e)

    groups: list[tuple[tuple, list[_Elem]]] = []
    run: list[_Elem] = []
    for k in range(len(spanning) + 1):
        col_groups = [(j, segs[(k, j)]) for j in range(len(columns)) if segs.get((k, j))]
        if len(col_groups) > 1 and not _side_by_side([items for _, items in col_groups]):
            # Columns that don't share any height here are stacked blocks (e.g. a header
            # above a title): read them top to bottom like full-width content.
            run.extend(e for _, items in col_groups for e in items)
            col_groups = []
        if col_groups and run:
            groups.append((("span", k), _order_group(run)))
            run = []
        for j, items in col_groups:
            groups.append(((k, j), _order_group(items)))
        if k < len(spanning):
            run.append(spanning[k])
    if run:
        groups.append((("span", len(spanning)), _order_group(run)))
    return groups


def _side_by_side(cols: list[list[_Elem]]) -> bool:
    """Do neighbouring columns overlap vertically (rather than sit one above the other)?"""
    for a, b in zip(cols, cols[1:]):
        a0, a1 = min(e.y0 for e in a), max(e.y1 for e in a)
        b0, b1 = min(e.y0 for e in b), max(e.y1 for e in b)
        if min(a1, b1) - max(a0, b0) < 0.3 * min(a1 - a0, b1 - b0):
            return False
    return True


def _detect_gutters(elems: list[_Elem]) -> list[tuple[float, float]]:
    """Find vertical gutters between text columns from the x-coverage of narrow elements."""
    lines = [e for e in elems if e.kind == "line"]
    if len(lines) < 6:
        return []
    xmin = min(e.x0 for e in elems)
    xmax = max(e.x1 for e in elems)
    width = xmax - xmin
    if width < 150:
        return []
    narrow = [e for e in elems if e.x1 - e.x0 <= 0.62 * width]
    if len(narrow) < 6:
        return []
    nbins = int(width) + 2
    diff = [0] * (nbins + 1)
    for e in narrow:
        a = max(0, int(e.x0 - xmin))
        b = min(nbins - 1, int(math.ceil(e.x1 - xmin)))
        diff[a] += 1
        diff[b + 1] -= 1
    cov, acc = [], 0
    for k in range(nbins):
        acc += diff[k]
        cov.append(acc)
    threshold = max(1, round(0.04 * len(narrow)))
    lo, hi = int(0.15 * width), int(0.85 * width)
    runs: list[tuple[int, int]] = []
    start = None
    for k in range(lo, hi + 1):
        if cov[k] <= threshold:
            if start is None:
                start = k
        elif start is not None:
            runs.append((start, k - 1))
            start = None
    if start is not None:
        runs.append((start, hi))
    candidates = [(xmin + a, xmin + b + 1) for a, b in runs if b - a + 1 >= 5]
    if not candidates:
        return []
    if _valid_columns(lines, candidates, xmin, xmax):
        return candidates
    # Fall back to the single best gutter (closest to the middle, widest).
    mid = (xmin + xmax) / 2
    for g in sorted(candidates, key=lambda g: (abs((g[0] + g[1]) / 2 - mid), -(g[1] - g[0]))):
        if _valid_columns(lines, [g], xmin, xmax):
            return [g]
    return []


def _valid_columns(lines: list[_Elem], gutters: list[tuple[float, float]], xmin: float, xmax: float) -> bool:
    bounds = [xmin - 1] + [x for g in gutters for x in g] + [xmax + 1]
    cols = [(bounds[2 * k], bounds[2 * k + 1]) for k in range(len(gutters) + 1)]
    members: list[list[_Elem]] = [[] for _ in cols]
    for e in lines:
        for j, (c0, c1) in enumerate(cols):
            if e.x0 >= c0 - 2 and e.x1 <= c1 + 2:
                members[j].append(e)
                break
    for m in members:
        rows = _text_rows(m)
        if len(rows) < 3:
            return False
        chars = [r[2] for r in rows]
        if sum(chars) / len(chars) < 12 or sum(chars) < 80:
            return False
        lefts = Counter(round(r[0] / 3) for r in rows)
        if lefts.most_common(1)[0][1] < 0.35 * len(rows):
            return False  # not a text column: rows don't share a left edge
    return _side_by_side(members)


def _text_rows(elems: list[_Elem]) -> list[tuple[float, float, int]]:
    """Group line elements into visual rows: (left x, right x, characters) per row."""
    rows: list[list[float]] = []  # [y0, y1, x0, x1, chars]
    for e in sorted(elems, key=lambda e: (e.y0 + e.y1) / 2):
        if rows:
            r = rows[-1]
            overlap = min(r[1], e.y1) - max(r[0], e.y0)
            if overlap >= 0.5 * min(r[1] - r[0], e.y1 - e.y0):
                r[0], r[1] = min(r[0], e.y0), max(r[1], e.y1)
                r[2], r[3] = min(r[2], e.x0), max(r[3], e.x1)
                r[4] += len(e.line.text) + 1
                continue
        rows.append([e.y0, e.y1, e.x0, e.x1, len(e.line.text)])
    return [(r[2], r[3], int(r[4])) for r in rows]


def _order_group(elems: list[_Elem]) -> list[_Elem]:
    """Sort a column's elements top to bottom, merging line fragments that share a row."""
    lines = [e.line for e in elems if e.kind == "line"]
    others = [e for e in elems if e.kind != "line"]
    merged = [_Elem("line", ln.x0, ln.y0, ln.x1, ln.y1, line=ln) for ln in _merge_rows(lines)]
    return sorted(merged + others, key=lambda e: (e.y0, e.x0))


def _merge_rows(lines: list[_Line]) -> list[_Line]:
    """Join fragments on the same baseline (bullet + text, label + value, split spans)."""
    if len(lines) < 2:
        return list(lines)
    ordered = sorted(lines, key=lambda ln: ((ln.y0 + ln.y1) / 2, ln.x0))
    rows: list[list[_Line]] = []
    for ln in ordered:
        if rows:
            row = rows[-1]
            ry0 = min(r.y0 for r in row)
            ry1 = max(r.y1 for r in row)
            overlap = min(ry1, ln.y1) - max(ry0, ln.y0)
            smaller = min(ry1 - ry0, ln.y1 - ln.y0)
            size = max(ln.size, max(r.size for r in row))
            gap = min(abs(ln.x0 - r.x1) for r in row) if all(ln.x0 >= r.x1 - 1 for r in row) else 0.0
            if smaller > 0 and overlap >= 0.5 * smaller and not any(_h_overlap(ln, r) for r in row) and gap <= 4 * size:
                row.append(ln)
                continue
        rows.append([ln])
    out = []
    for row in rows:
        if len(row) == 1:
            out.append(row[0])
            continue
        row.sort(key=lambda ln: ln.x0)
        spans = [s for ln in row for s in ln.spans]
        merged = _Line(
            page=row[0].page,
            x0=min(r.x0 for r in row),
            y0=min(r.y0 for r in row),
            x1=max(r.x1 for r in row),
            y1=max(r.y1 for r in row),
            spans=spans,
            pblock=row[0].pblock,
            role=row[0].role,
        )
        _finish_line(merged)
        merged.symbol_lead = row[0].symbol_lead or merged.symbol_lead
        out.append(merged)
    return out


def _h_overlap(a: _Line, b: _Line) -> bool:
    return min(a.x1, b.x1) - max(a.x0, b.x0) > 2


def _mark_footnotes(groups: list[tuple[tuple, list[_Elem]]], stats: _Stats) -> None:
    """Small-print lines below a column's body text that start with a note marker."""
    body = stats.body_size
    for _key, items in groups:
        lines = [e.line for e in items if e.kind == "line"]
        if not lines:
            continue
        body_lines = [ln for ln in lines if ln.size >= body * 0.95]
        if not body_lines:
            continue
        last_body = max(ln.y1 for ln in body_lines)
        active = False
        for ln in lines:
            if ln.y0 < last_body - 1 or ln.size > body * 0.92:
                active = False
                continue
            if _FOOTNOTE_START_RE.match(ln.text) or ln.lead_marker:
                active = True
            if active:
                ln.role = "footnote"


# --------------------------------------------------------------------------- lines -> chunks


def _ends_terminal(text: str) -> bool:
    t = text.rstrip(_CLOSERS)
    return bool(t) and t[-1] in _TERMINAL


def _list_marker(line: _Line) -> str | None:
    """'bullet' / 'number' / 'dash' when the line starts like a list item."""
    t = line.text
    if not t:
        return None
    if line.symbol_lead or t[0] in _BULLETS or 0xE000 <= ord(t[0]) <= 0xF8FF:
        return "bullet"
    if _NUMBERED_RE.match(t):
        return "number"
    if _DASH_BULLET_RE.match(t):
        return "dash"
    return None


def _is_caption_text(text: str) -> bool:
    m = _CAPTION_RE.match(text)
    if not m:
        return False
    rest = m.group("rest")
    if not rest:
        return True
    c = rest[0]
    return c in ".:—–|-)" or c.isupper() or c in "\"“'(‘" or c.isdigit()


def _is_equation_line(line: _Line) -> bool:
    if line.chars < 3 or len(line.text) > 200:
        return False
    long_words = len(re.findall(r"[A-Za-z]{4,}", line.text))
    if long_words > 1 and not (line.math >= 0.6 and long_words <= 3):
        return False
    math_chars = line.math * line.chars
    return math_chars >= max(2, 0.3 * line.chars) and (line.operators > 0 or line.math >= 0.5)


def _group_lines(lines: list[_Line], col: tuple, stats: _Stats, interrupted: bool) -> list[_Chunk]:
    """Group a column's ordered lines into chunks using spacing, indentation and style."""
    if not lines:
        return []
    col_left = min(ln.x0 for ln in lines)
    col_right = max(ln.x1 for ln in lines)
    width = max(col_right - col_left, 1.0)
    chunks: list[_Chunk] = []
    cur: _Chunk | None = None
    cur_list = False
    cur_eq = False
    for ln in lines:
        eq = _is_equation_line(ln)
        marker = _list_marker(ln)
        if cur is not None:
            prev = cur.lines[-1]
            if _continues(cur, prev, ln, eq, cur_eq, marker, cur_list, col_left, col_right, width, stats):
                cur.lines.append(ln)
                continue
        cur = _Chunk(
            kind=BlockKind.PARAGRAPH,
            page=ln.page,
            col=col,
            end_col=col,
            y0=ln.y0,
            size=ln.size,
            lines=[ln],
            bold=ln.bold,
            after_break=interrupted and not chunks,
        )
        cur_list = marker is not None and (marker == "bullet" or _closed_before(chunks, ln, col_right, width))
        cur_eq = eq
        if cur_list:
            cur.kind = BlockKind.LIST_ITEM
        if eq:
            cur.kind = BlockKind.EQUATION
        chunks.append(cur)
    for ch in chunks:
        ch.bold = all(ln.bold for ln in ch.lines)
        sizes = Counter()
        for ln in ch.lines:
            sizes[ln.size] += ln.chars
        ch.size = max(sizes.items(), key=lambda kv: (kv[1], kv[0]))[0]
        if ch.lines[0].role == "footnote":
            ch.kind = BlockKind.FOOTNOTE
            ch.is_float = True
        elif (
            ch.kind == BlockKind.PARAGRAPH
            and len(ch.lines) >= 2
            and width > 200
            and min(ln.x0 for ln in ch.lines) >= col_left + 1.5 * ch.size
            and max(ln.x1 for ln in ch.lines) <= col_right - 1.5 * ch.size
        ):
            ch.quote = True
    return chunks


def _closed_before(chunks: list[_Chunk], line: _Line, col_right: float, width: float) -> bool:
    """Was the line before ``line`` the end of a sentence/item (so a numbered line starts a list)?"""
    if not chunks:
        return True
    prev = chunks[-1].lines[-1]
    if chunks[-1].kind == BlockKind.LIST_ITEM:
        return True
    return _ends_terminal(prev.text) or prev.x1 < col_right - max(3 * prev.size, 0.15 * width)


def _continues(
    cur: _Chunk,
    prev: _Line,
    ln: _Line,
    eq: bool,
    cur_eq: bool,
    marker: str | None,
    cur_list: bool,
    col_left: float,
    col_right: float,
    width: float,
    stats: _Stats,
) -> bool:
    if ln.role != prev.role:
        return False
    if eq != cur_eq:
        return False
    size = max(prev.size, ln.size)
    if abs(prev.size - ln.size) > max(0.6, 0.08 * size):
        return False
    if prev.bold != ln.bold and not (prev.mono and ln.mono):
        return False
    if prev.mono != ln.mono:
        return False
    dy = ln.baseline - prev.baseline
    expected = stats.pitch * size
    limit = expected * (2.6 if (prev.mono and ln.mono) else 1.3) + 0.5
    if dy <= 0.2 * size or dy > limit:
        return False
    if min(ln.x1, max(l.x1 for l in cur.lines)) - max(ln.x0, min(l.x0 for l in cur.lines)) < -2:
        return False  # no horizontal overlap with the chunk
    if ln.role == "footnote" and (_FOOTNOTE_START_RE.match(ln.text) or ln.lead_marker):
        return False
    if prev.mono and ln.mono:
        return True
    if eq:
        return True
    closed = _ends_terminal(prev.text) or prev.x1 < col_right - max(3 * prev.size, 0.15 * width)
    if marker == "bullet" or (marker is not None and (closed or cur_list)):
        return False
    if _is_caption_text(ln.text) and closed:
        return False
    first = cur.lines[0]
    if cur_list:
        if ln.x0 >= first.x0 + 0.3 * size and ln.x0 <= first.x0 + 8 * size:
            return True  # hanging indent continuation
        return abs(ln.x0 - first.x0) <= 2 and not closed
    if ln.x0 > prev.x0 + 0.6 * size and ln.x0 - col_left < 0.4 * width and len(cur.lines) >= 1:
        if not (len(cur.lines) == 1 and abs(ln.x0 - col_left) <= 2):
            return False  # first-line indent of a new paragraph
    short = prev.x1 < col_right - max(3 * prev.size, 0.15 * width)
    if short and _ends_terminal(prev.text):
        return False
    return True


# --------------------------------------------------------------------------- classification


def _chunk_text(ch: _Chunk) -> str:
    if ch.kind == BlockKind.CODE:
        left = min(ln.x0 for ln in ch.lines)
        out = []
        for ln in ch.lines:
            indent = max(0, round((ln.x0 - left) / max(ln.size * 0.6, 1)))
            out.append(" " * indent + fix_characters(ln.text))
        return "\n".join(out)
    return clean_display_text("\n".join(ln.text for ln in ch.lines))


def _classify(chunks: list[_Chunk], stats: _Stats) -> None:
    body = stats.body_size
    for k, ch in enumerate(chunks):
        if ch.kind == BlockKind.TABLE or not ch.lines:
            continue
        if ch.kind not in (BlockKind.FOOTNOTE, BlockKind.EQUATION) and all(ln.mono for ln in ch.lines):
            if len(ch.lines) >= 2 or ch.lines[0].chars >= 12:
                ch.kind = BlockKind.CODE
        ch.raw = [ln.text for ln in ch.lines]
        ch.text = _chunk_text(ch)
        if ch.kind in (BlockKind.FOOTNOTE, BlockKind.EQUATION, BlockKind.CODE):
            continue
        text = ch.text
        if ch.kind == BlockKind.LIST_ITEM:
            first = ch.lines[0]
            if first.symbol_lead or text[:1] in _BULLETS or (text and 0xE000 <= ord(text[0]) <= 0xF8FF):
                text = text[1:].lstrip()
            elif _DASH_BULLET_RE.match(text):
                text = text[1:].lstrip()
            ch.text = text
            if _heading_like(ch, body):  # "1. Introduction" set as a heading
                ch.kind = BlockKind.HEADING
            continue
        if _is_caption_text(text):
            ch.kind = BlockKind.CAPTION
            ch.is_float = True
            continue
        if _heading_like(ch, body):
            ch.kind = BlockKind.HEADING
            continue
        if not any(c.isalpha() for c in text) and len(text) <= 8:
            ch.text = ""  # stray number or symbol (chart ticks, lone glyphs)
            continue
        if ch.quote:
            ch.kind = BlockKind.QUOTE


def _heading_like(ch: _Chunk, body: float) -> bool:
    text = ch.text
    if not text or len(ch.lines) > 4 or len(text) > 200:
        return False
    letters = sum(1 for c in text if c.isalpha())
    if letters < 2 or _LEADER_RE.search(text):
        return False  # table-of-contents entry ("3.2 Transport ........ 45")
    words = text.split()
    if ch.size >= body * 1.12 and ch.size - body >= 1.0:
        if len(words) > 25 or (len(words) > 14 and _ends_terminal(text)):
            return False  # large-print paragraph (intro text, pull quote)
        return True
    if ch.size < body * 0.95:
        return False
    if text.endswith((",", ";")) or len(words) > 15 or len(text) > 120:
        return False
    numbered = bool(_CHAPTERISH_RE.match(text) or _SECTION_NUM_RE.match(text))
    if ch.bold:
        if text.endswith(".") and not numbered and len(words) > 6:
            return False
        return numbered or text[0].isupper() or text[0].isdigit()
    if ch.lines[0].caps and len(ch.lines) <= 2 and len(words) <= 8 and not _ends_terminal(text):
        return any(len(re.sub(r"\W", "", w)) >= 4 for w in words)
    return False


def _heading_style(ch: _Chunk) -> tuple:
    depth = 0
    m = _SECTION_NUM_RE.match(ch.text)
    if _CHAPTERISH_RE.match(ch.text):
        depth = -1
    elif m:
        depth = max(0, m.group(1).count("."))
    return (ch.size, ch.bold, depth)


def _assign_heading_levels(chunks: list[_Chunk]) -> None:
    """Rank heading styles (size, then bold, then numbering depth) into levels 1..6."""
    heads = [ch for ch in chunks if ch.kind == BlockKind.HEADING]
    if not heads:
        return
    sizes = sorted({ch.size for ch in heads}, reverse=True)
    clusters: list[list[float]] = []
    for s in sizes:
        if clusters and clusters[-1][0] - s <= max(0.8, 0.05 * s):
            clusters[-1].append(s)
        else:
            clusters.append([s])
    rank_of_size = {s: i for i, c in enumerate(clusters) for s in c}
    counts: Counter[tuple] = Counter()
    for ch in heads:
        size, bold, depth = _heading_style(ch)
        ch.style = (rank_of_size[size], 0 if bold else 1, depth)
        counts[ch.style] += 1
    styles = sorted(counts)
    frequent = [s for s in styles if counts[s] >= 2]
    level_of: dict[tuple, int] = {}
    if frequent:
        # One-off styles bigger than every recurring one (a title, a title page) share
        # level 1; recurring styles rank below them. Other one-offs (a lone "Appendix"
        # in its own style) take the level of the next recurring style.
        offset = 1 if any(s < frequent[0] for s in styles) else 0
        for i, s in enumerate(frequent):
            level_of[s] = min(i + 1 + offset, 6)
        for s in styles:
            if s in level_of:
                continue
            if s < frequent[0]:
                level_of[s] = 1
                continue
            below = [f for f in frequent if f >= s]
            level_of[s] = level_of[below[0]] if below else min(len(frequent) + 1 + offset, 6)
    else:
        for i, s in enumerate(styles):
            level_of[s] = min(i + 1, 6)
    for ch in heads:
        ch.level = level_of[ch.style]
        if ch.in_box:
            ch.level = max(ch.level, 3)  # a sidebar title must not start a new section


def _join_heading_labels(chunks: list[_Chunk]) -> list[_Chunk]:
    """'Chapter 3' + 'Cells and Membranes' -> one heading; a big chapter number joins its title."""
    out: list[_Chunk] = []
    k = 0
    while k < len(chunks):
        ch = chunks[k]
        nxt = chunks[k + 1] if k + 1 < len(chunks) else None
        if nxt is not None and nxt.kind == BlockKind.HEADING and nxt.page == ch.page:
            if ch.kind == BlockKind.HEADING and _CHAPTER_LABEL_RE.match(ch.text):
                nxt.text = f"{ch.text.rstrip('.')}: {nxt.text}"
                nxt.level = min(ch.level or 1, nxt.level or 1)
                nxt.y0 = ch.y0
                nxt.col = ch.col
                k += 1
                continue
            if ch.kind == BlockKind.PARAGRAPH and not ch.text and ch.lines and re.fullmatch(
                r"\d{1,3}", ch.lines[0].text.strip()
            ) and ch.size >= nxt.size * 0.9:
                nxt.text = f"{ch.lines[0].text.strip()} {nxt.text}"
                nxt.y0 = ch.y0
                k += 1
                continue
        out.append(ch)
        k += 1
    return out


# --------------------------------------------------------------------------- assembly


def _continues_across(p: _Chunk, q: _Chunk, broken: bool) -> bool:
    """Does chunk ``q`` continue paragraph ``p`` (split by a column/page break or a float)?"""
    if q.kind != BlockKind.PARAGRAPH or p.kind not in (BlockKind.PARAGRAPH, BlockKind.LIST_ITEM):
        return False
    if p.in_box != q.in_box or not p.text or not q.text:
        return False
    if abs(p.size - q.size) > max(0.6, 0.08 * p.size):
        return False
    if _ends_terminal(p.text):
        return False
    first = q.text[0]
    if first.islower():
        return True
    if not broken:
        return False
    tail = p.text.rstrip()
    if tail.endswith((",", "-", "–", "—", "(")):
        return True
    last_word = re.sub(r"\W", "", tail.split()[-1].lower()) if tail.split() else ""
    if last_word in _FUNCTION_WORDS:
        return True
    return p.kind == BlockKind.PARAGRAPH and len(p.text) >= 60 and (tail[-1].isalnum()) and not p.bold


def _assemble(chunks: list[_Chunk]) -> list[_Chunk]:
    """Merge paragraphs split by breaks; floats that interrupt a paragraph move after it."""
    out: list[_Chunk] = []
    pending: list[_Chunk] = []
    open_para: _Chunk | None = None
    for ch in chunks:
        if not ch.text and not ch.rows:
            continue
        if ch.is_float:
            if open_para is not None:
                pending.append(ch)
            else:
                out.append(ch)
            continue
        if open_para is not None:
            broken = bool(pending) or ch.after_break or ch.col != open_para.end_col
            if _continues_across(open_para, ch, broken):
                open_para.text = clean_display_text(open_para.text + "\n" + ch.text)
                open_para.end_col = ch.col
                continue
        out.extend(pending)
        pending = []
        out.append(ch)
        open_para = ch if ch.kind in (BlockKind.PARAGRAPH, BlockKind.LIST_ITEM) else None
    out.extend(pending)
    return out


def _to_blocks(chunks: list[_Chunk]) -> tuple[list[Block], list[tuple[int, float]]]:
    blocks: list[Block] = []
    positions: list[tuple[int, float]] = []
    for ch in chunks:
        if not ch.text and not ch.rows:
            continue
        blocks.append(
            Block(
                kind=ch.kind,
                text=ch.text,
                level=ch.level if ch.kind == BlockKind.HEADING else None,
                rows=ch.rows,
                page=ch.page + 1,
            )
        )
        positions.append((ch.page + 1, ch.y0))
    return blocks, positions


# --------------------------------------------------------------------------- sections


def _norm_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _strip_numbering(text: str) -> str:
    t = re.sub(r"^(?:chapter|unit|part|module|lesson|section|appendix)\s+\w+\s*", "", text)
    return re.sub(r"^\d+(?:\s+\d+)*\s+", "", t).strip()


def _titles_match(heading: str, title: str) -> bool:
    a, b = _norm_title(heading), _norm_title(title)
    if not a or not b:
        return False
    if a == b:
        return True
    na, nb = re.findall(r"\d+", a), re.findall(r"\d+", b)
    if na and nb and na != nb:
        return False  # "1.1 Transport" is not "1.2 Transport"
    sa, sb = _strip_numbering(a), _strip_numbering(b)
    if sa and sb and sa == sb and len(sa) >= 3:
        return True
    if min(len(a), len(b)) >= 6 and (a in b or b in a):
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.85


def _sections_from_outline(
    doc: pymupdf.Document, blocks: list[Block], positions: list[tuple[int, float]], title: str
) -> list[Section] | None:
    try:
        toc = doc.get_toc(simple=False)
    except Exception:
        return None
    entries = []
    for item in toc:
        try:
            lvl, etitle, page = int(item[0]), clean_display_text(str(item[1])), int(item[2])
        except (TypeError, ValueError, IndexError):
            continue
        if page < 1 or not etitle:
            continue
        to_y = None
        dest = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
        point = dest.get("to")
        if point is not None:
            try:
                to_y = float(point.y)
            except (AttributeError, TypeError, ValueError):
                to_y = None
        entries.append((lvl, etitle, page, to_y))
    if not entries:
        return None
    if not blocks:
        return []

    # Adopt outline levels for matching headings; split points for levels 1-2.
    splits: list[tuple[int, int, str]] = []  # (block index, level, title)
    matched: set[int] = set()
    cursor = 0
    for lvl, etitle, page, to_y in entries:
        top = lvl <= 2
        idx = _find_outline_block(blocks, positions, etitle, page, to_y, cursor if top else 0)
        if idx is None:
            continue
        b = blocks[idx]
        if idx < len(blocks) and b.page == page and _titles_match(b.text, etitle) and len(b.text) <= 200:
            if b.kind != BlockKind.HEADING:
                b.kind = BlockKind.HEADING
            b.level = min(lvl, 6)
            matched.add(idx)
        if top:
            splits.append((idx, lvl, etitle))
            cursor = idx
    if not splits:
        return None

    sections: list[Section] = []
    first = splits[0][0]
    if first > 0:
        front = blocks[:first]
        heads = [b.text for b in front if b.kind == BlockKind.HEADING]
        sections.append(make_section(heads[0] if heads else title, 1, front))
    for k, (idx, lvl, etitle) in enumerate(splits):
        end = splits[k + 1][0] if k + 1 < len(splits) else len(blocks)
        if end <= idx:
            continue
        sec_blocks = blocks[idx:end]
        for j, b in enumerate(sec_blocks):
            if b.kind == BlockKind.HEADING and (idx + j) not in matched and (b.level or 1) <= lvl:
                b.level = min(lvl + 1, 6)
        sections.append(make_section(etitle, lvl, sec_blocks))
    return sections


def _find_outline_block(
    blocks: list[Block], positions: list[tuple[int, float]], title: str, page: int, to_y: float | None, start: int
) -> int | None:
    """Index of the block where an outline entry begins (or None if past the end)."""
    on_page = [i for i in range(start, len(blocks)) if positions[i][0] == page]
    if not on_page:
        later = [i for i in range(start, len(blocks)) if positions[i][0] > page]
        return later[0] if later else (len(blocks) if start < len(blocks) else None)
    for want_heading in (True, False):
        for i in on_page:
            b = blocks[i]
            if want_heading != (b.kind == BlockKind.HEADING):
                continue
            if len(b.text) <= 200 and _titles_match(b.text, title):
                return i
    norm = _norm_title(title)
    if len(norm) >= 6:
        for i in on_page:
            if _norm_title(blocks[i].text).startswith(norm):
                return i  # run-in heading: the paragraph starts with the title
    if to_y is not None and to_y > 1:
        for i in on_page:
            if positions[i][1] >= to_y - 6:
                return i
    return on_page[0]


def _page_range_sections(blocks: list[Block]) -> list[Section]:
    groups: dict[int, list[Block]] = defaultdict(list)
    for b in blocks:
        groups[((b.page or 1) - 1) // _PAGES_PER_PART].append(b)
    sections = []
    for g in sorted(groups):
        items = groups[g]
        first = min(b.page or 1 for b in items)
        last = max(b.page or 1 for b in items)
        name = f"Page {first}" if first == last else f"Pages {first}-{last}"
        sections.append(make_section(name, 1, items))
    return sections


# --------------------------------------------------------------------------- metadata


def _metadata_title(value: str | None) -> str | None:
    if not value:
        return None
    title = clean_display_text(str(value))
    if len(title) < 2 or len(title) > 300 or not any(c.isalpha() for c in title):
        return None
    if _JUNK_TITLE_RE.search(title):
        return None
    return title


def _metadata_author(value: str | None) -> str | None:
    if not value:
        return None
    author = clean_display_text(str(value))
    if not author or len(author) > 300 or _JUNK_AUTHOR_RE.match(author):
        return None
    return author


def _heading_title(blocks: list[Block]) -> str | None:
    for b in blocks:
        if (b.page or 1) > 3:
            break
        if b.kind == BlockKind.HEADING and b.level == 1 and 2 <= len(b.text) <= 200:
            return b.text
    return None


# --------------------------------------------------------------------------- warnings text


def _pages_phrase(pages0: list[int]) -> str:
    """'Page 3' / 'Pages 3-5, 9' for 0-based page indices."""
    nums = sorted({p + 1 for p in pages0})
    parts: list[str] = []
    start = prev = nums[0]
    for n in nums[1:] + [None]:  # type: ignore[list-item]
        if n is not None and n == prev + 1:
            prev = n
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if n is not None:
            start = prev = n
    return ("Page " if len(nums) == 1 else "Pages ") + ", ".join(parts)


def _was_were(pages0: list[int]) -> str:
    return "was" if len(set(pages0)) == 1 else "were"


def _is_are(pages0: list[int], one: str = "is", many: str = "are") -> str:
    return one if len(set(pages0)) == 1 else many
