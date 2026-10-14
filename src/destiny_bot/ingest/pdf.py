"""Load PDF files (textbooks, articles, handouts) into a :class:`Document`.

The pipeline:

1. Extract every text line with its spans (font size, weight, position), find
   ruled tables, framed boxes and vector diagrams, and OCR pages that are only
   scanned images (or whose text layer can't be decoded).
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
from contextlib import contextmanager
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterator

import pymupdf

from ..models import Block, BlockKind, Document, Section
from . import IngestError, ProgressFn
from .common import (
    HyphenationEvidence,
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

_BAND = 0.14  # top/bottom share of the page where running headers and footers live
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
# Bullets as Tesseract tends to read them ("•" often comes out as "©", "¢", "e" or "«").
_OCR_BULLET_RE = re.compile(r"^(?:[•●○◦▪■□◆►‣⁃∙·*¢©«»°]|[eo])\s+(?=[A-Z0-9“\"(])|^[-–—]\s+(?=\S)")
_CAPTION_RE = re.compile(
    r"^(?:fig(?:ure)?s?\.?|tables?|tab\.|exhibit|chart|graph|diagram|illustration|plate|map|photo|scheme|listing)"
    r"\s*(?:[A-Z]?\d+(?:[.\-–]\d+)*[a-z]?|[IVXLC]+\b)\s*(?P<rest>.*)$",
    re.I | re.S,
)
_FOOTNOTE_START_RE = re.compile(r"^(?:\d{1,3}|[*†‡§¶]{1,3})(?:[.)]?\s+|(?=[A-Z\"“(]))")
_REF_MARK_RE = re.compile(r"^[\d,\-–*†‡§¶]+$")
_ROMAN_RE = re.compile(r"^(?=[ivxlcdm]+$)m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})$", re.I)
_PAGE_ROMAN_RE = re.compile(r"^(?=[ivxlc]+$)c{0,3}(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})$", re.I)  # i .. cccxcix
_PAGE_NUM_RE = re.compile(
    r"^[\[(]?(?:page|pg\.?|p\.)?\s*[-–—]?\s*(?P<n>\d{1,4}|[ivxlcdm]{1,8})\s*[-–—]?\s*(?:(?:of|/)\s*\d{1,4})?"
    r"[\])]?\.?$",
    re.I,
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
# (starts only where a run of leaders starts, so that long runs of dots stay linear-time)
_LEADER_RE = re.compile(r"(?<![.·…_])(?<![.·…_]\s)(?:[.·…_]\s?){4,}\s*\d{1,4}$")
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
    r"the author|none|null|n/?a|python-docx|python-pptx|\W*)$|^(?:array|hash)\(0x[0-9a-f]+\)$",
    re.I,
)
_GENERIC_HEADINGS = frozenset(
    "contents|table of contents|abstract|preface|foreword|introduction|acknowledgments|acknowledgements|"
    "index|references|bibliography|copyright|dedication|summary|about the author|about the authors|"
    "brief contents|detailed contents".split("|")
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\ufffd\ue000-\uf8ff]")  # + private-use glyphs


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
    lead_spaces: int = 0  # leading blanks in the raw text (code indentation)
    role: str = "body"  # "body" | "footnote" | "note" (unmarked small print at the bottom)
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
    figures: list[tuple[float, float, float, float]] = field(default_factory=list)  # vector diagrams
    ocr_paragraphs: list[str] | None = None
    raw_chars: int = 0  # characters in the text layer ...
    bad_chars: int = 0  # ... of which undecodable (control codes, U+FFFD): broken font encodings


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
    top: bool = False  # first chunk of its column run (nothing above it in the flow)
    indent: float | None = None  # first-line indent relative to the column's left edge
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
    body_font: str = ""  # most used font (without the subset prefix)


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
    with _quiet_mupdf():
        doc = _open_pdf(path, display_name)
        try:
            return _load(doc, path, source_name, ocr, progress)
        except IngestError:
            raise
        except MemoryError as exc:
            raise IngestError(f"'{display_name}' is too large to open on this computer.") from exc
        finally:
            doc.close()


@contextmanager
def _quiet_mupdf() -> Iterator[None]:
    """Keep MuPDF's messages about damaged files out of the console while loading."""
    tools = getattr(pymupdf, "TOOLS", None)
    try:
        errors = tools.mupdf_display_errors()
        warnings = tools.mupdf_display_warnings()
        tools.mupdf_display_errors(False)
        tools.mupdf_display_warnings(False)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            tools.mupdf_display_errors(errors)
            tools.mupdf_display_warnings(warnings)
        except Exception:
            pass


def _open_pdf(path: Path, display_name: str) -> pymupdf.Document:
    not_pdf = IngestError(
        f"'{display_name}' doesn't look like a valid PDF file, or it is damaged. "
        "Try opening it in a PDF reader and saving a new copy."
    )
    try:
        with open(path, "rb") as fh:
            head = fh.read(1024)
    except FileNotFoundError as exc:
        raise IngestError(f"File not found: {path}") from exc
    except OSError as exc:
        raise IngestError(f"Couldn't open '{display_name}': {exc.strerror or exc}") from exc
    if b"%PDF" not in head:
        raise not_pdf
    try:
        doc = pymupdf.open(path, filetype="pdf")
    except FileNotFoundError as exc:
        raise IngestError(f"File not found: {path}") from exc
    except Exception as exc:  # FileDataError, RuntimeError, ...
        raise not_pdf from exc
    if doc.needs_pass and not doc.authenticate(""):
        doc.close()
        raise IngestError(
            f"This PDF is password-protected, so Destiny Bot can't read '{display_name}'. Open it in a "
            "PDF reader with the password, save an unprotected copy (for example with Print > Save as PDF), "
            "and import that copy."
        )
    if doc.page_count == 0:
        doc.close()
        raise IngestError(f"'{display_name}' has no pages, or it is too damaged to read.")
    return doc


@dataclass
class _ReadReport:
    """What happened while reading the pages (turned into user-facing warnings)."""

    ocr_enabled: bool
    failed: list[int] = field(default_factory=list)  # 0-based pages that couldn't be read at all
    scanned: list[int] = field(default_factory=list)  # pages without a text layer
    garbled: list[int] = field(default_factory=list)  # pages whose text layer can't be decoded
    outcome: dict[int, str] = field(default_factory=dict)  # page -> "done" | "empty" | "failed" (OCR)
    ocr_reason: str = ""  # why OCR is unavailable, when it is

    def pages(self, which: list[int], outcome: str | None) -> list[int]:
        return [i for i in which if self.outcome.get(i) == outcome]


def _load(doc: pymupdf.Document, path: Path, source_name: str, ocr: bool, progress: ProgressFn | None) -> Document:
    n = doc.page_count
    pages, stats, report = _read_pages(doc, ocr, progress)
    if progress:
        progress(0.92, "Organizing the text")

    _mark_headers_footers(pages, stats)
    # How the document spells words within lines decides breaks like "self-\nesteem".
    evidence = HyphenationEvidence(ln.text for pg in pages for ln in pg.lines)
    for pg in pages:
        for para in pg.ocr_paragraphs or ():
            evidence.add(para)
    layouts = [[] if pg.ocr_paragraphs else _page_elements(pg) for pg in pages]
    gutters = _page_gutters(layouts)
    chunks: list[_Chunk] = []
    for pg, elems, page_gutters in zip(pages, layouts, gutters):
        if pg.ocr_paragraphs:
            chunks.extend(_ocr_chunks(pg, stats, evidence))
        else:
            chunks.extend(_page_chunks(pg, elems, page_gutters, stats))
    _drop_repeated_ocr_margins(chunks)
    _mark_ocr_headings(chunks)
    _classify(chunks, stats, evidence)
    _assign_heading_levels(chunks)
    chunks = _assemble(_join_heading_labels(chunks), evidence)
    blocks, positions = _to_blocks(chunks)

    meta = doc.metadata or {}
    title = _metadata_title(meta.get("title")) or _heading_title(blocks) or title_from_filename(source_name)
    sections = _sections_from_outline(doc, blocks, positions, title)
    if sections is None:
        if any(b.kind == BlockKind.HEADING for b in blocks):
            sections = _fold_lone_headings(sections_from_blocks(blocks, title, split_level=2))
        elif n > _PAGES_PER_PART + 2:
            sections = _page_range_sections(blocks)
        else:
            sections = [make_section(title, 1, blocks)]
    warnings = _warnings(report, has_text=bool(blocks))
    if getattr(doc, "is_repaired", False):
        warnings.insert(0, "This PDF looks damaged. Destiny Bot repaired it, but some text may be missing.")
    if progress:
        progress(1.0, f"Finished reading {n} page{'s' if n != 1 else ''}")
    return Document(
        id=doc_id_for_file(path),
        title=title,
        author=_metadata_author(meta.get("author")),
        source_name=source_name,
        source_type="pdf",
        page_count=n,
        sections=finalize_sections(sections, title),
        warnings=warnings,
    )


def _read_pages(
    doc: pymupdf.Document, ocr: bool, progress: ProgressFn | None
) -> tuple[list[_Page], _Stats, _ReadReport]:
    """Extract every page (OCR where the text layer is missing or undecodable) and body-text stats."""
    n = doc.page_count
    report = _ReadReport(ocr_enabled=ocr)
    pages: list[_Page] = []
    size_weights: Counter[float] = Counter()
    font_weights: Counter[str] = Counter()
    pitch_samples: list[float] = []
    ocr_state: dict[str, object] = {}
    for i in range(n):
        if progress:
            progress(0.9 * i / n, f"Reading page {i + 1} of {n}")
        try:
            page = doc[i]
            pg = _extract_page(page, i, size_weights, pitch_samples)
            for ln in pg.lines:
                font_weights[_line_font(ln)] += ln.chars
        except Exception:
            report.failed.append(i)
            pages.append(_Page(index=i, width=612, height=792))
            continue
        garbled = _garbled(pg)
        if garbled or _needs_ocr(page, pg):
            (report.garbled if garbled else report.scanned).append(i)
            if not garbled:
                pg.lines = []  # (a garbled text layer is kept in case OCR can't replace it)
            if ocr:
                if progress:
                    progress(0.9 * i / n, f"Reading page {i + 1} of {n} with OCR")
                result = _ocr_page(page, ocr_state)
                if isinstance(result, Exception):
                    report.outcome[i] = "failed"
                elif result:
                    pg.ocr_paragraphs = result
                    pg.lines = []
                    report.outcome[i] = "done"
                elif result is not None:
                    report.outcome[i] = "empty"
        pages.append(pg)
    report.ocr_reason = str(ocr_state.get("reason") or "")

    stats = _Stats()
    if size_weights:
        stats.body_size = max(size_weights.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    if len(pitch_samples) >= 3:
        pitch_samples.sort()
        stats.pitch = min(max(pitch_samples[len(pitch_samples) // 2], 1.0), 3.0)
    if font_weights:
        stats.body_font = font_weights.most_common(1)[0][0]
    return pages, stats, report


def _line_font(line: _Line) -> str:
    """The font used for most of a line's characters (without the subset prefix)."""
    return max(line.spans, key=lambda sp: len(sp[0].strip()))[3].split("+", 1)[-1]


def _warnings(r: _ReadReport, *, has_text: bool) -> list[str]:
    out: list[str] = []

    def how_to_fix(pages0: list[int]) -> str:
        them = "it" if len(pages0) == 1 else "them"
        if not r.ocr_enabled:
            return f"Turn on OCR to have {them} read aloud."
        reason = r.ocr_reason
        extra = f" ({reason})" if reason and "tesseract" not in reason.lower() else ""
        return f"Install Tesseract OCR to have {them} read aloud.{extra}"

    if r.failed:
        out.append(f"{_pages_phrase(r.failed)} couldn't be read and {_was_were(r.failed)} skipped.")
    done = r.pages(r.scanned, "done")
    if done:
        out.append(
            f"{_pages_phrase(done)} had no text layer and {_was_were(done)} read with OCR. "
            "Recognition mistakes are possible."
        )
    done = r.pages(r.garbled, "done")
    if done:
        out.append(
            f"The text of {_pages_phrase(done, lower=True)} couldn't be decoded, so {_it_they(done)} "
            f"{_was_were(done)} read with OCR instead. Recognition mistakes are possible."
        )
    unread = r.pages(r.scanned, None)
    if unread:
        what = "is a scanned image" if len(unread) == 1 else "are scanned images"
        out.append(f"{_pages_phrase(unread)} {what} with no text. {how_to_fix(unread)}")
    undecoded = [i for i in r.garbled if r.outcome.get(i) != "done"]
    if undecoded:
        msg = (
            f"The text of {_pages_phrase(undecoded, lower=True)} is stored in a way that can't be turned "
            "back into letters, so parts may be missing or garbled."
        )
        unread = r.pages(r.garbled, None)
        out.append(msg + (f" {how_to_fix(unread)}" if unread else ""))
    bad = r.pages(r.scanned, "failed")
    if bad:
        out.append(f"OCR failed on {_pages_phrase(bad, lower=True)}, so {_it_they(bad)} {_was_were(bad)} skipped.")
    empty = r.pages(r.scanned, "empty")
    if empty:
        out.append(
            f"No text was found on scanned {_pages_phrase(empty, lower=True)} "
            f"({_it_they(empty)} may be {'a picture' if len(empty) == 1 else 'pictures'} or blank)."
        )
    if not has_text and not (r.scanned or r.failed or r.garbled):
        out.append("No readable text was found in this PDF.")
    return out


# --------------------------------------------------------------------------- extraction


@lru_cache(maxsize=512)
def _font_info(font: str) -> tuple[bool, bool, bool, bool, bool]:
    """(bold, italic, mono, math, dingbat) guessed from a font name."""
    name = font.split("+", 1)[-1]
    low = name.lower()
    bold = bool(re.search(r"bold|black|heavy|semibold|demi|extrabold|ultrabold|[-,_]bd\b|cmbx", low))
    italic = bool(re.search(r"italic|oblique|[-,_]it\b|[-,_]ital|cmti|cmmi", low))
    mono = bool(
        re.search(
            r"courier|mono|consol|menlo|monaco|inconsolata|typewriter|cmtt|lmtt|txtt|lucidaconsole|andale|"
            r"fixed|code|1mn|nimbusmon|pcrr|pcrb|letter ?gothic|ocr-?[ab]",
            low,
        )
    )
    math = bool(
        re.match(r"(cmmi|cmsy|cmex|msam|msbm|rsfs|eufm|stix|esint|mtsy|mtex|rtxmi|txsy|txex|pxsy|pxex|lms|lmm)", low)
        or "math" in low
    )
    dingbat = bool(re.search(r"dingbat|wingding|webding|zapf|bullet", low))
    return bold, italic, mono, math, dingbat


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
            for sp in ln["spans"]:
                t = sp["text"]
                pg.raw_chars += len(t)
                pg.bad_chars += sum(1 for c in t if (c < " " and c not in "\t\n\r") or c == "\ufffd")
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
    else:  # sideways page: bring picture positions into reading coordinates (tables are skipped)
        rects = []
        for x0, y0, x1, y1 in pg.image_rects:
            (ax, ay), (bx, by) = rot(x0, y0), rot(x1, y1)
            rects.append((min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)))
        pg.image_rects = rects
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
        f_bold, f_italic, f_mono, f_math, f_dingbat = _font_info(font)
        n = len(stripped.replace(" ", ""))
        if all(0xE000 <= ord(c) <= 0xF8FF for c in stripped) and size >= 1.4 * main:
            prev_x1 = x1  # an icon (Font Awesome and the like), not text
            continue
        if first:
            first = False
            c = stripped[0]
            if len(stripped) <= 2 and (f_dingbat or c in _BULLETS or 0xE000 <= ord(c) <= 0xF8FF):
                symbol_lead = True  # a bullet drawn as its own glyph: kept out of the text
                prev_x1 = x1
                continue
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

    raw = "".join(s[0] for s in spans).replace("\t", "    ")
    line.lead_spaces = len(raw) - len(raw.lstrip(" "))
    text = re.sub(r"[ \t]+", " ", _CONTROL_RE.sub("", "".join(pieces))).strip()
    if symbol_lead and not text:
        text = "•"  # a bullet on its own (its text comes on the same row, merged later)
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
        if not page.get_images():
            return []  # cheap check: the page uses no images at all
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
    h_edges: set[int] = set()  # distinct y positions of horizontal rules / rectangle edges
    v_edges: set[int] = set()  # distinct x positions of vertical ones
    h_rules: list[tuple[float, float, float]] = []  # (y, x0, x1) of horizontal lines
    box_candidates: list[tuple[float, float, float, float]] = []
    for p in paths[:_MAX_TABLE_DRAWINGS]:
        for item in p.get("items", ()):
            op = item[0]
            if op == "l":
                (ax, ay), (bx, by) = item[1], item[2]
                if abs(ay - by) < 1 and abs(ax - bx) > 10:
                    h_edges.add(round(ay))
                    h_rules.append((ay, min(ax, bx), max(ax, bx)))
                elif abs(ax - bx) < 1 and abs(ay - by) > 10:
                    v_edges.add(round(ax))
            elif op == "re":
                rx0, ry0, rx1, ry1 = item[1]
                rw, rh = abs(rx1 - rx0), abs(ry1 - ry0)
                if rw > 10:
                    h_edges.update((round(ry0), round(ry1)))
                    if rh < 2:
                        h_rules.append(((ry0 + ry1) / 2, min(rx0, rx1), max(rx0, rx1)))
                if rh > 10:
                    v_edges.update((round(rx0), round(rx1)))
                if rw >= 2 and rh >= 2:
                    filled = "f" in p.get("type", "") and p.get("fill") not in (None, (1.0, 1.0, 1.0), [1.0, 1.0, 1.0])
                    stroked = "s" in p.get("type", "")
                    if (filled or stroked) and rw >= 0.2 * pg.width and rh >= 18 and rw * rh <= 0.8 * area:
                        box_candidates.append((min(rx0, rx1), min(ry0, ry1), max(rx0, rx1), max(ry0, ry1)))

    # A ruled table needs a grid: at least 3 horizontal and 3 vertical edge positions.
    # (A shaded code box or a framed note has only 2 of each; find_tables is slow.)
    if len(paths) <= _MAX_TABLE_DRAWINGS and len(h_edges) >= 3 and len(v_edges) >= 3:
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
                    # PyMuPDF may take the line above a table as its header row; keep it
                    # unless it is really a caption ("Table 4. ...") or a sliced text line.
                    names = [n or "" for n in header.names]
                    hb = tuple(header.bbox)
                    if not _is_caption_text(" ".join(names).strip()) and not _cuts_through_text(hb, pg.lines):
                        rows = [names] + rows
                        bbox = (min(bbox[0], hb[0]), min(bbox[1], hb[1]), max(bbox[2], hb[2]), max(bbox[3], hb[3]))
            except Exception:
                continue
            cleaned = _clean_table(rows)
            if cleaned is not None and not _cuts_through_text(bbox, pg.lines):
                pg.tables.append(_Table(bbox=bbox, rows=cleaned))

    for t in _rule_tables(pg, h_rules):
        if not any(_overlap_frac(t.bbox, other.bbox) > 0.3 for other in pg.tables):
            pg.tables.append(t)

    for b in box_candidates:
        if any(_overlap_frac(b, t.bbox) > 0.5 for t in pg.tables):
            continue
        if b not in pg.boxes:
            pg.boxes.append(b)
    if len(paths) <= _MAX_TABLE_DRAWINGS:
        pg.figures = _vector_figures(paths, pg, [t.bbox for t in pg.tables] + pg.boxes)


def _vector_figures(paths: list[dict], pg: _Page, exclude: list[tuple]) -> list[tuple[float, float, float, float]]:
    """Regions dense with vector graphics (diagrams, charts): their short text lines are labels."""
    if len(paths) < 12:
        return []
    cell = 12.0
    area = pg.width * pg.height
    occupied: set[tuple[int, int]] = set()
    centers: list[tuple[int, int]] = []
    for p in paths:
        r = p.get("rect")
        if not r:
            continue
        x0, y0, x1, y1 = r
        w, h = x1 - x0, y1 - y0
        if w < 2 or h < 2 or w * h > 0.3 * area:
            continue  # rules, underlines, axis lines and page backgrounds don't make a figure
        if any(_overlap_frac(r, ex) > 0.8 for ex in exclude):
            continue
        # Grow each shape a little so that the parts of one diagram connect.
        i0, i1 = int(max(x0 - 15, 0) // cell), int(min(x1 + 15, pg.width) // cell)
        j0, j1 = int(max(y0 - 15, 0) // cell), int(min(y1 + 15, pg.height) // cell)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                occupied.add((i, j))
        centers.append((int((x0 + x1) / 2 // cell), int((y0 + y1) / 2 // cell)))
    comp_of: dict[tuple[int, int], int] = {}
    comps: list[list[tuple[int, int]]] = []
    for start in occupied:
        if start in comp_of:
            continue
        comp_of[start] = len(comps)
        stack, cells = [start], []
        while stack:
            c = stack.pop()
            cells.append(c)
            for n in ((c[0] + 1, c[1]), (c[0] - 1, c[1]), (c[0], c[1] + 1), (c[0], c[1] - 1)):
                if n in occupied and n not in comp_of:
                    comp_of[n] = len(comps)
                    stack.append(n)
        comps.append(cells)
    counts = Counter(comp_of[c] for c in centers if c in comp_of)
    regions = []
    for idx, cells in enumerate(comps):
        if counts[idx] < 12:
            continue
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        rect = (min(xs) * cell, min(ys) * cell, (max(xs) + 1) * cell, (max(ys) + 1) * cell)
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        if w * h <= 0.6 * area and w >= 80 and h >= 60:  # (a column of drawn bullets is too narrow)
            regions.append(rect)
    return regions


def _rule_tables(pg: _Page, h_rules: list[tuple[float, float, float]]) -> list[_Table]:
    """Tables ruled only by horizontal lines (booktabs style): cells come from text alignment."""
    groups: dict[tuple[int, int], list[tuple[float, float, float]]] = defaultdict(list)
    for rule in h_rules:
        if rule[2] - rule[1] >= 100:
            groups[(round(rule[1] / 3), round(rule[2] / 3))].append(rule)
    tables: list[_Table] = []
    for rules in groups.values():
        ys = sorted({round(r[0], 1) for r in rules})
        # rules of one table are close together; a big gap starts another table
        clusters: list[list[float]] = [[ys[0]]]
        for y in ys[1:]:
            if y - clusters[-1][-1] > 0.4 * pg.height:
                clusters.append([y])
            else:
                clusters[-1].append(y)
        x0 = min(r[1] for r in rules)
        x1 = max(r[2] for r in rules)
        for cl in clusters:
            if len(cl) < 2:
                continue
            region = (x0 - 2, cl[0] - 1, x1 + 2, cl[-1] + 1)
            inside = [ln for ln in pg.lines if _inside(ln, region, tol=0)]
            if len(inside) < 4 or any(ln.x1 - ln.x0 > 0.6 * (x1 - x0) for ln in inside):
                continue  # empty, or prose between two rules
            rows = _rows_of([_Elem("line", ln.x0, ln.y0, ln.x1, ln.y1, line=ln) for ln in inside])
            if len(rows) < 2 or sum(1 for r in rows if len(r) >= 2) < 0.5 * len(rows):
                continue
            bands: list[list[float]] = []
            for a, b in sorted((ln.x0, ln.x1) for ln in inside):
                if bands and a <= bands[-1][1] + 2:
                    bands[-1][1] = max(bands[-1][1], b)
                else:
                    bands.append([a, b])
            if len(bands) < 2:
                continue
            grid = []
            for row in rows:
                cells = [""] * len(bands)
                for e in sorted(row, key=lambda e: e.x0):
                    j = next((k for k, (a, b) in enumerate(bands) if e.x0 >= a - 0.5 and e.x1 <= b + 0.5), 0)
                    cells[j] = f"{cells[j]} {e.line.text}".strip()
                grid.append(cells)
            cleaned = _clean_table(grid)
            if cleaned is not None:
                tables.append(_Table(bbox=region, rows=cleaned))
    return tables


def _cuts_through_text(bbox: tuple, lines: list[_Line]) -> bool:
    """A "table" whose border slices through lines of text is a chart grid or a frame, not a table."""
    for ln in lines:
        w = max(ln.x1 - ln.x0, 1e-6)
        inside_x = min(ln.x1, bbox[2]) - max(ln.x0, bbox[0])
        inside_y = min(ln.y1, bbox[3]) - max(ln.y0, bbox[1])
        if inside_y > 0.5 * (ln.y1 - ln.y0) and 0.1 * w < inside_x < 0.9 * w and len(ln.text) > 3:
            return True
    return False


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
    cells = [c for r in grid for c in r if c]
    fill = len(cells) / (len(grid) * len(keep))
    if fill < 0.5 or (len(grid) <= 4 and fill < 0.75):
        return None  # mostly empty: a diagram or chart drawn with lines
    chars = "".join(cells).replace(" ", "")
    if sum(1 for c in chars if c.isalpha()) < 0.2 * len(chars) and not any(c.isalpha() for c in "".join(grid[0])):
        return None  # tick labels of a chart grid
    wordy = any(re.search(r"[^\W\d_]{3,}", c) for c in cells)
    numeric = sum(1 for c in cells if re.fullmatch(r"[-+−]?[\d.,%$€£ ]+", c)) >= 0.5 * len(cells)
    if not wordy and not numeric:
        return None  # labels of a geometric figure ("x 1", "U")
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


def _garbled(pg: _Page) -> bool:
    """Is the text layer mostly undecodable (fonts without a Unicode mapping)?"""
    return pg.raw_chars >= 20 and pg.bad_chars >= 0.3 * pg.raw_chars


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
            if current and _OCR_BULLET_RE.match(ln):
                paras.append("\n".join(current))
                current = []
            current.append(ln)
        if current:
            paras.append("\n".join(current))
    return [p for p in paras if _plausible_ocr_text(p)]


def _plausible_ocr_text(text: str) -> bool:
    """Reject OCR noise from pictures (logos, cover art): mostly symbols, or no real word."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    letters = sum(1 for c in chars if c.isalpha())
    if letters < 0.5 * len(chars):
        return bool(re.fullmatch(r"[\d\s.,:;%$€£()+\-–/]+", text)) and len(chars) >= 4  # a line of numbers
    if len(chars) < 15:
        return bool(re.search(r"[A-Za-z]{4,}", text))
    return True


def _ocr_chunks(pg: _Page, stats: _Stats, evidence: HyphenationEvidence | None = None) -> list[_Chunk]:
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
        ch.text = clean_display_text(para, evidence=evidence)
        bullet = _OCR_BULLET_RE.match(ch.text)
        if bullet:
            ch.kind = BlockKind.LIST_ITEM
            ch.text = ch.text[bullet.end() :].strip()
        elif _is_caption_text(ch.text):
            ch.kind = BlockKind.CAPTION
            ch.is_float = True
        ch.after_break = k == 0
        out.append(ch)
    return out


def _mark_ocr_headings(chunks: list[_Chunk]) -> None:
    """OCR gives no font sizes: treat a short title-like line before a long paragraph as a heading."""
    for k, ch in enumerate(chunks):
        if ch.col[0] != "ocr" or ch.kind != BlockKind.PARAGRAPH or "\n" in ch.raw[0]:
            continue
        nxt = chunks[k + 1] if k + 1 < len(chunks) else None
        prev = chunks[k - 1] if k else None
        if nxt is None or nxt.page != ch.page or nxt.kind != BlockKind.PARAGRAPH or len(nxt.text) < 150:
            continue
        if prev is not None and prev.page == ch.page and (len(prev.text) < 60 or not _ends_terminal(prev.text)):
            continue  # a cluster of short lines (diagram labels) or the middle of a sentence
        text = ch.text
        words = text.split()
        if not 1 <= len(words) <= 8 or _ends_terminal(text.rstrip("?")):
            continue
        if not (text[0].isupper() or text[0].isdigit()):
            continue
        if not re.search(r"[^\W\d_]{4,}", text) or re.search(r"[b-df-hj-np-tv-z]{6,}", text, re.I):
            continue  # no real word, or an unpronounceable OCR misread
        long_words = [w for w in words if len(re.sub(r"\W", "", w)) >= 4]
        title_case = sum(1 for w in long_words if w[0].isupper()) >= 0.6 * max(len(long_words), 1)
        numbered = bool(_CHAPTERISH_RE.match(text) or _SECTION_NUM_RE.match(text))
        if title_case or numbered or len(words) <= 3:
            ch.kind = BlockKind.HEADING
            ch.level = 1 if _CHAPTERISH_RE.match(text) else 2
            ch.style = ("ocr",)


def _drop_repeated_ocr_margins(chunks: list[_Chunk]) -> None:
    """Remove running headers/footers that OCR picked up (short paragraphs repeated at page edges)."""
    by_page: dict[int, list[_Chunk]] = defaultdict(list)
    for ch in chunks:
        if ch.col[0] == "ocr":
            by_page[ch.page].append(ch)
    if len(by_page) < 3:
        return

    def candidate(ch: _Chunk) -> bool:
        return len(ch.text) <= 160 and ch.raw[0].count("\n") <= 1

    seen: dict[str, set[int]] = defaultdict(set)
    for page, items in by_page.items():
        for ch in items[:2] + items[-2:]:
            if candidate(ch):
                seen[_margin_key(ch.text)].add(page)
    bad = {k for k, v in seen.items() if k and len(v) >= 3}
    if bad:
        chunks[:] = [
            ch for ch in chunks if not (ch.col[0] == "ocr" and candidate(ch) and _margin_key(ch.text) in bad)
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
        return n.isdigit() or bool(_PAGE_ROMAN_RE.match(n))
    m = _PAGE_NUM_SIDE_RE.match(t)
    if m:
        n = m.group("a") or m.group("b") or ""
        return n.isdigit() or bool(_PAGE_ROMAN_RE.match(n))
    return False


def _starts_row(ln: _Line, lines: list[_Line]) -> bool:
    """Is there text right after ``ln`` on the same row (as after a list number)?"""
    return any(0 <= other.x0 - ln.x1 <= 2.5 * ln.size for other in _same_row(ln, lines))


def _ends_row(ln: _Line, lines: list[_Line]) -> bool:
    """Is there text right before ``ln`` on the same row (a list number, a bullet)?"""
    return any(0 <= ln.x0 - other.x1 <= 2.5 * ln.size for other in _same_row(ln, lines))


def _same_row(ln: _Line, lines: list[_Line]) -> list[_Line]:
    return [o for o in lines if o is not ln and min(o.y1, ln.y1) - max(o.y0, ln.y0) >= 0.5 * (ln.y1 - ln.y0)]


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
    slots: dict[tuple, set[int]] = defaultdict(set)  # header "slot" -> pages using it
    slot_lines: list[tuple[tuple, _Line]] = []
    notices: dict[tuple, set[int]] = defaultdict(set)  # small-print block in a margin band -> pages
    notice_lines: list[tuple[tuple, list[_Line]]] = []
    for pg in pages:
        if not pg.lines:
            continue
        top_band = pg.height * _BAND
        bottom_band = pg.height * (1 - _BAND)
        top_y = min(ln.y0 for ln in pg.lines)
        bottom_y = max(ln.y1 for ln in pg.lines)
        for key, block in _margin_notices(pg, stats, top_band, bottom_band):
            notices[key].add(pg.index)
            notice_lines.append((key, block))
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
                if not _starts_row(ln, pg.lines):  # "6." followed by its list item is not a page number
                    ln.drop = True
                continue
            if ln.size > stats.body_size * 1.6:
                continue  # big display text (a chapter title) is never a running header
            key = (zone, _margin_key(ln.text), round(ln.size))
            if not any(c.isalpha() for c in key[1]):
                continue  # numbers alone are only dropped as page numbers (above)
            edge = ln.y0 if zone == "top" else pg.height - ln.y1
            occurrences[key].append((pg.index, edge))
            candidates.append((key, ln))
            outermost = ln.y0 <= top_y + 2 if zone == "top" else ln.y1 >= bottom_y - 2
            font = _line_font(ln)
            styled = ln.size <= stats.body_size * 0.95 or ln.caps or ln.italic or font != stats.body_font
            if outermost and styled and len(ln.text) <= 80 and not _ends_row(ln, pg.lines):
                slot = (zone, round(edge / 3), round(ln.size * 2), font)
                slots[slot].add(pg.index)
                slot_lines.append((slot, ln))

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
    # Headers whose text changes (the current section's name) still sit in the same
    # spot, in the same font, on a large share of the pages.
    busy_slots = {slot for slot, pset in slots.items() if len(pset) >= max(3, 0.4 * n_pages)}
    drop = [ln for key, ln in candidates if key in repeated]
    drop += [ln for slot, ln in slot_lines if slot in busy_slots]
    # A copyright or license notice printed in small type at the foot of some pages (every
    # chapter opener, say) repeats word for word, however far apart those pages are.
    drop += [ln for key, block in notice_lines if len(notices[key]) >= 3 for ln in block]
    total = sum(ln.chars for pg in pages for ln in pg.lines if not ln.drop)
    removed = sum(ln.chars for ln in {id(ln): ln for ln in drop}.values())
    if total and removed > 0.5 * total:
        return  # the "header" is the content itself (e.g. one line per page)
    for ln in drop:
        ln.drop = True


def _margin_notices(
    pg: _Page, stats: _Stats, top_band: float, bottom_band: float
) -> list[tuple[tuple, list[_Line]]]:
    """Text blocks of small print lying wholly in the top or bottom band (notices, imprints)."""
    blocks: dict[int, list[_Line]] = defaultdict(list)
    for ln in pg.lines:
        blocks[ln.pblock].append(ln)
    out = []
    for lines in blocks.values():
        if all(ln.y1 <= top_band for ln in lines):
            zone = "top"
        elif all(ln.y0 >= bottom_band for ln in lines):
            zone = "bottom"
        else:
            continue
        text = " ".join(ln.text for ln in lines)
        if len(text) >= 40 and all(ln.size <= stats.body_size * 0.95 for ln in lines):
            if not any(_inside(ln, t.bbox) for ln in lines for t in pg.tables):
                out.append(((zone, _margin_key(text)), lines))
    return out


# --------------------------------------------------------------------------- page layout


def _page_elements(pg: _Page) -> list[_Elem]:
    """The page's content as layout elements: text lines, framed boxes and tables."""
    lines = [ln for ln in pg.lines if not ln.drop]
    if not lines and not pg.tables:
        return []
    lines = [ln for ln in lines if not any(_inside(ln, t.bbox) for t in pg.tables)]
    lines = _drop_figure_labels(lines, pg)
    mono = [ln for ln in lines if ln.mono]
    if len(mono) > 1:
        # Code with right-aligned comments: join each row first, so the gap between
        # code and comment is never mistaken for a column gutter.
        lines = [ln for ln in lines if not ln.mono] + _merge_rows(mono)

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
    return elems


def _page_chunks(
    pg: _Page, elems: list[_Elem], gutters: list[tuple[float, float]], stats: _Stats
) -> list[_Chunk]:
    """Chunks of one page in reading order."""
    if not elems:
        return []
    groups = _reading_order(elems, gutters)
    _mark_footnotes(groups, stats, pg.height)

    chunks: list[_Chunk] = []
    interrupted = False
    for key, items in groups:
        group_lines = [e.line for e in items if e.kind == "line"]
        bounds = _column_geometry(group_lines) if group_lines else None
        pending_lines: list[_Line] = []
        for el in items:
            if el.kind == "line":
                pending_lines.append(el.line)
                continue
            if pending_lines:
                chunks.extend(_group_lines(pending_lines, (pg.index,) + key, stats, interrupted, bounds))
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
                        text=clean_display_text(" / ".join(" | ".join(row) for row in t.rows)),
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
            chunks.extend(_group_lines(pending_lines, (pg.index,) + key, stats, interrupted, bounds))
            interrupted = False
    return chunks


def _drop_figure_labels(lines: list[_Line], pg: _Page) -> list[_Line]:
    """Short text on top of a picture or inside a vector diagram is a label, not prose."""
    page_area = pg.width * pg.height
    figs = [r for r in pg.image_rects if (r[2] - r[0]) * (r[3] - r[1]) < 0.6 * page_area] + pg.figures
    if not figs:
        return lines
    out = []
    for ln in lines:
        if (
            len(ln.text) <= 40
            and len(ln.text.split()) <= 5
            and any(_inside(ln, r, tol=0) for r in figs)
            and not _is_caption_text(ln.text)
        ):
            continue
        out.append(ln)
    return out


def _reading_order(elems: list[_Elem], gutters: list[tuple[float, float]]) -> list[tuple[tuple, list[_Elem]]]:
    """Order page elements by column. Returns groups (column key, elements in order)."""
    if not elems:
        return []
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

    cols = {id(e): col_of(e) for e in elems}
    # The short last line of a full-width paragraph fits inside one column; keep it
    # with the full-width line right above it.
    xright = max(e.x1 for e in elems)
    full_lines = [e for e in elems if cols[id(e)] == -1 and e.kind == "line"]
    for e in sorted(elems, key=lambda e: e.y0):
        if e.kind != "line" or cols[id(e)] == -1:
            continue
        for s in full_lines:
            if (
                abs(s.x0 - e.x0) <= 2
                and s.x1 >= xright - 0.1 * (xright - s.x0)
                and -2 <= e.y0 - s.y1 <= 0.6 * e.line.size
                and abs(s.line.size - e.line.size) <= 0.5
            ):
                cols[id(e)] = -1
                break
    spanning = sorted((e for e in elems if cols[id(e)] == -1), key=lambda e: e.y0)
    span_mid = [(s.y0 + s.y1) / 2 for s in spanning]
    segs: dict[tuple[int, int], list[_Elem]] = defaultdict(list)
    for e in elems:
        j = cols[id(e)]
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


def _page_gutters(layouts: list[list[_Elem]]) -> list[list[tuple[float, float]]]:
    """Column gutters of every page. A page whose own evidence is too thin (the last page
    of a two-column article with three lines on the right) borrows a gutter found on
    other pages when its text fits around it."""
    found = [_detect_gutters(elems) if elems else [] for elems in layouts]
    known = [g for gs in found if len(gs) == 1 for g in gs]
    for k, elems in enumerate(layouts):
        if found[k] or not known or not elems:
            continue
        for g in known:
            if _fits_gutter(elems, g):
                found[k] = [g]
                break
    return found


def _fits_gutter(elems: list[_Elem], gutter: tuple[float, float]) -> bool:
    lines = [e for e in elems if e.kind == "line"]
    left = [e for e in lines if e.x1 <= gutter[0] + 2]
    right = [e for e in lines if e.x0 >= gutter[1] - 2]
    crossing = len(lines) - len(left) - len(right)
    if not left or not right or crossing > 0.15 * len(lines):
        return False
    return any(min(a.y1, b.y1) - max(a.y0, b.y0) > 0 for a in left for b in right)


def _side_by_side(cols: list[list[_Elem]]) -> bool:
    """Do neighbouring columns overlap vertically (rather than sit one above the other)?"""
    for a, b in zip(cols, cols[1:]):
        a0, a1 = min(e.y0 for e in a), max(e.y1 for e in a)
        b0, b1 = min(e.y0 for e in b), max(e.y1 for e in b)
        if min(a1, b1) - max(a0, b0) < 0.3 * min(a1 - a0, b1 - b0):
            return False
    return True


def _detect_gutters(elems: list[_Elem]) -> list[tuple[float, float]]:
    """Find vertical gutters between text columns.

    Two kinds of evidence: x ranges that narrow elements leave empty (works when the
    columns fill the page), and x ranges that fall in the gap between side-by-side
    fragments of many rows (works when full-width figures or code cross the gutter).
    """
    lines = [e for e in elems if e.kind == "line"]
    if len(lines) < 6:
        return []
    xmin = min(e.x0 for e in elems)
    xmax = max(e.x1 for e in elems)
    width = xmax - xmin
    if width < 150:
        return []
    center = (xmin + xmax) / 2
    lefts = Counter(round(e.x0 / 2) for e in lines)

    def column_like(e: _Elem) -> bool:
        # Lines centred on the page (a title block above two columns) would hide the
        # gutter; they become full-width elements later. Lines of a middle column are
        # centred too, but share their left edge with many others.
        if abs((e.x0 + e.x1) / 2 - center) > 0.03 * width:
            return True
        k = round(e.x0 / 2)
        return lefts[k - 1] + lefts[k] + lefts[k + 1] >= 4

    nbins = int(width) + 2
    lo, hi = int(0.15 * width), int(0.85 * width)
    candidate_sets: list[list[tuple[float, float]]] = []

    narrow = [e for e in elems if e.x1 - e.x0 <= 0.62 * width and column_like(e)]
    if len(narrow) >= 6:
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
        candidate_sets.append(_runs(cov, lo, hi, lambda v: v <= threshold, best=min, xmin=xmin))

    votes = [0] * (nbins + 1)
    rows = _rows_of(elems)
    for row in rows:
        segs: list[list[float]] = []
        for e in sorted(row, key=lambda e: e.x0):
            if segs and e.x0 - segs[-1][1] < 6:
                segs[-1][1] = max(segs[-1][1], e.x1)
            else:
                segs.append([e.x0, e.x1])
        for (_, a), (b, _) in zip(segs, segs[1:]):
            for k in range(max(0, int(math.ceil(a - xmin))), min(nbins, int(b - xmin)) + 1):
                votes[k] += 1
    need = max(3, 0.2 * len(rows))
    candidate_sets.append(_runs(votes, lo, hi, lambda v: v >= need, best=max, xmin=xmin))

    mid = (xmin + xmax) / 2
    for candidates in candidate_sets:
        if not candidates:
            continue
        if _valid_columns(lines, candidates, xmin, xmax):
            return candidates
        # Fall back to the single best gutter (closest to the middle, widest).
        for g in sorted(candidates, key=lambda g: (abs((g[0] + g[1]) / 2 - mid), -(g[1] - g[0]))):
            if _valid_columns(lines, [g], xmin, xmax):
                return [g]
    return []


def _runs(
    values: list[int], lo: int, hi: int, good: Callable[[int], bool], *, best: Callable, xmin: float
) -> list[tuple[float, float]]:
    """Maximal runs of ``good`` bins in [lo, hi], each narrowed to its best part (>= 5 bins wide)."""
    runs: list[tuple[int, int]] = []
    start = None
    for k in range(lo, hi + 1):
        if good(values[k]):
            if start is None:
                start = k
        elif start is not None:
            runs.append((start, k - 1))
            start = None
    if start is not None:
        runs.append((start, hi))
    out = []
    for a, b in runs:
        target = best(values[a : b + 1])
        k = a
        while k <= b:
            if values[k] == target:
                j = k
                while j + 1 <= b and values[j + 1] == target:
                    j += 1
                if j - k + 1 >= 5:
                    out.append((xmin + k, xmin + j + 1))
                k = j + 1
            else:
                k += 1
    return out


def _rows_of(elems: list[_Elem]) -> list[list[_Elem]]:
    """Cluster elements into visual rows (vertical overlap of at least 30%)."""
    rows: list[list[_Elem]] = []
    bounds: list[list[float]] = []
    for e in sorted(elems, key=lambda e: (e.y0 + e.y1) / 2):
        if e.kind != "line":
            continue
        if bounds:
            b = bounds[-1]
            overlap = min(b[1], e.y1) - max(b[0], e.y0)
            if overlap >= 0.3 * min(b[1] - b[0], e.y1 - e.y0):
                rows[-1].append(e)
                b[0], b[1] = min(b[0], e.y0), max(b[1], e.y1)
                continue
        rows.append([e])
        bounds.append([e.y0, e.y1])
    return rows


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
        rows = _rows_of(m)
        if len(rows) < 3:
            return False
        chars = [sum(len(e.line.text) + 1 for e in row) for row in rows]
        if sum(chars) / len(chars) < 12 or sum(chars) < 80:
            return False
        lefts = Counter(round(min(e.x0 for e in row) / 3) for row in rows)
        if lefts.most_common(1)[0][1] < 0.35 * len(rows):
            return False  # not a text column: rows don't share a left edge
    return _side_by_side(members)


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
            near = gap <= 4 * size or (ln.mono and all(r.mono for r in row))
            if smaller > 0 and overlap >= 0.5 * smaller and not any(_h_overlap(ln, r) for r in row) and near:
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
        out.append(merged)
    return out


def _h_overlap(a: _Line, b: _Line) -> bool:
    return min(a.x1, b.x1) - max(a.x0, b.x0) > 2


def _mark_footnotes(groups: list[tuple[tuple, list[_Elem]]], stats: _Stats, page_height: float) -> None:
    """Mark small print below a column's body text.

    Lines starting with a note marker ("1", "*", a raised number) become footnotes.
    Unmarked small print set apart at the bottom (a copyright notice, a source line)
    becomes a "note": still read, but it no longer splits a paragraph that continues
    in the next column or on the next page.
    """
    body = stats.body_size
    for _key, items in groups:
        lines = [e.line for e in items if e.kind == "line"]
        if not lines:
            continue
        body_lines = [ln for ln in lines if ln.size >= body * 0.95]
        if not body_lines:
            continue
        last_body = max(ln.y1 for ln in body_lines)
        below = [ln for ln in lines if ln.y0 >= last_body - 1 and ln.size <= body * 0.92]
        if not below:
            continue
        first = below[0]
        note = (
            first.y0 - last_body >= body
            and below[-1].y1 - first.y0 <= 0.3 * page_height
            and not _is_caption_text(first.text)
        )
        role = ""
        for ln in lines:
            if ln.y0 < last_body - 1 or ln.size > body * 0.92:
                role = ""
                continue
            if _FOOTNOTE_START_RE.match(ln.text) or ln.lead_marker:
                role = "footnote"
            elif ln is first and note:
                role = "note"
            if role:
                ln.role = role


# --------------------------------------------------------------------------- lines -> chunks


def _ends_terminal(text: str) -> bool:
    t = text.rstrip(_CLOSERS)
    return bool(t) and t[-1] in _TERMINAL


def _list_marker(line: _Line) -> str | None:
    """'bullet' / 'number' / 'dash' when the line starts like a list item."""
    t = line.text
    if not t:
        return None
    if line.symbol_lead or t[0] in _BULLETS:
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


def _column_geometry(lines: list[_Line]) -> tuple[float, float, float]:
    """(left edge, right edge, raggedness) of a column of text lines.

    Raggedness is how far full lines typically fall short of the right edge: about
    zero for justified text, a few words' width for ragged-right text.
    """
    left = min(ln.x0 for ln in lines)
    right = max(ln.x1 for ln in lines)
    ends = sorted(ln.x1 for ln in lines if ln.x1 - ln.x0 >= 0.5 * (right - left))
    ragged = 0.0
    if len(ends) >= 5:
        ragged = min(ends[int(0.9 * (len(ends) - 1))] - ends[int(0.25 * (len(ends) - 1))], 0.35 * (right - left))
    return left, right, ragged


def _is_short(line: _Line, col_right: float, width: float, ragged: float) -> bool:
    """Does ``line`` stop clearly before the column's right edge (a paragraph's last line)?"""
    return line.x1 < col_right - max(3 * line.size, 0.15 * width, 1.8 * ragged + 0.5 * line.size)


def _group_lines(
    lines: list[_Line],
    col: tuple,
    stats: _Stats,
    interrupted: bool,
    bounds: tuple[float, float, float] | None = None,
) -> list[_Chunk]:
    """Group a column's ordered lines into chunks using spacing, indentation and style.

    ``bounds`` is the column's geometry (see :func:`_column_geometry`) when ``lines``
    is only part of it.
    """
    if not lines:
        return []
    col_left, col_right, ragged = bounds or _column_geometry(lines)
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
            if _continues(cur, prev, ln, eq, cur_eq, marker, cur_list, (col_left, col_right, ragged), stats):
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
            top=not chunks,
            indent=ln.x0 - col_left,
        )
        cur_list = marker is not None and (marker == "bullet" or _closed_before(chunks, col_right, width, ragged))
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
        elif ch.lines[0].role == "note":
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


def _closed_before(chunks: list[_Chunk], col_right: float, width: float, ragged: float) -> bool:
    """Did the previous line end a sentence or item (so a numbered line starts a list)?"""
    if not chunks:
        return True
    prev = chunks[-1].lines[-1]
    if chunks[-1].kind == BlockKind.LIST_ITEM:
        return True
    return _ends_terminal(prev.text) or _is_short(prev, col_right, width, ragged)


def _continues(
    cur: _Chunk,
    prev: _Line,
    ln: _Line,
    eq: bool,
    cur_eq: bool,
    marker: str | None,
    cur_list: bool,
    geometry: tuple[float, float, float],
    stats: _Stats,
) -> bool:
    """Does line ``ln`` continue chunk ``cur`` (whose last line is ``prev``)?"""
    col_left, col_right, ragged = geometry
    width = max(col_right - col_left, 1.0)
    if ln.role != prev.role:
        return False
    if eq != cur_eq:
        return False
    size = max(prev.size, ln.size)
    if abs(prev.size - ln.size) > max(0.6, 0.08 * size):
        return False
    short = _is_short(prev, col_right, width, ragged)
    closed = short or _ends_terminal(prev.text)
    if prev.mono != ln.mono and (closed or cur.lines[0].mono or len(ln.text) > 60):
        return False  # a code block starts or ends (a short mono line mid-sentence is inline code)
    if _LEADER_RE.search(prev.text) or _LEADER_RE.search(ln.text):
        return False  # table-of-contents entries: one per line
    if prev.bold != ln.bold and not (prev.mono and ln.mono) and closed:
        return False  # a bold heading line, or a bold line after a finished paragraph
    dy = ln.baseline - prev.baseline
    expected = stats.pitch * size
    limit = expected * (2.6 if (prev.mono and ln.mono) else 1.3) + 0.5
    if dy <= 0.2 * size or dy > limit:
        return False
    if min(ln.x1, max(l.x1 for l in cur.lines)) - max(ln.x0, min(l.x0 for l in cur.lines)) < -2:
        return False  # no horizontal overlap with the chunk
    if ln.role == "footnote" and (_FOOTNOTE_START_RE.match(ln.text) or ln.lead_marker):
        return False
    if prev.mono and ln.mono and cur.lines[0].mono:
        return True
    if eq:
        return True
    if marker == "bullet" or (marker is not None and (closed or cur_list)):
        return False
    if _is_caption_text(ln.text) and closed:
        return False
    first = cur.lines[0]
    if cur_list:
        if ln.x0 >= first.x0 + 0.3 * size and ln.x0 <= first.x0 + 8 * size:
            return True  # hanging indent continuation
        return abs(ln.x0 - first.x0) <= 2 and not closed
    centered = (
        abs((prev.x0 + prev.x1) / 2 - (ln.x0 + ln.x1) / 2) < max(3.0, 0.02 * width)
        and prev.x0 > col_left + size
        and ln.x0 > col_left + size
    )
    display = size >= stats.body_size * 1.12  # heading-sized text has no first-line indents
    if ln.x0 > prev.x0 + 0.6 * size and ln.x0 - col_left < 0.4 * width and not centered and not display:
        return False  # first-line indent of a new paragraph
    if short and _ends_terminal(prev.text):
        return False
    return True


# --------------------------------------------------------------------------- classification


def _chunk_text(ch: _Chunk, evidence: HyphenationEvidence | None = None) -> str:
    if ch.kind == BlockKind.CODE:
        left = min(ln.x0 for ln in ch.lines)
        out = []
        for ln in ch.lines:
            indent = max(0, round((ln.x0 - left) / max(ln.size * 0.6, 1))) + ln.lead_spaces
            out.append(" " * indent + fix_characters(ln.text))
        return "\n".join(out)
    return clean_display_text("\n".join(ln.text for ln in ch.lines), evidence=evidence)


def _classify(chunks: list[_Chunk], stats: _Stats, evidence: HyphenationEvidence | None = None) -> None:
    body = stats.body_size
    for ch in chunks:
        if ch.kind == BlockKind.TABLE or not ch.lines:
            continue
        if ch.kind not in (BlockKind.FOOTNOTE, BlockKind.EQUATION) and all(ln.mono for ln in ch.lines):
            if len(ch.lines) >= 2 or ch.lines[0].chars >= 12:
                ch.kind = BlockKind.CODE
        ch.raw = [ln.text for ln in ch.lines]
        ch.text = _chunk_text(ch, evidence)
        if ch.kind in (BlockKind.FOOTNOTE, BlockKind.EQUATION, BlockKind.CODE):
            continue
        text = ch.text
        if ch.kind == BlockKind.LIST_ITEM:
            if text[:1] in _BULLETS or _DASH_BULLET_RE.match(text):
                text = text[1:].lstrip()
            ch.text = text
            if _heading_like(ch, body):  # "1. Introduction" set as a heading
                ch.kind = BlockKind.HEADING
            continue
        if _LEADER_RE.search(text):
            ch.kind = BlockKind.LIST_ITEM  # a table-of-contents entry: "Cells … 45"
            ch.text = re.sub(r"\s*(?:[.·…_]\s?){3,}\s*", " … ", text)
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
    heads = [ch for ch in chunks if ch.kind == BlockKind.HEADING and ch.style != ("ocr",)]
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


def _continues_across(p: _Chunk, q: _Chunk, broken: bool, indent_style: bool) -> bool:
    """Does chunk ``q`` continue paragraph ``p`` (split by a column/page break or a float)?"""
    if q.kind != BlockKind.PARAGRAPH or p.kind not in (BlockKind.PARAGRAPH, BlockKind.LIST_ITEM):
        return False
    if p.in_box != q.in_box or not p.text or not q.text:
        return False
    if abs(p.size - q.size) > max(0.6, 0.08 * p.size):
        return False
    if broken and indent_style and p.kind == BlockKind.PARAGRAPH and q.indent is not None:
        # In a book whose paragraphs start with an indent, a flush-left chunk at the top
        # of a column / page (or after a figure) is the rest of the previous paragraph.
        return q.indent < 0.3 * q.size
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


def _uses_first_line_indents(chunks: list[_Chunk]) -> bool:
    """Do this document's paragraphs start with an indented first line?"""
    indented = flush = 0
    for ch in chunks:
        if ch.kind != BlockKind.PARAGRAPH or len(ch.lines) < 2 or ch.top or ch.indent is None or ch.in_box:
            continue
        step = ch.lines[0].x0 - ch.lines[1].x0
        if step > 0.6 * ch.size and ch.indent > 0.6 * ch.size:
            indented += 1
        elif abs(step) <= 1.5:
            flush += 1
    return indented >= 3 and indented >= 0.6 * (indented + flush)


def _assemble(chunks: list[_Chunk], evidence: HyphenationEvidence | None = None) -> list[_Chunk]:
    """Merge paragraphs split by breaks; floats that interrupt a paragraph move after it."""
    indent_style = _uses_first_line_indents(chunks)
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
            if _continues_across(open_para, ch, broken, indent_style):
                open_para.text = clean_display_text(open_para.text + "\n" + ch.text, evidence=evidence)
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


def _is_title_block(block: Block, title: str) -> bool:
    """Is ``block`` the visible heading of an outline entry called ``title``?"""
    if len(block.text) > 200:
        return False
    if block.kind != BlockKind.HEADING and len(block.text) > 1.5 * len(title) + 10:
        return False  # a paragraph that merely mentions the title
    return _titles_match(block.text, title)


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
    entries = [e for e in entries if not _JUNK_TITLE_RE.search(e[1])]  # "chapter3.docx", "htmldoc797.html"
    if sum(1 for e in entries if e[0] <= 2) < 2:
        return None  # a single bookmark says nothing about the structure: use the headings
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
        if idx < len(blocks) and blocks[idx].page == page and _is_title_block(blocks[idx], etitle):
            blocks[idx].kind = BlockKind.HEADING  # (an undetected heading becomes one)
            blocks[idx].level = min(lvl, 6)
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
            if want_heading == (blocks[i].kind == BlockKind.HEADING) and _is_title_block(blocks[i], title):
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


def _fold_lone_headings(sections: list[Section]) -> list[Section]:
    """A section that is only a heading, followed by a sibling rather than a child (the author
    line and date under a title, say), joins the section before it."""
    out: list[Section] = []
    for k, sec in enumerate(sections):
        nxt = sections[k + 1] if k + 1 < len(sections) else None
        if out and len(sec.blocks) == 1 and (nxt is None or nxt.level <= sec.level):
            prev = out[-1]
            out[-1] = make_section(prev.title, prev.level, prev.blocks + sec.blocks)
            continue
        out.append(sec)
    return out


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
    title = clean_display_text(_CONTROL_RE.sub("", str(value)))
    if len(title) < 2 or len(title) > 300 or not any(c.isalpha() for c in title):
        return None
    if _JUNK_TITLE_RE.search(title):
        return None
    return title


def _metadata_author(value: str | None) -> str | None:
    if not value:
        return None
    author = clean_display_text(_CONTROL_RE.sub("", str(value)))
    if not author or len(author) > 300 or _JUNK_AUTHOR_RE.match(author):
        return None
    return author


def _heading_title(blocks: list[Block]) -> str | None:
    for b in blocks:
        if (b.page or 1) > 3:
            break
        if b.kind == BlockKind.HEADING and b.level == 1 and 2 <= len(b.text) <= 200:
            letters = sum(1 for c in b.text if c.isalpha())
            if _norm_title(b.text) in _GENERIC_HEADINGS or letters < 0.5 * len(b.text.replace(" ", "")):
                continue  # "Contents", or a scale bar like "0 50 100 200 Miles"
            return b.text
    return None


# --------------------------------------------------------------------------- warnings text


def _pages_phrase(pages0: list[int], *, lower: bool = False) -> str:
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
    word = "Page " if len(nums) == 1 else "Pages "
    return (word.lower() if lower else word) + ", ".join(parts)


def _was_were(pages0: list[int]) -> str:
    return "was" if len(set(pages0)) == 1 else "were"


def _it_they(pages0: list[int]) -> str:
    return "it" if len(set(pages0)) == 1 else "they"
