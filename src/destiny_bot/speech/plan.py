"""Turn a :class:`Section` into its reading plan: the utterances the player speaks.

Every utterance points back at the display text it reads (``block``, ``start``,
``end``) so the reader can highlight it, and carries the normalized ``speech``
sent to the voice. See docs/ARCHITECTURE.md for the guarantees.
"""

from __future__ import annotations

import re
from typing import Callable

from ..models import Block, BlockKind, ReadingOptions, Section, Utterance
from .normalize import normalize_for_speech, verbalize_math
from .segment import chunk_span, split_sentences

__all__ = ["plan_section", "section_plain_text"]

# The pause after an utterance is the only gap between them (the server trims the
# silence a voice adds at the start and end of each clip), so every one is non-zero.
PAUSE_CHUNK_MS = 60  # after a piece of one long sentence split at , ; : or a space
PAUSE_SENTENCE_MS = 220  # between sentences (and code lines) inside a block
PAUSE_BLOCK_MS = 450  # after a paragraph, quote, caption, footnote, figure, equation, code or table
PAUSE_ROW_MS = 300  # after a list item, a table announcement and each table row but the last
PAUSE_HEADING_MS = 750  # after a heading or the section-title announcement

SPEAKER_NOTES_PREFIX = "Speaker notes: "
FOOTNOTE_PREFIX = "Footnote: "
IMAGE_PREFIX = "Image: "
EQUATION_PREFIX = "Equation: "
CODE_ANNOUNCEMENT = "There is a code example here."
TRUNCATION_MARKER = "\n[...the rest of this section was cut for length]"

_TEXT_KINDS = frozenset({BlockKind.PARAGRAPH, BlockKind.QUOTE, BlockKind.LIST_ITEM, BlockKind.CAPTION,
                         BlockKind.FOOTNOTE})
_SPEAKABLE = re.compile(r"[^\W_]")
_NUMERIC_CELL = re.compile(r"[\s\d.,%$€£¥+\-−–—()/:×x]*\d[\s\d.,%$€£¥+\-−–—()/:×x]*")
_YEAR_CELL = re.compile(r"(?:1[5-9]|20)\d\d")
_LEADING_BULLET = re.compile(r"^[•▪◦■□►▸▶➢➤✓✔●○◆◇★☆♦❖⁃‣\-*]\s+")

Normalizer = Callable[[str], str]
# (start, end, speech, pause_after_ms) for one utterance inside a block
_Piece = tuple[int, int, str, int]


def plan_section(section: Section, options: ReadingOptions | None = None) -> list[Utterance]:
    """The reading plan for ``section``: utterances in reading order, indexed 0..n-1."""
    opts = options or ReadingOptions()
    planned: list[tuple[int, BlockKind, _Piece]] = []

    blocks = section.blocks
    if opts.announce_section_title and (not blocks or blocks[0].kind != BlockKind.HEADING):
        title = normalize_for_speech(section.title, opts)
        parts = _split_speech(title, opts.max_chars)
        pieces = _end_pause([(0, 0, part, PAUSE_CHUNK_MS) for part in parts], PAUSE_HEADING_MS)
        planned.extend((-1, BlockKind.HEADING, piece) for piece in pieces)

    notes_prefix_pending = False
    previous_was_notes = False
    for index, block in enumerate(blocks):
        is_notes = bool(block.meta.get("speaker_notes"))
        if is_notes and not previous_was_notes:
            notes_prefix_pending = True
        elif not is_notes:
            notes_prefix_pending = False
        previous_was_notes = is_notes

        lead = SPEAKER_NOTES_PREFIX if notes_prefix_pending else ""
        pieces = _plan_block(block, opts, lead)
        if pieces:
            notes_prefix_pending = False
            planned.extend((index, block.kind, piece) for piece in pieces)

    return [
        Utterance(index=i, block=b, start=start, end=end, kind=kind, speech=speech, pause_after_ms=pause)
        for i, (b, kind, (start, end, speech, pause)) in enumerate(planned)
    ]


# --------------------------------------------------------------------------- blocks


def _plan_block(block: Block, opts: ReadingOptions, lead: str) -> list[_Piece]:
    """Utterance pieces for one block (empty when options skip it), with their pauses."""
    kind = block.kind
    text = block.text or ""
    normalize = _normalizer(opts)

    if kind == BlockKind.HEADING:
        if not opts.read_headings:
            return []
        start, end = _trimmed(text)
        pieces = _speak_span(text, start, end, opts.max_chars - len(lead), normalize) if start < end else []
        return _end_pause(_with_prefix(pieces, lead), PAUSE_HEADING_MS)

    if kind in _TEXT_KINDS:
        if kind == BlockKind.CAPTION and not opts.read_captions:
            return []
        if kind == BlockKind.FOOTNOTE and not opts.read_footnotes:
            return []
        prefix = lead + (FOOTNOTE_PREFIX if kind == BlockKind.FOOTNOTE else "")
        pause = PAUSE_ROW_MS if kind == BlockKind.LIST_ITEM else PAUSE_BLOCK_MS
        return _end_pause(_sentences(text, opts.max_chars, normalize, prefix), pause)

    if kind == BlockKind.FIGURE:
        if not opts.read_figures or not text.strip():
            return []
        return _end_pause(_sentences(text, opts.max_chars, normalize, lead + IMAGE_PREFIX), PAUSE_BLOCK_MS)

    if kind == BlockKind.EQUATION:
        if opts.equations != "read":
            return []
        spoken = verbalize_math(text)
        return _end_pause(_whole_block(text, spoken, opts.max_chars, lead + EQUATION_PREFIX), PAUSE_BLOCK_MS)

    if kind == BlockKind.CODE:
        if opts.code == "skip" or not text.strip():
            return []
        if opts.code == "announce":
            pieces = _whole_block(text, CODE_ANNOUNCEMENT, opts.max_chars, lead)
        else:
            pieces = _code_lines(text, opts, lead)
        return _end_pause(pieces, PAUSE_BLOCK_MS)

    if kind == BlockKind.TABLE:
        return _end_pause(_table(block, opts, normalize, lead), PAUSE_BLOCK_MS)

    return []


def _normalizer(opts: ReadingOptions) -> Normalizer:
    return lambda s: normalize_for_speech(s, opts)


def _trimmed(text: str) -> tuple[int, int]:
    stripped = text.strip()
    if not stripped:
        return 0, 0
    start = text.index(stripped[0])
    return start, start + len(stripped)


def _with_prefix(pieces: list[_Piece], prefix: str) -> list[_Piece]:
    if pieces and prefix:
        start, end, speech, pause = pieces[0]
        pieces[0] = (start, end, prefix + speech, pause)
    return pieces


def _end_pause(pieces: list[_Piece], pause: int) -> list[_Piece]:
    """Give the last piece ``pause`` (the gap after the whole sentence / block)."""
    if pieces:
        start, end, speech, _ = pieces[-1]
        pieces[-1] = (start, end, speech, pause)
    return pieces


def _sentences(text: str, max_chars: int, normalize: Normalizer, prefix: str) -> list[_Piece]:
    """One or more utterances per sentence; ``prefix`` goes before the first one spoken."""
    pieces: list[_Piece] = []
    for start, end in split_sentences(text):
        found = _speak_span(text, start, end, max_chars - len(prefix), normalize)
        if found and prefix:
            found = _with_prefix(found, prefix)
            prefix = ""
        pieces.extend(_end_pause(found, PAUSE_SENTENCE_MS))
    return pieces


def _speak_span(text: str, start: int, end: int, budget: int, normalize: Normalizer) -> list[_Piece]:
    """Normalized speech for ``text[start:end]``, split so every piece fits in ``budget``.

    Long spans are cut at display level (so each piece still highlights exactly what
    it says) using a target length scaled by how much normalization expands the
    text. A single word whose speech is still too long is split in the speech only.
    All pieces get the short chunk pause; callers set the pause after the last one.
    """
    speech = normalize(text[start:end])
    if not speech:
        return []
    if len(speech) <= budget:
        return [(start, end, speech, PAUSE_CHUNK_MS)]
    length = end - start
    target = max(1, min(length - 1, int(budget * length / len(speech) * 0.95)))
    spans = chunk_span(text, start, end, target, split_words=False) if length > 1 else [(start, end)]
    if len(spans) <= 1:
        return [(start, end, part, PAUSE_CHUNK_MS) for part in _split_speech(speech, budget)]
    pieces: list[_Piece] = []
    for s, e in spans:
        pieces.extend(_speak_span(text, s, e, budget, normalize))
    return pieces


def _split_speech(speech: str, max_chars: int) -> list[str]:
    """Split speech-only text (no display mapping) into speakable parts of at most ``max_chars``."""
    if not speech:
        return []
    if len(speech) <= max_chars:
        return [speech]
    parts = [speech[s:e] for s, e in chunk_span(speech, 0, len(speech), max_chars)]
    return [p for p in parts if _SPEAKABLE.search(p)]


def _whole_block(text: str, speech: str, max_chars: int, prefix: str) -> list[_Piece]:
    """Speech that highlights the whole block (equations, code and table announcements)."""
    if not speech or not _SPEAKABLE.search(speech):
        return []
    parts = _split_speech(speech, max_chars - len(prefix))
    return _with_prefix([(0, len(text), part, PAUSE_CHUNK_MS) for part in parts], prefix)


# --------------------------------------------------------------------------- code

_CODE_COMMENT = re.compile(r"^(?:#+|//+|--|;+|%+|/\*+|\*+/?)\s*")
_CODE_OPERATORS = [
    ("===", " is identical to "), ("!==", " is not identical to "), ("==", " is equal to "),
    ("!=", " is not equal to "), ("<=", " is less than or equal to "), (">=", " is greater than or equal to "),
    ("+=", " plus equals "), ("-=", " minus equals "), ("*=", " times equals "), ("/=", " divide equals "),
    ("->", " arrow "), ("=>", " arrow "), ("&&", " and "), ("||", " or "), ("++", " plus plus "),
    ("**", " to the power of "), ("::", " "),
]


def _code_lines(text: str, opts: ReadingOptions, lead: str) -> list[_Piece]:
    """Read code line by line; each line highlights itself."""
    code_opts = opts.model_copy(update={"citations": "keep", "expand_abbreviations": False})

    def normalize(line: str) -> str:
        stripped = line.strip()
        comment = _CODE_COMMENT.match(stripped)
        if comment and stripped[comment.end() :]:
            body = normalize_for_speech(stripped[comment.end() :], opts)
            return f"Comment: {body}" if body else ""
        for op, words in _CODE_OPERATORS:
            stripped = stripped.replace(op, words)
        return normalize_for_speech(stripped, code_opts)

    pieces: list[_Piece] = []
    prefix = lead
    pos = 0
    for line in text.splitlines(keepends=True):
        start, end = _trimmed(line)
        if start < end:
            found = _speak_span(text, pos + start, pos + end, opts.max_chars - len(prefix), normalize)
            if found and prefix:
                found = _with_prefix(found, prefix)
                prefix = ""
            pieces.extend(_end_pause(found, PAUSE_SENTENCE_MS))
        pos += len(line)
    return pieces


# --------------------------------------------------------------------------- tables


def _table(block: Block, opts: ReadingOptions, normalize: Normalizer, lead: str) -> list[_Piece]:
    """Table utterances; the caller sets the pause after the last one."""
    if opts.tables == "skip":
        return []
    text = block.text or ""
    rows = _table_rows(block)
    if not rows:
        if opts.tables == "announce":
            return _whole_block(text, "There is a table here.", opts.max_chars, lead)
        return _sentences(text, opts.max_chars, normalize, lead + "Table: ")

    header = rows[0] if _has_header(rows, block.meta) else None
    data = rows[1:] if header is not None else rows
    n_rows, n_cols = len(data), max(len(r) for r in rows)
    size = f"{n_rows} {'row' if n_rows == 1 else 'rows'} and {n_cols} {'column' if n_cols == 1 else 'columns'}"
    if opts.tables == "announce":
        return _whole_block(text, f"There is a table here with {size}.", opts.max_chars, lead)

    whole = (0, len(text))
    pieces = _end_pause(_whole_block(text, f"Table with {size}.", opts.max_chars, lead), PAUSE_ROW_MS)
    labels = [normalize(cell) if cell else "" for cell in header] if header else []
    for number, row in enumerate(data, start=1):
        cells: list[str] = []
        for col, cell in enumerate(row):
            value = normalize(cell) if cell else ""
            if not value:
                continue
            label = labels[col] if col < len(labels) else ""
            cells.append(f"{label}, {value}" if label else value)
        parts = [(*whole, part, PAUSE_CHUNK_MS) for part in _pack(f"Row {number}: ", cells, opts.max_chars)]
        pieces.extend(_end_pause(parts, PAUSE_ROW_MS))
    return pieces


def _table_rows(block: Block) -> list[list[str]]:
    """The table's non-empty rows, with trailing empty cells removed."""
    rows = []
    for row in block.rows or []:
        cells = [(c or "").strip() if isinstance(c, str) else str(c).strip() for c in row]
        while cells and not cells[-1]:
            cells.pop()
        if cells:
            rows.append(cells)
    return rows


def _has_header(rows: list[list[str]], meta: dict) -> bool:
    """Does the first row label the columns? Uses the loader's hint when it gave one."""
    for key in ("header", "has_header"):
        if isinstance(meta.get(key), bool):
            return bool(meta[key]) and len(rows) >= 2
    if isinstance(meta.get("header_rows"), int) and not isinstance(meta.get("header_rows"), bool):
        return meta["header_rows"] >= 1 and len(rows) >= 2
    if len(rows) < 2:
        return False
    cells = [c for c in rows[0] if c]
    if not cells or len(cells) < min(2, len(rows[0])):
        return False
    if any(len(c) > 60 for c in cells):
        return False
    if any(_NUMERIC_CELL.fullmatch(c) and not _YEAR_CELL.fullmatch(c) for c in cells):
        return False
    return len({c.casefold() for c in cells}) == len(cells)


def _pack(head: str, parts: list[str], max_chars: int) -> list[str]:
    """``head`` + parts joined by "; ", starting a new utterance whenever the next part won't fit."""
    out: list[str] = []
    current, fresh = head, True  # fresh: nothing but the head so far
    for part in parts:
        sep = "" if fresh else "; "
        if len(current) + len(sep) + len(part) <= max_chars:
            current, fresh = current + sep + part, False
            continue
        if not fresh:
            out.append(current)
            current, fresh = "", True
        room = max_chars - len(current)
        if len(part) <= room:
            current, fresh = current + part, False
            continue
        chunks = _split_speech(part, room)  # a single cell longer than an utterance
        if not chunks:
            continue
        out.append(current + chunks[0])
        out.extend(chunks[1:-1])
        current, fresh = (chunks[-1], False) if len(chunks) > 1 else ("", True)
    if not fresh:
        out.append(current)
    return out


# --------------------------------------------------------------------------- plain text


def section_plain_text(section: Section, *, max_chars: int | None = None) -> str:
    """Readable text of a section for an AI prompt (Markdown-like, tables as pipe rows)."""
    parts: list[str] = []
    blocks = section.blocks
    if section.title.strip() and (not blocks or blocks[0].kind != BlockKind.HEADING):
        parts.append(f"# {section.title.strip()}")
    previous: BlockKind | None = None
    for block in blocks:
        piece = _plain_block(block)
        if not piece:
            continue
        if block.kind == BlockKind.LIST_ITEM and previous == BlockKind.LIST_ITEM and parts:
            parts[-1] += "\n" + piece
        else:
            parts.append(piece)
        previous = block.kind
    text = "\n\n".join(parts)
    if max_chars is None or len(text) <= max_chars:
        return text
    keep = max_chars - len(TRUNCATION_MARKER)
    if keep <= 0:
        return TRUNCATION_MARKER.lstrip("\n")
    cut = text.rfind("\n", 0, keep + 1)
    if cut < keep * 0.8:
        cut = text.rfind(" ", 0, keep + 1)
    if cut < keep * 0.5:
        cut = keep
    return text[:cut].rstrip() + TRUNCATION_MARKER


def _plain_block(block: Block) -> str:
    text = (block.text or "").strip()
    kind = block.kind
    if kind == BlockKind.HEADING:
        return f"{'#' * min(max(block.level or 1, 1), 6)} {text}" if text else ""
    if kind == BlockKind.TABLE:
        rows = _table_rows(block)
        if not rows:
            return f"Table: {text}" if text else ""
        width = max(len(r) for r in rows)
        lines = ["| " + " | ".join(_pipe_cell(c) for c in r + [""] * (width - len(r))) + " |" for r in rows]
        if _has_header(rows, block.meta):
            lines.insert(1, "|" + " --- |" * width)
        return "\n".join(lines)
    if kind == BlockKind.FIGURE:
        return f"[Image: {text}]" if text else "[Image]"
    if not text:
        return ""
    if kind == BlockKind.CODE:
        return f"```\n{(block.text or '').strip(chr(10))}\n```"
    if kind == BlockKind.LIST_ITEM:
        return "- " + _LEADING_BULLET.sub("", text)
    if kind == BlockKind.QUOTE:
        return "> " + text
    if kind == BlockKind.CAPTION:
        return "Caption: " + text
    if kind == BlockKind.FOOTNOTE:
        return "Footnote: " + text
    if kind == BlockKind.EQUATION:
        return "Equation: " + text
    if block.meta.get("speaker_notes"):
        return "Speaker notes: " + text
    return text


def _pipe_cell(cell: str) -> str:
    return " ".join(cell.split()).replace("|", "/")
