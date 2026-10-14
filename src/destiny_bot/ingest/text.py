"""Plain text and Markdown: ``.txt`` / ``.md`` files and pasted text.

Also home to small helpers the other loaders share: byte decoding
(:func:`decode_text_bytes`), the plain-text paragraph/heading/bullet logic used
for OCR output (:func:`plain_text_blocks`), and table rendering
(:func:`table_block`).
"""

from __future__ import annotations

import codecs
import html as html_lib
import re
from pathlib import Path

from ..models import Block, BlockKind, Document
from . import IngestError, ProgressFn
from .common import (
    HyphenationEvidence,
    clean_display_text,
    doc_id_for_bytes,
    doc_id_for_file,
    finalize_sections,
    fix_characters,
    sections_from_blocks,
    title_from_filename,
)

MARKDOWN_EXTENSIONS = (".md", ".markdown")

# --------------------------------------------------------------------------- decoding

_BOMS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32"),  # before UTF-16 LE: it starts with the same two bytes
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


def _bom_encoding(data: bytes) -> str | None:
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            return encoding
    return None


def _guess_utf16(data: bytes) -> str | None:
    """Spot BOM-less UTF-16 (mostly-ASCII text has a NUL in every other byte)."""
    if len(data) < 4:
        return None
    half = len(data) // 2
    even_nuls, odd_nuls = data[0::2].count(0), data[1::2].count(0)
    if odd_nuls > 0.4 * half and even_nuls < 0.05 * half:
        return "utf-16-le"
    if even_nuls > 0.4 * half and odd_nuls < 0.05 * half:
        return "utf-16-be"
    return None


def decode_text_bytes(data: bytes, *, declared: str | None = None) -> str:
    """Decode text of unknown encoding.

    Order: byte-order mark, then the ``declared`` encoding (e.g. an HTML ``<meta
    charset>``), then UTF-8, Windows-1252 and finally Latin-1, which never fails.
    """
    bom = _bom_encoding(data)
    if bom:
        return data.decode(bom, errors="replace")
    if declared:
        try:
            codec = codecs.lookup(declared).name
        except LookupError:
            codec = None
        if codec in ("latin-1", "iso8859-1", "ascii"):
            codec = "cp1252"  # what browsers do: these labels mean Windows-1252 in practice
        if codec and not codec.startswith(("utf-16", "utf-32")):
            try:
                return data.decode(codec)
            except UnicodeDecodeError:
                pass
    utf16 = _guess_utf16(data)
    if utf16:
        return data.decode(utf16, errors="replace")
    for encoding in ("utf-8", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")


# --------------------------------------------------------------------------- shared block helpers


def text_block(kind: BlockKind, text: str, *, evidence: HyphenationEvidence | None = None, **fields) -> Block | None:
    """A cleaned block, or ``None`` when nothing readable is left."""
    if kind == BlockKind.CODE:
        cleaned = fix_characters(text).strip("\n")
    else:
        cleaned = clean_display_text(text, evidence=evidence)
    if not cleaned.strip():
        return None
    return Block(kind=kind, text=cleaned, **fields)


def rows_to_text(rows: list[list[str]]) -> str:
    """One-line rendering of a table: cells joined by " | ", rows by "; "."""
    return clean_display_text("; ".join(" | ".join(cell for cell in row if cell) for row in rows))


def table_block(rows: list[list[str]], *, page: int | None = None, **meta) -> Block | None:
    """A TABLE block with cleaned cells; empty rows and trailing empty columns are dropped."""
    cleaned = [[clean_display_text(cell or "") for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return None
    width = max(len(row) for row in cleaned)
    while width > 1 and not any(len(row) >= width and row[width - 1] for row in cleaned):
        width -= 1
    cleaned = [row[:width] for row in cleaned]
    return Block(kind=BlockKind.TABLE, text=rows_to_text(cleaned), rows=cleaned, page=page, meta=meta)


def code_text(text: str) -> str:
    """Code keeps its line breaks: trim trailing spaces and blank edge lines."""
    lines = [line.rstrip() for line in _normalize_newlines(text).split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return fix_characters("\n".join(lines))


_SUPERSCRIPT = str.maketrans("0123456789+-−=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁼⁽⁾ⁿⁱ")
_SUPERSCRIPTABLE = re.compile(r"[0-9+\-−=()ni]{1,6}")


def wrap_math(value: str) -> str:
    """Parenthesise a math sub-expression unless it is a single token (``x``, ``2``, ``ab``)."""
    return value if re.fullmatch(r"[\w.′']+", value) else f"({value})"


def superscript_text(text: str) -> str:
    """Display form of superscript text: ``2`` -> ``²``; ordinals and words stay; other exponents get ``^``."""
    value = text.strip()
    if not value:
        return ""
    if _SUPERSCRIPTABLE.fullmatch(value):
        return value.translate(_SUPERSCRIPT)
    if value.isalpha() or any(ch.isspace() for ch in value) or len(value) > 12:
        return value
    return "^" + wrap_math(value)


# --------------------------------------------------------------------------- plain text

_ROMAN = r"(?=[MDCLXVI])M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
_NUMBER_WORDS = (
    r"(?i:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|first|second|third|fourth|fifth|"
    r"sixth|seventh|eighth|ninth|tenth)"
)
_TOP_KEYWORDS = ("part", "book", "volume")
_SECTION_KEYWORDS = ("section", "topic")
_KEYWORD_HEADING = re.compile(
    r"^(?P<kw>(?i:part|book|volume|chapter|unit|module|lesson|lecture|week|appendix|section|topic))"
    rf"\s+(?P<num>\d{{1,3}}[A-Za-z]?|{_ROMAN}|{_NUMBER_WORDS}|[A-Z])"
    r"(?P<rest>\s*(?:[:.\-–—]\s*\S.*|\s+[A-Z0-9“\"'(].*|[:.]?))$"
)
_NUMBERED_HEADING = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){1,3})\.?\s+(?P<title>[A-Z0-9“\"'(].*)$")
_BULLET = re.compile(
    r"^(?P<indent>[ \t]*)(?P<marker>[-*•◦▪▫●○■□‣⁃–—+]|\d{1,3}[.)]|[a-z][.)]|\(\d{1,3}\)|\([a-z]\))"
    r"[ \t]+(?P<text>\S.*)$"
)
_RULE_LINE = re.compile(r"^\s*([=\-_*~#+.•·]\s*){3,}$")
_TERMINAL_PUNCT = re.compile(r"[.!?;,]\s*$")
_LETTERS = re.compile(r"[^\W\d_]")


def _is_heading_shaped(line: str, max_len: int = 100) -> bool:
    return 0 < len(line) <= max_len and not _TERMINAL_PUNCT.search(line) and len(line.split()) <= 14


def _heading_rank(line: str, *, allow_caps: bool) -> float | None:
    """Rank of a heading line (0 = part, 1 = chapter, ...), or ``None`` for ordinary text."""
    line = line.strip()
    m = _KEYWORD_HEADING.match(line)
    if m and len(line) <= 100:
        rest = m.group("rest").strip()
        if rest in ("", ".", ":") or not _TERMINAL_PUNCT.search(rest):
            keyword = m.group("kw").lower()
            if keyword in _TOP_KEYWORDS:
                return 0
            return 2 if keyword in _SECTION_KEYWORDS else 1
    m = _NUMBERED_HEADING.match(line)
    if m and _is_heading_shaped(line, 90):
        return 1 + m.group("num").count(".")
    if allow_caps and _is_caps_heading(line):
        return 2
    return None


def _is_caps_heading(line: str) -> bool:
    letters = _LETTERS.findall(line)
    return (
        len(letters) >= 4
        and _is_heading_shaped(line, 70)
        and line.upper() == line
        and line.lower() != line
        and len(letters) >= 0.5 * len(line.replace(" ", ""))
        and not _BULLET.match(line)
        and not _RULE_LINE.match(line)
    )


def _title_like(line: str) -> bool:
    """A short line that could finish a two-line heading ("CHAPTER 1" / "The Cell")."""
    return _is_heading_shaped(line, 70) and bool(_LETTERS.search(line)) and not _BULLET.match(line)


def _split_paragraphs(text: str) -> list[list[str]]:
    paragraphs: list[list[str]] = []
    current: list[str] = []
    for line in text.split("\n"):
        if line.strip():
            current.append(line.rstrip())
        elif current:
            paragraphs.append(current)
            current = []
    if current:
        paragraphs.append(current)
    return paragraphs


def _body_blocks(lines: list[str], page: int | None, evidence: HyphenationEvidence | None = None) -> list[Block]:
    """Paragraph and bullet blocks for the non-heading lines of one paragraph."""
    if len(lines) > 1 and max(len(line) for line in lines) > 200:
        # Not hard-wrapped: every line is its own paragraph.
        groups = [[line] for line in lines]
    else:
        groups = [lines]
    pieces: list[tuple[BlockKind, list[str]]] = []
    for group in groups:
        prose: list[str] = []
        item: list[str] | None = None
        item_indent = 0
        for line in group:
            bullet = _BULLET.match(line)
            if bullet:
                if prose:
                    pieces.append((BlockKind.PARAGRAPH, prose))
                    prose = []
                if item:
                    pieces.append((BlockKind.LIST_ITEM, item))
                marker, body = bullet.group("marker"), bullet.group("text")
                item = [f"{marker} {body}" if marker[0].isalnum() or marker[0] == "(" else body]
                item_indent = len(bullet.group("indent").expandtabs(4))
                continue
            indent = len(line) - len(line.lstrip())
            if item is not None:
                # Continuation of a bullet unless it is clearly a new, outdented paragraph.
                if indent > item_indent or not _TERMINAL_PUNCT.search(item[-1]):
                    item.append(line)
                    continue
                pieces.append((BlockKind.LIST_ITEM, item))
                item = None
            if prose and indent >= 2 and not prose[-1][:1].isspace() and _TERMINAL_PUNCT.search(prose[-1]):
                pieces.append((BlockKind.PARAGRAPH, prose))  # an indented first line starts a new paragraph
                prose = []
            prose.append(line)
        if prose:
            pieces.append((BlockKind.PARAGRAPH, prose))
        if item:
            pieces.append((BlockKind.LIST_ITEM, item))
    blocks = [text_block(kind, "\n".join(piece), page=page, evidence=evidence) for kind, piece in pieces]
    return [block for block in blocks if block is not None]


def plain_text_blocks(text: str, *, page: int | None = None) -> list[Block]:
    """Turn plain text (a .txt file, OCR output) into headings, paragraphs and bullets.

    Paragraphs are separated by blank lines; hard-wrapped lines are joined. Short
    lines without closing punctuation that look like "CHAPTER 1", "Chapter 3:
    Cells", "PART II", "1.2 Membrane Transport" or an ALL-CAPS title become
    headings; heading levels follow the kinds of headings the text uses.
    """
    text = _normalize_newlines(text).replace("\f", "\n\n").replace("\v", "\n")
    evidence = HyphenationEvidence(text.split("\n"))  # how the text spells "self-esteem" elsewhere
    raw: list[tuple[str, object]] = []  # ("heading", (rank, text)) | ("body", lines)
    for para in _split_paragraphs(text):
        # Underlined headings ("Notes" over "=====") and decorative rule lines.
        lines: list[str] = []
        for line in para:
            if _RULE_LINE.match(line):
                stripped = line.strip()
                if lines and stripped[0] in "=-" and _title_like(lines[-1].strip()) and len(lines) == 1:
                    raw.append(("heading", (1 if stripped[0] == "=" else 2, lines.pop().strip())))
                continue
            lines.append(line)
        if not lines:
            continue
        first = lines[0].strip()
        if len(lines) == 1:
            rank = _heading_rank(first, allow_caps=True)
            if rank is not None:
                raw.append(("heading", (rank, first)))
            else:
                raw.append(("body", lines))
            continue
        # A wrapped first line of prose is nearly as long as the lines after it; a heading isn't.
        short_first = len(first) <= 50 or len(first) < 0.8 * max(len(line.strip()) for line in lines[1:])
        rank = _heading_rank(first, allow_caps=False) if short_first else None
        if rank is not None:
            second = lines[1].strip()
            match = _KEYWORD_HEADING.match(first)
            bare_number = match is not None and match.group("rest").strip() in ("", ".", ":")
            if bare_number and _title_like(second) and (len(lines) == 2 or _is_caps_heading(second)):
                raw.append(("heading", (rank, f"{first.rstrip('.:')}: {second}")))
                rest = lines[2:]
            else:
                raw.append(("heading", (rank, first)))
                rest = lines[1:]
            if rest:
                raw.append(("body", rest))
            continue
        if _is_caps_heading(first) and not _is_caps_heading(lines[1].strip()):
            raw.append(("heading", (2, first)))
            raw.append(("body", lines[1:]))
            continue
        raw.append(("body", lines))

    ranks = sorted({item[0] for kind, item in raw if kind == "heading"})  # type: ignore[index]
    level_for = {rank: min(6, i + 1) for i, rank in enumerate(ranks)}
    blocks: list[Block] = []
    for kind, item in raw:
        if kind == "heading":
            rank, heading = item  # type: ignore[misc]
            block = text_block(BlockKind.HEADING, heading, level=level_for[rank], page=page)
            if block:
                blocks.append(block)
        else:
            blocks.extend(_body_blocks(item, page, evidence))  # type: ignore[arg-type]
    return blocks


# --------------------------------------------------------------------------- Markdown

_MD_ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
_MD_ATX_ANY = re.compile(r"^ {0,3}#{1,6}[ \t]+\S", re.M)
_MD_FENCE = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")
_MD_FENCE_ANY = re.compile(r"^ {0,3}(?:```|~~~)", re.M)
_MD_SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_MD_SETEXT_ANY = re.compile(r"^(?=[^\n]*\w)[^\n]*\n {0,3}={3,}[ \t]*$", re.M)  # (lookahead: linear time)
_MD_HR = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_MD_QUOTE = re.compile(r"^ {0,3}>[ \t]?")
_MD_LIST = re.compile(r"^(?P<indent>[ \t]*)(?P<marker>[-*+]|\d{1,9}[.)])(?:[ \t]+(?P<text>.*))?$")
_MD_TASK = re.compile(r"^\[[ xX]\][ \t]+")
_MD_FOOTNOTE_DEF = re.compile(r"^ {0,3}\[\^([^\]\s]+)\]:[ \t]*(.*)$")
_MD_REF_DEF = re.compile(r"^ {0,3}\[([^\]]+)\]:[ \t]*\S+")
_MD_REF_IMAGE = re.compile(r"!\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\[[^\[\]]*\]")
_MD_SHORTCUT_REF = re.compile(r"(?<![\]!\\])\[([^\[\]]+)\](?![\[(:])")
_MD_HIDDEN_HTML = (("<!--", "-->"), ("<script", "</script>"), ("<style", "</style>"))
_MD_HTML_ONLY = re.compile(r"^\s*(?:</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>\s*)+$")
_MD_IMG_HTML = re.compile(r"<img\b[^>]*?\balt\s*=\s*(?:\"([^\"]*)\"|'([^']*)')[^>]*>", re.I)
_MD_TABLE_DELIM_ANY = re.compile(r"^ {0,3}\|?[ \t]*:?-{3,}:?[ \t]*(?:\|[ \t]*:?-{3,}:?[ \t]*)+\|?[ \t]*$", re.M)

_DEST = (
    r"\(\s*(?:<[^>\n]*>|[^()\s]*(?:\([^()\s]*\)[^()\s]*)*)"
    r"(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^()\n]*\)))?\s*\)"
)
_MD_IMAGE = re.compile(r"!\[((?:\\.|[^\[\]\\])*)\]" + _DEST)
_MD_LINK = re.compile(r"\[((?:\\.|[^\[\]\\]|\[[^\[\]]*\])*)\]" + _DEST)
_MD_REF_LINK = re.compile(r"!?\[((?:[^\[\]]|\[[^\[\]]*\])+)\]\[[^\[\]]*\]")
_MD_AUTOLINK = re.compile(r"<((?:https?|ftp|mailto):[^<>\s]+|[^<>\s@]+@[^<>\s@]+\.[^<>\s@]+)>")
_MD_HTML_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s+[^<>]*)?/?>")
_MD_BR = re.compile(r"<br\s*/?>", re.I)
_MD_FOOTREF = re.compile(r"\[\^[^\]\s]+\]")
_MD_CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.S)  # (whole backtick runs: linear time)
_MD_ESCAPE = re.compile(r"\\([!-/:-@\[-`{-~])")
_MD_EMPHASIS = (
    re.compile(r"(\*\*\*|___)(?=\S)(.+?)(?<=\S)\1"),
    re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*"),
    re.compile(r"(?<!\w)__(?=\S)(.+?)(?<=\S)__(?!\w)"),
    re.compile(r"~~(?=\S)(.+?)(?<=\S)~~"),
    re.compile(r"==(?=\S)(.+?)(?<=\S)=="),
    re.compile(r"(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])"),
    re.compile(r"(?<![\w_])_(?=[^\s_])(.+?)(?<=[^\s_])_(?![\w_])"),
)
_PLACEHOLDER = re.compile("\ue000(\\d+)\ue001")


def looks_like_markdown(text: str) -> bool:
    """True when ``text`` clearly uses Markdown (headings, fences, tables, links...)."""
    sample = text[:200_000]
    headings = len(_MD_ATX_ANY.findall(sample))
    if headings >= 2 or len(_MD_FENCE_ANY.findall(sample)) >= 2 or _MD_TABLE_DELIM_ANY.search(sample):
        return True
    links = len(_MD_LINK.findall(sample)) + len(_MD_IMAGE.findall(sample))
    bold = len(re.findall(r"\*\*(?=\S)[^\n*]+?(?<=\S)\*\*", sample))
    setext = len(_MD_SETEXT_ANY.findall(sample))
    return 2 * headings + 2 * setext + links + bold >= 3


def _md_inline(text: str, refs: frozenset[str] = frozenset()) -> str:
    """Markdown inline syntax -> plain text (links keep their words, emphasis marks go).

    ``refs`` holds the (lower-cased) labels of reference definitions, so that
    shortcut links like ``[Smith 2019]`` lose their brackets.
    """
    stash: list[str] = []

    def keep(value: str) -> str:
        stash.append(value)
        return f"\ue000{len(stash) - 1}\ue001"

    text = _MD_CODE_SPAN.sub(lambda m: keep(m.group(2).strip() if m.group(2).strip() else m.group(2)), text)
    text = _MD_ESCAPE.sub(lambda m: keep(m.group(1)), text)
    text = _MD_FOOTREF.sub("", text)
    text = _MD_IMG_HTML.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _MD_IMAGE.sub(lambda m: m.group(1), text)
    for _ in range(2):  # links may wrap images or other links' text
        text = _MD_LINK.sub(lambda m: m.group(1), text)
    text = _MD_REF_LINK.sub(lambda m: m.group(1), text)
    if refs:
        text = _MD_SHORTCUT_REF.sub(lambda m: m.group(1) if m.group(1).lower() in refs else m.group(0), text)
    text = _MD_AUTOLINK.sub(lambda m: m.group(1), text)
    text = _MD_BR.sub(" ", text)
    text = _MD_HTML_TAG.sub("", text)
    for pattern in _MD_EMPHASIS:
        text = pattern.sub(lambda m: m.group(m.lastindex or 1), text)
    text = html_lib.unescape(text)
    for _ in range(3):  # escaped text can itself contain stashed code
        text = _PLACEHOLDER.sub(lambda m: stash[int(m.group(1))], text)
        if "\ue000" not in text:
            break
    return text


def _md_figures(text: str) -> list[str] | None:
    """Alt texts when a paragraph is nothing but images, else ``None``."""
    images: list[str] = []

    def take_markdown(m: re.Match[str]) -> str:
        images.append(m.group(1))
        return " "

    def take_html(m: re.Match[str]) -> str:
        images.append(m.group(1) or m.group(2) or "")
        return " "

    remainder = _MD_IMAGE.sub(take_markdown, text)
    remainder = _MD_REF_IMAGE.sub(take_markdown, remainder)
    remainder = _MD_IMG_HTML.sub(take_html, remainder)
    remainder = _MD_LINK.sub(lambda m: m.group(1), remainder)  # [![alt](img)](link)
    remainder = _MD_HTML_TAG.sub("", remainder)
    if images and not remainder.strip():
        return images
    return None


def _strip_front_matter(text: str) -> tuple[str, dict[str, str]]:
    """Remove YAML front matter (or a Pandoc "% title" block) and return its title/author."""
    text = text.lstrip("\ufeff")
    if text.startswith("% "):
        lines = text.split("\n")
        head: list[str] = []
        while lines and lines[0].startswith("%") and len(head) < 3:
            head.append(lines.pop(0)[1:].strip())
        meta = {"title": head[0]} if head and head[0] else {}
        if len(head) > 1 and head[1]:
            meta["author"] = ", ".join(part.strip() for part in head[1].split(";") if part.strip())
        return "\n".join(lines), meta
    m = re.match(r"^---[ \t]*\n(.*?)\n(?:---|\.\.\.)[ \t]*(?:\n|$)", text, re.S)
    if not m:
        return text, {}
    meta: dict[str, str] = {}
    for line in m.group(1).split("\n"):
        kv = re.match(r"^(title|author)\s*:\s*(.+?)\s*$", line, re.I)
        if kv:
            meta[kv.group(1).lower()] = kv.group(2).strip().strip("'\"")
    return text[m.end() :], meta


def _has_hidden_html(line: str) -> bool:
    lowered = line.lower()
    return any(opener in lowered for opener, _ in _MD_HIDDEN_HTML)


def _remove_hidden_html(lines: list[str], i: int) -> tuple[str, int]:
    """Strip comments, ``<script>`` and ``<style>`` (possibly spanning lines) starting at line ``i``."""
    line = lines[i]
    while True:
        lowered = line.lower()
        found = [(lowered.find(opener), opener, closer) for opener, closer in _MD_HIDDEN_HTML if opener in lowered]
        if not found:
            return line, i
        start, _opener, closer = min(found)
        before, after = line[:start], line[start:]
        while closer not in after.lower() and i + 1 < len(lines):
            i += 1
            after = lines[i]
        end = after.lower().find(closer)
        line = before + (after[end + len(closer) :] if end >= 0 else "")


def _is_table_delimiter(line: str) -> bool:
    stripped = line.strip()
    if "-" not in stripped or "|" not in stripped or set(stripped) - set("|:- \t"):
        return False
    cells = [c.strip() for c in stripped.strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-+:?", c) for c in cells)


def _split_table_row(line: str, refs: frozenset[str] = frozenset()) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [_md_inline(cell.strip().replace("\\|", "|"), refs) for cell in re.split(r"(?<!\\)\|", s)]


def markdown_blocks(text: str, *, _refs: frozenset[str] | None = None) -> list[Block]:
    """Parse Markdown into blocks (headings, lists, quotes, code, tables, images...)."""
    lines = _normalize_newlines(text).split("\n")
    refs = _refs
    if refs is None:
        refs = frozenset(m.group(1).strip().lower() for line in lines if (m := _MD_REF_DEF.match(line)))

    def inline(value: str) -> str:
        return _md_inline(value, refs)

    blocks: list[Block] = []
    para: list[str] = []
    item: list[str] | None = None
    item_level = 1
    list_indents: list[int] = []

    def add(kind: BlockKind, value: str, **fields) -> None:
        block = text_block(kind, value, **fields)
        if block:
            blocks.append(block)

    def flush() -> None:
        nonlocal item
        if item is not None:
            add(BlockKind.LIST_ITEM, inline("\n".join(item)), meta={"list_level": item_level})
            item = None
        if para:
            raw = "\n".join(para)
            figures = _md_figures(raw)
            if figures is not None:
                for alt in figures:
                    add(BlockKind.FIGURE, inline(alt))
            else:
                add(BlockKind.PARAGRAPH, inline(raw))
            para.clear()

    def is_block_start(line: str) -> bool:
        return bool(
            _MD_ATX.match(line)
            or _MD_FENCE.match(line)
            or _MD_QUOTE.match(line)
            or _MD_HR.match(line)
            or (_MD_LIST.match(line) and (_MD_LIST.match(line).group("text") or "").strip())
        )

    i, n = 0, len(lines)
    previous_blank = True
    while i < n:
        line = lines[i]
        if "<" in line and _has_hidden_html(line):
            line, i = _remove_hidden_html(lines, i)
            if not line.strip():
                i += 1
                continue
        stripped = line.strip()
        if not stripped:
            if item is None or (i + 1 < n and not lines[i + 1].startswith((" ", "\t"))):
                flush()
            previous_blank = True
            i += 1
            continue

        fence = _MD_FENCE.match(line)
        if fence and not (fence.group(2)[0] == "`" and "`" in fence.group(3)):
            flush()
            list_indents.clear()
            indent, marks = len(fence.group(1)), fence.group(2)
            body: list[str] = []
            i += 1
            while i < n:
                close = re.match(r"^ {0,3}(`{3,}|~{3,})[ \t]*$", lines[i])
                if close and close.group(1)[0] == marks[0] and len(close.group(1)) >= len(marks):
                    break
                raw_line = lines[i]
                strip = len(raw_line) - len(raw_line.lstrip(" "))
                body.append(raw_line[min(indent, strip) :])
                i += 1
            language = fence.group(3).strip().split(" ")[0]
            add(BlockKind.CODE, code_text("\n".join(body)), meta={"language": language} if language else {})
            i += 1
            previous_blank = False
            continue

        if stripped.startswith("$$") and not para and item is None:
            flush()
            body = [stripped[2:]]
            if stripped.endswith("$$") and len(stripped) > 4:
                body = [stripped[2:-2]]
            else:
                i += 1
                while i < n and "$$" not in lines[i]:
                    body.append(lines[i])
                    i += 1
                if i < n:
                    body.append(lines[i].split("$$")[0])
            add(BlockKind.EQUATION, "\n".join(body))
            i += 1
            previous_blank = False
            continue

        atx = _MD_ATX.match(line)
        if atx:
            flush()
            list_indents.clear()
            add(BlockKind.HEADING, inline(atx.group(2) or ""), level=len(atx.group(1)))
            i += 1
            previous_blank = False
            continue

        setext = _MD_SETEXT.match(line)
        if setext and para and item is None:
            heading = "\n".join(para)
            para.clear()
            add(BlockKind.HEADING, inline(heading), level=1 if setext.group(1)[0] == "=" else 2)
            i += 1
            previous_blank = False
            continue

        if _MD_HR.match(line):
            flush()
            list_indents.clear()
            i += 1
            previous_blank = True
            continue

        if "|" in line and i + 1 < n and _is_table_delimiter(lines[i + 1]):
            flush()
            list_indents.clear()
            rows = [_split_table_row(line, refs)]
            i += 2
            while i < n and lines[i].strip() and "|" in lines[i]:
                rows.append(_split_table_row(lines[i], refs))
                i += 1
            block = table_block(rows)
            if block:
                blocks.append(block)
            previous_blank = False
            continue

        if _MD_QUOTE.match(line):
            flush()
            list_indents.clear()
            quoted: list[str] = []
            while i < n:
                current = lines[i]
                if _MD_QUOTE.match(current):
                    quoted.append(_MD_QUOTE.sub("", current, count=1))
                elif current.strip() and quoted and quoted[-1].strip() and not is_block_start(current):
                    quoted.append(current)  # lazy continuation
                else:
                    break
                i += 1
            # The quote's content is Markdown too (lists, code, nested quotes); its prose becomes QUOTE.
            for block in markdown_blocks("\n".join(quoted), _refs=refs):
                if block.kind == BlockKind.PARAGRAPH:
                    block = block.model_copy(update={"kind": BlockKind.QUOTE})
                blocks.append(block)
            previous_blank = False
            continue

        footnote = _MD_FOOTNOTE_DEF.match(line)
        if footnote:
            flush()
            note = [footnote.group(2)]
            i += 1
            while i < n and lines[i].strip() and (lines[i].startswith((" ", "\t")) or not is_block_start(lines[i])):
                if _MD_FOOTNOTE_DEF.match(lines[i]):
                    break
                note.append(lines[i])
                i += 1
            add(BlockKind.FOOTNOTE, inline("\n".join(note)))
            previous_blank = False
            continue

        if _MD_REF_DEF.match(line) and not para and item is None:
            i += 1
            continue

        listed = _MD_LIST.match(line)
        if (
            listed
            and (listed.group("text") or "").strip()
            and (not para or previous_blank or listed.group("marker")[0] in "-*+1")
        ):
            flush()
            indent = len(listed.group("indent").expandtabs(4))
            while list_indents and indent + 1 < list_indents[-1]:
                list_indents.pop()
            if not list_indents or indent >= list_indents[-1] + 2:
                list_indents.append(indent)
            item_level = len(list_indents)
            marker, body_text = listed.group("marker"), _MD_TASK.sub("", listed.group("text").strip())
            item = [f"{marker} {body_text}" if marker[0].isdigit() else body_text]
            i += 1
            previous_blank = False
            continue

        if line.startswith(("    ", "\t")) and previous_blank and not para and item is None and not list_indents:
            body = []
            while i < n and (lines[i].startswith(("    ", "\t")) or not lines[i].strip()):
                body.append(lines[i][4:] if lines[i].startswith("    ") else lines[i][1:])
                i += 1
            add(BlockKind.CODE, code_text("\n".join(body)))
            previous_blank = True
            continue

        if _MD_HTML_ONLY.match(line) and not _MD_IMG_HTML.search(line):
            i += 1  # stray <div>, </details>, ... lines
            continue

        if item is not None:
            if previous_blank and not line.startswith((" ", "\t")):
                flush()
                list_indents.clear()
                para.append(line)
            else:
                item.append(line)  # continuation (lazy or indented) of the list item
        else:
            if previous_blank and list_indents and not line.startswith((" ", "\t")):
                list_indents.clear()
            para.append(line)
        previous_blank = False
        i += 1
    flush()
    return blocks


# --------------------------------------------------------------------------- documents


def _first_heading(blocks: list[Block], level: int | None = None) -> str | None:
    for block in blocks:
        if block.kind == BlockKind.HEADING and (level is None or block.level == level):
            return block.text
    return None


def _looks_binary(text: str) -> bool:
    sample = text[:20_000]
    if not sample:
        return False
    control = sum(1 for ch in sample if ord(ch) < 32 and ch not in "\n\t\f\v")
    return control > 0.1 * len(sample) or sample.count("\ufffd") > 0.1 * len(sample)


def _build_document(
    text: str,
    *,
    title: str | None,
    fallback_title: str,
    doc_id: str,
    source_name: str,
    source_type: str,
    markdown: bool,
) -> Document:
    text = _normalize_newlines(text)
    meta: dict[str, str] = {}
    if markdown:
        text, meta = _strip_front_matter(text)
        blocks = markdown_blocks(text)
    else:
        blocks = plain_text_blocks(text)
    doc_title = (
        clean_display_text(title or "")
        or meta.get("title")
        or (_first_heading(blocks, 1) if markdown else None)
        or fallback_title
    )
    sections = finalize_sections(sections_from_blocks(blocks, doc_title), doc_title)
    warnings = [] if blocks else ["No readable text was found."]
    return Document(
        id=doc_id,
        title=doc_title,
        author=meta.get("author") or None,
        source_name=source_name,
        source_type=source_type,
        sections=sections,
        warnings=warnings,
    )


def document_from_text(text: str, title: str, *, source_type: str = "paste") -> Document:
    """Build a document from pasted text. Markdown is recognised when the text uses it."""
    text = _normalize_newlines(text or "")
    if not text.strip():
        raise IngestError("There's no text to read. Paste some text and try again.")
    title = clean_display_text(title or "")
    markdown = source_type == "md" or looks_like_markdown(text)
    first_line = next((clean_display_text(line) for line in text.split("\n") if line.strip()), "")
    fallback = first_line[:80].rstrip() if 0 < len(first_line) <= 80 else "Pasted text"
    return _build_document(
        text,
        title=title or None,
        fallback_title=fallback,
        doc_id=doc_id_for_bytes((title + "\n" + text).encode("utf-8")),
        source_name=title or "Pasted text",
        source_type=source_type,
        markdown=markdown,
    )


def load_text_file(path: Path, *, source_name: str, progress: ProgressFn | None = None) -> Document:
    """Load a ``.txt`` or Markdown file of any common encoding."""
    path = Path(path)
    if progress:
        progress(0.0, "Reading text")
    data = path.read_bytes()
    text = _normalize_newlines(decode_text_bytes(data))
    if not text.strip():
        raise IngestError(f"'{source_name}' is empty, so there's nothing to read.")
    if _looks_binary(text):
        raise IngestError(f"'{source_name}' doesn't look like a text file. It may be damaged or in another format.")
    suffix = Path(source_name).suffix.lower() or path.suffix.lower()
    is_md_file = suffix in MARKDOWN_EXTENSIONS
    doc = _build_document(
        text,
        title=None,
        fallback_title=title_from_filename(source_name),
        doc_id=doc_id_for_file(path),
        source_name=source_name,
        source_type="md" if is_md_file else "txt",
        markdown=is_md_file or looks_like_markdown(text),
    )
    if progress:
        progress(1.0, "Done")
    return doc
