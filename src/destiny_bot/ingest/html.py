"""HTML pages (``.html``, ``.htm``, ``.xhtml``) and the XHTML chapters inside EPUB books.

:func:`html_to_blocks` walks the page in reading order and maps its structure to
blocks: headings, paragraphs, list items, quotes, captions, figures (alt text),
tables, code, MathML equations and EPUB/DPUB footnotes. Scripts, navigation,
hidden elements, page-number markers and note-reference markers are dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from bs4 import BeautifulSoup, Comment, Declaration, Doctype, NavigableString, ProcessingInstruction, Tag
from bs4.dammit import EncodingDetector

from ..models import Block, BlockKind, Document
from . import ProgressFn
from .common import clean_display_text, doc_id_for_file, finalize_sections, sections_from_blocks, title_from_filename
from .text import code_text, decode_text_bytes, superscript_text, table_block, text_block, wrap_math

_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_SKIP_TAGS = frozenset(
    "script style noscript template svg nav head title iframe object embed canvas audio video map "
    "button select option datalist textarea input rp rt link meta dialog".split()
)
_BLOCK_TAGS = frozenset(
    "address article aside blockquote body caption center dd details dir div dl dt fieldset figcaption "
    "figure footer form h1 h2 h3 h4 h5 h6 header hgroup hr html legend li main menu ol p pre section "
    "summary table tbody td tfoot th thead tr ul".split()
)
_BLOCK_LIST = sorted(_BLOCK_TAGS)
_CONTENT_ANCESTORS = frozenset("article section main aside blockquote figure".split())
_INLINE_NAMES = frozenset("a abbr b bdi bdo cite code data dfn em i kbd mark q s samp small span strong sub sup time u var".split())
_LAYOUT_SIGNS = ["table", "h1", "h2", "h3", "h4", "h5", "h6"]

_NOTE_TYPES = frozenset({"footnote", "endnote", "rearnote"})
_NOTE_ROLES = frozenset({"doc-footnote", "doc-endnote"})
_NOTE_CLASSES = frozenset({"footnote", "endnote"})
_NOTES_TYPES = frozenset({"footnotes", "endnotes", "rearnotes"})
_NOTES_ROLES = frozenset({"doc-endnotes"})
_NOTES_CLASSES = frozenset({"footnotes", "endnotes", "references"})
_BACKLINK_CLASSES = frozenset({"footnote-back", "footnote-backref", "mw-cite-backlink"})
_BACKLINK_TEXTS = frozenset({"↩", "↩︎", "↑", "^", "⤴", "back", "[back]", "return"})
_CHROME_ROLES = frozenset({"navigation", "search", "banner"})
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)
_TEX_ENCODINGS = frozenset({"application/x-tex", "tex", "latex", "application/x-latex", "text/x-tex"})
_MATH_INVISIBLE = dict.fromkeys(map(ord, "⁡⁢⁣⁤"), None)
_VOID = "area|base|br|col|embed|hr|img|input|link|meta|param|source|track|wbr"
_SELF_CLOSING = re.compile(rf"<(?!(?:{_VOID})\b)([A-Za-z][\w:.-]*)(\s[^<>]*?)?\s*/>")
_IGNORED_STRINGS = (Comment, Declaration, Doctype, ProcessingInstruction)


def _local(tag: Tag) -> str:
    return (tag.name or "").rsplit(":", 1)[-1].lower()


def _tokens(tag: Tag, attr: str) -> set[str]:
    value = tag.get(attr)
    if not value:
        return set()
    if isinstance(value, list):
        value = " ".join(value)
    return set(value.lower().split())


def _epub_types(tag: Tag) -> set[str]:
    types: set[str] = set()
    for key in tag.attrs:
        if key == "epub:type" or (key.endswith(":type") and key != "type"):
            types |= _tokens(tag, key)
    return types


def decode_html_bytes(data: bytes) -> str:
    """Decode HTML bytes: BOM, then ``<meta charset>`` / XML declaration, then UTF-8 / Windows-1252."""
    declared = EncodingDetector.find_declared_encoding(data[:8192], is_html=True)
    return decode_text_bytes(data, declared=declared)


def _mathml_linear(node) -> str:
    """Best-effort linear text for a MathML element (``x^2 + (a)/(b)``)."""
    if isinstance(node, NavigableString):
        return "" if isinstance(node, _IGNORED_STRINGS) else str(node).strip()
    if not isinstance(node, Tag):
        return ""
    name = _local(node)
    kids = [child for child in node.children if isinstance(child, Tag)]
    lin = [_mathml_linear(child) for child in kids]
    if name in ("annotation", "annotation-xml", "none", "mprescripts", "mphantom"):
        return ""
    if name == "mspace":
        return " "
    if name in ("mi", "mn", "mo", "mtext", "ms"):
        return node.get_text().translate(_MATH_INVISIBLE).strip()
    if name == "semantics":
        return lin[0] if lin else ""
    if name == "mfrac" and len(lin) == 2:
        return f"{wrap_math(lin[0])}/{wrap_math(lin[1])}"
    if name == "msup" and len(lin) == 2:
        return f"{wrap_math(lin[0])}^{wrap_math(lin[1])}"
    if name == "msub" and len(lin) == 2:
        return f"{lin[0]}_{wrap_math(lin[1])}"
    if name in ("msubsup", "munderover") and len(lin) == 3:
        return f"{lin[0]}_{wrap_math(lin[1])}^{wrap_math(lin[2])}"
    if name == "munder" and len(lin) == 2:
        return f"{lin[0]}_{wrap_math(lin[1])}"
    if name == "mover" and len(lin) == 2:
        if lin[1] in ("¯", "‾", "^", "ˆ", "~", "˜", "→", "⃗", "˙", "¨", "."):
            return f"{lin[0]}{lin[1]}"
        return f"{wrap_math(lin[0])}^{wrap_math(lin[1])}"
    if name == "msqrt":
        return f"sqrt({' '.join(lin).strip()})"
    if name == "mroot" and len(lin) == 2:
        return f"root({lin[1]}, {lin[0]})"
    if name == "mfenced":
        opener, closer = node.get("open", "("), node.get("close", ")")
        separator = (node.get("separators", ",") or ",")[0]
        return f"{opener}{(separator + ' ').join(lin)}{closer}"
    if name == "mtable":
        return "; ".join(lin)
    if name == "mtr" or name == "mlabeledtr":
        return ", ".join(lin)
    parts: list[str] = []
    for child, text in zip(kids, lin):
        if not text:
            continue
        if _local(child) == "mo" and text not in "()[]{}|‖⟨⟩":
            parts.append(", " if text == "," else f" {text} ")
        else:
            parts.append(text)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _math_text(math: Tag) -> str:
    alt = (math.get("alttext") or "").strip()
    if alt:
        return alt
    for child in math.find_all(True):
        if _local(child) == "annotation" and (child.get("encoding") or "").strip().lower() in _TEX_ENCODINGS:
            tex = child.get_text().strip()
            if tex:
                return tex
    return _mathml_linear(math)


@dataclass(frozen=True)
class _Ctx:
    para_kind: BlockKind = BlockKind.PARAGRAPH
    list_level: int = 0
    in_notes: bool = False


@dataclass
class HtmlContent:
    """Everything :func:`parse_html` extracts from one HTML document."""

    blocks: list[Block]
    title: str | None
    author: str | None = None
    # element id -> index of the first block at or after that element (for TOC anchors)
    anchors: dict[str, int] = field(default_factory=dict)


class _Walker:
    def __init__(self, page: int | None) -> None:
        self.page = page
        self.blocks: list[Block] = []
        self.anchors: dict[str, int] = {}
        self._prefix: str | None = None

    # ------------------------------------------------------------ classification

    @staticmethod
    def _hidden(tag: Tag) -> bool:
        if tag.has_attr("hidden") or (tag.get("aria-hidden") or "").strip().lower() == "true":
            return True
        style = tag.get("style")
        return bool(style and _HIDDEN_STYLE.search(style))

    def _skip(self, tag: Tag) -> bool:
        name = _local(tag)
        if name in _SKIP_TAGS or self._hidden(tag):
            return True
        types, roles = _epub_types(tag), _tokens(tag, "role")
        if "pagebreak" in types or "doc-pagebreak" in roles:
            return True  # printed page numbers: not part of the text
        if roles & _CHROME_ROLES:
            return True
        if name == "default" and tag.parent is not None and _local(tag.parent) == "switch":
            return True  # <epub:switch> fallback content duplicates the <epub:case>
        if name in ("header", "footer") or "contentinfo" in roles:
            return self._is_page_chrome(tag, name)
        return False

    def _is_page_chrome(self, tag: Tag, name: str) -> bool:
        """Site headers/footers outside the main content (kept when they hold a chapter heading or notes)."""
        for parent in tag.parents:
            if not isinstance(parent, Tag):
                continue
            if _local(parent) in _CONTENT_ANCESTORS or _tokens(parent, "role") & {"main", "article"}:
                return False
        if name == "header" and "contentinfo" not in _tokens(tag, "role"):
            has_heading = tag.find(list(_HEADINGS)) is not None
            return not has_heading or tag.find("nav") is not None or "banner" in _tokens(tag, "role")
        return not self._contains_notes(tag)

    def _contains_notes(self, tag: Tag) -> bool:
        return any(self._is_note(t) or self._is_note_container(t) for t in tag.find_all(True))

    @staticmethod
    def _is_note(tag: Tag) -> bool:
        name = _local(tag)
        if name in _INLINE_NAMES:
            return False
        return bool(
            _epub_types(tag) & _NOTE_TYPES or _tokens(tag, "role") & _NOTE_ROLES or _tokens(tag, "class") & _NOTE_CLASSES
        )

    @staticmethod
    def _is_note_container(tag: Tag) -> bool:
        return bool(
            _epub_types(tag) & _NOTES_TYPES
            or _tokens(tag, "role") & _NOTES_ROLES
            or (_local(tag) not in _INLINE_NAMES and _tokens(tag, "class") & _NOTES_CLASSES)
        )

    @staticmethod
    def _is_noteref(tag: Tag) -> bool:
        name = _local(tag)
        if "noteref" in _epub_types(tag) or "doc-noteref" in _tokens(tag, "role"):
            return True
        if name == "a" and _tokens(tag, "class") & {"footnote-ref", "noteref", "footnote-reference"}:
            return True
        if name == "a" and (tag.get("href") or "").startswith("#") and tag.parent is not None:
            if _local(tag.parent) == "sup":
                return True
        if name == "sup":
            kids = [c for c in tag.children if not (isinstance(c, NavigableString) and not c.strip())]
            if len(kids) == 1 and isinstance(kids[0], Tag) and _local(kids[0]) == "a":
                return (kids[0].get("href") or "").startswith("#")
        return False

    @staticmethod
    def _is_backlink(tag: Tag) -> bool:
        if "doc-backlink" in _tokens(tag, "role") or "backlink" in _epub_types(tag):
            return True
        if _tokens(tag, "class") & _BACKLINK_CLASSES:
            return True
        return (
            _local(tag) == "a"
            and (tag.get("href") or "").startswith("#")
            and tag.get_text(strip=True).lower() in _BACKLINK_TEXTS
        )

    def _is_blockish(self, tag: Tag) -> bool:
        name = _local(tag)
        if name in _BLOCK_TAGS or name in _HEADINGS or self._is_note(tag) or self._is_note_container(tag):
            return True
        if "heading" in _tokens(tag, "role"):
            return True
        if name in ("img", "br", "math"):
            return False
        return tag.find(_BLOCK_LIST) is not None

    # ------------------------------------------------------------ output

    def _add(self, kind: BlockKind, text: str, ctx: _Ctx | None = None, **fields) -> None:
        if kind == BlockKind.LIST_ITEM:
            if self._prefix and text.strip():
                text, self._prefix = self._prefix + text.strip(), None
            fields.setdefault("meta", {"list_level": max(1, ctx.list_level if ctx else 1)})
        block = text_block(kind, text, page=self.page, **fields)
        if block:
            self.blocks.append(block)

    def _record_anchor(self, tag: Tag) -> None:
        for key in ("id", "name") if _local(tag) == "a" else ("id",):
            value = tag.get(key)
            if isinstance(value, str) and value:
                self.anchors.setdefault(value, len(self.blocks))

    # ------------------------------------------------------------ text

    def inline_text(self, node: Tag) -> str:
        parts: list[str] = []
        self._collect(node, parts)
        return "".join(parts)

    def _collect(self, node: Tag, parts: list[str]) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                if not isinstance(child, _IGNORED_STRINGS):
                    # Source line breaks are just spaces in HTML (and must not trigger
                    # clean_display_text's "hyphen at line end" repair).
                    parts.append(str(child).replace("\r", " ").replace("\n", " "))
                continue
            if not isinstance(child, Tag):
                continue
            self._record_anchor(child)
            if self._skip(child) or self._is_noteref(child) or self._is_backlink(child):
                continue
            name = _local(child)
            if name == "br":
                parts.append(" ")
            elif name == "img":
                alt = self._alt(child)
                if alt:
                    parts.append(alt)
            elif name == "math":
                parts.append(_math_text(child))
            elif name == "sup":
                parts.append(superscript_text(self.inline_text(child)))
            elif name in _BLOCK_TAGS or name in _HEADINGS:
                parts.append(" ")
                self._collect(child, parts)
                parts.append(" ")
            else:
                self._collect(child, parts)

    def _raw_text(self, node: Tag) -> str:
        """Text with its line breaks kept (for ``<pre>``)."""
        parts: list[str] = []

        def walk(tag: Tag) -> None:
            for child in tag.children:
                if isinstance(child, NavigableString):
                    if not isinstance(child, _IGNORED_STRINGS):
                        parts.append(str(child))
                elif isinstance(child, Tag):
                    self._record_anchor(child)
                    if self._skip(child):
                        continue
                    if _local(child) == "br":
                        parts.append("\n")
                    else:
                        walk(child)

        walk(node)
        return "".join(parts)

    def _alt(self, img: Tag) -> str:
        if self._hidden(img) or _tokens(img, "role") & {"presentation", "none"}:
            return ""
        return clean_display_text(img.get("alt") or "")

    # ------------------------------------------------------------ structure

    def walk_container(self, element: Tag, ctx: _Ctx) -> None:
        buffer: list = []
        for child in list(element.children):
            if isinstance(child, NavigableString):
                if not isinstance(child, _IGNORED_STRINGS):
                    buffer.append(child)
                continue
            if not isinstance(child, Tag):
                continue
            self._record_anchor(child)
            if self._skip(child) or self._is_noteref(child) or self._is_backlink(child):
                continue
            if self._is_blockish(child):
                self._flush(buffer, ctx)
                buffer = []
                self._walk_block(child, ctx)
            else:
                buffer.append(child)
        self._flush(buffer, ctx)

    def _only_images(self, nodes: list) -> list[Tag] | None:
        images: list[Tag] = []
        for node in nodes:
            if isinstance(node, NavigableString):
                if node.strip():
                    return None
                continue
            name = _local(node)
            if name == "img":
                images.append(node)
            elif name != "br":
                inner = node.find_all("img")
                if not inner or self.inline_text(node).strip() != "".join(self._alt(i) for i in inner).strip():
                    return None
                images.extend(inner)
        return images or None

    def _flush(self, buffer: list, ctx: _Ctx) -> None:
        meaningful = [
            node
            for node in buffer
            if not (isinstance(node, NavigableString) and not node.strip())
            and not (isinstance(node, Tag) and _local(node) == "br")
        ]
        if not meaningful:
            return
        images = self._only_images(meaningful)
        if images is not None:
            for image in images:
                self._record_anchor(image)
                self._add(BlockKind.FIGURE, self._alt(image))
            return
        if len(meaningful) == 1 and isinstance(meaningful[0], Tag):
            node = meaningful[0]
            name = _local(node)
            if name == "math":
                self._add(BlockKind.EQUATION, _math_text(node))
                return
            if name == "code" and not ctx.in_notes:
                self._add(BlockKind.CODE, code_text(self._raw_text(node)))
                return
        parts: list[str] = []
        for node in buffer:
            if isinstance(node, NavigableString):
                parts.append(str(node).replace("\r", " ").replace("\n", " "))
            elif _local(node) == "br":
                parts.append(" ")
            else:
                parts.append(self._node_text(node))
        self._add(ctx.para_kind, "".join(parts), ctx)

    def _node_text(self, node: Tag) -> str:
        """Inline text of a single node (including the node itself)."""
        name = _local(node)
        if name == "img":
            return self._alt(node)
        if name == "math":
            return _math_text(node)
        if name == "sup":
            return superscript_text(self.inline_text(node))
        return self.inline_text(node)

    def _walk_block(self, tag: Tag, ctx: _Ctx) -> None:
        name = _local(tag)
        roles = _tokens(tag, "role")
        if self._is_note(tag) and name != "li":
            self._add(BlockKind.FOOTNOTE, self.inline_text(tag))
            return
        if self._is_note_container(tag) and name not in ("ul", "ol"):
            self.walk_container(tag, replace(ctx, para_kind=BlockKind.FOOTNOTE, in_notes=True))
            return
        if name in _HEADINGS or "heading" in roles:
            level = _HEADINGS.get(name)
            if level is None:
                try:
                    level = max(1, min(6, int(tag.get("aria-level", 2))))
                except (TypeError, ValueError):
                    level = 2
            self._add(BlockKind.HEADING, self.inline_text(tag), level=level)
            return
        if name in ("ul", "ol", "menu", "dir"):
            in_notes = ctx.in_notes or self._is_note_container(tag)
            self._walk_list(tag, replace(ctx, in_notes=in_notes, para_kind=BlockKind.FOOTNOTE) if in_notes else ctx)
            return
        if name == "li":
            self._walk_item(tag, ctx, None)
            return
        if name == "blockquote":
            kind = BlockKind.FOOTNOTE if ctx.in_notes else BlockKind.QUOTE
            self.walk_container(tag, replace(ctx, para_kind=kind))
            return
        if name in ("figcaption", "caption"):
            self._add(BlockKind.CAPTION, self.inline_text(tag))
            return
        if name == "table":
            self._walk_table(tag, ctx)
            return
        if name == "pre":
            self._add(BlockKind.CODE, code_text(self._raw_text(tag)))
            return
        if name == "hr":
            return
        self.walk_container(tag, ctx)

    def _walk_list(self, tag: Tag, ctx: _Ctx) -> None:
        ordered = _local(tag) == "ol"
        items = [c for c in tag.children if isinstance(c, Tag) and _local(c) == "li"]
        reverse = tag.has_attr("reversed")
        try:
            number = int(tag.get("start"))
        except (TypeError, ValueError):
            number = len(items) if reverse else 1
        style = (tag.get("type") or "1").strip() or "1"
        stray: list = []
        for child in list(tag.children):
            if isinstance(child, Tag) and _local(child) == "li":
                self._flush(stray, ctx)
                stray = []
                self._record_anchor(child)
                if self._skip(child):
                    continue
                try:
                    number = int(child.get("value"))
                except (TypeError, ValueError):
                    pass
                self._walk_item(child, ctx, f"{_list_marker(number, style)} " if ordered else None)
                number += -1 if reverse else 1
            elif isinstance(child, Tag):
                self._record_anchor(child)
                if self._skip(child):
                    continue
                if self._is_blockish(child):
                    self._flush(stray, ctx)
                    stray = []
                    self._walk_block(child, ctx)
                else:
                    stray.append(child)
            elif isinstance(child, NavigableString) and not isinstance(child, _IGNORED_STRINGS):
                stray.append(child)
        self._flush(stray, ctx)

    def _walk_item(self, li: Tag, ctx: _Ctx, prefix: str | None) -> None:
        if ctx.in_notes or self._is_note(li):
            self._add(BlockKind.FOOTNOTE, (prefix or "") + self.inline_text(li))
            return
        self._prefix = prefix
        self.walk_container(li, replace(ctx, para_kind=BlockKind.LIST_ITEM, list_level=ctx.list_level + 1))
        self._prefix = None

    def _own_rows(self, table: Tag) -> list[Tag]:
        return [tr for tr in table.find_all("tr") if tr.find_parent("table") is table and not self._skip(tr)]

    def _cells(self, tr: Tag) -> list[Tag]:
        return [c for c in tr.children if isinstance(c, Tag) and _local(c) in ("td", "th") and not self._skip(c)]

    def _walk_table(self, table: Tag, ctx: _Ctx) -> None:
        rows = self._own_rows(table)
        layout = (
            _tokens(table, "role") & {"presentation", "none"}
            or not rows
            or max(len(self._cells(tr)) for tr in rows) <= 1
            or any(cell.find(_LAYOUT_SIGNS) for tr in rows for cell in self._cells(tr))
        )
        if layout:  # a table used for page layout: read its cells as ordinary content
            for child in table.children:
                if isinstance(child, Tag) and _local(child) == "caption":
                    self._walk_block(child, ctx)
            for tr in rows:
                for cell in self._cells(tr):
                    self._record_anchor(cell)
                    self.walk_container(cell, ctx)
            return
        for child in table.children:
            if isinstance(child, Tag) and _local(child) == "caption" and not self._skip(child):
                self._add(BlockKind.CAPTION, self.inline_text(child))
        grid: list[list[str]] = []
        for tr in rows:
            self._record_anchor(tr)
            row: list[str] = []
            for cell in self._cells(tr):
                self._record_anchor(cell)
                row.append(self.inline_text(cell))
                try:
                    span = int(cell.get("colspan", 1))
                except (TypeError, ValueError):
                    span = 1
                row.extend([""] * max(0, min(span, 20) - 1))
            grid.append(row)
        block = table_block(grid, page=self.page)
        if block:
            self.blocks.append(block)


def _to_roman(number: int) -> str:
    values = ((1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
              (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"))  # fmt: skip
    out = []
    for value, letters in values:
        while number >= value:
            out.append(letters)
            number -= value
    return "".join(out)


def _list_marker(number: int, style: str) -> str:
    if style in ("a", "A") and 1 <= number <= 26:
        letter = chr(ord("a") + number - 1)
        return f"{letter.upper() if style == 'A' else letter}."
    if style in ("i", "I") and 1 <= number < 4000:
        roman = _to_roman(number)
        return f"{roman.upper() if style == 'I' else roman}."
    return f"{number}."


def _soup(html: str | bytes) -> BeautifulSoup:
    markup = decode_html_bytes(html) if isinstance(html, (bytes, bytearray)) else html
    # XHTML (EPUB) uses <a id="p1"/>, <div/> ...; the HTML parser would leave those open.
    markup = _SELF_CLOSING.sub(r"<\1\2></\1>", markup)
    return BeautifulSoup(markup, "lxml")


def _content_root(soup: BeautifulSoup) -> Tag:
    body = soup.body or soup
    candidates = body.find_all("main") or body.find_all(attrs={"role": "main"})
    if not candidates:
        articles = body.find_all("article")
        candidates = [a for a in articles if a.find_parent("article") is None]
    if len(candidates) == 1:
        main = candidates[0]
        total = len(body.get_text(" ", strip=True))
        if total and len(main.get_text(" ", strip=True)) >= 0.4 * total:
            return main
    return body


def parse_html(html: str | bytes, *, page: int | None = None) -> HtmlContent:
    """Blocks, title, author and anchor positions of one HTML document."""
    soup = _soup(html)
    title = None
    for candidate in soup.find_all("title"):
        if candidate.find_parent("svg") is None:
            title = clean_display_text(candidate.get_text()) or None
            break
    author = None
    meta = soup.find("meta", attrs={"name": re.compile(r"^(author|dc\.creator)$", re.I)})
    if meta is not None:
        author = clean_display_text(meta.get("content") or "") or None
    walker = _Walker(page)
    body = soup.body or soup
    root = _content_root(soup)
    if root is not body:
        walker._record_anchor(root)
    walker.walk_container(root, _Ctx())
    if root is not body:
        # Footnotes often sit after <main>/<article>; keep them.
        done: set[int] = set()
        for tag in body.find_all(True):
            if tag is root or any(id(p) in done or p is root for p in tag.parents):
                continue
            if (walker._is_note(tag) or walker._is_note_container(tag)) and not walker._skip(tag):
                done.add(id(tag))
                walker._record_anchor(tag)
                walker._walk_block(tag, _Ctx())
    return HtmlContent(blocks=walker.blocks, title=title, author=author, anchors=walker.anchors)


def html_to_blocks(html: str | bytes, *, page: int | None = None) -> tuple[list[Block], str | None]:
    """Convert an HTML/XHTML document to blocks. Returns ``(blocks, <title> text or None)``."""
    content = parse_html(html, page=page)
    return content.blocks, content.title


def load_html(path: Path, *, source_name: str, progress: ProgressFn | None = None) -> Document:
    """Load a saved web page or XHTML file."""
    path = Path(path)
    if progress:
        progress(0.0, "Reading page")
    content = parse_html(path.read_bytes())
    first_h1 = next((b.text for b in content.blocks if b.kind == BlockKind.HEADING and b.level == 1), None)
    title = content.title or first_h1 or title_from_filename(source_name)
    sections = finalize_sections(sections_from_blocks(content.blocks, title), title)
    if progress:
        progress(1.0, "Done")
    return Document(
        id=doc_id_for_file(path),
        title=title,
        author=content.author,
        source_name=source_name,
        source_type="html",
        sections=sections,
        warnings=[] if content.blocks else ["No readable text was found on this page."],
    )
