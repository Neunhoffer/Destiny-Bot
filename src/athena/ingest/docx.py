"""Word documents (``.docx``).

The document body is read in order straight from its XML, so paragraphs,
tables, hyperlinks, fields, tracked insertions, text boxes and equations (OMML)
all keep their place. Paragraph styles map to block kinds: "Heading N" (or any
style with an outline level) -> heading, "Title" -> level-1 heading and the
document title, list styles and numbered paragraphs -> list items, "Quote" ->
quote, "Caption" -> caption. Footnotes and endnotes follow the paragraph that
cites them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

from ..models import Block, BlockKind, Document
from . import IngestError, ProgressFn
from .common import (
    check_zip_size,
    clean_alt_text,
    clean_display_text,
    doc_id_for_file,
    finalize_sections,
    sections_from_blocks,
    title_from_filename,
)
from .text import code_text, superscript_text, table_block, text_block, wrap_math

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
_WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
_MC = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
_V = "{urn:schemas-microsoft-com:vml}"

_HEADING_NAME = re.compile(r"heading\s*([1-9])")
_SKIP_STYLE = re.compile(r"toc\s*\d|toc heading|table of figures|index\s*\d|index heading|header|footer")
_QUOTE_STYLES = frozenset({"quote", "intense quote", "block text"})
_CODE_STYLES = frozenset({"source code", "html preformatted", "code", "macro text"})
_FALSE = frozenset({"0", "false", "off"})
_INLINE_CONTAINERS = frozenset(
    _W + name
    for name in ("hyperlink", "ins", "moveTo", "smartTag", "customXml", "fldSimple", "bdo", "dir", "sdtContent")
)
_XML_PARSER = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=True)


def _local(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _val(element, tag: str, ns: str = _W) -> str | None:
    if element is None:
        return None
    child = element.find(tag)
    return None if child is None else child.get(ns + "val")


def _on(element, tag: str) -> bool:
    """A Word on/off property such as ``<w:vanish/>`` (on unless ``w:val`` says otherwise)."""
    if element is None:
        return False
    child = element.find(tag)
    return child is not None and (child.get(_W + "val") or "1").lower() not in _FALSE


# --------------------------------------------------------------------------- equations


def omml_text(element) -> str:
    """Best-effort linear text for an Office Math (OMML) element."""
    name = _local(element.tag)

    def part(tag: str) -> str:
        child = element.find(_M + tag)
        return omml_text(child) if child is not None else ""

    def children() -> str:
        return "".join(
            omml_text(child) for child in element if isinstance(child.tag, str) and not _local(child.tag).endswith("Pr")
        )

    def prop(container: str, tag: str, default: str) -> str:
        props = element.find(_M + container)
        if props is None:
            return default
        node = props.find(_M + tag)
        return default if node is None else (node.get(_M + "val") if node.get(_M + "val") is not None else default)

    if name == "r":
        return "".join(t.text or "" for t in element.iter(_M + "t", _W + "t"))
    if name == "t":
        return element.text or ""
    if name == "oMathPara":
        return "; ".join(omml_text(child) for child in element.findall(_M + "oMath"))
    if name == "f":
        return f"{wrap_math(part('num'))}/{wrap_math(part('den'))}"
    if name == "sSup":
        return f"{wrap_math(part('e'))}^{wrap_math(part('sup'))}"
    if name == "sSub":
        return f"{part('e')}_{wrap_math(part('sub'))}"
    if name == "sSubSup":
        return f"{part('e')}_{wrap_math(part('sub'))}^{wrap_math(part('sup'))}"
    if name == "sPre":
        return f"_{wrap_math(part('sub'))}^{wrap_math(part('sup'))}{part('e')}"
    if name == "rad":
        degree = part("deg")
        return f"root({degree}, {part('e')})" if degree else f"sqrt({part('e')})"
    if name == "d":
        begin, end = prop("dPr", "begChr", "("), prop("dPr", "endChr", ")")
        separator = prop("dPr", "sepChr", "|")
        return begin + separator.join(omml_text(e) for e in element.findall(_M + "e")) + end
    if name == "nary":
        symbol = prop("naryPr", "chr", "∫")
        sub, sup = part("sub"), part("sup")
        limits = (f"_{wrap_math(sub)}" if sub else "") + (f"^{wrap_math(sup)}" if sup else "")
        return f"{symbol}{limits} {part('e')}".strip()
    if name == "func":
        argument = part("e")
        wrapped = wrap_math(argument)
        return part("fName") + (" " + wrapped if wrapped == argument else wrapped)
    if name == "limLow":
        return f"{part('e')}_{wrap_math(part('lim'))}"
    if name == "limUpp":
        return f"{part('e')}^{wrap_math(part('lim'))}"
    if name in ("eqArr",):
        return "; ".join(omml_text(e) for e in element.findall(_M + "e"))
    if name == "m":
        return "; ".join(", ".join(omml_text(e) for e in row.findall(_M + "e")) for row in element.findall(_M + "mr"))
    return children()


# --------------------------------------------------------------------------- styles


@dataclass
class _Style:
    name: str
    based_on: str | None
    outline: int | None
    numbered: bool


class _Styles:
    def __init__(self, styles_element) -> None:
        self.by_id: dict[str, _Style] = {}
        self.default: str | None = None
        self._cache: dict[str | None, tuple[str, int | None]] = {}
        if styles_element is None:
            return
        for style in styles_element.iter(_W + "style"):
            if style.get(_W + "type") not in (None, "paragraph"):
                continue
            style_id = style.get(_W + "styleId") or ""
            ppr = style.find(_W + "pPr")
            outline = _val(ppr, _W + "outlineLvl")
            num_id = _val(ppr.find(_W + "numPr") if ppr is not None else None, _W + "numId")
            self.by_id[style_id] = _Style(
                name=(_val(style, _W + "name") or style_id).strip().lower(),
                based_on=_val(style, _W + "basedOn"),
                outline=int(outline) if outline and outline.isdigit() else None,
                numbered=num_id not in (None, "0"),
            )
            if (style.get(_W + "default") or "").lower() in ("1", "true", "on"):
                self.default = style_id

    def name(self, style_id: str | None) -> str:
        style = self.by_id.get(style_id or self.default or "")
        return style.name if style else ""

    def classify(self, style_id: str | None) -> tuple[str, int | None]:
        """``(kind, level)`` where kind is heading/title/paragraph/skip/quote/caption/code/list."""
        if style_id in self._cache:
            return self._cache[style_id]
        result: tuple[str, int | None] = ("paragraph", None)
        current, seen = style_id or self.default, set()
        while current and current not in seen and current in self.by_id:
            seen.add(current)
            style = self.by_id[current]
            name = style.name
            heading = _HEADING_NAME.fullmatch(name)
            if heading:
                result = ("heading", min(6, int(heading.group(1))))
            elif name == "title":
                result = ("title", 1)
            elif name == "subtitle":
                result = ("paragraph", None)
            elif _SKIP_STYLE.fullmatch(name):
                result = ("skip", None)
            elif name in _QUOTE_STYLES:
                result = ("quote", None)
            elif name == "caption":
                result = ("caption", None)
            elif name in _CODE_STYLES:
                result = ("code", None)
            elif "list" in name:
                digits = re.search(r"(\d)$", name)
                result = ("list", int(digits.group(1)) if digits else None)
            elif style.outline is not None and style.outline < 9:
                result = ("heading", min(6, style.outline + 1))
            elif style.numbered:
                result = ("list", None)
            else:
                current = style.based_on
                continue
            break
        self._cache[style_id] = result
        return result


# --------------------------------------------------------------------------- reader


@dataclass
class _Para:
    parts: list[str] = field(default_factory=list)  # everything, in order
    plain: list[str] = field(default_factory=list)  # text outside equations
    math: list[str] = field(default_factory=list)
    extras: list[Block] = field(default_factory=list)
    notes: list[tuple[str, str]] = field(default_factory=list)


class _DocxReader:
    def __init__(self, document) -> None:
        self.document = document
        try:
            styles_element = document.styles.element
        except Exception:
            styles_element = None
        self.styles = _Styles(styles_element)
        self.blocks: list[Block] = []
        self.title: str | None = None
        self.notes: dict[tuple[str, str], str] = {}
        self._notes_done: set[tuple[str, str]] = set()
        self._last_code = False
        self._load_notes()

    # ------------------------------------------------------------ notes

    def _load_notes(self) -> None:
        try:
            rels = list(self.document.part.rels.values())
        except Exception:
            return
        for rel in rels:
            kind = rel.reltype.rsplit("/", 1)[-1]
            if kind not in ("footnotes", "endnotes") or rel.is_external:
                continue
            try:
                root = etree.fromstring(rel.target_part.blob, _XML_PARSER)
            except Exception:
                continue
            singular = kind[:-1]
            for note in root.iter(_W + singular):
                if note.get(_W + "type") in ("separator", "continuationSeparator", "continuationNotice"):
                    continue
                texts = []
                for p in note.iter(_W + "p"):
                    para = _Para()
                    self._inline(p, para)
                    texts.append("".join(para.parts))
                text = clean_display_text(" ".join(texts))
                if text:
                    self.notes[(singular, note.get(_W + "id") or "")] = text

    # ------------------------------------------------------------ body

    def body(self, container) -> None:
        for child in container.iterchildren():
            tag = child.tag
            if tag == _W + "p":
                self.paragraph(child)
            elif tag == _W + "tbl":
                self.table(child)
            elif tag == _W + "sdt":
                if self._is_toc(child):
                    continue
                content = child.find(_W + "sdtContent")
                if content is not None:
                    self.body(content)
            elif tag == _W + "customXml":
                self.body(child)

    @staticmethod
    def _is_toc(sdt) -> bool:
        gallery = sdt.find(f"{_W}sdtPr/{_W}docPartObj/{_W}docPartGallery")
        return gallery is not None and "contents" in (gallery.get(_W + "val") or "").lower()

    def _emit(self, block: Block | None) -> None:
        if block is not None:
            self.blocks.append(block)
            self._last_code = False

    def paragraph(self, p) -> None:
        ppr = p.find(_W + "pPr")
        style_id = _val(ppr, _W + "pStyle")
        kind, level = self.styles.classify(style_id)
        if kind == "skip":
            return
        outline = _val(ppr, _W + "outlineLvl")
        if kind not in ("heading", "title") and outline and outline.isdigit() and int(outline) < 9:
            kind, level = "heading", min(6, int(outline) + 1)
        numpr = ppr.find(_W + "numPr") if ppr is not None else None
        if numpr is not None and kind not in ("heading", "title", "code"):
            if _val(numpr, _W + "numId") not in (None, "0"):
                kind = "list"
                ilvl = _val(numpr, _W + "ilvl")
                level = int(ilvl) + 1 if ilvl and ilvl.isdigit() else level
            elif kind == "list" and "list" not in self.styles.name(style_id):
                kind = "paragraph"  # numbering explicitly removed

        para = _Para()
        self._inline(p, para)
        text = "".join(para.parts)
        if para.math and not "".join(para.plain).strip():
            self._emit(text_block(BlockKind.EQUATION, " ".join(para.math)))
        elif kind == "code":
            code = code_text(text)
            if code:
                if self._last_code and self.blocks and self.blocks[-1].kind == BlockKind.CODE:
                    self.blocks[-1].text += "\n" + code  # consecutive code lines form one block
                else:
                    self.blocks.append(Block(kind=BlockKind.CODE, text=code))
                self._last_code = True
        elif kind in ("heading", "title"):
            block = text_block(BlockKind.HEADING, text, level=level or 1)
            if block and kind == "title" and self.title is None:
                self.title = block.text
            self._emit(block)
        elif kind == "list":
            self._emit(text_block(BlockKind.LIST_ITEM, text, meta={"list_level": level or 1}))
        else:
            mapped = {"quote": BlockKind.QUOTE, "caption": BlockKind.CAPTION}.get(kind, BlockKind.PARAGRAPH)
            self._emit(text_block(mapped, text))
        for extra in para.extras:
            self._emit(extra)
        for key in para.notes:
            if key in self.notes and key not in self._notes_done:
                self._notes_done.add(key)
                self._emit(text_block(BlockKind.FOOTNOTE, self.notes[key]))

    def _inline(self, element, para: _Para) -> None:
        for child in element.iterchildren():
            tag = child.tag
            if not isinstance(tag, str):
                continue
            if tag == _W + "r":
                self._run(child, para)
            elif tag in (_M + "oMath", _M + "oMathPara"):
                math = omml_text(child).strip()
                if math:
                    para.parts.append(math)
                    para.math.append(math)
            elif tag in _INLINE_CONTAINERS:
                self._inline(child, para)
            elif tag == _W + "sdt":
                content = child.find(_W + "sdtContent")
                if content is not None:
                    self._inline(content, para)
            elif tag == _MC + "AlternateContent":
                chosen = child.find(_MC + "Choice")
                if chosen is None:
                    chosen = child.find(_MC + "Fallback")
                if chosen is not None:
                    self._inline(chosen, para)

    def _run(self, run, para: _Para) -> None:
        rpr = run.find(_W + "rPr")
        if _on(rpr, _W + "vanish"):
            return  # hidden text
        parts: list[str] = []
        for child in run.iterchildren():
            tag = child.tag
            if tag == _W + "t":
                parts.append(child.text or "")
            elif tag in (_W + "tab", _W + "ptab"):
                parts.append("\t")
            elif tag in (_W + "br", _W + "cr"):
                parts.append("\n")
            elif tag == _W + "noBreakHyphen":
                parts.append("-")
            elif tag == _W + "sym":
                try:
                    code = int(child.get(_W + "char") or "", 16)
                except ValueError:
                    continue
                if not 0xF000 <= code <= 0xF0FF:  # symbol-font private-use glyphs can't be mapped reliably
                    parts.append(chr(code))
            elif tag == _W + "ruby":  # East Asian text with a pronunciation guide: read the text itself
                base = child.find(_W + "rubyBase")
                if base is not None:
                    inner = _Para()
                    self._inline(base, inner)
                    parts.append("".join(inner.parts))
            elif tag in (_W + "footnoteReference", _W + "endnoteReference"):
                para.notes.append((_local(tag)[: -len("Reference")], child.get(_W + "id") or ""))
            elif tag in (_W + "drawing", _W + "pict", _W + "object", _MC + "AlternateContent"):
                para.extras.extend(self._drawing_blocks(child))
        text = "".join(parts)
        if _val(rpr, _W + "vertAlign") == "superscript" and text.strip():
            text = superscript_text(text)
        para.parts.append(text)
        para.plain.append(text)

    def _drawing_blocks(self, element) -> list[Block]:
        """Alt text of pictures (as figures) and the content of text boxes."""
        out: list[Block] = []

        def visit(node) -> None:
            for child in node.iterchildren():
                tag = child.tag
                if not isinstance(tag, str) or tag == _MC + "Fallback":
                    continue
                if tag == _W + "txbxContent":
                    saved, self.blocks = self.blocks, []
                    self.body(child)
                    out.extend(self.blocks)
                    self.blocks = saved
                    continue
                if tag in (_WP + "inline", _WP + "anchor"):
                    doc_pr = child.find(_WP + "docPr")
                    alt = ""
                    if doc_pr is not None:
                        alt = clean_alt_text(doc_pr.get("descr") or doc_pr.get("title") or "")
                    if alt.strip() and child.find(".//" + _W + "txbxContent") is None:
                        block = text_block(BlockKind.FIGURE, alt)
                        if block:
                            out.append(block)
                elif tag.startswith(_V) and clean_alt_text(child.get("alt") or ""):
                    block = text_block(BlockKind.FIGURE, clean_alt_text(child.get("alt") or ""))
                    if block:
                        out.append(block)
                visit(child)

        visit(element)
        return out

    # ------------------------------------------------------------ tables

    @staticmethod
    def _children(element, tag: str):
        """Direct children named ``tag``, looking through content controls / custom XML wrappers."""
        for child in element.iterchildren():
            if child.tag == tag:
                yield child
            elif child.tag == _W + "sdt":
                content = child.find(_W + "sdtContent")
                if content is not None:
                    yield from _DocxReader._children(content, tag)
            elif child.tag == _W + "customXml":
                yield from _DocxReader._children(child, tag)

    def _cell_text(self, cell) -> str:
        texts: list[str] = []
        for child in cell.iterchildren():
            if child.tag == _W + "p":
                para = _Para()
                self._inline(child, para)
                texts.append("".join(para.parts))
            elif child.tag == _W + "tbl":
                for row in self._children(child, _W + "tr"):
                    texts.append(" ".join(self._cell_text(c) for c in self._children(row, _W + "tc")))
            elif child.tag in (_W + "sdt", _W + "customXml"):
                content = child.find(_W + "sdtContent") if child.tag == _W + "sdt" else child
                if content is not None:
                    texts.append(self._cell_text(content))
        return " ".join(texts)

    def table(self, tbl) -> None:
        rows = [list(self._children(tr, _W + "tc")) for tr in self._children(tbl, _W + "tr")]
        if not rows:
            return
        if max(len(row) for row in rows) <= 1:
            for row in rows:  # a one-column table is a box around ordinary content
                for cell in row:
                    self.body(cell)
            return
        grid: list[list[str]] = []
        for row in rows:
            cells: list[str] = []
            for cell in row:
                tcpr = cell.find(_W + "tcPr")
                merge = tcpr.find(_W + "vMerge") if tcpr is not None else None
                continued = merge is not None and merge.get(_W + "val") != "restart"
                cells.append("" if continued else self._cell_text(cell))
                span = _val(tcpr, _W + "gridSpan")
                if span and span.isdigit():
                    cells.extend([""] * max(0, min(int(span), 20) - 1))
            grid.append(cells)
        self._emit(table_block(grid))


def load_docx(path: Path, *, source_name: str, progress: ProgressFn | None = None) -> Document:
    """Load a Word ``.docx`` document."""
    import docx

    path = Path(path)
    check_zip_size(path, source_name)
    if progress:
        progress(0.0, "Opening document")
    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise IngestError(
            "This Word file couldn't be opened. It may be damaged, password-protected, or an older .doc file "
            "(open it in Word and use Save As > Word Document (.docx))."
        ) from exc
    reader = _DocxReader(document)
    reader.body(document.element.body)
    if progress:
        progress(0.9, "Organizing sections")

    core_title = core_author = None
    try:
        core = document.core_properties
        core_title = clean_display_text(core.title or "")
        core_author = clean_display_text(core.author or "") or None
    except Exception:
        pass
    if core_title and core_title.lower().startswith("microsoft word - "):
        core_title = None
    title = reader.title or core_title or title_from_filename(source_name)
    sections = finalize_sections(sections_from_blocks(reader.blocks, title), title)
    if progress:
        progress(1.0, "Done")
    return Document(
        id=doc_id_for_file(path),
        title=title,
        author=core_author,
        source_name=source_name,
        source_type="docx",
        sections=sections,
        warnings=[] if reader.blocks else ["No readable text was found in this document."],
    )
