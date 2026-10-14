"""Generate realistic test PDFs in code (no binary fixtures are committed).

:class:`PdfBuilder` lays out text like a word processor or a textbook: wrapped
paragraphs (optionally justified, with first-line indents), headings, bullet
lists, one or more columns that flow onto new pages, footnote-style
superscript markers (``word^1``), running headers / footers and page numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pymupdf

# PyMuPDF base-14 font codes
REGULAR = "helv"
BOLD = "hebo"
ITALIC = "heit"
SERIF = "tiro"
SERIF_BOLD = "tibo"
MONO = "cour"


_FONTS: dict[str, pymupdf.Font] = {}


def _font(name: str) -> pymupdf.Font:
    if name not in _FONTS:
        _FONTS[name] = pymupdf.Font(name)
    return _FONTS[name]


def text_width(text: str, font: str = REGULAR, size: float = 11) -> float:
    return pymupdf.get_text_length(text, fontname=font, fontsize=size)


class PdfBuilder:
    """Flow text onto pages top to bottom, column by column."""

    def __init__(
        self,
        *,
        width: float = 612,
        height: float = 792,
        margin: float = 72,
        body: float = 11,
        leading: float = 1.3,
        font: str = REGULAR,
    ) -> None:
        self.doc = pymupdf.open()
        self.width = width
        self.height = height
        self.margin = margin
        self.body = body
        self.leading = leading
        self.font = font
        self.columns: list[tuple[float, float]] = [(margin, width - margin)]
        self.col = 0
        self.region_top = margin
        self.y = margin
        self._writer: pymupdf.TextWriter | None = None
        self._writer_page: pymupdf.Page | None = None
        self.page: pymupdf.Page = self.doc.new_page(width=width, height=height)

    # ----------------------------------------------------------------- layout state

    @property
    def x0(self) -> float:
        return self.columns[self.col][0]

    @property
    def x1(self) -> float:
        return self.columns[self.col][1]

    @property
    def bottom(self) -> float:
        return self.height - self.margin

    def flush(self) -> None:
        """Write pending text of the current page (text is batched for speed)."""
        if self._writer is not None and self._writer_page is not None:
            self._writer.write_text(self._writer_page)
        self._writer = self._writer_page = None

    def text(self, pos: tuple[float, float], text: str, *, font: str = REGULAR, size: float = 11) -> None:
        """Queue one text run at ``pos`` (baseline start) on the current page."""
        if self._writer_page is not self.page:
            self.flush()
            self._writer = pymupdf.TextWriter(self.page.rect)
            self._writer_page = self.page
        assert self._writer is not None
        self._writer.append(pos, text, font=_font(font), fontsize=size)

    def new_page(self) -> None:
        self.flush()
        self.page = self.doc.new_page(width=self.width, height=self.height)
        self.col = 0
        self.region_top = self.margin
        self.y = self.margin

    def set_columns(self, n: int, gutter: float = 24) -> None:
        """Switch to ``n`` columns starting at the current position."""
        left, right = self.margin, self.width - self.margin
        w = (right - left - gutter * (n - 1)) / n
        self.columns = [(left + k * (w + gutter), left + k * (w + gutter) + w) for k in range(n)]
        self.col = 0
        self.region_top = self.y

    def _advance(self, height: float) -> None:
        """Make room for ``height`` points, moving to the next column/page when needed."""
        if self.y + height <= self.bottom:
            return
        if self.col + 1 < len(self.columns):
            self.col += 1
            self.y = self.region_top
        else:
            self.new_page()

    def space(self, pts: float) -> None:
        self.y += pts

    # ----------------------------------------------------------------- primitives

    def line(self, text: str, *, size: float | None = None, font: str | None = None, x: float | None = None) -> None:
        """Place one line of text (``word^1`` renders a raised superscript marker)."""
        size = size or self.body
        font = font or self.font
        pitch = size * self.leading
        self._advance(pitch)
        baseline = self.y + size
        self._draw_line(text, x if x is not None else self.x0, baseline, size, font)
        self.y += pitch

    def _draw_line(self, text: str, x: float, baseline: float, size: float, font: str, justify_to: float | None = None) -> None:
        if "^" not in text and justify_to is None:
            self.text((x, baseline), text, font=font, size=size)
            return
        words = text.split(" ")
        gap = text_width(" ", font, size)
        if justify_to is not None and len(words) > 1:
            natural = sum(self._word_width(w, size, font) for w in words)
            gap = (justify_to - x - natural) / (len(words) - 1)
        for k, word in enumerate(words):
            base, _, sup = word.partition("^")
            if base:
                self.text((x, baseline), base, font=font, size=size)
                x += text_width(base, font, size)
            if sup:
                small = size * 0.6
                self.text((x + 0.3, baseline - size * 0.38), sup, font=font, size=small)
                x += text_width(sup, font, small) + 0.6
            if k < len(words) - 1:
                x += gap

    def _word_width(self, word: str, size: float, font: str) -> float:
        base, _, sup = word.partition("^")
        return text_width(base, font, size) + (text_width(sup, font, size * 0.6) + 0.9 if sup else 0)

    def wrap(self, text: str, width: float, size: float, font: str, first_indent: float = 0) -> list[str]:
        lines: list[str] = []
        current: list[str] = []
        avail = width - first_indent
        used = 0.0
        space = text_width(" ", font, size)
        for word in text.split():
            w = self._word_width(word, size, font)
            if current and used + space + w > avail:
                lines.append(" ".join(current))
                current, used, avail = [word], w, width
            else:
                used += (space if current else 0) + w
                current.append(word)
        if current:
            lines.append(" ".join(current))
        return lines

    # ----------------------------------------------------------------- blocks

    def paragraph(
        self,
        text: str,
        *,
        size: float | None = None,
        font: str | None = None,
        first_indent: float = 0,
        space_after: float | None = None,
        justify: bool = False,
        left_indent: float = 0,
    ) -> None:
        size = size or self.body
        font = font or self.font
        pitch = size * self.leading
        width = self.x1 - self.x0 - left_indent
        lines = self.wrap(text, width, size, font, first_indent)
        for k, ln in enumerate(lines):
            self._advance(pitch)
            x = self.x0 + left_indent + (first_indent if k == 0 else 0)
            last = k == len(lines) - 1
            self._draw_line(ln, x, self.y + size, size, font, None if (last or not justify) else self.x1)
            self.y += pitch
        self.y += self.body * 0.6 if space_after is None else space_after

    def lines(self, lines: list[str], *, size: float | None = None, font: str | None = None, space_after: float = 6) -> None:
        """Place exact lines (to control line breaks, e.g. for hyphenation tests)."""
        for ln in lines:
            self.line(ln, size=size, font=font)
        self.y += space_after

    def heading(self, text: str, *, size: float = 16, font: str = BOLD, before: float = 10, after: float = 4) -> None:
        if self.y > self.region_top + 1:
            self.y += before
        self._advance(size * 1.3 + self.body * 3)  # keep a heading with its first lines
        self.line(text, size=size, font=font)
        self.y += after

    def bullet(self, text: str, *, marker: str = "•", indent: float = 18, size: float | None = None) -> None:
        size = size or self.body
        pitch = size * self.leading
        lines = self.wrap(text, self.x1 - self.x0 - indent, size, self.font)
        for k, ln in enumerate(lines):
            self._advance(pitch)
            if k == 0:
                self.text((self.x0 + 4, self.y + size), marker, font=self.font, size=size)
            self._draw_line(ln, self.x0 + indent, self.y + size, size, self.font)
            self.y += pitch
        self.y += 2

    def footnote(self, text: str, *, size: float = 8) -> None:
        """Place a footnote below the text area of the current page (in the bottom margin)."""
        lines = self.wrap(text, self.x1 - self.x0, size, self.font)
        y = self.bottom + 12
        self.page.draw_line((self.x0, y - 6), (self.x0 + 120, y - 6), width=0.5)
        for ln in lines:
            self._draw_line(ln, self.x0, y + size, size, self.font)
            y += size * 1.25

    def table(self, rows: list[list[str]], *, col_widths: list[float] | None = None, size: float = 10) -> pymupdf.Rect:
        """Draw a ruled table at the current position (full column width)."""
        ncols = len(rows[0])
        widths = col_widths or [(self.x1 - self.x0) / ncols] * ncols
        rh = size * 2
        self._advance(rh * len(rows))
        top = self.y
        for r, row in enumerate(rows):
            x = self.x0
            for c, cell in enumerate(row):
                rect = pymupdf.Rect(x, top + r * rh, x + widths[c], top + (r + 1) * rh)
                self.page.draw_rect(rect, color=(0, 0, 0), width=0.6)
                font = BOLD if r == 0 else self.font
                self.text((rect.x0 + 4, rect.y1 - size * 0.6), cell, font=font, size=size)
                x += widths[c]
        self.y = top + rh * len(rows) + self.body
        return pymupdf.Rect(self.x0, top, self.x0 + sum(widths), self.y)

    def image(self, height: float, *, width: float | None = None, png: bytes | None = None) -> pymupdf.Rect:
        """A grey picture (or the given PNG) at the current position."""
        width = width or (self.x1 - self.x0)
        self._advance(height)
        rect = pymupdf.Rect(self.x0, self.y, self.x0 + width, self.y + height)
        if png is None:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 30), False)
            pix.clear_with(180)
            self.page.insert_image(rect, pixmap=pix)
        else:
            self.page.insert_image(rect, stream=png)
        self.y += height + 6
        return rect

    # ----------------------------------------------------------------- finishing

    def decorate(
        self,
        *,
        header: Callable[[int], str | None] | None = None,
        footer: Callable[[int], str | None] | None = None,
        size: float = 9,
    ) -> None:
        """Add running headers/footers; callables get the 1-based page number."""
        self.flush()
        for i, page in enumerate(self.doc):
            n = i + 1
            if header and (h := header(n)):
                x = self.margin if n % 2 == 0 else self.width - self.margin - text_width(h, self.font, size)
                page.insert_text((x, self.margin * 0.55), h, fontname=self.font, fontsize=size)
            if footer and (f := footer(n)):
                x = (self.width - text_width(f, self.font, size)) / 2
                page.insert_text((x, self.height - self.margin * 0.45), f, fontname=self.font, fontsize=size)

    def save(self, path: Path, *, toc: list | None = None, metadata: dict | None = None, **save_kwargs) -> Path:
        self.flush()
        if toc:
            self.doc.set_toc(toc)
        if metadata:
            self.doc.set_metadata(metadata)
        self.doc.save(path, **save_kwargs)
        self.doc.close()
        return Path(path)


def text_image_png(lines: list[str], *, size: int = 40, width: int = 1700, height: int = 2200) -> bytes:
    """Render text into a PNG (black on white) to simulate a scanned page."""
    doc = pymupdf.open()
    page = doc.new_page(width=width / 4, height=height / 4)
    y = 60.0
    for ln in lines:
        if not ln:
            y += size / 4 * 1.2
            continue
        page.insert_text((50, y), ln, fontname=SERIF, fontsize=size / 4)
        y += size / 4 * 1.5
    pix = page.get_pixmap(dpi=288, colorspace=pymupdf.csGRAY)
    data = pix.tobytes("png")
    doc.close()
    return data


def scanned_pdf(path: Path, pages: list[list[str]]) -> Path:
    """A PDF whose pages are only images of text (no text layer)."""
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page(width=612, height=792)
        page.insert_image(page.rect, stream=text_image_png(lines))
    doc.save(path)
    doc.close()
    return Path(path)


