"""Core data model shared by every part of Destiny Bot.

A loaded file becomes a :class:`Document`: an ordered list of :class:`Section`
objects (chapters / sub-chapters / slides), each holding :class:`Block` objects
(headings, paragraphs, captions, tables, ...). ``Block.text`` is *display* text:
already cleaned for reading on screen (hyphenation repaired, whitespace
collapsed, footnote markers dropped) but otherwise faithful to the source.

The speech layer turns a section into a list of :class:`Utterance` objects. Each
utterance points back at the display text it came from (``block`` + ``start`` /
``end`` character offsets) so the UI can highlight the sentence being read,
while ``speech`` holds the normalized text that is sent to the voice engine.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


class BlockKind(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    QUOTE = "quote"
    CAPTION = "caption"  # "Figure 2.1 ..." / "Table 3.4 ..."
    FIGURE = "figure"  # an image; ``text`` is its alt text / description ("" if none)
    TABLE = "table"  # ``rows`` holds the cells; ``text`` is a plain-text rendering
    EQUATION = "equation"  # best-effort linear form (LaTeX, MathML alttext, or raw glyphs)
    FOOTNOTE = "footnote"
    CODE = "code"


class Block(BaseModel):
    kind: BlockKind
    text: str
    level: int | None = None  # headings only: 1 (chapter) .. 6
    rows: list[list[str]] | None = None  # tables only: rows of cell strings
    page: int | None = None  # 1-based PDF page / PPTX slide number where the block starts
    meta: dict[str, Any] = Field(default_factory=dict)


class Section(BaseModel):
    id: str  # unique within its document: "s0", "s1", ... in reading order
    title: str
    level: int = 1  # 1 = top level (chapter); 2+ = nested sub-sections
    blocks: list[Block] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None


class Document(BaseModel):
    id: str  # first 16 hex chars of the SHA-256 of the source bytes
    title: str
    author: str | None = None
    source_name: str  # original file name, e.g. "Biology 101 - Chapter 3.pdf"
    source_type: str  # "pdf" | "epub" | "docx" | "pptx" | "html" | "txt" | "md" | "image" | "paste"
    page_count: int | None = None
    sections: list[Section]
    warnings: list[str] = Field(default_factory=list)  # human-readable import notes
    schema_version: int = SCHEMA_VERSION

    def section(self, section_id: str) -> Section:
        for sec in self.sections:
            if sec.id == section_id:
                return sec
        raise KeyError(section_id)


class ReadingOptions(BaseModel):
    """What to read aloud and how. Changing these never changes the display text."""

    read_headings: bool = True
    announce_section_title: bool = True  # say the section title when it has no heading block
    read_captions: bool = True
    read_footnotes: bool = False
    read_figures: bool = True  # speak alt text of images, when there is some
    tables: Literal["skip", "announce", "rows"] = "rows"
    equations: Literal["skip", "read"] = "read"
    code: Literal["skip", "announce", "read"] = "announce"
    citations: Literal["keep", "remove"] = "remove"  # "(Smith et al., 2019)", "[12]", "[3-5]"
    urls: Literal["full", "domain", "skip"] = "domain"
    expand_abbreviations: bool = True  # "e.g." -> "for example", "Fig." -> "Figure", ...
    max_chars: int = Field(default=400, ge=80, le=2000)  # longest utterance sent to a voice


class Utterance(BaseModel):
    index: int  # position in the section's reading plan, 0-based and contiguous
    block: int  # index into Section.blocks; -1 for a synthesized section-title announcement
    start: int  # char offset into blocks[block].text to highlight (inclusive)
    end: int  # exclusive; for block == -1 both are 0
    kind: BlockKind
    speech: str  # normalized text for the TTS engine; never empty
    pause_after_ms: int = 0  # extra silence after this utterance (paragraph/heading breaks)
