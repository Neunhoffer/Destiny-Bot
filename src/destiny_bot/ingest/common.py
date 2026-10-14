"""Helpers shared by every file loader."""

from __future__ import annotations

import hashlib
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

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
# Control codes (except tab, line breaks and other whitespace): never text. PDFs with math
# fonts and badly decoded files leave them behind.
_INVISIBLE.update(dict.fromkeys([*range(0x00, 0x09), *range(0x0E, 0x1C), 0x7F, *range(0x80, 0x85), *range(0x86, 0xA0)]))
# A word hyphenated at a line break: "infor-\nmation", "self-\nesteem", "pre-\nColumbian".
_HYPHEN_BREAK = re.compile(r"(?<!\w)(\w+)[-\u2010\u2011][ \t]*\n\s*(\w+)")
_SOFT_HYPHEN_BREAK = re.compile(r"\u00ad[ \t]*\n\s*")  # a discretionary hyphen: always rejoin
_DASH_BREAK = re.compile(r"(?<=\w)([\u2013\u2014])[ \t]*\n\s*(?=\w)")  # "1990–\n2000": no space
_WS_RUN = re.compile(r"\s+")  # (one pass: r"\s*\n\s*" would be quadratic on long runs of spaces)

# A section longer than this many characters is split into parts (~40 minutes of speech).
MAX_SECTION_CHARS = 40_000


# Word, PowerPoint and EPUB files are zip archives whose parts are unpacked into memory. A
# small file built to unpack to gigabytes (a "zip bomb") must not freeze the computer.
MAX_UNPACKED_BYTES = 1 << 30  # everything, pictures and videos included
MAX_PART_BYTES = 320 << 20  # one part (a video can be nearly as big as the 300 MB upload limit)
MAX_MARKUP_BYTES = 256 << 20  # the XML / XHTML parts that get parsed
_MARKUP_SUFFIXES = frozenset({"", ".xml", ".rels", ".xhtml", ".html", ".htm", ".opf", ".ncx", ".svg"})


def check_zip_size(path: Path, source_name: str) -> None:
    """Raise :class:`IngestError` if the zip archive at ``path`` unpacks to too much data.

    Damaged archives pass silently: the loader reports them in its own words.
    (Python's zip reader never inflates a part beyond its declared size.)
    """
    from . import IngestError

    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
    except (zipfile.BadZipFile, OSError, ValueError, EOFError):
        return
    total = sum(info.file_size for info in infos)
    markup = sum(info.file_size for info in infos if Path(info.filename).suffix.lower() in _MARKUP_SUFFIXES)
    biggest = max((info.file_size for info in infos), default=0)
    if total > MAX_UNPACKED_BYTES or markup > MAX_MARKUP_BYTES or biggest > MAX_PART_BYTES:
        size = f"{total / (1 << 30):.1f} GB" if total >= 1 << 30 else f"{total >> 20} MB"
        raise IngestError(
            f"'{Path(source_name).name}' would unpack to {size} of data, far more than a normal document, so it "
            "wasn't opened (that keeps your computer from freezing). The file may be damaged. If it opens "
            "normally in Word, PowerPoint or your e-book reader, save a new copy there and import that."
        )


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


# --------------------------------------------------------------------------- line-break hyphens

# Prefixes after which a line-break hyphen is usually a real one ("self-esteem", "half-life",
# "short-term", "pre-existing"). The hyphen goes only when the right part makes an ordinary
# word with the prefix: a word ending ("self-\nish") or a common closed word ("pre-\nvious",
# "anti-\nbodies", "multi-\nple").
_HYPHEN_PREFIXES: dict[str, tuple[str, ...]] = {
    "self": ("ish", "less", "hood", "same"),
    "well": ("ness", "spring", "ington"),
    "non": ("sens", "ethele", "chalan", "descript"),
    "all": ("ow", "ot", "oy", "ude", "ur", "ianc", "ied", "ies", "ocat", "usi", "uvi", "erg", "itera", "ay", "eg"),
    "cross": ("ing", "ed", "es", "er", "ly", "ness"),
    "half": ("penn",),
    "full": ("ness", "er", "est"),
    "high": ("light", "way", "land", "ness", "er", "est", "ly"),
    "low": ("land", "er", "est", "ly", "li", "ness"),
    "long": ("ing", "er", "est", "itud", "ish", "hand", "horn", "bow", "boat", "shore", "evit"),
    "short": ("age", "en", "er", "est", "ly", "ness", "fall", "coming", "hand", "cut", "stop", "bread"),
    "pre": (
        "vious", "vent", "dict", "fer", "par", "serv", "sum", "cis", "fix", "view", "mium", "mis", "lud", "fac",
        "ced", "cip", "dat", "decess", "domin", "limin", "matur", "scri", "tend", "vail", "val", "sent", "gnan",
        "jud", "ach", "caution", "cursor", "dispos", "histor", "mier", "natal", "occup", "paid", "posit",
        "requis", "school", "synap", "rogativ", "hensil", "amble", "cinct", "cious", "clud", "concept", "determin",
        "dica", "dilect", "empt", "fect", "frontal", "load", "medit", "monit", "ponder", "scien", "supp", "text",
    ),
    "post": (
        "pon", "age", "script", "doc", "graduat", "war", "modern", "operativ", "natal", "synap", "mortem", "mark",
        "man", "card", "season", "partum", "secondary", "colonial", "industrial", "prandial", "menopaus", "ulat",
        "ur", "erior",
    ),
    "anti": (
        "bod", "biot", "gen", "dot", "septic", "oxidant", "bacterial", "viral", "freez", "matter", "thes",
        "derivativ", "symmetr", "clockwis", "nod", "particl", "path", "social", "trust", "histamin", "sens",
        "parallel", "log", "quark", "neutrino", "proton", "electron", "cipat", "cancer", "coagul", "convuls",
        "depress", "inflamm", "fung", "micro", "pod",
    ),
    "multi": (
        "pl", "tud", "variat", "nomial", "media", "national", "cellular", "lateral", "lingual", "task", "facet",
        "purpose", "cultural", "dimensional", "linear", "set", "modal", "valent", "core", "thread",
    ),
}
# Prefixes that are mostly written closed ("example", "coefficient", "cosine"): the hyphen
# stays only for the few compounds listed ("ex-wife", "co-author").
_CLOSED_PREFIXES: dict[str, tuple[str, ...]] = {
    "ex": ("wife", "wive", "husband", "president", "boyfriend", "girlfriend", "partner", "spouse", "officio",
           "smoker", "convict", "soldier", "minister", "employee", "member"),
    "co": ("worker", "author", "founder", "chair", "host", "pilot", "star", "sign", "conspirator", "defendant",
           "teach", "parent"),
}  # fmt: skip
# Second halves that make compounds, not word endings ("evidence-based", "decision-making").
_COMPOUND_TAILS = frozenset(
    "based like specific making free friendly oriented driven induced mediated dependent level scale term "
    "wide known related off sized solving containing bound centered focused aware sensitive resistant".split()
)
# "first- and second-order": a suspended hyphen keeps its space.
_SUSPENDED = frozenset({"and", "or", "to"})


class HyphenationEvidence:
    """How a document spells words elsewhere, to decide breaks such as "self-\nesteem".

    Build it from the document's text (lines as they appear, with their own line
    breaks); a break is joined when the document writes the joined word more often
    than the hyphenated one, and kept hyphenated in the opposite case.
    """

    def __init__(self, texts: Iterable[str] = ()) -> None:
        self.words: Counter[str] = Counter()
        self.compounds: Counter[str] = Counter()
        for text in texts:
            self.add(text)

    def add(self, text: str) -> None:
        if not text:
            return
        lowered = fix_characters(text).lower()
        self.words.update(_WORD.findall(lowered))
        if "-" in lowered:
            self.compounds.update(f"{a}-{b}" for a, b in _COMPOUND.findall(lowered))

    def prefers(self, left: str, right: str) -> bool | None:
        """True: keep the hyphen; False: join; None: the document doesn't say."""
        hyphenated = self.compounds.get(f"{left}-{right}", 0)
        joined = self.words.get(left + right, 0)
        if hyphenated == joined:
            return None
        return hyphenated > joined


_WORD = re.compile(r"[^\W\d_]+")
_COMPOUND = re.compile(r"(?<![^\W\d_])([^\W\d_]+)-(?=([^\W\d_]+))")  # (anchored: linear time)


def _line_break_hyphen(left: str, right: str, before: str, evidence: HyphenationEvidence | None) -> str:
    """How to rejoin "left-" at the end of a line with "right" at the start of the next.

    ``before`` is the character before ``left``.
    """
    hyphenated, joined = f"{left}-{right}", left + right
    if left[-1].isdigit() or right[0].isdigit() or before in ("-", "‐", "‑"):
        return hyphenated  # "COVID-19", "3-methyl", "state-of-the-art"
    if right[0].isupper() and not (left.isupper() and right.isupper()):
        return hyphenated  # "pre-Columbian", "non-English" ("INFOR-MATION" is still one word)
    lo_left, lo_right = left.lower(), right.lower()
    if lo_right in _SUSPENDED and len(left) >= 3:
        return f"{left}- {right}"  # "first- and second-order"
    if evidence is not None:
        keep = evidence.prefers(lo_left, lo_right)
        if keep is not None:
            return hyphenated if keep else joined
    if lo_left in _HYPHEN_PREFIXES:
        return joined if lo_right.startswith(_HYPHEN_PREFIXES[lo_left]) else hyphenated
    if lo_left in _CLOSED_PREFIXES:
        return hyphenated if lo_right.startswith(_CLOSED_PREFIXES[lo_left]) else joined
    if lo_right in _COMPOUND_TAILS and len(left) >= 4:
        return hyphenated  # "evidence-based"
    return joined


def rejoin_line_breaks(text: str, evidence: HyphenationEvidence | None = None) -> str:
    """Rejoin words hyphenated (or dashed) across line breaks; other line breaks stay."""
    if "\n" not in text:
        return text
    text = _SOFT_HYPHEN_BREAK.sub("", text)
    text = _DASH_BREAK.sub(r"\1", text)

    def repl(m: re.Match[str]) -> str:
        start = m.start(1)
        return _line_break_hyphen(m.group(1), m.group(2), m.string[start - 1 : start] if start else "", evidence)

    return _HYPHEN_BREAK.sub(repl, text)


def clean_display_text(text: str, *, evidence: HyphenationEvidence | None = None) -> str:
    """Turn raw extracted text into a single clean line for display and reading.

    Repairs ligatures, removes soft hyphens / zero-width characters, rejoins words
    hyphenated across line breaks (keeping the hyphen of compounds such as
    "self-esteem"; ``evidence`` tells how the rest of the document spells them),
    turns remaining line breaks into spaces and collapses runs of whitespace.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\u00ad" in text:
        text = _SOFT_HYPHEN_BREAK.sub("", text)  # (before fix_characters drops the soft hyphens)
    text = rejoin_line_breaks(fix_characters(text), evidence)
    text = _WS_RUN.sub(" ", text)  # line breaks, tabs, no-break and other spaces -> one space
    return text.strip()


# Programs fill in a picture's description with its file name or address ("IMG_2034.JPG",
# "K:\\Pics\\cell.png", "https://..."), or append a note to the one they generate.
_FILE_NAME_ALT = re.compile(
    r"(?:^|[\\/])[^\\/]*\.(?:png|jpe?g|gif|bmp|tiff?|emf|wmf|svg|webp|heic|ico|eps|psd)$", re.I
)
_URL_ALT = re.compile(r"^(?:https?|file|ftp)://\S*$|^www\.\S+$", re.I)
_AUTO_ALT_NOTE = re.compile(
    r"\.? ?(?:Description|Text) (?:automatically generated|generated with (?:very )?(?:high|medium|low) confidence)"
    r"\.?$",
    re.I,
)


def clean_alt_text(alt: str) -> str:
    """A picture's description worth reading aloud ("" for a file name, path or web address)."""
    alt = _AUTO_ALT_NOTE.sub("", clean_display_text(alt)).strip()
    if _URL_ALT.match(alt) or (_FILE_NAME_ALT.search(alt) and len(alt.split()) <= 8):
        return ""
    return alt


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
