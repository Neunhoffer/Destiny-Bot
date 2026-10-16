"""Tests for the PDF loader (all PDFs are generated in code)."""

from __future__ import annotations

import random
import time
from pathlib import Path

import pymupdf
import pytest
from pdf_factory import BOLD, MONO, REGULAR, PdfBuilder, scanned_pdf, text_width

from athena.ingest import IngestError, load_document
from athena.ingest import pdf as pdf_mod
from athena.ingest.common import doc_id_for_file
from athena.ingest.pdf import load_pdf
from athena.models import BlockKind, Document

WORDS = (
    "cell membrane protein lipid transport energy diffusion gradient water molecule channel pump "
    "signal receptor enzyme nucleus organelle solute osmosis vesicle carrier ion"
).split()


def sentence(rng: random.Random, n: int = 12) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n)).capitalize() + "."


def para(rng: random.Random, k: int = 4, tag: str = "") -> str:
    text = " ".join(sentence(rng) for _ in range(k))
    return f"{tag} {text}" if tag else text


def roman(n: int) -> str:
    out = ""
    for value, digits in ((10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while n >= value:
            out, n = out + digits, n - value
    return out


def load(path: Path, **kwargs) -> Document:
    return load_pdf(path, source_name=path.name, **kwargs)


def blocks(doc: Document):
    return [b for s in doc.sections for b in s.blocks]


def texts(doc: Document) -> list[str]:
    return [b.text for b in blocks(doc)]


def full_text(doc: Document) -> str:
    return "\n".join(texts(doc))


# --------------------------------------------------------------------------- headers / footers


@pytest.mark.parametrize(
    "footer",
    [
        lambda n: str(n + 10),
        lambda n: f"Page {n + 10}",
        lambda n: roman(n),
        lambda n: f"{n + 10} | Chapter 3 Cells",
        lambda n: f"Chapter 3 | {n + 10}",
    ],
    ids=["number", "page-n", "roman", "num-bar-title", "title-bar-num"],
)
def test_running_headers_footers_and_page_numbers_removed(tmp_path, footer):
    rng = random.Random(3)
    b = PdfBuilder()
    b.heading("Cells", size=20)
    for i in range(70):
        b.paragraph(para(rng, 3, f"P{i}"), first_indent=14)
    b.decorate(
        header=lambda n: "Biology 101: Fall Edition" if n % 2 == 0 else "Chapter 3 Cells and Membranes",
        footer=footer,
    )
    path = b.save(tmp_path / "hf.pdf")
    doc = load(path)
    assert doc.page_count >= 6
    all_text = full_text(doc)
    assert "Biology 101" not in all_text
    assert "Chapter 3 Cells and Membranes" not in all_text
    assert "Chapter 3 |" not in all_text and "| Chapter 3" not in all_text
    for t in texts(doc):
        assert not t.strip().isdigit(), t
        assert not t.lower().startswith("page "), t
        assert t not in {roman(n) for n in range(1, 40)}
    # every body paragraph survived, in order
    tags = [t.split()[0] for t in texts(doc) if t.startswith("P")]
    assert tags == [f"P{i}" for i in range(70)]


def test_heading_at_top_of_one_page_is_kept(tmp_path):
    rng = random.Random(4)
    b = PdfBuilder()
    for i in range(8):
        if i:
            b.new_page()
        if i == 4:
            b.line("Summary of Membrane Transport", size=14, font=BOLD)
            b.space(6)
        for j in range(4):
            b.paragraph(para(rng, 3), first_indent=14)
    b.decorate(header=lambda n: "Unit 2 Cell Biology", footer=lambda n: f"- {n} -")
    doc = load(b.save(tmp_path / "top.pdf"))
    heads = [bl for bl in blocks(doc) if bl.kind == BlockKind.HEADING]
    assert [h.text for h in heads] == ["Summary of Membrane Transport"]
    assert heads[0].page == 5
    assert "Unit 2 Cell Biology" not in full_text(doc)


# --------------------------------------------------------------------------- text repair


def test_hyphenation_repaired_and_lines_joined(tmp_path):
    b = PdfBuilder()
    b.lines(
        [
            "The plasma membrane regulates the move-",
            "ment of substances into and out of the cell, and it is",
            "built from a phospholipid bi-",
            "layer with embedded proteins.",
        ]
    )
    doc = load(b.save(tmp_path / "hyph.pdf"))
    paras = [bl for bl in blocks(doc) if bl.kind == BlockKind.PARAGRAPH]
    assert len(paras) == 1
    assert paras[0].text == (
        "The plasma membrane regulates the movement of substances into and out of the cell, "
        "and it is built from a phospholipid bilayer with embedded proteins."
    )


def test_superscript_footnote_markers_removed_but_exponents_kept(tmp_path):
    b = PdfBuilder()
    b.lines(
        [
            "Water crosses the membrane^1 through aquaporins.^2 A cell may",
            "contain about 10^3 channels, and the area grows as r^2 while",
            "the volume grows faster.^12",
        ]
    )
    doc = load(b.save(tmp_path / "sup.pdf"))
    (p,) = [bl for bl in blocks(doc) if bl.kind == BlockKind.PARAGRAPH]
    assert "membrane through aquaporins. A cell" in p.text
    assert "10³ channels" in p.text
    assert "r² while" in p.text
    assert p.text.endswith("grows faster.")


# --------------------------------------------------------------------------- layout


def test_two_column_reading_order(tmp_path):
    rng = random.Random(5)
    b = PdfBuilder()
    b.line("Membrane Transport in Living Cells", size=20, font=BOLD)
    b.space(8)
    b.paragraph("FULLWIDTH " + para(rng, 2))
    b.set_columns(2)
    for i in range(16):
        b.paragraph(para(rng, 3, f"COL{i:02d}"), first_indent=12, justify=True)
    path = b.save(tmp_path / "twocol.pdf")
    doc = load(path)
    order = [t.split()[0] for t in texts(doc)]
    assert order[0] == "Membrane"
    assert order[1] == "FULLWIDTH"
    tags = [t for t in order if t.startswith("COL")]
    assert tags == [f"COL{i:02d}" for i in range(16)]
    # Paragraphs that wrap from the left column into the right column (or the next
    # page) are rejoined, so every COL paragraph is one block with complete sentences.
    for t in texts(doc):
        if t.startswith("COL"):
            assert t.endswith("."), t
            assert t.count("COL") == 1


def test_single_column_keeps_order_with_indents_and_short_lines(tmp_path):
    rng = random.Random(6)
    b = PdfBuilder()
    for i in range(10):
        # styles: indented first line, or no indent but space before it
        b.paragraph(para(rng, 2, f"S{i}"), first_indent=18 if i % 3 else 0, space_after=8 if i % 3 == 2 else 0)
    doc = load(b.save(tmp_path / "single.pdf"))
    assert [t.split()[0] for t in texts(doc)] == [f"S{i}" for i in range(10)]


def test_three_columns(tmp_path):
    rng = random.Random(7)
    b = PdfBuilder()
    b.set_columns(3, gutter=18)
    for i in range(12):
        b.paragraph(para(rng, 2, f"T{i:02d}"), first_indent=10)
    doc = load(b.save(tmp_path / "three.pdf"))
    assert [t.split()[0] for t in texts(doc) if t.startswith("T")] == [f"T{i:02d}" for i in range(12)]


def test_cross_page_paragraph_merge_with_footnote(tmp_path):
    rng = random.Random(8)
    b = PdfBuilder()
    i = 0
    while b.y < b.bottom - 150:
        b.paragraph(para(rng, 3, f"Intro{i}"), first_indent=14)
        i += 1
    long = para(rng, 14, "SPLIT")
    b.paragraph(long, first_indent=14)
    b.paragraph("AFTER " + para(rng, 2), first_indent=14)
    b.footnote("1 Data adapted from Smith (2019), Journal of Cell Science.", page=0)
    doc = load(b.save(tmp_path / "flow.pdf"))
    assert doc.page_count == 2
    split = [bl for bl in blocks(doc) if "SPLIT" in bl.text]
    assert len(split) == 1
    assert split[0].page == 1
    assert split[0].text.endswith(long.split()[-1])
    kinds = [(bl.kind, bl.text.split()[0]) for bl in blocks(doc)]
    i_split = next(i for i, (_, w) in enumerate(kinds) if w == "SPLIT")
    assert kinds[i_split + 1][0] == BlockKind.FOOTNOTE
    assert kinds[i_split + 2][1] == "AFTER"
    fn = blocks(doc)[i_split + 1]
    assert fn.text.startswith("1 Data adapted") and fn.page == 1


def test_boxed_text_does_not_break_paragraph(tmp_path):
    rng = random.Random(9)
    b = PdfBuilder()
    b.paragraph(para(rng, 2, "BEFORE"))
    first_half = "MAIN The sodium-potassium pump moves three sodium ions out of the cell and two potassium ions into"
    b.lines([first_half], space_after=8)
    top = b.y
    b.space(6)
    b.paragraph("KEY IDEA Active transport needs energy from ATP.", left_indent=10, space_after=4)
    box = pymupdf.Rect(b.x0, top, b.x1, b.y + 4)
    b.page.draw_rect(box, color=(0.2, 0.2, 0.6), fill=(0.9, 0.9, 1.0), overlay=False)
    b.space(12)
    b.lines(["the cell for every ATP molecule it uses. This keeps the gradient steady."])
    doc = load(b.save(tmp_path / "box.pdf"))
    ts = texts(doc)
    main = next(t for t in ts if t.startswith("MAIN"))
    assert main.endswith("potassium ions into the cell for every ATP molecule it uses. This keeps the gradient steady.")
    assert ts.index(main) < next(i for i, t in enumerate(ts) if t.startswith("KEY IDEA"))


# --------------------------------------------------------------------------- headings & sections


def test_heading_levels_without_outline(tmp_path):
    rng = random.Random(10)
    b = PdfBuilder()
    b.line("Introduction to Cell Biology", size=22, font=BOLD)
    b.space(8)
    b.paragraph(para(rng, 2))
    for sec in range(1, 3):
        b.heading(f"{sec} Membranes part {sec}", size=15)
        b.paragraph(para(rng, 3))
        for sub in range(1, 3):
            b.heading(f"{sec}.{sub} Detail {sub}", size=11, before=6, after=2)
            b.paragraph(para(rng, 3))
            b.paragraph("Some more text in the middle: " + para(rng, 1))
    doc = load(b.save(tmp_path / "levels.pdf"))
    heads = [(h.text, h.level) for h in blocks(doc) if h.kind == BlockKind.HEADING]
    assert heads == [
        ("Introduction to Cell Biology", 1),
        ("1 Membranes part 1", 2),
        ("1.1 Detail 1", 3),
        ("1.2 Detail 2", 3),
        ("2 Membranes part 2", 2),
        ("2.1 Detail 1", 3),
        ("2.2 Detail 2", 3),
    ]
    assert doc.title == "Introduction to Cell Biology"
    assert [s.title for s in doc.sections] == [
        "Introduction to Cell Biology",
        "1 Membranes part 1",
        "2 Membranes part 2",
    ]
    for s in doc.sections:
        assert s.blocks[0].kind == BlockKind.HEADING and s.blocks[0].text == s.title


def test_chapter_label_joins_title_and_bold_numbered_headings(tmp_path):
    rng = random.Random(11)
    b = PdfBuilder()
    b.line("CHAPTER 3", size=14, font=BOLD)
    b.line("Cells and Membranes", size=24, font=BOLD)
    b.space(10)
    b.paragraph(para(rng, 3))
    for k in range(1, 4):
        b.heading(f"3.{k} Membrane Transport {k}", size=11, before=8, after=2)
        b.paragraph(para(rng, 3) + " Cells use energy daily.")
    doc = load(b.save(tmp_path / "chapter.pdf"))
    heads = [(h.text, h.level) for h in blocks(doc) if h.kind == BlockKind.HEADING]
    assert heads[0] == ("CHAPTER 3: Cells and Membranes", 1)
    assert heads[1:] == [(f"3.{k} Membrane Transport {k}", 2) for k in range(1, 4)]


def test_no_false_heading_for_bold_words_and_labels(tmp_path):
    b = PdfBuilder()
    x = b.x0
    y = b.y + 11
    # A paragraph line that mixes bold and regular text
    b.page.insert_text((x, y), "Osmosis", fontname=BOLD, fontsize=11)
    rest = "is the diffusion of water across a membrane."
    b.page.insert_text((x + text_width("Osmosis ", BOLD, 11), y), rest, fontsize=11)
    b.y += 30
    b.paragraph("Water moves from low solute concentration to high solute concentration until it balances.")
    doc = load(b.save(tmp_path / "bold.pdf"))
    assert all(bl.kind != BlockKind.HEADING for bl in blocks(doc))
    assert texts(doc)[0] == "Osmosis is the diffusion of water across a membrane."


def _outline_book(tmp_path: Path) -> Path:
    rng = random.Random(12)
    b = PdfBuilder()
    # page 1: title page + preface (front matter, not in the outline)
    b.line("Cell Biology Course Reader", size=26, font=BOLD)
    b.space(20)
    b.paragraph("PREFACE " + para(rng, 3))
    # page 2: chapter 1 heading, text, then section 1.1 mid-page
    b.new_page()
    b.line("Chapter 1 Cells", size=20, font=BOLD)
    b.space(6)
    b.paragraph("CH1INTRO " + para(rng, 3))
    b.space(20)
    sec_y = b.y
    b.line("1.1 Membranes", size=14, font=BOLD)
    b.paragraph("MEMBRANES " + para(rng, 3))
    b.line("1.1.1 Lipids", size=12, font=BOLD)
    b.paragraph("LIPIDS " + para(rng, 3))
    # page 3: section 1.2 has no visible heading (split at the page start)
    b.new_page()
    b.paragraph("TRANSPORT " + para(rng, 3))
    # page 4: chapter 2
    b.new_page()
    b.line("CHAPTER 2: ENERGY", size=20, font=BOLD)
    b.paragraph("ENERGY " + para(rng, 3))
    toc = [
        [1, "Chapter 1: Cells", 2],
        [2, "1.1 Membranes", 2, {"kind": pymupdf.LINK_GOTO, "page": 1, "to": pymupdf.Point(72, sec_y)}],
        [3, "1.1.1 Lipids", 2],
        [2, "1.2 Transport", 3],
        [1, "Chapter 2 Energy", 4],
    ]
    return b.save(tmp_path / "outline.pdf", toc=toc, metadata={"title": "Cell Biology Reader", "author": "Dr. Ada Lin"})


def test_sections_from_outline(tmp_path):
    doc = load(_outline_book(tmp_path))
    assert doc.title == "Cell Biology Reader"
    assert doc.author == "Dr. Ada Lin"
    assert [(s.title, s.level) for s in doc.sections] == [
        ("Cell Biology Course Reader", 1),  # front matter
        ("Chapter 1: Cells", 1),
        ("1.1 Membranes", 2),
        ("1.2 Transport", 2),
        ("Chapter 2 Energy", 1),
    ]
    front, ch1, s11, s12, ch2 = doc.sections
    assert any("PREFACE" in bl.text for bl in front.blocks)
    # chapter 1 begins with its own heading and ends before the mid-page section 1.1
    assert ch1.blocks[0].kind == BlockKind.HEADING and ch1.blocks[0].text == "Chapter 1 Cells"
    assert [bl.text.split()[0] for bl in ch1.blocks[1:]] == ["CH1INTRO"]
    assert s11.blocks[0].text == "1.1 Membranes" and s11.blocks[0].level == 2
    # the level-3 outline entry stays a heading inside 1.1
    lipids = [bl for bl in s11.blocks if bl.text == "1.1.1 Lipids"]
    assert lipids and lipids[0].kind == BlockKind.HEADING and lipids[0].level == 3
    assert s12.blocks[0].text.startswith("TRANSPORT") and s12.page_start == 3
    assert ch2.blocks[0].text == "CHAPTER 2: ENERGY" and ch2.blocks[0].level == 1
    assert [s.id for s in doc.sections] == ["s0", "s1", "s2", "s3", "s4"]


def test_page_range_sections_for_long_unstructured_pdf(tmp_path):
    rng = random.Random(13)
    b = PdfBuilder()
    for i in range(25):
        if i:
            b.new_page()
        b.paragraph(para(rng, 3, f"Page{i + 1}text"))
    doc = load(b.save(tmp_path / "plain.pdf"))
    assert [s.title for s in doc.sections] == ["Pages 1-10", "Pages 11-20", "Pages 21-25"]
    assert doc.sections[1].page_start == 11 and doc.sections[1].page_end == 20


def test_short_pdf_without_headings_is_one_section(tmp_path):
    rng = random.Random(14)
    b = PdfBuilder()
    b.paragraph(para(rng, 3))
    doc = load_pdf(b.save(tmp_path / "x.pdf"), source_name="Week_3 reading.pdf")
    assert len(doc.sections) == 1
    assert doc.title == "Week 3 reading"
    assert doc.sections[0].title == "Week 3 reading"


# --------------------------------------------------------------------------- block kinds


def test_captions_footnotes_lists_and_equations(tmp_path):
    rng = random.Random(15)
    b = PdfBuilder()
    b.paragraph("Table 2.1 shows how the concentration changes. " + para(rng, 1))
    b.image(120)
    b.lines(["Figure 1.2 The structure of the plasma membrane."], size=9)
    b.paragraph("The main types of transport are:")
    b.bullet("Passive transport, which needs no energy.")
    b.bullet("Active transport, which uses ATP to move molecules against their gradient and is therefore costly.")
    b.bullet("Bulk transport.", marker="▪")
    b.paragraph("Steps:", space_after=2)
    b.lines(["1. The ligand binds.", "2. The receptor changes shape.", "a) the channel opens", "(ii) ions flow"])
    # A full-width line that wraps right before "2." (a number, not a list item).
    first = "Cells count the numbered steps in order, as shown in the laboratory protocol"
    while text_width(first + " of the", REGULAR, 11) < 0.95 * (b.x1 - b.x0):
        first += " of the"
    b.lines([first + " of", "2. The next sentence continues normally."])
    html = '<p style="font-family: serif; font-size: 12pt">∑ x<sub>i</sub> = ∫ f(t) dt + √(α + β)</p>'
    rect = pymupdf.Rect(b.x0, b.y, b.x1, b.y + 30)
    b.page.insert_htmlbox(rect, html)
    b.y += 34
    b.paragraph("Footnote markers appear in text.^3")
    b.footnote("3 This is the footnote text, which explains a detail of the experiment.")
    doc = load(b.save(tmp_path / "kinds.pdf"))
    by_text = {bl.text: bl for bl in blocks(doc)}
    assert by_text["Figure 1.2 The structure of the plasma membrane."].kind == BlockKind.CAPTION
    assert next(bl for bl in blocks(doc) if bl.text.startswith("Table 2.1")).kind == BlockKind.PARAGRAPH
    items = [bl.text for bl in blocks(doc) if bl.kind == BlockKind.LIST_ITEM]
    assert items == [
        "Passive transport, which needs no energy.",
        "Active transport, which uses ATP to move molecules against their gradient and is therefore costly.",
        "Bulk transport.",
        "1. The ligand binds.",
        "2. The receptor changes shape.",
        "a) the channel opens",
        "(ii) ions flow",
    ]
    wrapped = next(bl for bl in blocks(doc) if bl.text.startswith("Cells count the"))
    assert wrapped.kind == BlockKind.PARAGRAPH
    assert wrapped.text.endswith(" of 2. The next sentence continues normally.")
    eq = [bl for bl in blocks(doc) if bl.kind == BlockKind.EQUATION]
    assert len(eq) == 1 and "∫" in eq[0].text
    fns = [bl for bl in blocks(doc) if bl.kind == BlockKind.FOOTNOTE]
    assert [f.text for f in fns] == ["3 This is the footnote text, which explains a detail of the experiment."]
    assert "Footnote markers appear in text." in by_text


def test_code_block_keeps_line_breaks(tmp_path):
    b = PdfBuilder()
    b.paragraph("The following program prints a greeting:")
    b.lines(["def greet(name):", "    print('Hello', name)", "", "greet('Ada')"], font=MONO, size=10)
    b.paragraph("It runs on any computer.")
    doc = load(b.save(tmp_path / "code.pdf"))
    code = [bl for bl in blocks(doc) if bl.kind == BlockKind.CODE]
    assert len(code) == 1
    assert code[0].text.splitlines()[:2] == ["def greet(name):", "    print('Hello', name)"]
    assert "greet('Ada')" in code[0].text


def test_ruled_table_extracted_once(tmp_path):
    rng = random.Random(16)
    b = PdfBuilder()
    b.paragraph(para(rng, 2))
    b.lines(["Table 4.1 Organelles and their functions"], size=9, space_after=4)
    rows = [
        ["Organelle", "Function", "Found in"],
        ["Nucleus", "Stores DNA", "Eukaryotes"],
        ["Ribosome", "Makes proteins", "All cells"],
        ["Mitochondrion", "Releases energy", "Eukaryotes"],
    ]
    b.table(rows)
    b.paragraph("AFTER the table, the text continues.")
    doc = load(b.save(tmp_path / "table.pdf"))
    tables = [bl for bl in blocks(doc) if bl.kind == BlockKind.TABLE]
    assert len(tables) == 1
    assert tables[0].rows == rows
    assert "Organelle | Function | Found in" in tables[0].text
    assert "\n" not in tables[0].text
    others = " ".join(bl.text for bl in blocks(doc) if bl.kind != BlockKind.TABLE)
    assert "Ribosome" not in others and "Makes proteins" not in others
    order = [bl.kind for bl in blocks(doc)]
    assert order.index(BlockKind.CAPTION) < order.index(BlockKind.TABLE)
    assert texts(doc)[-1] == "AFTER the table, the text continues."


# --------------------------------------------------------------------------- scanned pages / OCR


def _ocr_ready() -> bool:
    try:
        from athena.ingest import ocr
    except ImportError:
        return False
    return ocr.ocr_available()[0]


@pytest.mark.skipif(not _ocr_ready(), reason="Tesseract OCR (and athena.ingest.ocr) not available")
def test_scanned_page_read_with_real_ocr(tmp_path):
    rng = random.Random(17)
    b = PdfBuilder()
    b.paragraph("TEXTLAYER " + para(rng, 2))
    b.save(tmp_path / "text.pdf")
    scan = scanned_pdf(
        tmp_path / "scan.pdf",
        [
            [
                "Photosynthesis converts light energy",
                "into chemical energy stored in sugar.",
                "",
                "Plants release oxygen as a by-product.",
            ]
        ],
    )
    merged = pymupdf.open(tmp_path / "text.pdf")
    merged.insert_pdf(pymupdf.open(scan))
    merged.save(tmp_path / "mixed.pdf")
    doc = load(tmp_path / "mixed.pdf")
    joined = full_text(doc).lower()
    assert "photosynthesis converts light energy into chemical energy" in joined
    assert "oxygen" in joined
    ocr_blocks = [bl for bl in blocks(doc) if "photosynthesis" in bl.text.lower()]
    assert ocr_blocks[0].page == 2
    assert any("Page 2 had no text layer and was read with OCR" in w for w in doc.warnings)


def test_scanned_pages_with_mocked_ocr(tmp_path, monkeypatch):
    calls = []

    def fake_ocr(data: bytes, *, lang: str = "eng") -> str:
        calls.append(data[:8])
        if len(calls) == 1:
            return "Mitosis has four\nphases in total.\n\n• Prophase\n• Metaphase\n\n12\n"
        return "the cell then divides into\ntwo daughter cells.\n\nCytokinesis follows.\n\n13\n"

    monkeypatch.setattr(pdf_mod, "_ocr_function", lambda: (fake_ocr, ""))
    scan = scanned_pdf(tmp_path / "scan.pdf", [["anything"], ["more"]])
    doc = load(scan)
    assert len(calls) == 2 and all(c.startswith(b"\x89PNG") for c in calls)
    ts = texts(doc)
    assert ts[0] == "Mitosis has four phases in total."
    assert [bl.text for bl in blocks(doc) if bl.kind == BlockKind.LIST_ITEM] == [
        "Prophase",
        "Metaphase the cell then divides into two daughter cells.",  # continues across the page break
    ]
    assert ts[-1] == "Cytokinesis follows."
    assert "12" not in ts and "13" not in ts
    assert all(bl.page in (1, 2) for bl in blocks(doc))
    assert any("Pages 1-2 had no text layer" in w for w in doc.warnings)


def test_scanned_pdf_without_ocr_returns_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_mod, "_ocr_function", lambda: (None, "Tesseract OCR isn't installed."))
    scan = scanned_pdf(tmp_path / "scan.pdf", [["one"], ["two"], ["three"]])
    doc = load(scan)
    assert isinstance(doc, Document)
    assert doc.page_count == 3
    assert len(doc.sections) == 1 and doc.sections[0].blocks == []
    assert any(
        w.startswith("Pages 1-3 are scanned images with no text. Install Tesseract OCR") for w in doc.warnings
    ), doc.warnings


def test_ocr_module_missing_is_treated_as_unavailable(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 1 and fromlist and "ocr" in fromlist:
            raise ImportError("no ocr module")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    fn, reason = pdf_mod._ocr_function()
    assert fn is None and reason


def test_ocr_turned_off(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_mod, "_ocr_function", lambda: pytest.fail("OCR must not run"))
    scan = scanned_pdf(tmp_path / "scan.pdf", [["one"]])
    doc = load(scan, ocr=False)
    assert "Page 1 is a scanned image with no text. Turn on OCR to have it read aloud." in doc.warnings


def test_ocr_failure_on_a_page_is_a_warning(tmp_path, monkeypatch):
    def broken(data: bytes, *, lang: str = "eng") -> str:
        raise IngestError("Tesseract crashed")

    monkeypatch.setattr(pdf_mod, "_ocr_function", lambda: (broken, ""))
    doc = load(scanned_pdf(tmp_path / "scan.pdf", [["one"]]))
    assert "OCR failed on page 1, so it was skipped." in doc.warnings


# --------------------------------------------------------------------------- errors & metadata


def test_encrypted_pdf_needs_password(tmp_path):
    b = PdfBuilder()
    b.paragraph("Secret notes.")
    path = b.save(tmp_path / "locked.pdf", encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="owner")
    with pytest.raises(IngestError, match="password-protected"):
        load(path)


def test_owner_password_only_pdf_opens(tmp_path):
    b = PdfBuilder()
    b.paragraph("Restricted but readable notes.")
    path = b.save(
        tmp_path / "restricted.pdf",
        encryption=pymupdf.PDF_ENCRYPT_AES_128,
        owner_pw="owner",
        user_pw="",
        permissions=pymupdf.PDF_PERM_PRINT,
    )
    doc = load(path)
    assert texts(doc) == ["Restricted but readable notes."]


@pytest.mark.parametrize(
    "content",
    [b"", b"this is not a pdf at all", b"%PDF-1.7\n" + bytes(range(256)) * 4, b"\x89PNG\r\n\x1a\n" + b"\0" * 64],
    ids=["empty", "text", "garbage", "png"],
)
def test_corrupt_or_fake_pdf_gives_friendly_error(tmp_path, content):
    path = tmp_path / "broken.pdf"
    path.write_bytes(content)
    with pytest.raises(IngestError) as info:
        load(path)
    assert "broken.pdf" in str(info.value)


def test_truncated_pdf_does_not_crash(tmp_path):
    rng = random.Random(18)
    b = PdfBuilder()
    for _ in range(12):
        b.paragraph(para(rng, 4))
    path = b.save(tmp_path / "full.pdf")
    data = path.read_bytes()
    cut = tmp_path / "cut.pdf"
    cut.write_bytes(data[: len(data) * 2 // 3])
    try:
        doc = load(cut)
    except IngestError:
        return
    assert isinstance(doc, Document)


def test_metadata_title_fallbacks(tmp_path):
    rng = random.Random(19)

    def build(name: str, metadata: dict, heading: bool) -> Document:
        b = PdfBuilder()
        if heading:
            b.line("Genetics Lecture Notes", size=20, font=BOLD)
            b.space(6)
        b.paragraph(para(rng, 2))
        return load_pdf(b.save(tmp_path / name, metadata=metadata), source_name=name)

    assert build("a.pdf", {"title": "Microsoft Word - lecture7.docx", "author": "Administrator"}, True).title == (
        "Genetics Lecture Notes"
    )
    doc = build("b.pdf", {"title": "", "author": "Prof. Kim"}, True)
    assert doc.title == "Genetics Lecture Notes" and doc.author == "Prof. Kim"
    doc = build("Lecture_7 slides.pdf", {"title": "untitled", "author": "Administrator"}, False)
    assert doc.title == "Lecture 7 slides" and doc.author is None
    assert build("d.pdf", {"title": "Genes and Heredity"}, True).title == "Genes and Heredity"


def test_ids_stable_pages_set_and_dispatch(tmp_path):
    rng = random.Random(20)
    b = PdfBuilder()
    b.heading("Topic", size=18)
    for _ in range(8):
        b.paragraph(para(rng, 4))
    path = b.save(tmp_path / "stable.pdf")
    d1 = load(path)
    d2 = load_document(path)
    assert d1.id == d2.id == doc_id_for_file(path)
    assert d1.model_dump() == d2.model_dump()
    assert d1.source_type == "pdf" and d1.source_name == "stable.pdf"
    assert all(bl.page is not None and 1 <= bl.page <= d1.page_count for bl in blocks(d1))


def test_progress_reports_pages(tmp_path):
    rng = random.Random(21)
    b = PdfBuilder()
    for _ in range(10):
        b.paragraph(para(rng, 6))
    path = b.save(tmp_path / "prog.pdf")
    events: list[tuple[float, str]] = []
    doc = load(path, progress=lambda f, m: events.append((f, m)))
    n = doc.page_count
    assert ("Reading page 1 of %d" % n) in [m for _, m in events]
    assert ("Reading page %d of %d" % (n, n)) in [m for _, m in events]
    fractions = [f for f, _ in events]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0 and 0 <= fractions[0] < 0.1


def test_blank_pages_are_ignored_quietly(tmp_path):
    b = PdfBuilder()
    b.paragraph("Only page with text.")
    b.new_page()
    doc = load(b.save(tmp_path / "blank.pdf"))
    assert texts(doc) == ["Only page with text."]
    assert doc.warnings == []


# --------------------------------------------------------------------------- performance


@pytest.mark.slow
def test_performance_300_pages(tmp_path):
    rng = random.Random(22)
    b = PdfBuilder()
    for ch in range(1, 11):
        if ch > 1:
            b.new_page()
        b.heading(f"Chapter {ch} Topic {ch}", size=20)
        while len(b.doc) < ch * 30:
            b.paragraph(para(rng, 5), first_indent=14)
    b.decorate(header=lambda n: "Introductory Biology", footer=lambda n: str(n))
    path = b.save(tmp_path / "big.pdf")
    assert pymupdf.open(path).page_count >= 300
    start = time.perf_counter()
    doc = load(path)
    elapsed = time.perf_counter() - start
    print(f"\n300-page PDF loaded in {elapsed:.2f}s")
    assert elapsed < 30
    assert [s.title for s in doc.sections if not s.title.endswith(")")][:3] == [
        "Chapter 1 Topic 1",
        "Chapter 2 Topic 2",
        "Chapter 3 Topic 3",
    ]
    assert "Introductory Biology" not in full_text(doc)


# --------------------------------------------------------------------------- integration


def test_textbook_like_integration(tmp_path):
    """A small 'textbook chapter' exercising most features at once."""
    rng = random.Random(23)
    b = PdfBuilder(margin=60)
    b.line("Chapter 5", size=14, font=BOLD)
    b.line("The Working Cell", size=26, font=BOLD)
    b.space(10)
    b.paragraph("OPENING " + para(rng, 3), size=12)
    b.set_columns(2, gutter=20)
    b.heading("5.1 Membrane Structure", size=13)
    for i in range(4):
        b.paragraph(para(rng, 4, f"A{i}") + " Proteins^4 float in the bilayer.", first_indent=12)
    b.bullet("Phospholipids form the bilayer.")
    b.bullet("Proteins act as channels and pumps that move ions across.")
    b.image(80)
    b.lines(["Figure 5.1 A fluid mosaic membrane."], size=8.5)
    b.heading("5.2 Passive Transport", size=13)
    for i in range(10):
        b.paragraph(para(rng, 5, f"B{i}"), first_indent=12, justify=True)
    b.table([["Process", "Energy"], ["Diffusion", "None"], ["Pumping", "ATP"]])
    b.heading("5.3 Active Transport", size=13)
    page_53 = len(b.doc)
    for i in range(8):
        b.paragraph(para(rng, 5, f"C{i}"), first_indent=12)
    b.decorate(
        header=lambda n: "Chapter 5 The Working Cell" if n % 2 else "Campbell Essentials",
        footer=lambda n: str(n + 96),
    )
    toc = [
        [1, "Chapter 5 The Working Cell", 1],
        [2, "5.1 Membrane Structure", 1],
        [2, "5.2 Passive Transport", 1],
        [2, "5.3 Active Transport", page_53],
    ]
    path = b.save(tmp_path / "textbook.pdf", toc=toc, metadata={"title": "Microsoft Word - ch5.docx"})
    doc = load(path)
    assert doc.page_count >= 3
    assert doc.title == "Chapter 5: The Working Cell"
    titles = [s.title for s in doc.sections]
    assert titles[:4] == [
        "Chapter 5 The Working Cell",
        "5.1 Membrane Structure",
        "5.2 Passive Transport",
        "5.3 Active Transport",
    ]
    all_text = full_text(doc)
    assert "Campbell Essentials" not in all_text
    for n in range(97, 97 + doc.page_count):
        assert str(n) not in texts(doc)
    # paragraphs in order, each exactly once, never split
    tags = [t.split()[0] for t in texts(doc) if t[:1] in "ABC" and t[1:2].isdigit()]
    assert tags == [f"A{i}" for i in range(4)] + [f"B{i}" for i in range(10)] + [f"C{i}" for i in range(8)]
    for t in texts(doc):
        if t[:1] in "ABC" and t[1:2].isdigit():
            assert t.endswith("."), t
    assert "Proteins float in the bilayer." in all_text
    kinds = {bl.kind for bl in blocks(doc)}
    assert {BlockKind.HEADING, BlockKind.PARAGRAPH, BlockKind.LIST_ITEM, BlockKind.CAPTION, BlockKind.TABLE} <= kinds
    table = next(bl for bl in blocks(doc) if bl.kind == BlockKind.TABLE)
    assert table.rows == [["Process", "Energy"], ["Diffusion", "None"], ["Pumping", "ATP"]]
    assert all(bl.page for bl in blocks(doc))
    for s in doc.sections[1:4]:
        assert s.blocks[0].kind == BlockKind.HEADING


# --------------------------------------------------------------------------- harder layouts


def test_running_header_with_changing_section_name_removed(tmp_path):
    rng = random.Random(30)
    names = ["CELL STRUCTURE", "MEMBRANES", "TRANSPORT", "ENERGY", "ENZYMES", "SIGNALS"]
    b = PdfBuilder()
    for i in range(12):
        if i:
            b.new_page()
        for _ in range(3):
            b.paragraph(para(rng, 4), first_indent=14)
    b.decorate(header=lambda n: names[(n - 1) // 2], footer=lambda n: str(n))
    doc = load(b.save(tmp_path / "slots.pdf"))
    all_text = full_text(doc)
    for name in names:
        assert name not in all_text


def test_table_of_contents_entries(tmp_path):
    b = PdfBuilder()
    b.heading("Contents", size=16)
    entries = [("1 Cells", 1), ("1.1 Membranes", 4), ("2 Energy", 12), ("Index", 140)]
    for name, page in entries:
        dots = "." * int((b.x1 - b.x0 - text_width(f"{name} {page}", REGULAR, 11)) / text_width(".", REGULAR, 11) - 2)
        b.line(f"{name} {dots} {page}")
    doc = load(b.save(tmp_path / "toc.pdf"))
    items = [bl.text for bl in blocks(doc) if bl.kind == BlockKind.LIST_ITEM]
    assert items == ["1 Cells … 1", "1.1 Membranes … 4", "2 Energy … 12", "Index … 140"]


def test_code_with_comments_stays_one_block(tmp_path):
    b = PdfBuilder()
    b.paragraph("The loop below adds the numbers:")
    rows = [
        ("total = 0", "# start at zero"),
        ("for n in numbers:", "# visit each item"),
        ("    total += n", "# add it"),
    ]
    for code, comment in rows:
        y = b.y + 10
        b.text((b.x0, y), code, font=MONO, size=10)
        b.text((b.x0 + 260, y), comment, font=MONO, size=10)
        b.y += 13
    b.space(8)
    b.paragraph("After the loop, total holds the sum.")
    doc = load(b.save(tmp_path / "code2.pdf"))
    code = [bl for bl in blocks(doc) if bl.kind == BlockKind.CODE]
    assert len(code) == 1
    lines = code[0].text.splitlines()
    assert lines[0].startswith("total = 0") and lines[0].endswith("# start at zero")
    assert lines[2].startswith("    total += n")


def test_vector_diagram_labels_are_dropped(tmp_path):
    rng = random.Random(31)
    b = PdfBuilder()
    b.paragraph("BEFORE " + para(rng, 2))
    top = b.y
    for k in range(16):  # a diagram: shapes joined by connector lines
        x = b.x0 + 20 + (k % 8) * 50
        y = top + 20 + (k // 8) * 60
        b.page.draw_circle((x, y), 12, color=(0, 0, 0), fill=(0.8, 0.9, 1.0))
        if k % 8:
            b.page.draw_line((x - 38, y), (x - 12, y))
        if k < 8:
            b.page.draw_line((x, y + 12), (x, y + 48))
    labels = [("Nucleus", b.x0 + 10, top + 52), ("Ribosome", b.x0 + 160, top + 52), ("ER", b.x0 + 300, top + 110)]
    for label, x, y in labels:
        b.text((x, y), label, size=8)
    b.y = top + 150
    b.lines(["Figure 2.4 Parts of an animal cell."], size=9)
    b.paragraph("AFTER " + para(rng, 2))
    doc = load(b.save(tmp_path / "diagram.pdf"))
    ts = texts(doc)
    assert not any(t in ("Nucleus", "Ribosome", "ER") for t in ts)
    assert any(t.startswith("BEFORE") for t in ts) and any(t.startswith("AFTER") for t in ts)
    assert "Figure 2.4 Parts of an animal cell." in ts


def test_booktabs_table_without_vertical_lines(tmp_path):
    b = PdfBuilder()
    b.paragraph("The table compares two kinds of transport.")
    x0, x1 = b.x0, b.x1
    y = b.y + 4
    cols = [x0 + 4, x0 + 150, x0 + 300]
    rows = [
        ["Process", "Energy", "Example"],
        ["Diffusion", "None", "Oxygen entering"],
        ["Pumping", "ATP", "Sodium leaving"],
    ]
    b.page.draw_line((x0, y), (x1, y), width=1)
    yy = y + 14
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            b.text((cols[c], yy), cell, font=BOLD if r == 0 else REGULAR, size=10)
        if r == 0:
            b.page.draw_line((x0, yy + 5), (x1, yy + 5), width=0.5)
        yy += 16
    b.page.draw_line((x0, yy - 6), (x1, yy - 6), width=1)
    b.y = yy + 6
    b.paragraph("AFTER the table.")
    doc = load(b.save(tmp_path / "booktabs.pdf"))
    tables = [bl for bl in blocks(doc) if bl.kind == BlockKind.TABLE]
    assert len(tables) == 1 and tables[0].rows == rows
    assert not any("Oxygen" in t for t in texts(doc) if not t.startswith("Process"))


def test_small_print_at_column_bottom_does_not_split_paragraph(tmp_path):
    rng = random.Random(32)
    b = PdfBuilder()
    b.set_columns(2)
    while b.y < b.bottom - 90:
        b.paragraph(para(rng, 2), first_indent=12)
    long = para(rng, 8, "SPLIT")
    b.paragraph(long, first_indent=12)
    assert b.col == 1  # the paragraph flowed into the right column
    left = b.columns[0][0]
    b.text((left, b.bottom + 16), "Copyright 2026 by the authors. Classroom use is permitted.", size=7)
    b.paragraph("NEXT " + para(rng, 2), first_indent=12)
    doc = load(b.save(tmp_path / "note.pdf"))
    ts = texts(doc)
    split = [t for t in ts if "SPLIT" in t]
    assert len(split) == 1 and split[0].endswith(long.split()[-1])
    i = ts.index(split[0])
    assert ts[i + 1].startswith("Copyright 2026") and ts[i + 2].startswith("NEXT")


def test_centered_multiline_title_is_one_heading(tmp_path):
    rng = random.Random(33)
    b = PdfBuilder()
    for line in ["An Overview of the Technology", "Acceptance Model in Education"]:
        w = text_width(line, BOLD, 18)
        b.line(line, size=18, font=BOLD, x=(b.width - w) / 2)
    b.space(10)
    b.paragraph(para(rng, 3))
    doc = load(b.save(tmp_path / "centered.pdf"))
    heads = [bl.text for bl in blocks(doc) if bl.kind == BlockKind.HEADING]
    assert heads == ["An Overview of the Technology Acceptance Model in Education"]
    assert doc.title == heads[0]


def test_title_page_lines_fold_into_first_section(tmp_path):
    rng = random.Random(34)
    b = PdfBuilder()
    b.line("Lecture Notes on Genetics", size=20, font=BOLD)
    b.line("Dr. Maria Gomez", size=13)
    b.line("September 2026", size=13)
    b.space(10)
    for name in ["Inheritance", "Mutations", "Gene Expression"]:
        b.heading(name, size=14)
        b.paragraph(para(rng, 3))
    doc = load(b.save(tmp_path / "titlepage.pdf"))
    titles = ["Lecture Notes on Genetics", "Inheritance", "Mutations", "Gene Expression"]
    assert [s.title for s in doc.sections] == titles
    assert [bl.text for bl in doc.sections[0].blocks] == ["Lecture Notes on Genetics", "Dr. Maria Gomez September 2026"]


def test_single_or_junk_outline_entries_fall_back_to_headings(tmp_path):
    rng = random.Random(35)
    b = PdfBuilder()
    for name in ["Photosynthesis", "Respiration"]:
        b.heading(name, size=16)
        for _ in range(3):
            b.paragraph(para(rng, 3))
    path = b.save(tmp_path / "one.pdf", toc=[[1, "lecture_notes_final.docx", 1], [1, "Untitled", 1]])
    doc = load(path)
    assert [s.title for s in doc.sections] == ["Photosynthesis", "Respiration"]


def test_rotated_page_and_sideways_text(tmp_path):
    rng = random.Random(36)
    b = PdfBuilder()
    for i in range(3):
        b.paragraph(f"ROT{i} " + para(rng, 2))
    b.flush()
    b.page.set_rotation(90)
    b.new_page()
    words = [f"SIDE{i} " + sentence(rng, 6) for i in range(4)]
    b.flush()
    for i, w in enumerate(words):
        b.page.insert_text((80 + 15 * i, 700), w, fontsize=11, rotate=90)
    doc = load(b.save(tmp_path / "rot.pdf"))
    ts = texts(doc)
    assert [t.split()[0] for t in ts if t.startswith("ROT")] == ["ROT0", "ROT1", "ROT2"]
    # text running up the page is read line by line, in order
    assert [w for t in ts for w in t.split() if w.startswith("SIDE")] == ["SIDE0", "SIDE1", "SIDE2", "SIDE3"]
    assert all(bl.page == 2 for bl in blocks(doc) if "SIDE" in bl.text)


def _garbled_pdf(path: Path, pages: int = 2) -> Path:
    """Pages whose text layer is made of undecodable codes (like PDFs with broken fonts)."""
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        for i in range(12):
            page.insert_text((72, 80 + 18 * i), "\x02\x03\x04\x05 \x06\x07\x08 \x0e\x0f\x10\x11 \x12\x13", fontsize=11)
    doc.save(path)
    doc.close()
    return path


def test_undecodable_text_layer_is_read_with_ocr(tmp_path, monkeypatch):
    def fake_ocr(data: bytes, *, lang: str = "eng") -> str:
        return "Proteins fold into shapes that\ndetermine their function.\n"

    monkeypatch.setattr(pdf_mod, "_ocr_function", lambda: (fake_ocr, ""))
    doc = load(_garbled_pdf(tmp_path / "garbled.pdf"))
    assert texts(doc) == ["Proteins fold into shapes that determine their function."] * 2
    assert doc.warnings == [
        "The text of pages 1-2 couldn't be decoded, so they were read with OCR instead. "
        "Recognition mistakes are possible."
    ]


def test_undecodable_text_layer_without_ocr_warns(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_mod, "_ocr_function", lambda: (None, "Tesseract OCR isn't installed."))
    doc = load(_garbled_pdf(tmp_path / "garbled.pdf", pages=1))
    assert len(doc.warnings) == 1
    assert doc.warnings[0].startswith("The text of page 1 is stored in a way that can't be turned back into letters")
    assert doc.warnings[0].endswith("Install Tesseract OCR to have it read aloud.")
    assert all(not any(c < " " for c in t) for t in texts(doc))


def test_ocr_running_headers_and_headings(tmp_path, monkeypatch):
    pages_text = [
        "Biology Reader 2026\n\nIntroduction\n\n" + "Cells are the units of life and they divide often. " * 5
        + "Around every cell\n\n7",
        "Biology Reader 2026\n\n" + "the membrane controls what enters the cell and what leaves it. " * 4 + "\n\n8",
        "Biology Reader 2026\n\nTransport Across Membranes\n\n"
        + "Molecules move by diffusion and by active pumps. " * 5
        + "\n\n9",
    ]
    calls = iter(pages_text)
    monkeypatch.setattr(pdf_mod, "_ocr_function", lambda: (lambda data, lang="eng": next(calls), ""))
    doc = load(scanned_pdf(tmp_path / "scan3.pdf", [["a"], ["b"], ["c"]]))
    kinds = [(bl.kind, bl.text[:20]) for bl in blocks(doc)]
    assert not any("Biology Reader" in t for _, t in kinds)
    assert [t for k, t in kinds if k == BlockKind.HEADING] == ["Introduction", "Transport Across Mem"]
    # the paragraph on page 1 continues on page 2 (lowercase start)
    assert sum(1 for k, _ in kinds if k == BlockKind.PARAGRAPH) == 2
    assert [s.title for s in doc.sections] == ["Introduction", "Transport Across Membranes"]


# --------------------------------------------------------------------------- line-level helpers


def _span(text, size=10.0, font="Helvetica", x0=0.0, flags=0, oy=10.0):
    width = len(text) * size * 0.5
    return (text, size, flags, font, x0, oy - size, x0 + width, oy + 2, oy)


def _line(*spans) -> pdf_mod._Line:
    ln = pdf_mod._Line(page=0, x0=0, y0=0, x1=100, y1=12, spans=list(spans))
    pdf_mod._finish_line(ln)
    return ln


def test_line_text_handles_icons_bullets_and_superscripts():
    icon = _line(
        _span("\uf05a", size=24, font="FontAwesome5Free-Solid"),  # an icon glyph (private use area)
        _span("Note the version.", x0=30),
        _span(" Check it.", x0=120),
    )
    assert icon.text == "Note the version. Check it." and not icon.symbol_lead
    bullet = _line(_span("l", font="Wingdings"), _span("First point", x0=15))
    assert bullet.text == "First point" and bullet.symbol_lead
    assert pdf_mod._list_marker(bullet) == "bullet"
    greek = _line(_span("α", font="Symbol"), _span(" = 0.05", x0=6))
    assert greek.text == "α = 0.05" and not greek.symbol_lead
    sup = _line(_span("energy"), _span("12", size=6, x0=31, flags=1, oy=6), _span(" is stored", x0=38))
    assert sup.text == "energy is stored"
    exp = _line(_span("10"), _span("-3", size=6, x0=11, flags=1, oy=6), _span(" m", x0=18))
    assert exp.text == "10⁻³ m"
    lead = _line(_span("3", size=6, flags=1, oy=6), _span("See chapter 4.", x0=5, size=8))
    assert lead.text == "3 See chapter 4." and lead.lead_marker
    ctrl = _line(_span("Ha\x03bi\x00bi"))
    assert ctrl.text == "Habibi"


def test_page_number_and_margin_key_patterns():
    for t in ["12", "xii", "Page 12", "page 3 of 10", "- 7 -", "12 | Chapter 3 Cells", "Chapter 3 | 12", "iv", "[12]"]:
        assert pdf_mod._is_page_number(t), t
    for t in ["2026 Edition", "Chapter 3", "12 apples", "Figure 3", "mix"]:
        assert not pdf_mod._is_page_number(t), t
    assert pdf_mod._margin_key("Chapter 3 · Cells · 47") == pdf_mod._margin_key("Chapter 4 · Cells · 52")
    assert pdf_mod._margin_key("xii | Preface") == pdf_mod._margin_key("xiv | Preface")


def test_outline_entries_past_the_text_and_mentions_in_paragraphs(tmp_path):
    rng = random.Random(37)
    b = PdfBuilder()
    b.heading("Getting Started", size=16)
    b.paragraph("This part explains Advanced Topics only briefly; " + para(rng, 2))
    b.new_page()
    b.paragraph("Advanced Topics")
    b.paragraph(para(rng, 3))
    b.new_page()  # an empty last page that the outline points at
    toc = [[1, "Getting Started", 1], [1, "Advanced Topics", 2], [1, "Appendix", 3]]
    doc = load(b.save(tmp_path / "outline2.pdf", toc=toc))
    assert [s.title for s in doc.sections] == ["Getting Started", "Advanced Topics"]
    first, second = doc.sections
    assert first.blocks[1].kind == BlockKind.PARAGRAPH  # mentions the title, isn't it
    assert second.blocks[0].kind == BlockKind.HEADING and second.blocks[0].text == "Advanced Topics"


def test_numbered_item_at_top_of_page_is_not_a_page_number(tmp_path):
    rng = random.Random(38)
    b = PdfBuilder()
    for page in range(4):
        if page:
            b.new_page()
        b.text((b.x0, b.y + 11), f"{page + 1}.", size=11)  # a numbered step starts each page
        b.line(f"Step {page + 1} " + sentence(rng, 6), x=b.x0 + 18)
        b.space(14)
        b.table([["Stage", "Place"], ["Glycolysis", "Cytoplasm"]])
        b.paragraph(para(rng, 2))
    b.decorate(footer=lambda n: f"{n}.")
    doc = load(b.save(tmp_path / "steps.pdf"))
    items = [bl.text for bl in blocks(doc) if bl.kind == BlockKind.LIST_ITEM]
    assert [t.split()[:2] for t in items] == [[f"{n}.", "Step"] for n in range(1, 5)]
    assert not any(bl.text.strip(".").isdigit() for bl in blocks(doc))


def test_line_break_hyphens_follow_the_books_own_spelling(tmp_path):
    b = PdfBuilder()
    b.lines(
        [
            "Students with high self-esteem and a co-op job were asked about their self-",
            "esteem, their infor-",
            "mation needs, the co-",
            "efficient of the test and how their co-",
            "op works. The half-",
            "life of a drug, the pre-",
            "Columbian era, multi-",
            "ple anti-",
            "bodies and the state-of-the-",
            "art.",
        ]
    )
    doc = load(b.save(tmp_path / "hyphens.pdf"))
    assert texts(doc) == [
        "Students with high self-esteem and a co-op job were asked about their self-esteem, their information "
        "needs, the coefficient of the test and how their co-op works. The half-life of a drug, the "
        "pre-Columbian era, multiple antibodies and the state-of-the-art."
    ]


def test_hyphenated_word_across_a_page_break(tmp_path):
    rng = random.Random(41)
    b = PdfBuilder()
    while b.y < b.bottom - 40:
        b.paragraph(para(rng, 2), first_indent=14)
    b.lines(["Building self-esteem takes practice, and low self-"], space_after=0)
    b.new_page()
    b.lines(["esteem is common in first-year students."])
    doc = load(b.save(tmp_path / "pagebreak.pdf"))
    joined = [t for t in texts(doc) if "low self" in t]
    assert len(joined) == 1 and joined[0].endswith("low self-esteem is common in first-year students.")


def test_repeated_copyright_notice_on_chapter_openers_is_dropped(tmp_path):
    # A multi-line notice in small print at the foot of every chapter opener (pages far apart),
    # as in many publishers' free textbook editions; a one-off note at a page foot stays.
    rng = random.Random(42)
    b = PdfBuilder()
    notice = [
        "This material is published by Example University Press as Cell Biology for Everyone by",
        "Ana Lima and Ben Okafor (2024). This version is free to view and download for personal use",
        "only. Not for re-distribution, re-sale, or use in derivative works. (c) the authors.",
    ]
    for chapter in range(1, 5):
        if chapter > 1:
            b.new_page()
        b.heading(f"Chapter {chapter} Topic {chapter}", size=18)
        for k, line in enumerate(notice):
            b.text((b.margin, b.bottom + 20 + 10 * k), line, size=8)
        for _ in range(24):
            b.paragraph(para(rng, 3), first_indent=14)
    b.text((b.margin, b.bottom + 20), "Figure credits: photographs by the Cell Imaging Library.", size=8)
    doc = load(b.save(tmp_path / "openers.pdf"))
    assert doc.page_count >= 12
    all_text = full_text(doc)
    assert "This material is published" not in all_text and "re-distribution" not in all_text
    assert "Figure credits: photographs by the Cell Imaging Library." in all_text
    assert [bl.text for bl in blocks(doc) if bl.kind == BlockKind.HEADING] == [
        f"Chapter {n} Topic {n}" for n in range(1, 5)
    ]


def test_control_codes_from_math_fonts_are_not_in_table_cells():
    rows = pdf_mod._clean_table([["Distribution", "Eta"], ["Bernoulli", "log\x00 phi \x01"], ["Gaussian", "mu\x02"]])
    assert rows == [["Distribution", "Eta"], ["Bernoulli", "log phi"], ["Gaussian", "mu"]]


def test_margin_notes_do_not_split_sentences(tmp_path):
    # Textbook layout: a narrower text block and key terms in small print in the outer
    # margin, beside the paragraph that introduces them (right margin on odd pages, left on even).
    rng = random.Random(43)
    b = PdfBuilder()
    for page in range(2):
        if page:
            b.new_page()
        x0, x1 = (72, 400) if page == 0 else (200, 540)
        b.columns = [(x0, x1)]
        note_x = 412 if page == 0 else 72
        for k in range(3):
            b.paragraph(para(rng, 2, f"P{page}{k}"), first_indent=14, space_after=4)
            top = b.y
            b.paragraph(f"M{page}{k} " + para(rng, 4), first_indent=14)
            b.text((note_x, top + 28), f"key term {page}{k}", size=8)
            b.text((note_x, top + 37), "in the margin", size=8)
    doc = load(b.save(tmp_path / "margins.pdf"))
    assert doc.page_count == 2
    ts = texts(doc)
    for page in range(2):
        for k in range(3):
            main = next(t for t in ts if t.startswith(f"M{page}{k} "))
            assert ts[ts.index(main) + 1] == f"key term {page}{k} in the margin"
            assert "key term" not in main and main.endswith(".")
    expected = [f"{c}{p}{k}" for p in range(2) for k in range(3) for c in "PM"]
    assert [t.split()[0] for t in ts if t[:1] in "PM"] == expected


def test_long_chapter_splits_at_its_outline_sections(tmp_path):
    # Part > chapter > section outline: a 50,000-character chapter used to become
    # "Chapter 2 (part 2)" pieces cut at an arbitrary paragraph.
    rng = random.Random(44)
    b = PdfBuilder()
    toc = [[1, "Part I Foundations", 1]]
    b.heading("Part I Foundations", size=24)
    b.new_page()
    toc.append([2, "Chapter 2 Membranes", 2])
    b.heading("Chapter 2 Membranes", size=20)
    b.paragraph(para(rng, 3, "INTRO"))
    for n in range(1, 4):
        if n > 1:
            b.new_page()
        toc.append([3, f"2.{n} Topic {n}", b.doc.page_count])
        b.heading(f"2.{n} Topic {n}", size=14)
        for _ in range(60):
            b.paragraph(para(rng, 3), first_indent=14)
    b.new_page()
    toc.append([2, "Chapter 3 Short", b.doc.page_count])
    b.heading("Chapter 3 Short", size=20)
    b.paragraph(para(rng, 3, "SHORT"))
    doc = load(b.save(tmp_path / "parts.pdf", toc=toc))
    chapter_chars = sum(len(bl.text) for s in doc.sections[1:5] for bl in s.blocks)
    assert chapter_chars > 40_000
    assert [(s.title, s.level) for s in doc.sections] == [
        ("Part I Foundations", 1),
        ("Chapter 2 Membranes", 2),
        ("2.1 Topic 1", 3),
        ("2.2 Topic 2", 3),
        ("2.3 Topic 3", 3),
        ("Chapter 3 Short", 2),
    ]
    assert doc.sections[2].blocks[0].text == "2.1 Topic 1"


def _schedule_pdf(path: Path, *, weeks: int = 14) -> Path:
    """A Word-style syllabus: a numbered list, then a class schedule laid out as a table
    without borders that runs onto the next page, then more text."""
    rng = random.Random(45)
    b = PdfBuilder()
    b.heading("Learning Outcomes", size=14)
    for n in range(1, 4):
        b._advance(b.body * 1.3)
        b.text((b.x0, b.y + b.body), f"{n}.", size=b.body)  # the number is its own text run, as Word writes it
        b.text((b.x0 + 18, b.y + b.body), f"Outcome {n}: explain cell transport in your own words.", size=b.body)
        b.y += b.body * 1.3 + 3
    b.heading("Weekly Schedule", size=14)
    b.paragraph("The schedule may change; check the course page every week. " + para(rng, 2))
    cols = (72, 160, 380)
    for cells in [("Week", "Topic", "Reading")] + [
        (str(w), f"Topic {w}: membranes, transport|and signaling part {w}", f"Chapter {w}, pages {20 * w}-{20 * w + 18}")
        for w in range(1, weeks + 1)
    ]:
        lines = [c.split("|") for c in cells]
        height = len(max(lines, key=len)) * 14 + 10
        if b.y + height > b.bottom:
            # the page ends in the middle of a row: its last line goes to the next page
            if len(lines[1]) > 1:
                for x, part in zip(cols, lines):
                    b.text((x, b.y + 11), part[0], size=11)
                b.new_page()
                b.text((cols[1], b.y + 11), lines[1][1], size=11)
                b.y += 24
                continue
            b.new_page()
        for x, part in zip(cols, lines):
            for k, text in enumerate(part):
                b.text((x, b.y + 11 + 14 * k), text, size=11)
        b.y += height
    b.space(12)
    b.heading("Policies", size=14)
    b.paragraph("LATE work loses ten percent per day. " + para(rng, 3))
    return b.save(path)


def test_table_without_borders_is_read_row_by_row(tmp_path):
    doc = load(_schedule_pdf(tmp_path / "syllabus.pdf"))
    tables = [bl for bl in blocks(doc) if bl.kind == BlockKind.TABLE]
    assert len(tables) == 1  # one table, although it runs onto a second page
    rows = tables[0].rows
    assert rows[0] == ["Week", "Topic", "Reading"]
    assert rows[1:] == [
        [str(w), f"Topic {w}: membranes, transport and signaling part {w}", f"Chapter {w}, pages {20 * w}-{20 * w + 18}"]
        for w in range(1, 15)
    ]
    ts = texts(doc)
    assert [t for t in ts if t.startswith(("1.", "2.", "3."))] == [
        f"{n}. Outcome {n}: explain cell transport in your own words." for n in range(1, 4)
    ]
    assert not any(t.startswith(("Topic", "Chapter")) for t in ts)  # no cells left over as paragraphs
    assert any(t.startswith("LATE work") for t in ts)


def test_aligned_text_that_is_not_a_table_stays_text():
    def line(text: str, x0: float, y0: float, size: float = 10) -> pdf_mod._Line:
        ln = pdf_mod._Line(page=0, x0=x0, y0=y0, x1=x0 + 5.2 * len(text), y1=y0 + size, spans=[], size=size)
        ln.text, ln.chars, ln.baseline = text, len(text), y0 + size
        return ln

    def page(rows: list[tuple[str, ...]], xs: tuple[float, ...]) -> pdf_mod._Page:
        pg = pdf_mod._Page(index=0, width=612, height=792)
        pg.lines.append(line("A full-width sentence of ordinary text that runs across the whole page.", 72, 60))
        for r, row in enumerate(rows):
            pg.lines.extend(line(text, x, 100 + 14 * r) for text, x in zip(row, xs) if text)
        return pg

    equations = [(f"f{k}(x) = a{k} + b{k}x", f"({k}.{k + 1})") for k in range(5)]
    index = [(f"gradient {k}, {100 + k}", f"matrix {k}, {200 + k}, {210 + k}") for k in range(8)]
    ticks = [(f"{v:.2f}", "") for v in (0.2, 0.15, 0.1, 0.05, 0.0)]
    ticks[0] = ("0.20", "Density of the mixture model")
    for rows, xs in [(equations, (150, 480)), (index, (72, 320)), (ticks, (72, 300))]:
        assert pdf_mod._borderless_tables(page(rows, xs)) == []
    real = [("Week", "Topic"), ("1", "Cells and membranes"), ("2", "Transport"), ("3", "Energy")]
    assert pdf_mod._borderless_tables(page(real, (72, 200)))[0].rows == [list(r) for r in real]


def test_kerned_table_cells_keep_their_letters_in_order():
    # PyMuPDF sorts a cell's letters by position: a kerned "i" with no width came out "Actvi e".
    words = [(79.2, 400, 124.1, 410, "Active", 3, 0, 0), (300, 400, 320, 410, "Yes", 4, 0, 0)]
    cells = [[(72, 395, 263, 415), (263, 395, 455, 415)]]
    assert pdf_mod._cell_texts([["Actvi e", "Yes"]], cells, words) == [["Active", "Yes"]]
    # different letters: PyMuPDF's text is kept (a word merely touching the cell is not pulled in)
    assert pdf_mod._cell_texts([["Act", "Yes"]], cells, words) == [["Act", "Yes"]]


def test_heading_right_above_a_ruled_table_is_not_its_header_row(tmp_path):
    b = PdfBuilder()
    b.heading("Grading", size=14, after=0)
    b.table([["Component", "Weight"], ["Midterm exam", "25%"], ["Final exam", "30%"], ["Labs", "45%"]])
    b.paragraph("All grades are posted online.")
    doc = load(b.save(tmp_path / "grading.pdf"))
    assert [(bl.kind, bl.text) for bl in blocks(doc)][:1] == [(BlockKind.HEADING, "Grading")]
    table = next(bl for bl in blocks(doc) if bl.kind == BlockKind.TABLE)
    assert table.rows[0] == ["Component", "Weight"]


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        ("(818) 708-8131 TELEX #298949 LINN UR", True),  # a letterhead line: real, though mostly digits
        ("Yes / No", True),
        ("Total: 25 pts", True),
        ("12.5 13.0 14.2", True),
        ("Cit", False),  # pieces of a logo
        ("((11////44 44 jf Ad A ddds", False),
        ("~ |", False),
    ],
)
def test_ocr_noise_filter_keeps_short_real_lines(text, kept):
    assert pdf_mod._plausible_ocr_text(text) is kept
