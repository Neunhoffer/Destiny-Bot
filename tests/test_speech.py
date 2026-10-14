"""Tests for the speech layer: normalization, sentence splitting and reading plans."""

from __future__ import annotations

import itertools
import random
import re
import shutil
import subprocess
import time

import pytest

from destiny_bot.models import Block, BlockKind, ReadingOptions, Section
from destiny_bot.speech import (
    chunk_span,
    normalize_for_speech,
    plan_section,
    section_plain_text,
    split_sentences,
    verbalize_math,
)
from destiny_bot.speech import plan as plan_module

K = BlockKind


def norm(text: str, **options) -> str:
    return normalize_for_speech(text, ReadingOptions(**options))


# --------------------------------------------------------------------------- abbreviations

ABBREVIATIONS = [
    ("Many organisms, e.g. bacteria, divide.", "Many organisms, for example bacteria, divide."),
    ("E.g., cells divide.", "For example, cells divide."),
    ("Small things (e.g. atoms) are hard to see.", "Small things (for example atoms) are hard to see."),
    ("The nucleus, i.e. the control center, is large.", "The nucleus, that is the control center, is large."),
    ("I.e., it is small.", "That is, it is small."),
    ("Apples, oranges, etc.", "Apples, oranges, et cetera."),
    ("Apples, oranges, etc. are fruits.", "Apples, oranges, et cetera are fruits."),
    ("Cf. Chapter 3.", "Compare Chapter 3."),
    ("Cats vs. dogs", "Cats versus dogs"),
    ("cats vs dogs", "cats versus dogs"),
    ("Brown v. Board of Education", "Brown versus Board of Education"),
    ("two chambers, viz. the atria", "two chambers, namely the atria"),
    ("approx. 5 cells", "approximately 5 cells"),
    ("It took approx. ten years.", "It took approximately ten years."),
    ("Built c. 1500.", "Built circa fifteen hundred."),
    ("Built ca. 1600.", "Built circa sixteen hundred."),
    ("Ca. 1600 it was built.", "Circa sixteen hundred it was built."),
    ("Smith et al. showed this.", "Smith and colleagues showed this."),
    ("This was shown by Smith et al.", "This was shown by Smith and colleagues."),
    ("See Fig. 3.", "See Figure 3."),
    ("See Figs. 3 and 4.", "See Figures 3 and 4."),
    ("See FIG. 2.", "See Figure 2."),
    ("as in Fig. 2.1", "as in Figure 2 point 1"),
    ("From Eq. 2", "From Equation 2"),
    ("Eqs. (3) and (4)", "Equations (3) and (4)"),
    ("Eqn. 4 holds", "Equation 4 holds"),
    ("Ch. 4 is long", "Chapter 4 is long"),
    ("Chap. 12 ends", "Chapter 12 ends"),
    ("the ch. 3 notes", "the chapter 3 notes"),
    ("Sec. 2.3 explains", "Section 2 point 3 explains"),
    ("see § 4", "see Section 4"),
    ("see §§ 4–6", "see Sections 4 to 6"),
    ("See p. 12.", "See page 12."),
    ("See pp. 10–12.", "See pages 10 to 12."),
    ("pp. 45-67", "pages 45 to 67"),
    ("Vol. 3", "Volume 3"),
    ("Vol. II", "Volume 2"),
    ("No. 5 is right", "Number 5 is right"),
    ("item no. 7", "item number 7"),
    ("Nos. 3 and 4", "Numbers 3 and 4"),
    ("the 2nd ed. of the book", "the 2nd edition of the book"),
    ("the revised ed. is out", "the revised edition is out"),
    ("Smith (Ed.)", "Smith (editor)"),
    ("Smith and Lee (Eds.)", "Smith and Lee (editors)"),
    ("coffee w/ milk", "coffee with milk"),
    ("tea w/o sugar", "tea without sugar"),
    ("tea w/out sugar", "tea without sugar"),
    ("at 5 p.m.", "at 5 P M."),
    ("at 9 a.m. sharp", "at 9 A M sharp"),
    ("at 5pm", "at 5 P M"),
    ("at 5:30 pm", "at 5 30 P M"),
    ("R&D", "R and D"),
    ("salt & pepper", "salt and pepper"),
    ("see refs. 3 and 4", "see references 3 and 4"),
    # Titles: voices pause at the period ("Doctor. Smith") or spell them out ("L T Dan").
    ("Mr. Smith met Dr. Jones.", "Mister Smith met Doctor Jones."),
    ("Mrs. Lee and Ms. Park", "Missus Lee and Miz Park"),
    ("St. Louis is a city.", "Saint Louis is a city."),
    ("Prof. Lee spoke.", "Professor Lee spoke."),
    ("Gen. Grant and Lt. Dan met Sen. Smith.", "General Grant and Lieutenant Dan met Senator Smith."),
    ("Pres. Lincoln and Gov. Brown", "President Lincoln and Governor Brown"),
    ("Mt. Everest and Ft. Sumter", "Mount Everest and Fort Sumter"),
    ("Martin Luther King Jr. spoke.", "Martin Luther King Junior spoke."),
    ("It was written by John Smith Jr.", "It was written by John Smith Junior."),
    ("Ken Griffey Sr., his father", "Ken Griffey Senior, his father"),
    ("Gen. 1:1 says", "Gen. 1:1 says"),
    ("on Jan. 15, 2020", "on January 15th, twenty twenty"),
    ("by Oct. 1929", "by October nineteen twenty-nine"),
    ("See Thm. 3 and Def. 2.1.", "See Theorem 3 and Definition 2 point 1."),
    ("by Prop. 4 and Cor. 1", "by Proposition 4 and Corollary 1"),
    ("Art. I, § 8, cl. 3", "Article 1, Section 8, clause 3"),
    ("call ext. 22", "call extension 22"),
    ("the avg. score", "the average score"),
    ("use === not ==", "use triple equals not double equals"),
    ("if (a && b)", "if (a and b)"),
]

ABBREVIATION_NEGATIVES = [
    "He lives on Main St. Springfield.",
    "Pick a fig. Then eat.",
    "No. That is wrong.",
    "He was v. good.",
    "I am at home.",
    "The AM radio",
    "approx 5",
    "Ed. Smith said hello.",
    "Fig 3",
    "The pp file",
    "Ivy grew on the wall.",
    "etcetera",
    "web/app",
]


@pytest.mark.parametrize("text, expected", ABBREVIATIONS)
def test_abbreviations(text, expected):
    assert norm(text) == expected


@pytest.mark.parametrize("text", ABBREVIATION_NEGATIVES)
def test_abbreviation_negatives(text):
    assert norm(text) == text


def test_abbreviations_can_be_turned_off():
    text = "See Fig. 3, e.g. this & that, pp. 4"
    assert norm(text, expand_abbreviations=False) == "See Fig. 3, e.g. this & that, pp. 4"
    assert norm(text) == "See Figure 3, for example this and that, pages 4"


# --------------------------------------------------------------------------- citations

CITATIONS_REMOVED = [
    ("Cells divide (Smith, 2019).", "Cells divide."),
    ("Cells divide (Smith 2019).", "Cells divide."),
    ("Cells divide (Smith & Jones, 2019; Lee et al., 2020a).", "Cells divide."),
    ("Cells divide (see Smith, 2019, p. 4).", "Cells divide."),
    ("Cells divide (e.g., Smith, 2019).", "Cells divide."),
    ("Cells divide (Smith, 2019, pp. 10-12).", "Cells divide."),
    ("Cells divide (Smith, 2019, 2020).", "Cells divide."),
    ("Cells divide (Smith, 2019a, b).", "Cells divide."),
    ("Cells divide (Smith, n.d.).", "Cells divide."),
    ("Cells divide (Smith, in press).", "Cells divide."),
    ("Cells divide (Smith 2019: 45).", "Cells divide."),
    ("Cells (Smith, 2019) divide.", "Cells divide."),
    ("Results were clear (van der Waals, 1873).", "Results were clear."),
    ("Results (O'Brien & McDonald, 2018b) were clear.", "Results were clear."),
    ("It spread (World Health Organization, 2020).", "It spread."),
    ("It spread (WHO, 2020).", "It spread."),
    ("It spread (Smith, Jones, and Lee, 2019).", "It spread."),
    ("It spread (Smith, 2019; see also Lee, 2020, as cited in Park, 2021).", "It spread."),
    ("It spread (Kant, 1781/1998).", "It spread."),
    ("It spread [Smith, 2019].", "It spread."),
    ("Cells divide [12].", "Cells divide."),
    ("Cells divide [3-5, 9].", "Cells divide."),
    ("Cells divide [3–5].", "Cells divide."),
    ("Cells divide [1,2].", "Cells divide."),
    ("Cells divide.[12] Then", "Cells divide. Then"),
    ("Earlier studies[12] agree.", "Earlier studies agree."),
    ("Studies [citation needed] show it.", "Studies show it."),
    # A bracket that is part of the sentence becomes "reference N" instead of vanishing.
    ("As shown in [3], cells divide.", "As shown in reference 3, cells divide."),
    ("See [3, 5] for details.", "See references 3 and 5 for details."),
    ("as described in [3–5]", "as described in references 3 to 5"),
    ("see reference [7]", "see reference 7"),
    # Narrative citations keep the name and drop the year.
    ("Smith (2019) showed that cells divide.", "Smith showed that cells divide."),
    ("Smith et al. (2019) showed it.", "Smith and colleagues showed it."),
    ("According to Smith and Jones (2019), cells divide.", "According to Smith and Jones, cells divide."),
    ("As Smith (2019) noted, it works.", "As Smith noted, it works."),
    ("Recently, Lee (2020a) found more.", "Recently, Lee found more."),
    ("Smith (2019, p. 4) argues this.", "Smith argues this."),
    ("Smith (in press) argues this.", "Smith argues this."),
    ("Darwin's (1859) theory changed biology.", "Darwin's theory changed biology."),
    ("Smith & Jones (2019) disagree.", "Smith and Jones disagree."),
]

CITATIONS_KEPT = [
    ("The firm (founded in 1990) grew.", "The firm (founded in nineteen ninety) grew."),
    ("As shown (see Figure 2), it grew.", "As shown (see Figure 2), it grew."),
    ("The sample (n = 30) was small.", "The sample (n equals 30) was small."),
    ("It was significant (p < .05).", "It was significant (p is less than 0 point 0 5)."),
    ("The war (1990-1995) ended.", "The war (nineteen ninety to nineteen ninety-five) ended."),
    ("Read this (Chapter 3).", "Read this (Chapter 3)."),
    ("The Clean Air Act (1970) requires it.", "The Clean Air Act (nineteen seventy) requires it."),
    ("the Smith (2019) study", "the Smith (twenty nineteen) study"),
    ("We met (Summer 2019) twice.", "We met (Summer twenty nineteen) twice."),
    ("It happened (March 2020).", "It happened (March twenty twenty)."),
    ("It happened (Fall 2019).", "It happened (Fall 2019)."),
    ("Data (Census 2010) show it.", "Data (Census 2010) show it."),
    ("It is in (Room 2019) now.", "It is in (Room 2019) now."),
    ("Austen's Pride and Prejudice (1813) is a novel.", "Austen's Pride and Prejudice (eighteen thirteen) is a novel."),
    ("Hurricane Katrina (2005) hit.", "Hurricane Katrina (two thousand five) hit."),
    ("Jane Austen's Emma (1815) is long.", "Jane Austen's Emma (eighteen fifteen) is long."),
    ("Then Darwin's On the Origin (1859) appeared.", "Then Darwin's On the Origin (eighteen fifty-nine) appeared."),
    ("The years (1990, 1995) were good.", "The years (nineteen ninety, nineteen ninety-five) were good."),
    ("In 2019 (the year of the study) it rained.", "In twenty nineteen (the year of the study) it rained."),
    ("The ratio (2:1) holds.", "The ratio (2 to 1) holds."),
    ("Smith (Figure 2) shows it.", "Smith (Figure 2) shows it."),
    ("The Great Exhibition (London, 1851) was huge.", "The Great Exhibition (London, eighteen fifty-one) was huge."),
    ("An event (in 2019) occurred.", "An event (in twenty nineteen) occurred."),
    ("The cell (a unit of life) divides.", "The cell (a unit of life) divides."),
    ("Values lie in [0, 1].", "Values lie in [0, 1]."),
    ("Take x in [1, 10].", "Take x in [1, 10]."),
    ("The interval [2, 5] is closed.", "The interval [2, 5] is closed."),
    ("Read a[1] first.", "Read a[1] first."),
    ("He wrote [sic] it.", "He wrote [sic] it."),
    ("Insert [your name] here.", "Insert [your name] here."),
]


@pytest.mark.parametrize("text, expected", CITATIONS_REMOVED)
def test_citations_removed(text, expected):
    assert norm(text) == expected


@pytest.mark.parametrize("text, expected", CITATIONS_KEPT)
def test_parentheticals_that_are_not_citations_are_kept(text, expected):
    assert norm(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Cells divide (Smith et al., 2019).", "Cells divide (Smith and colleagues, twenty nineteen)."),
        ("Smith (2019) found it [12].", "Smith (twenty nineteen) found it [12]."),
        ("Cells divide [3-5].", "Cells divide [3 to 5]."),
    ],
)
def test_citations_keep_mode(text, expected):
    assert norm(text, citations="keep") == expected


@pytest.mark.parametrize("text", ["(Smith, 2019)", "[12]", "[3-5, 9].", " (Smith et al., 2019; Lee, 2020). "])
def test_citation_only_text_is_unspeakable(text):
    assert norm(text) == ""


# --------------------------------------------------------------------------- links

LINKS = [
    ("Visit https://www.example.com/path/to-page?x=1#top.", {
        "full": "Visit example dot com slash path slash to dash page.",
        "domain": "Visit a link to example dot com.",
        "skip": "Visit."}),
    ("See (https://example.org/a_b).", {
        "full": "See (example dot org slash a underscore b).",
        "domain": "See (a link to example dot org).",
        "skip": "See."}),
    ("Go to example.com/docs, then click.", {
        "full": "Go to example dot com slash docs, then click.",
        "domain": "Go to a link to example dot com, then click.",
        "skip": "Go to, then click."}),
    ("At www.nasa.gov!", {"full": "At nasa dot gov!", "domain": "At a link to nasa dot gov!", "skip": "At!"}),
    ("Read https://en.wikipedia.org/wiki/Cell_(biology).", {
        "full": "Read en dot wikipedia dot org slash wiki slash Cell underscore (biology).",
        "domain": "Read a link to en dot wikipedia dot org.",
        "skip": "Read."}),
    ("Email jane.doe@uni.edu for help.", {
        "full": "Email jane dot doe at uni dot edu for help.",
        "domain": "Email jane dot doe at uni dot edu for help.",
        "skip": "Email for help."}),
    ("See doi.org/10.1038/nature12373.", {
        "full": "See doi dot org slash 10 dot 1038 slash nature12373.",
        "domain": "See a link to doi dot org.",
        "skip": "See."}),
]


@pytest.mark.parametrize("mode", ["full", "domain", "skip"])
@pytest.mark.parametrize("text, expected", LINKS)
def test_links(text, expected, mode):
    assert norm(text, urls=mode) == expected[mode]


@pytest.mark.parametrize(
    "text",
    ["Node.js is fast.", "Use e.g. and i.e. here.", "The U.S. is big.", "It ends.In the next", "3.14 is pi"],
)
def test_things_that_are_not_links(text):
    assert norm(text, urls="skip") == norm(text, urls="domain") == norm(text, urls="full")
    assert "link" not in norm(text)


# --------------------------------------------------------------------------- symbols and units

SYMBOLS = [
    ("50% of cells", "50 percent of cells"),
    ("a 5‰ rise", "a 5 per mille rise"),
    ("±2", "plus or minus 2"),
    ("3 × 4", "3 times 4"),
    ("a 2×2 grid", "a 2 by 2 grid"),
    ("6 ÷ 2", "6 divided by 2"),
    ("x = 5", "x equals 5"),
    ("y ≈ 3", "y is approximately equal to 3"),
    ("costs ≈ 5 dollars", "costs approximately 5 dollars"),
    ("DNA (≈ 3 billion pairs)", "DNA (approximately 3 billion pairs)"),
    ("(≤ 5 mm)", "(at most 5 millimeters)"),
    ("a ≠ b", "a is not equal to b"),
    ("x ≤ 5", "x is less than or equal to 5"),
    ("x ≥ 5", "x is greater than or equal to 5"),
    ("≥18 years old", "at least 18 years old"),
    ("p < 0.05", "p is less than 0 point 0 5"),
    ("3<5", "3 is less than 5"),
    ("<5% of cases", "less than 5 percent of cases"),
    ("5 > 3", "5 is greater than 3"),
    ("File > Open > Save", "File Open Save"),
    ("A → B", "A leads to B"),
    ("A ← B", "A comes from B"),
    ("A ↔ B", "A corresponds to B"),
    ("A ⇌ B", "A is in equilibrium with B"),
    ("A -> B", "A leads to B"),
    ("tends to ∞", "tends to infinity"),
    ("√x", "the square root of x"),
    ("2√3", "2 times the square root of 3"),
    ("∑ x", "sum of x"),
    ("∫ f", "integral of f"),
    ("∂f", "partial f"),
    ("Δx", "delta x"),
    ("ΔG is negative", "delta G is negative"),
    ("x − y", "x minus y"),
    ("−5", "-5"),
    ("a·b", "a times b"),
    ("Home · About", "Home, About"),
    ("5 * 3", "5 times 3"),
    ("x - y", "x minus y"),
    ("x ~ y", "x is similar to y"),
]

UNITS = [
    ("37°C", "37 degrees Celsius"),
    ("98.6 °F", "98 point 6 degrees Fahrenheit"),
    ("1 °C warmer", "1 degree Celsius warmer"),
    ("a 45° angle", "a 45 degree angle"),
    ("-5 °C", "-5 degrees Celsius"),
    ("300 K", "300 kelvin"),
    ("a 5 µm cell", "a 5 micrometer cell"),
    ("a 5 μm cell", "a 5 micrometer cell"),
    ("10 mm", "10 millimeters"),
    ("3 km", "3 kilometers"),
    ("200 mg", "200 milligrams"),
    ("2 kg", "2 kilograms"),
    ("1 kg", "1 kilogram"),
    ("5 mL", "5 milliliters"),
    ("5mL", "5 milliliters"),
    ("100 nm", "100 nanometers"),
    ("9.8 m/s²", "9 point 8 meters per second squared"),
    ("5 m²", "5 square meters"),
    ("3 cm³", "3 cubic centimeters"),
    ("2 mg/mL", "2 milligrams per milliliter"),
    ("60 km/h", "60 kilometers per hour"),
    ("0.5 M NaCl", "0 point 5 molar N A C L"),
    ("10 µL", "10 microliters"),
    ("1.5 Å", "1 point 5 angstroms"),
    ("bond length in Å", "bond length in angstroms"),
    ("measured in µm", "measured in micrometers"),
    ("cells per mm³", "cells per cubic millimeter"),
    ("copies/mL", "copies per milliliter"),
    ("30 min", "30 minutes"),
    ("5 Ω resistor", "5 ohms resistor"),
    ("60 Hz", "60 hertz"),
    ("120 mmHg", "120 millimeters of mercury"),
    ("3 × 10⁸ m/s", "3 times 10 to the power of 8 meters per second"),
    ("6.02 × 10²³ mol⁻¹", "6 point 0 2 times 10 to the power of 23 per mole"),
]

UNIT_NEGATIVES = [
    "a 10K race",
    "5 men came",
    "in 5 minutes",
    "5G networks",
    "at 10 pm",
]


@pytest.mark.parametrize("text, expected", SYMBOLS)
def test_symbols(text, expected):
    assert norm(text) == expected


@pytest.mark.parametrize("text, expected", UNITS)
def test_units(text, expected):
    assert norm(text) == expected


@pytest.mark.parametrize("text", UNIT_NEGATIVES)
def test_unit_negatives(text):
    assert norm(text) == text


SCRIPTS_GREEK_FRACTIONS = [
    ("x²", "x squared"),
    ("x³", "x cubed"),
    ("x² + y² = r²", "x squared plus y squared equals r squared"),
    ("z⁴", "z to the power of 4"),
    ("10⁻³", "10 to the power of minus 3"),
    ("eˣ", "e to the power of x"),
    ("(x+1)²", "(x plus 1) squared"),
    ("H₂O", "H 2 O"),
    ("CO₂ levels", "C O 2 levels"),
    ("C₆H₁₂O₆", "C 6 H 12 O 6"),
    ("Ca²⁺ ions", "C A 2 plus ions"),
    ("Na⁺", "N A plus"),
    ("Cl⁻", "C L minus"),
    ("x₁ and x₂", "x 1 and x 2"),
    ("the 1ˢᵗ one", "the 1st one"),
    ("Cells¹ divide.", "Cells divide."),
    ("They divide.² Then", "They divide. Then"),
    ("α, β, γ and Ω", "alpha, beta, gamma and omega"),
    ("β-carotene", "beta-carotene"),
    ("2π radians", "2 pi radians"),
    ("αβ", "alpha beta"),
    ("½ cup", "one half cup"),
    ("2½ cups", "2 and a half cups"),
    ("¾ done", "three quarters done"),
    ("⅓ of them", "one third of them"),
    ("3⁄16 inch", "three sixteenths inch"),
    ("x^2 in prose", "x squared in prose"),
    ("10^6 cells", "10 to the power of 6 cells"),
    ("The \\alpha helix", "The alpha helix"),
    ("so $x^2 + 1$ is positive", "so x squared plus 1 is positive"),
    ("costs $5 and $10", "costs 5 dollars and 10 dollars"),
]


@pytest.mark.parametrize("text, expected", SCRIPTS_GREEK_FRACTIONS)
def test_scripts_greek_fractions(text, expected):
    assert norm(text) == expected


PUNCTUATION = [
    ("The cell—the basic unit—divides.", "The cell, the basic unit, divides."),
    ("The cell -- the basic unit -- divides.", "The cell, the basic unit, divides."),
    ("The cell - the basic unit - divides.", "The cell, the basic unit, divides."),
    ("Use and/or here.", "Use and or here."),
    ("He/she said", "He or she said"),
    ("~5 cells", "about 5 cells"),
    ("#3 item", "number 3 item"),
    ("C# and F#", "C sharp and F sharp"),
    ("• First point", "First point"),
    ("▪ ◦ ■ ► ✓ Done", "Done"),
    ("Wait… what?", "Wait... what?"),
    ("Wait . . . what?", "Wait... what?"),
    ("Some **bold** and `code` text", "Some bold and code text"),
    ("A <b>bold</b> move", "A bold move"),
    ("Tom &amp; Jerry", "Tom and Jerry"),
    ("snake_case_name", "snake case name"),
    ("a | b | c", "a b c"),
    ("{braces}", "braces"),
    ("Copyright © 2020 Acme™", "Copyright copyright 2020 Acme"),
    ("Contact @handle", "Contact at handle"),
    ("p < .05", "p is less than 0 point 0 5"),
    ("  lots   of\n\nspace  ", "lots of space"),
]


@pytest.mark.parametrize("text, expected", PUNCTUATION)
def test_punctuation_and_leftovers(text, expected):
    assert norm(text) == expected


@pytest.mark.parametrize(
    "text", ["", "   ", "...", "—", "• ▪ ◦", "()", "[ ]", "* * *", "| |", "---", "#", "…", "※"]
)
def test_unspeakable_text_becomes_empty(text):
    assert norm(text) == ""


# --------------------------------------------------------------------------- ranges

RANGES = [
    ("From 1990–1995", "From nineteen ninety to nineteen ninety-five"),
    ("From 1990-1995", "From nineteen ninety to nineteen ninety-five"),
    ("He lived 1809–1882.", "He lived eighteen oh nine to eighteen eighty-two."),
    ("from 1990-95", "from nineteen ninety to 95"),
    ("Grades 9-12", "Grades 9 to 12"),
    ("ages 18-25", "ages 18 to 25"),
    ("pH 7-8", "pH 7 to 8"),
    ("10-12% of cases", "10 to 12 percent of cases"),
    ("0.5-1.5 mg", "0 point 5 to 1 point 5 milligrams"),
    ("9:00-10:30", "9 o'clock to 10 30"),
    ("1,000-2,000 people", "1000 to 2000 people"),
    ("between 5-10 cells", "between 5 and 10 cells"),
    ("Figures 2-3 show", "Figures 2 to 3 show"),
    ("the 1990s–2000s", "the nineteen nineties to two thousands"),
    ("pages 10 - 12", "pages 10 to 12"),
    ("eat 200-1500 calories", "eat 200 to 1500 calories"),
]

RANGE_NEGATIVES = [
    "Born 1990-05-01",
    "ISBN 978-3-16-148410-0",
    "The score was 3-2",
    "Figure 2-3 shows",
    "Table 4-1 lists",
    "Section 3-4 covers",
    "ISSN 1535-6078",
    "x-1 and 2-1",
    "1-2-3",
    "1990-2019-2020",
    "temp -5 and -10",
    "an x-ray of the x-axis",
    "Rule 12(b)(6) applies",
]


@pytest.mark.parametrize("text, expected", RANGES)
def test_number_ranges(text, expected):
    assert norm(text) == expected


@pytest.mark.parametrize("text", RANGE_NEGATIVES)
def test_number_range_negatives(text):
    assert norm(text) == text


def test_spaced_hyphen_in_equation_is_minus():
    assert norm("10 - 2 = 8") == "10 minus 2 equals 8"


# --------------------------------------------------------------------------- Roman numerals

ROMAN = [
    ("Chapter IV", "Chapter 4"),
    ("Part II begins", "Part 2 begins"),
    ("Section III", "Section 3"),
    ("Unit XII", "Unit 12"),
    ("Book IX", "Book 9"),
    ("Volume II", "Volume 2"),
    ("Appendix II", "Appendix 2"),
    ("Act I, Scene 2", "Act 1, Scene 2"),
    ("Phase III trials", "Phase 3 trials"),
    ("Type I diabetes", "Type 1 diabetes"),
    ("Type II error", "Type 2 error"),
    ("Class III", "Class 3"),
    ("Stage IV cancer", "Stage 4 cancer"),
    ("Title IX", "Title 9"),
    ("CHAPTER IV", "Chapter 4"),
    ("Chapters II and III", "Chapters 2 and 3"),
    ("Parts I–IV", "Parts 1 to 4"),
    ("In Chapter I we see", "In Chapter 1 we see"),
    ("Henry VIII", "Henry the Eighth"),
    ("Henry VIII's wives", "Henry the Eighth's wives"),
    ("Louis XIV ruled", "Louis the Fourteenth ruled"),
    ("Pope John Paul II", "Pope John Paul the Second"),
    ("Elizabeth I ruled", "Elizabeth the First ruled"),
    ("Elizabeth II", "Elizabeth the Second"),
    ("Ramesses II built it", "Ramesses the Second built it"),
    ("Pius XII", "Pius the Twelfth"),
    ("Charles XXI", "Charles the Twenty-first"),
    ("World War II", "World War 2"),
    ("World War I ended", "World War 1 ended"),
]

ROMAN_NEGATIVES = [
    "I think so.",
    "Then I left.",
    "V for Vendetta",
    "X marks the spot",
    "Part C is next",
    "Class C fire",
    "Appendix D lists",
    "Appendix I lists",
    "In this Chapter I argue that",
    "Rocky IV was a film",
    "Super Bowl LV",
    "Vitamin C is good",
    "the third act I saw",
    "chapter iv",
    "Part mix",
    "Mary I think is right",
]


@pytest.mark.parametrize("text, expected", ROMAN)
def test_roman_numerals(text, expected):
    assert norm(text) == expected


@pytest.mark.parametrize("text", ROMAN_NEGATIVES)
def test_roman_numeral_negatives(text):
    assert norm(text) == text


# --------------------------------------------------------------------------- ALL CAPS

CAPS = [
    ("CHAPTER ONE: THE CELL", "Chapter One: The Cell"),
    ("THE CELL", "The Cell"),
    ("DNA REPLICATION AND ATP", "DNA Replication And ATP"),
    ("This is VERY IMPORTANT.", "This is Very Important."),
    ("Read CHAPTER 4 NOW", "Read Chapter 4 Now"),
    ("DON'T STOP", "Don't Stop"),
    ("WELL-KNOWN FACT", "Well-Known Fact"),
    ("CO2 LEVELS RISE", "C O 2 Levels Rise"),
    ("A NEW HOPE", "A New Hope"),
    ("Do NOT submit LATE work.", "Do NOT submit LATE work."),
    ("ABSOLUTELY not", "Absolutely not"),
]

CAPS_NEGATIVES = [
    "DNA and RNA are made by cells.",
    "NASA and the USA",
    "the WHO said",
    "HTTP and LDL and NADPH",
    "UNESCO and UNICEF",
    "ATP",
    "Use the IT desk",
    "NADPH is made.",
]


@pytest.mark.parametrize("text, expected", CAPS)
def test_all_caps(text, expected):
    assert norm(text) == expected


@pytest.mark.parametrize("text", CAPS_NEGATIVES)
def test_all_caps_negatives(text):
    assert norm(text) == text


def test_normalize_never_returns_only_punctuation():
    rng = random.Random(7)
    pieces = list("()[].,—…•*#|~- !?;:▪")
    for _ in range(300):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 12)))
        out = norm(text)
        assert out == "" or re.search(r"[^\W_]", out), (text, out)


def test_normalize_default_options_and_whitespace():
    assert normalize_for_speech("Fig. 2 (Smith, 2019)") == "Figure 2"
    out = normalize_for_speech("a\tb\n\nc")
    assert out == "a b c"


@pytest.mark.parametrize(
    "text",
    [
        "Text (" + "see also e.g. " * 25 + "Smith 2019 z)",
        "Text (" + "Smith, " * 50 + "2019, p. " + "1, " * 30 + "x)",
        "Text (" + "Smith " * 60 + "2019 x)",
        "Text (" + "van de " * 60 + "Smith, 2019 q)",
        "Text (Smith, 2019" + ", 2020" * 50 + " q)",
        "Text " + "Smith and Jones, " * 20 + "(2019) found",
        "(" + "A " * 200 + ")",
        "[" + "1, " * 300 + "x]",
    ],
)
def test_near_miss_citations_are_fast(text):
    assert _timed(lambda: norm(text)) < 0.25


def test_normalize_survives_random_unicode():
    rng = random.Random(99)
    alphabet = list("abcXYZ 0123456789.,;:!?()[]{}<>=+-*/\\^_~#@&%$|'\"`") + list(
        "—–…•²³⁴⁻⁺₁₂αβΔΩµμ°×÷±≈≠≤≥→←↔√∑∫∂∞½¼¾§¶©®™†‡\n\t "
    ) + ["e.g.", "Fig.", "(Smith, 2019)", "[12]", "https://x.org/a", "Chapter IV", "CELLS", " Dr. ", "a@b.co"]
    for _ in range(2000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        out = normalize_for_speech(text, random_options(rng))
        assert out == out.strip()
        assert out == "" or re.search(r"[^\W_]", out), (text, out)
        assert not any(0xF0000 <= ord(ch) <= 0xFFFFD for ch in out), (text, out)


# --------------------------------------------------------------------------- math

MATH = [
    ("x^2", "x squared"),
    ("x^3", "x cubed"),
    ("x^{n+1}", "x to the power of n plus 1"),
    ("a_i", "a sub i"),
    ("a_{ij}", "a sub i j"),
    (r"\frac{a}{b}", "a over b"),
    (r"\sqrt{x}", "the square root of x"),
    (r"\sqrt[3]{x}", "the cube root of x"),
    (r"\alpha + \beta = \omega", "alpha plus beta equals omega"),
    (r"\sum_{i=1}^{n} i^2", "the sum from i equals 1 to n of i squared"),
    (r"\int_a^b f(x)\,dx", "the integral from a to b of f of x d x"),
    (r"\lim_{x\to 0} f(x)", "the limit as x approaches 0 of f of x"),
    (r"a \cdot b \times c", "a times b times c"),
    (r"x \le y", "x is less than or equal to y"),
    (r"x \ge y", "x is greater than or equal to y"),
    (r"x \neq y", "x is not equal to y"),
    (r"x \approx y", "x is approximately equal to y"),
    (r"\pm \infty", "plus or minus infinity"),
    (r"\frac{\partial f}{\partial x}", "partial f over partial x"),
    (r"\Delta x", "delta x"),
    (r"\sin \theta + \cos \theta", "sine of theta plus cosine of theta"),
    (r"\log x + \ln y", "log of x plus natural log of y"),
    (r"\log_2 8 = 3", "log base 2 of 8 equals 3"),
    (r"\sin^2 x + \cos^2 x = 1", "sine squared of x plus cosine squared of x equals 1"),
    ("f(x) = 2(x + 1)", "f of x equals 2 times the quantity x plus 1"),
    ("(a+b)^2", "the quantity a plus b, squared"),
    ("E = mc^2", "E equals m c squared"),
    ("y = mx + b", "y equals m x plus b"),
    ("density = mass/volume", "density equals mass over volume"),
    (r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
     "x equals the fraction minus b plus or minus the square root of b squared minus 4 a c, over 2 a"),
    ("x² + y² = r²", "x squared plus y squared equals r squared"),
    ("√(x+1)", "the square root of x plus 1"),
    ("e^{-x}", "e to the power of minus x"),
    ("x^-1", "x to the power of minus 1"),
    (r"\foo{x}", "foo x"),
    (r"v_{max} = 5", "v sub max equals 5"),
    ("f'(x) = 2x", "f prime of x equals 2 x"),
    ("n!", "n factorial"),
    ("|x|", "the absolute value of x"),
    (r"\vec{F} = m\vec{a}", "vector F equals m vector a"),
    (r"\binom{n}{k}", "n choose k"),
    (r"\mathbb{R}", "the real numbers"),
    (r"\text{speed} = \frac{d}{t}", "speed equals d over t"),
    (r"T = 25^\circ C", "T equals 25 degrees Celsius"),
    ("E = mc² (3.2)", "E equals m c squared"),
    ("$$x^2$$", "x squared"),
    (r"\[ x^2 \]", "x squared"),
    (r"\left( \frac{1}{2} \right)^n", "the quantity 1 over 2, to the power of n"),
    ("max(a, b)", "the maximum of a, b"),
    (r"x \in A \cup B", "x in A union B"),
    ("a/b", "a over b"),
    ("ΔG = ΔH - TΔS", "delta G equals delta H minus T delta S"),
    ("π r²", "pi r squared"),
]


@pytest.mark.parametrize("text, expected", MATH)
def test_verbalize_math(text, expected):
    assert verbalize_math(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "{}", "$$", r"\left( \right)", r"\label{eq:1}", "\\,\\;"])
def test_verbalize_math_empty(text):
    assert verbalize_math(text) == ""


def test_verbalize_math_survives_garbage():
    rng = random.Random(3)
    alphabet = list("x^_{}()[]\\+-=/|!'2a ")
    alphabet += [r"\frac", r"\sqrt", r"\sum", r"\int", r"\lim", r"\left", r"\right"]
    for _ in range(500):
        src = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 25)))
        out = verbalize_math(src)
        assert isinstance(out, str)
        assert out == out.strip()
        assert "\\" not in out


# --------------------------------------------------------------------------- how voices read numbers and names
# Each case below was heard (as phonemes, with espeak-ng, which both Kokoro and Piper use)
# being read wrongly before it was fixed.

DATES_AND_YEARS = [
    # espeak reads years as quantities: "nineteen hundred ninety", "one thousand sixty-six".
    ("In 1990, the wall fell.", "In nineteen ninety, the wall fell."),
    ("The Norman Conquest began in 1066.", "The Norman Conquest began in ten sixty-six."),
    ("since 1905", "since nineteen oh five"),
    ("by 2008", "by two thousand eight"),
    ("in 2019", "in twenty nineteen"),
    ("In 1900 and in 2000", "In nineteen hundred and in two thousand"),
    ("from 1914 to 1918", "from nineteen fourteen to nineteen eighteen"),
    ("World War II lasted 1939-1945.", "World War 2 lasted nineteen thirty-nine to nineteen forty-five."),
    ("in 1200 BC and AD 79", "in twelve hundred BC and AD 79"),
    ("On July 4, 1776, it began.", "On July 4th, seventeen seventy-six, it began."),
    ("June 6-8", "June 6th to 8th"),
    ("on 3 March 2021", "on 3rd March twenty twenty-one"),
    ("Austen's Pride and Prejudice (1813)", "Austen's Pride and Prejudice (eighteen thirteen)"),
    # Decades: espeak says "nineteen hundred ninety z".
    ("the 1990s", "the nineteen nineties"),
    ("the 1990's", "the nineteen nineties"),
    ("the 1800s", "the eighteen hundreds"),
    ("the 2000s and 2010s", "the two thousands and twenty tens"),
    ("the mid-1960s", "the mid-nineteen sixties"),
    ("the '90s", "the nineties"),
    ("in her 60s", "in her sixties"),
    # Not years.
    ("divide by 1024 bytes", "divide by 1024 bytes"),
    ("in 1024 steps", "in 1024 steps"),
    ("Option 2 May be chosen", "Option 2 May be chosen"),
    ("1990 people came", "1990 people came"),
    ("in 1500 mL of water", "in 1500 milliliters of water"),
    ("pages 1200 and 1300", "pages 1200 and 1300"),
    ("ISSN 1535-6078", "ISSN 1535-6078"),
]

MONEY = [
    # espeak: "dollar three point five zero", and Kokoro's phonemizer cuts it to "dollar three. fifty".
    ("It costs $3.50 now.", "It costs 3 dollars and 50 cents now."),
    ("It costs $5.", "It costs 5 dollars."),
    ("only $1", "only 1 dollar"),
    ("just $0.99", "just 99 cents"),
    ("GDP was $21.4 trillion.", "GDP was 21 point 4 trillion dollars."),
    ("a $5 tax", "a 5 dollar tax"),
    ("$1,000 or $2,500,000", "1000 dollars or 2500000 dollars"),
    ("$5-10 each", "5 to 10 dollars each"),
    ("$300K salary", "300 thousand dollars salary"),
    ("a $2M grant", "a 2 million dollar grant"),
    ("€20 and £5.50 and ¥100", "20 euros and 5 pounds and 50 pence and 100 yen"),
    ("lost -$5", "lost minus 5 dollars"),
    ("25¢", "25 cents"),
    ("the $PATH variable", "the $PATH variable"),
]

PHONES_TIMES_RATIOS = [
    ("Call (555) 123-4567 now.", "Call 5 5 5, 1 2 3, 4 5 6 7 now."),
    ("or 1-800-555-0199", "or 1, 8 0 0, 5 5 5, 0 1 9 9"),
    ("phone 555-1234", "phone 5 5 5, 1 2 3 4"),
    # espeak: "nine zero zero"; Kokoro also pauses at the colon ("three: thirty").
    ("Open 9:00-5:00 daily.", "Open 9 o'clock to 5 o'clock daily."),
    ("at 9:00 a.m. sharp", "at 9 A M sharp"),
    ("at 3:30 p.m.", "at 3 30 P M."),
    ("at 10:05", "at 10 oh 5"),
    ("at 14:00", "at 14 hundred"),
    ("John 3:16", "John 3 16"),
    ("a 3:1 ratio", "a 3 to 1 ratio"),
    ("mix them 1:10", "mix them 1 to 10"),
]

CHEMISTRY = [
    # espeak: "kah two plus", "nah C L", "fee two O three", "C H three coo".
    ("CO2 and H2O", "C O 2 and H 2 O"),
    ("Ca2+ ions", "C A 2 plus ions"),
    ("NaCl dissolves.", "N A C L dissolves."),
    ("Fe2O3 and Fe3+", "F E 2 O 3 and F E 3 plus"),
    ("CH3COOH is weak.", "C H 3 C O O H is weak."),
    ("2H2 + O2 makes 2H2O", "2 H 2 plus O 2 makes 2 H 2 O"),
    ("Na+ and K+ and Cl- ions", "N A plus and K plus and C L minus ions"),
    ("the SO42- ion", "the S O 4 2 minus ion"),
    ("PO4³⁻ and Ca²⁺", "P O 4 3 minus and C A 2 plus"),
    ("NaCl(aq) and N2(g)", "N A C L aqueous and N 2 gas"),
    ("KOH and NaOH", "K O H and N A O H"),
    ("moves 2 K+ in", "moves 2 K plus in"),  # potassium, not kelvin
    # Not formulas.
    ("HeLa cells", "HeLa cells"),
    ("SARS-CoV-2", "SARS-CoV-2"),
    ("SVN- or CVS-style", "SVN- or CVS-style"),
    ("He said No to the USA", "He said No to the USA"),
    ("CO2-rich air", "CO2-rich air"),
    ("pH and mRNA", "pH and mRNA"),
]

PROSE_MATH = [
    ("e^0.1 is close to 1.1", "e to the power of 0 point 1 is close to 1 point 1"),  # was "0 0 point 1"
    ("x^(n+1) grows", "x to the power of n plus 1 grows"),  # the power was lost
    ("(x+1)^2 here", "(x plus 1) squared here"),
    ("Then ((AB))^(-1) = B^(-1)A^(-1).", "Then ((AB)) to the power of minus 1 equals B to the power of minus 1 A to the power of minus 1."),
    ("A^(∗)A is normal", "A star A is normal"),  # was "A to the power of A"
    ("x^2/2! + x^3/3!", "x squared over 2 factorial + x cubed over 3 factorial"),
    ("Let y = 2x - 1.", "Let y equals 2x minus 1."),  # the minus became a comma
    ("Qd = 100 - 2P", "Qd equals 100 minus 2P"),
    ("KE = ½mv²", "KE equals one half mv squared"),  # "squared" was deleted as a footnote mark
    ("f(x) = x^2 and f'(3) = 6", "f of x equals x squared and f prime of 3 equals 6"),
    ("P(A ∩ B) = P(A)P(B)", "P(A intersection B) equals P of A times P of B"),
    ("P(A|B) is high", "P of A given B is high"),
    ("sin x and cos(x) and tan θ", "sine x and cosine(x) and tangent theta"),
    ("the sin of pride", "the sin of pride"),
    ("(f(b) - f(a))/(b - a)", "(f of b minus f of a) over (b minus a)"),
    ("the function 1/x and x/2", "the function 1 over x and x over 2"),
    ("1/2 cup and 3/4 of them", "one half cup and three quarters of them"),
    ("24/7 and 9/11 and 4/5 stars", "24/7 and 9/11 and 4/5 stars"),
    ("The sample mean x̄ estimates μ.", "The sample mean x bar estimates mu."),
    ("p̂ and ŷ", "p hat and y hat"),
    ("Tōkyō", "Tōkyō"),
    ("In △ABC the angle", "In triangle ABC the angle"),
    ("A definition ends here. △", "A definition ends here."),
]

UNITS_MORE = [
    ("g = 9.8 m/s^2", "g equals 9 point 8 meters per second squared"),
    ("a BMI of 25 kg/m2", "a BMI of 25 kilograms per meter squared"),
    ("50 m2 and 3 cm3", "50 square meters and 3 cubic centimeters"),
    ("[H+] = 10^-3 M", "[H plus] equals 10 to the power of minus 3 molar"),
    ("1.5 × 10⁻³ M", "1 point 5 times 10 to the power of minus 3 molar"),
    ("c = 3.00 × 10^8 m/s", "c equals 3 point 0 0 times 10 to the power of 8 meters per second"),
    ("100-120/min", "100 to 120 per minute"),
    ("take 2/d", "take 2 per day"),
    ("3.5-5.0 mEq/L", "3 point 5 to 5 point 0 milliequivalents per liter"),
    ("a 2.0 kg ball and a 9 V battery", "a 2 point 0 kilogram ball and a 9 volt battery"),
    ("a 250-mL flask", "a 250-milliliter flask"),
    ("after 3.0 s and 2 h", "after 3 point 0 seconds and 2 hours"),
    ("the 1s orbital", "the 1s orbital"),
    ("BP 120/80 and 140/90 mmHg", "BP 120 over 80 and 140 over 90 millimeters of mercury"),
]

PERIODS_AND_LISTS = [
    # A period inside a sentence makes Kokoro and Piper pause as if the sentence ended.
    ("President Franklin D. Roosevelt spoke.", "President Franklin D Roosevelt spoke."),
    ("J. K. Rowling and J.R.R. Tolkien", "J K Rowling and J R R Tolkien"),
    ("The U.S. Army and the U.N. met in Washington, D.C. today.", "The U S Army and the U N met in Washington, D C today."),
    ("She moved to the U.S.", "She moved to the U S."),
    ("a Ph.D. from MIT", "a P H D from MIT"),
    ("Apple Inc. and Microsoft Corp. grew.", "Apple Inc and Microsoft Corp grew."),
    ("It was Apple Inc.", "It was Apple Inc."),
    ("Robert A. Beezer wrote it.", "Robert A Beezer wrote it."),
    ("Study Technique D. We'll wait.", "Study Technique D. We'll wait."),  # a lettered thing ends the sentence
    ("by Theorem G. B is a basis", "by Theorem G. B is a basis"),
    # A list letter "a" is not the article "uh".
    ("Steps: a) mix; b) heat", "Steps: A) mix; b) heat"),
    ("(a) the first and (b) the second", "(A) the first and (b) the second"),
    ("f(a) - f(b)", "f of a minus f of b"),
    # Rhyme schemes are letters, not words ("Abab").
    ("The rhyme scheme is ABAB CDCD EFEF GG.", "The rhyme scheme is A B A B C D C D E F E F G G."),
    ("a BAD DEAD FACE", "a Bad Dead Face"),
]


@pytest.mark.parametrize(
    "text, expected",
    DATES_AND_YEARS + MONEY + PHONES_TIMES_RATIOS + CHEMISTRY + PROSE_MATH + UNITS_MORE + PERIODS_AND_LISTS,
)
def test_voice_friendly_numbers_names_and_math(text, expected):
    assert norm(text) == expected


def test_link_domains_are_spelled_for_kokoro():
    # Kokoro's phonemizer splits "example.com" at the dot and reads "example. com".
    assert norm("Visit www.example.com/syllabus today.") == "Visit a link to example dot com today."


def test_numbers_survive_kokoro_punctuation_splitting():
    """Kokoro's phonemizer cuts the text at the first copy of each punctuation mark it finds,
    so "7.4." became "seven. four" and "2,500 cases," "two, five hundred cases"."""
    punctuation = pytest.importorskip("phonemizer.punctuation")
    marks = punctuation.Punctuation()
    texts = [
        "The pH of blood is 7.4.", "In 2,500 cases,", "It costs $3.50 now.", "Pi is 3.14159.",
        "Mix 0.5 g with 1,000 mL.", "From 1.5 to 2.5,", "At 98.6°F.", "Python 3.12 and v1.2.",
    ]
    for text in texts:
        speech = norm(text)
        chunks, _ = marks.preserve(speech)
        for left, right in itertools.pairwise(chunks):
            assert not (left[-1:].isdigit() and right[:1].isdigit()), (text, speech, chunks)


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng is not installed")
def test_espeak_reads_years_and_money_as_words():
    """Listen by proxy: the phonemes both offline voices are built from."""

    def ipa(text: str) -> str:
        out = subprocess.run(
            ["espeak-ng", "-q", "--ipa", "-v", "en-us"], input=text, capture_output=True, text=True, timeout=30,
            check=False,
        )
        return out.stdout

    assert "hˈʌndɹɪd" in ipa("In 1990, the wall fell.")  # "nineteen hundred ninety"
    assert "hˈʌndɹɪd" not in ipa(norm("In 1990, the wall fell."))
    money = ipa(norm("It costs $3.50."))  # was "dollar three point five zero"
    assert "ɔɪnt" not in money and 0 <= money.find("θɹˈiː") < money.find("dˈɑːlɚz")


# --------------------------------------------------------------------------- sentence splitting


def sentences(text: str) -> list[str]:
    return [text[s:e] for s, e in split_sentences(text)]


SPLITS = [
    ("One. Two. Three.", ["One.", "Two.", "Three."]),
    ("Dr. Smith arrived. He sat.", ["Dr. Smith arrived.", "He sat."]),
    ("Mr. and Mrs. Lee met Prof. Park at St. Mary's.", ["Mr. and Mrs. Lee met Prof. Park at St. Mary's."]),
    ("Fruit, e.g. apples, is good. Eat it.", ["Fruit, e.g. apples, is good.", "Eat it."]),
    ("Use one, i.e. the best. Then stop.", ["Use one, i.e. the best.", "Then stop."]),
    ("Apples, pears, etc. are fruit. Eat them.", ["Apples, pears, etc. are fruit.", "Eat them."]),
    ("We bought apples, pears, etc. Then we left.", ["We bought apples, pears, etc.", "Then we left."]),
    ("See Fig. 3 for details. It is clear.", ["See Fig. 3 for details.", "It is clear."]),
    ("From Eq. 4 we get x. Done.", ["From Eq. 4 we get x.", "Done."]),
    ("Smith et al. found it. Others agree.", ["Smith et al. found it.", "Others agree."]),
    ("Smith et al. Found it.", ["Smith et al. Found it."]),
    ("Cats vs. Dogs is a debate. It rages.", ["Cats vs. Dogs is a debate.", "It rages."]),
    ("It is approx. 5 cm. Measure it.", ["It is approx. 5 cm.", "Measure it."]),
    ("The U.S. Army grew. It won.", ["The U.S. Army grew.", "It won."]),
    ("She moved to the U.S. The move was hard.", ["She moved to the U.S.", "The move was hard."]),
    ("He earned a Ph.D. Students admired him.", ["He earned a Ph.D. Students admired him."]),
    ("He earned a Ph.D. He then left.", ["He earned a Ph.D.", "He then left."]),
    ("The U.K. Parliament met.", ["The U.K. Parliament met."]),
    ("J. K. Rowling wrote it. It sold.", ["J. K. Rowling wrote it.", "It sold."]),
    ("George W. Bush spoke.", ["George W. Bush spoke."]),
    ("Pi is 3.14 roughly. Yes.", ["Pi is 3.14 roughly.", "Yes."]),
    ("See No. 5 and p. 12 now. Next.", ["See No. 5 and p. 12 now.", "Next."]),
    ("Read pp. 10-12. Then rest.", ["Read pp. 10-12.", "Then rest."]),
    ("1. Introduction to cells", ["1. Introduction to cells"]),
    ("IV. The Cell Cycle", ["IV. The Cell Cycle"]),
    ("A. First choice", ["A. First choice"]),
    ("Steps: 1. Mix 2. Stir", ["Steps: 1. Mix 2.", "Stir"]),
    ("Wait... what happened?", ["Wait... what happened?"]),
    ("Wait... What happened?", ["Wait...", "What happened?"]),
    ("Wait… What?", ["Wait…", "What?"]),
    ('"Really?" she asked.', ['"Really?" she asked.']),
    ('"Really?" She left.', ['"Really?"', "She left."]),
    ("He said “Stop!” Then he left.", ["He said “Stop!”", "Then he left."]),
    ("(See Figure 2.) Next we look.", ["(See Figure 2.)", "Next we look."]),
    ("It works (mostly). Good.", ["It works (mostly).", "Good."]),
    ("Yahoo! is a company.", ["Yahoo! is a company."]),
    ("What?! No way.", ["What?!", "No way."]),
    ("It ended.¹ Then it began.", ["It ended.¹", "Then it began."]),
    ("Built c. 1500. It stands.", ["Built c. 1500.", "It stands."]),
    ("Smith v. Jones was decided. Appeals followed.", ["Smith v. Jones was decided.", "Appeals followed."]),
    ("So did I. Then we left.", ["So did I.", "Then we left."]),
    ("He is from Washington, D.C. and works there.", ["He is from Washington, D.C. and works there."]),
    ("It was 5 p.m. The bus came.", ["It was 5 p.m.", "The bus came."]),
    ("Visit example.com. It helps.", ["Visit example.com.", "It helps."]),
    ("First para.\n\nSecond para", ["First para.", "Second para"]),
    ("Heading without period\n\nBody text.", ["Heading without period", "Body text."]),
    ("No punctuation at all", ["No punctuation at all"]),
    ("   padded sentence.   ", ["padded sentence."]),
    ("Jan. 5 was cold. Feb. was warm.", ["Jan. 5 was cold.", "Feb. was warm."]),
    ("Dept. of Biology. Room 5.", ["Dept. of Biology.", "Room 5."]),
    ("John Smith Jr. said hi. He left.", ["John Smith Jr. said hi.", "He left."]),
    ("Apple Inc. The company grew.", ["Apple Inc.", "The company grew."]),
    # Numbered things in math and law books, and math after "!" (factorial), are not sentence ends.
    ("See Thm. 3 and Def. 2. Also Prop. 4 and Cor. 1.", ["See Thm. 3 and Def. 2.", "Also Prop. 4 and Cor. 1."]),
    ("By Lem. 2 and Prob. 3 it holds.", ["By Lem. 2 and Prob. 3 it holds."]),
    ("Art. I, § 8, cl. 3 grants power. Art. II covers it.", ["Art. I, § 8, cl. 3 grants power.", "Art. II covers it."]),
    ("Call ext. 22 today. Or not.", ["Call ext. 22 today.", "Or not."]),
    ("It is 1 + x + x^2/2! + x^3/3! + ... in all.", ["It is 1 + x + x^2/2! + x^3/3! + ... in all."]),
    ("Compute 5! = 120 now. Then stop.", ["Compute 5! = 120 now.", "Then stop."]),
    ("I like modern art. I hate it too.", ["I like modern art.", "I hate it too."]),
]


@pytest.mark.parametrize("text, expected", SPLITS)
def test_split_sentences(text, expected):
    assert sentences(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_split_sentences_empty(text):
    assert split_sentences(text) == []


def _random_prose(rng: random.Random, n_words: int) -> str:
    words = [
        "cell", "Dr.", "e.g.", "i.e.", "etc.", "Fig.", "3.14", "U.S.", "J.", "K.", "Rowling", "(Smith,",
        "2019).", "[12]", "What?", "Yes!", "Wait...", "“Stop!”", "end.", "The", "It", "and", "No.", "5", "p.",
        "12", "et", "al.", "a.m.", "vs.", "…", "x²", "—", "-", ";", ":", ",", "https://example.com.", "pp.",
        "10–12.", "\n\n",
    ]
    return " ".join(rng.choice(words) for _ in range(n_words))


def test_split_sentences_spans_cover_all_text_randomized():
    rng = random.Random(42)
    for _ in range(400):
        text = (" " * rng.randint(0, 2)) + _random_prose(rng, rng.randint(0, 60)) + (" " * rng.randint(0, 2))
        spans = split_sentences(text)
        covered = [False] * len(text)
        last_end = -1
        for s, e in spans:
            assert last_end <= s < e <= len(text)
            assert text[s:e] == text[s:e].strip()
            for i in range(s, e):
                covered[i] = True
            last_end = e
        for i, ch in enumerate(text):
            assert covered[i] or ch.isspace(), (text, i)


# --------------------------------------------------------------------------- chunking


def test_chunk_span_prefers_strong_breaks():
    text = "First clause is here; second clause follows, with a comma and more words to pad it out nicely."
    pieces = [text[s:e] for s, e in chunk_span(text, 0, len(text), 60)]
    assert pieces[0] == "First clause is here;"
    assert all(len(p) <= 60 for p in pieces)


def test_chunk_span_uses_commas_then_spaces():
    text = "alpha beta gamma delta, epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho"
    pieces = [text[s:e] for s, e in chunk_span(text, 0, len(text), 40)]
    assert pieces[0] == "alpha beta gamma delta,"
    assert all(len(p) <= 40 for p in pieces)
    no_punct = "word " * 30
    pieces = [no_punct[s:e] for s, e in chunk_span(no_punct, 0, len(no_punct), 22)]
    assert all(p.split() and all(w == "word" for w in p.split()) for p in pieces)


def test_chunk_span_avoids_breaking_inside_brackets():
    text = "Cells divide quickly (Smith & Jones, 2019; Lee et al., 2020) and then they rest for a while"
    pieces = [text[s:e] for s, e in chunk_span(text, 0, len(text), 70)]
    assert any("(Smith & Jones, 2019; Lee et al., 2020)" in p for p in pieces)


def test_chunk_span_long_word():
    word = "A" * 50
    text = f"short {word} tail"
    pieces = [text[s:e] for s, e in chunk_span(text, 0, len(text), 20)]
    assert all(len(p) <= 20 for p in pieces)
    assert "".join(pieces).replace(" ", "") == text.replace(" ", "")
    whole = [text[s:e] for s, e in chunk_span(text, 0, len(text), 20, split_words=False)]
    assert word in whole


def test_chunk_span_randomized_invariants():
    rng = random.Random(5)
    for _ in range(300):
        text = _random_prose(rng, rng.randint(1, 80))
        max_chars = rng.randint(5, 120)
        pieces = chunk_span(text, 0, len(text), max_chars)
        last = 0
        for s, e in pieces:
            assert last <= s < e and e - s <= max_chars
            assert text[s:e] == text[s:e].strip()
            last = e
        joined = "".join(text[s:e] for s, e in pieces)
        assert joined.replace(" ", "").replace("\n", "") == text.replace(" ", "").replace("\n", "")
        for s, e in pieces:  # never split inside a word that fits
            if s > 0 and not text[s - 1].isspace():
                word_end = e
                assert len(text[:s].split()[-1] + text[s:word_end].split()[0]) > max_chars


def test_chunk_span_rejects_bad_size():
    with pytest.raises(ValueError):
        chunk_span("abc", 0, 3, 0)


# --------------------------------------------------------------------------- plan helpers

CHUNK, CLAUSE, SENTENCE, ROW, BLOCK, HEADING = (
    plan_module.PAUSE_CHUNK_MS,
    plan_module.PAUSE_CLAUSE_MS,
    plan_module.PAUSE_SENTENCE_MS,
    plan_module.PAUSE_ROW_MS,
    plan_module.PAUSE_BLOCK_MS,
    plan_module.PAUSE_HEADING_MS,
)
PAUSES = {CHUNK, CLAUSE, SENTENCE, ROW, BLOCK, HEADING}
END_PAUSE = {
    K.HEADING: HEADING,
    K.PARAGRAPH: BLOCK,
    K.QUOTE: BLOCK,
    K.CAPTION: BLOCK,
    K.FOOTNOTE: BLOCK,
    K.FIGURE: BLOCK,
    K.EQUATION: BLOCK,
    K.CODE: BLOCK,
    K.TABLE: BLOCK,
    K.LIST_ITEM: ROW,
}
WHOLE_BLOCK_KINDS = {K.TABLE, K.EQUATION}


def check_plan(section: Section, opts: ReadingOptions) -> list:
    """Assert every documented plan_section guarantee; return the plan."""
    plan = plan_section(section, opts)
    assert [u.index for u in plan] == list(range(len(plan)))
    last_key = (-2, -1)
    for n, u in enumerate(plan):
        assert u.speech and u.speech == u.speech.strip()
        assert re.search(r"[^\W_]", u.speech), u
        assert len(u.speech) <= opts.max_chars, (len(u.speech), u)
        assert u.pause_after_ms in PAUSES, u
        if u.block == -1:
            assert u.start == u.end == 0 and u.kind == K.HEADING
            assert all(p.block == -1 for p in plan[:n])
            continue
        block = section.blocks[u.block]
        assert u.kind == block.kind
        assert 0 <= u.start <= u.end <= len(block.text)
        if block.kind in WHOLE_BLOCK_KINDS or (block.kind == K.CODE and opts.code == "announce"):
            if block.kind != K.TABLE or plan_module._table_rows(block):
                assert (u.start, u.end) == (0, len(block.text))
        else:
            span = block.text[u.start : u.end]
            assert span and span == span.strip(), u
        assert (u.block, u.start) >= last_key, "reading order"
        last_key = (u.block, u.start)
    # pause after the last utterance of each block
    for n, u in enumerate(plan):
        is_last_of_block = n == len(plan) - 1 or plan[n + 1].block != u.block
        if u.block == -1:
            if is_last_of_block:
                assert u.pause_after_ms == HEADING
            continue
        if is_last_of_block:
            assert u.pause_after_ms == END_PAUSE[u.kind], u
        else:
            assert u.pause_after_ms in (CHUNK, CLAUSE, SENTENCE, ROW), u
    return plan


def kinds_in(plan) -> set:
    return {u.kind for u in plan}


def make_section(*blocks: Block, title: str = "Cell Biology") -> Section:
    return Section(id="s1", title=title, blocks=list(blocks))


# --------------------------------------------------------------------------- plan: basics


def test_plan_paragraph_sentences_and_pauses():
    text = "Cells are small. They divide (Smith, 2019). Fig. 3 shows this."
    sec = make_section(
        Block(kind=K.HEADING, text="Chapter 1: Cells", level=1), Block(kind=K.PARAGRAPH, text=text)
    )
    plan = check_plan(sec, ReadingOptions())
    assert [(u.block, u.speech, u.pause_after_ms) for u in plan] == [
        (0, "Chapter 1: Cells", HEADING),
        (1, "Cells are small.", SENTENCE),
        (1, "They divide.", SENTENCE),
        (1, "Figure 3 shows this.", BLOCK),
    ]
    assert [text[u.start : u.end] for u in plan[1:]] == [
        "Cells are small.", "They divide (Smith, 2019).", "Fig. 3 shows this."
    ]


def test_plan_section_title_announcement():
    sec = make_section(Block(kind=K.PARAGRAPH, text="Body."), title="INTRODUCTION TO CELLS")
    plan = check_plan(sec, ReadingOptions())
    assert (plan[0].block, plan[0].start, plan[0].end, plan[0].speech) == (-1, 0, 0, "Introduction To Cells")
    assert plan[0].pause_after_ms == HEADING and plan[0].kind == K.HEADING
    no_title = check_plan(sec, ReadingOptions(announce_section_title=False))
    assert no_title[0].block == 0
    with_heading = make_section(
        Block(kind=K.HEADING, text="Intro", level=1), Block(kind=K.PARAGRAPH, text="Body.")
    )
    assert check_plan(with_heading, ReadingOptions())[0].block == 0


def test_plan_headings_can_be_skipped():
    sec = make_section(Block(kind=K.HEADING, text="Intro", level=1), Block(kind=K.PARAGRAPH, text="Body."))
    plan = check_plan(sec, ReadingOptions(read_headings=False))
    assert [u.speech for u in plan] == ["Body."]


def test_plan_list_items_and_quotes():
    sec = make_section(
        Block(kind=K.LIST_ITEM, text="• First item. It has two sentences."),
        Block(kind=K.LIST_ITEM, text="Second item"),
        Block(kind=K.QUOTE, text="To be or not to be."),
    )
    plan = check_plan(sec, ReadingOptions(announce_section_title=False))
    assert [(u.speech, u.pause_after_ms) for u in plan] == [
        ("First item.", SENTENCE),
        ("It has two sentences.", ROW),
        ("Second item", ROW),
        ("To be or not to be.", BLOCK),
    ]


def test_plan_captions_footnotes_figures():
    sec = make_section(
        Block(kind=K.CAPTION, text="Figure 2.1 A cell."),
        Block(kind=K.FOOTNOTE, text="See the appendix. It helps."),
        Block(kind=K.FIGURE, text="A diagram of a plant cell"),
        Block(kind=K.FIGURE, text=""),
    )
    opts = ReadingOptions(announce_section_title=False, read_footnotes=True)
    plan = check_plan(sec, opts)
    assert [u.speech for u in plan] == [
        "Figure 2 point 1 A cell.", "Footnote: See the appendix.", "It helps.", "Image: A diagram of a plant cell"
    ]
    assert plan[3].start == 0 and plan[3].end == len("A diagram of a plant cell")
    quiet = ReadingOptions(
        announce_section_title=False, read_captions=False, read_footnotes=False, read_figures=False
    )
    assert check_plan(sec, quiet) == []
    assert kinds_in(check_plan(sec, ReadingOptions(announce_section_title=False))) == {K.CAPTION, K.FIGURE}


def test_plan_equations():
    sec = make_section(Block(kind=K.EQUATION, text=r"\frac{a}{b} = c"))
    plan = check_plan(sec, ReadingOptions(announce_section_title=False))
    assert [(u.speech, u.start, u.end, u.pause_after_ms) for u in plan] == [
        ("Equation: a over b equals c", 0, len(r"\frac{a}{b} = c"), BLOCK)
    ]
    assert check_plan(sec, ReadingOptions(announce_section_title=False, equations="skip")) == []
    empty = make_section(Block(kind=K.EQUATION, text=r"\label{x}"))
    assert check_plan(empty, ReadingOptions(announce_section_title=False)) == []


def test_plan_code_modes():
    code = "def area(r):\n    return 3.14 * r ** 2  \n\n# done\n"
    sec = make_section(Block(kind=K.CODE, text=code))
    base = ReadingOptions(announce_section_title=False)
    assert check_plan(sec, base.model_copy(update={"code": "skip"})) == []
    announce = check_plan(sec, base)
    assert [(u.speech, u.start, u.end, u.pause_after_ms) for u in announce] == [
        ("There is a code example here.", 0, len(code), BLOCK)
    ]
    read = check_plan(sec, base.model_copy(update={"code": "read"}))
    assert [code[u.start : u.end] for u in read] == ["def area(r):", "return 3.14 * r ** 2", "# done"]
    assert [u.speech for u in read] == [
        "def area(r):", "return 3 point 1 4 times r to the power of 2", "Comment: done"
    ]
    assert [u.pause_after_ms for u in read] == [SENTENCE, SENTENCE, BLOCK]


def test_plan_code_read_keeps_array_indexes():
    sec = make_section(Block(kind=K.CODE, text="total = items[3] + items[4]"))
    plan = check_plan(sec, ReadingOptions(announce_section_title=False, code="read"))
    assert plan[0].speech == "total equals items[3] plus items[4]"


# --------------------------------------------------------------------------- plan: tables

PEOPLE = [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]


def table_block(rows, text: str | None = None, **meta) -> Block:
    text = text if text is not None else "\n".join(" | ".join(r) for r in rows)
    return Block(kind=K.TABLE, text=text, rows=rows, meta=meta)


def test_table_rows_with_header():
    block = table_block(PEOPLE)
    plan = check_plan(make_section(block), ReadingOptions(announce_section_title=False))
    assert [(u.speech, u.pause_after_ms) for u in plan] == [
        ("Table with 2 rows and 2 columns.", ROW),
        ("Row 1: Name, Alice; Age, 30", ROW),
        ("Row 2: Name, Bob; Age, 25", BLOCK),
    ]
    assert all((u.start, u.end) == (0, len(block.text)) for u in plan)


def test_table_without_header():
    rows = [["1", "2", "3"], ["4", "5", "6"]]
    plan = check_plan(make_section(table_block(rows)), ReadingOptions(announce_section_title=False))
    assert [u.speech for u in plan] == [
        "Table with 2 rows and 3 columns.", "Row 1: 1; 2; 3", "Row 2: 4; 5; 6"
    ]


@pytest.mark.parametrize(
    "rows, header",
    [
        (PEOPLE, True),
        ([["Country", "2019", "2020"], ["France", "1.2", "1.3"]], True),
        ([["Name", "Name"], ["a", "b"]], False),  # not distinct
        ([["1", "2"], ["3", "4"]], False),  # numeric
        ([["Only one row", "here"]], False),
        ([["x" * 70, "y"], ["a", "b"]], False),  # too long for a label
        ([["", "Mass", "Charge"], ["Proton", "1", "+1"]], True),
    ],
)
def test_table_header_detection(rows, header):
    assert plan_module._has_header(plan_module._table_rows(table_block(rows)), {}) is header


def test_table_header_hint_from_loader():
    rows = [["1", "2"], ["3", "4"]]
    opts = ReadingOptions(announce_section_title=False)
    plan = check_plan(make_section(table_block(rows, header=True)), opts)
    assert plan[1].speech == "Row 1: 1, 3; 2, 4"
    plan = check_plan(make_section(table_block(PEOPLE, header=False)), opts)
    assert plan[0].speech == "Table with 3 rows and 2 columns."


def test_table_announce_and_skip():
    block = table_block([["A", "B"], ["1", "2"], ["3", "4"], ["5", "6"]])
    opts = ReadingOptions(announce_section_title=False, tables="announce")
    plan = check_plan(make_section(block), opts)
    assert [(u.speech, u.pause_after_ms) for u in plan] == [
        ("There is a table here with 3 rows and 2 columns.", BLOCK)
    ]
    one = table_block([["Name", "Age"], ["Alice", "30"]])
    assert check_plan(make_section(one), opts)[0].speech == "There is a table here with 1 row and 2 columns."
    assert check_plan(make_section(block), opts.model_copy(update={"tables": "skip"})) == []


def test_table_cells_are_normalized_and_empty_cells_skipped():
    rows = [
        ["Term", "Notes"], ["Mitosis", "Cell division (Smith, 2019), e.g. in skin"], ["Meiosis", ""], ["", ""]
    ]
    plan = check_plan(make_section(table_block(rows)), ReadingOptions(announce_section_title=False))
    assert [u.speech for u in plan] == [
        "Table with 2 rows and 2 columns.",
        "Row 1: Term, Mitosis; Notes, Cell division, for example in skin",
        "Row 2: Term, Meiosis",
    ]


def test_table_long_rows_are_chunked():
    header = [f"Column {i}" for i in range(12)]
    row = [f"value number {i} with some extra words" for i in range(12)]
    huge = ["word " * 120]
    block = table_block([header, row, huge + [""] * 11])
    opts = ReadingOptions(announce_section_title=False, max_chars=120)
    plan = check_plan(make_section(block), opts)
    assert plan[0].speech == "Table with 2 rows and 12 columns."
    assert plan[1].speech.startswith("Row 1: Column 0, value number 0")
    assert sum(1 for u in plan if u.speech.startswith("Row ")) == 2
    assert len(plan) > 4
    row_one = " ".join(u.speech for u in plan[1:] if not u.speech.startswith("Row 2"))
    assert "Column 11, value number 11 with some extra words" in row_one


def test_table_without_rows_falls_back_to_text():
    block = Block(kind=K.TABLE, text="Name Age Alice 30", rows=None)
    plan = check_plan(make_section(block), ReadingOptions(announce_section_title=False))
    assert [u.speech for u in plan] == ["Table: Name Age Alice 30"]
    plan = check_plan(make_section(block), ReadingOptions(announce_section_title=False, tables="announce"))
    assert [u.speech for u in plan] == ["There is a table here."]


# --------------------------------------------------------------------------- plan: speaker notes, skipping


def test_speaker_notes_prefix_once_per_run():
    sec = make_section(
        Block(kind=K.HEADING, text="Slide 1", level=2),
        Block(kind=K.PARAGRAPH, text="Slide text."),
        Block(kind=K.PARAGRAPH, text="(Smith, 2019)", meta={"speaker_notes": True}),
        Block(kind=K.PARAGRAPH, text="Talk about cells. Then pause.", meta={"speaker_notes": True}),
        Block(kind=K.PARAGRAPH, text="More notes.", meta={"speaker_notes": True}),
        Block(kind=K.HEADING, text="Slide 2", level=2),
        Block(kind=K.PARAGRAPH, text="Second notes.", meta={"speaker_notes": True}),
    )
    plan = check_plan(sec, ReadingOptions())
    assert [u.speech for u in plan] == [
        "Slide 1",
        "Slide text.",
        "Speaker notes: Talk about cells.",
        "Then pause.",
        "More notes.",
        "Slide 2",
        "Speaker notes: Second notes.",
    ]


def test_empty_speech_is_skipped_but_indices_stay_contiguous():
    sec = make_section(
        Block(kind=K.PARAGRAPH, text="(Smith, 2019). [12]. Real sentence. [3-5]."),
        Block(kind=K.PARAGRAPH, text="• ▪ ◦"),
        Block(kind=K.PARAGRAPH, text="Last one."),
    )
    plan = check_plan(sec, ReadingOptions(announce_section_title=False))
    assert [(u.block, u.speech) for u in plan] == [(0, "Real sentence."), (2, "Last one.")]
    assert plan[0].pause_after_ms == BLOCK


def test_long_sentences_are_chunked_with_short_pauses():
    clauses = ", ".join(f"adding clause number {i}" for i in range(40))
    text = f"This sentence keeps going, {clauses}. Done."
    sec = make_section(Block(kind=K.PARAGRAPH, text=text))
    plan = check_plan(sec, ReadingOptions(announce_section_title=False, max_chars=100))
    assert len(plan) > 5
    assert all(u.pause_after_ms == CHUNK for u in plan[:-2])
    assert plan[-2].pause_after_ms == SENTENCE and plan[-1].pause_after_ms == BLOCK
    assert all(text[u.start : u.end].endswith(",") for u in plan[:-3])


def test_long_sentence_split_at_semicolons_gets_clause_pauses():
    # Read straight through, Kokoro pauses about 300 ms at a semicolon; a 60 ms gap ran clauses together.
    text = "; ".join(f"clause number {i} has a few more words in it" for i in range(12)) + "."
    sec = make_section(Block(kind=K.PARAGRAPH, text=text))
    plan = check_plan(sec, ReadingOptions(announce_section_title=False, max_chars=100))
    assert len(plan) > 2
    assert all(text[u.start : u.end].endswith(";") and u.pause_after_ms == CLAUSE for u in plan[:-1])
    assert plan[-1].pause_after_ms == BLOCK


def test_pauses_grow_with_the_size_of_the_break():
    # Measured: Kokoro leaves 300-520 ms between sentences when it reads them in one go, so a
    # 220 ms gap (plus ~50 ms of kept padding) sounded rushed next to the voice's own pacing.
    assert 0 < CHUNK < CLAUSE < SENTENCE < ROW < BLOCK < HEADING
    assert SENTENCE >= 300 and BLOCK >= 2 * SENTENCE


def test_expanding_text_still_fits_max_chars():
    text = " ".join(["α β γ δ ε ζ η θ ι κ λ μ"] * 20) + "."
    sec = make_section(Block(kind=K.PARAGRAPH, text=text))
    plan = check_plan(sec, ReadingOptions(announce_section_title=False, max_chars=80))
    assert len(plan) > 3


def test_single_giant_word_is_split_in_speech_only():
    word = "ACGT" * 300
    sec = make_section(Block(kind=K.PARAGRAPH, text=f"Sequence {word} ends."))
    plan = check_plan(sec, ReadingOptions(announce_section_title=False, max_chars=200))
    assert len(plan) >= 6


def test_long_heading_and_title_are_chunked():
    long = "A very long heading " * 30
    sec = make_section(Block(kind=K.HEADING, text=long.strip(), level=1))
    plan = check_plan(sec, ReadingOptions(max_chars=100))
    assert len(plan) > 1 and plan[-1].pause_after_ms == HEADING
    sec = make_section(Block(kind=K.PARAGRAPH, text="Body."), title=long)
    plan = check_plan(sec, ReadingOptions(max_chars=100))
    assert plan[0].block == -1 and plan[1].block == -1
    assert [u.pause_after_ms for u in plan if u.block == -1][-1] == HEADING


def test_plan_empty_section():
    assert plan_section(Section(id="s0", title="", blocks=[])) == []
    plan = plan_section(Section(id="s0", title="Only a title", blocks=[]))
    assert [u.speech for u in plan] == ["Only a title"]


def test_plan_uses_default_options():
    sec = make_section(
        Block(kind=K.FOOTNOTE, text="Hidden by default."), Block(kind=K.PARAGRAPH, text="Shown.")
    )
    assert [u.speech for u in plan_section(sec)] == ["Cell Biology", "Shown."]


# --------------------------------------------------------------------------- plan: randomized invariants

SENTENCE_BANK = [
    "Cells are the basic unit of life (Smith et al., 2019; Lee, 2020a).",
    "As shown in Fig. 3, e.g. the nucleus contains DNA, i.e. genetic material.",
    "The rate rose by approx. 50% at 37°C (see pp. 10–12).",
    "Mitochondria produce ATP [12], and chloroplasts capture light [3-5, 9].",
    "Dr. Jones measured 5 µm cells at 9 a.m. using a 2×2 grid.",
    "The equation x² + y² = r² describes a circle.",
    "Visit https://www.example.com/cells for more.",
    "During World War II, Chapter IV explains why Henry VIII mattered.",
    "THE CELL CYCLE has phases, viz. G1, S, G2 and M.",
    "Smith (2019) showed that Δx ≈ 0.5 mm when p < .05.",
    "Wait... what?",
    "“Really?” she asked.",
    "(Smith, 2019).",
    "[12].",
    "Short.",
    "J. K. Rowling wrote it.",
]


def random_text(rng: random.Random) -> str:
    parts = [rng.choice(SENTENCE_BANK) for _ in range(rng.randint(1, 6))]
    if rng.random() < 0.2:
        filler = ["lorem", "ipsum,", "dolor;", "sit:", "amet", "—"]
        parts.append(" ".join(rng.choice(filler) for _ in range(150)) + ".")
    if rng.random() < 0.1:
        parts.append("X" * rng.randint(100, 500))
    return " ".join(parts)


def random_block(rng: random.Random) -> Block:
    kind = rng.choice(list(K))
    meta = {"speaker_notes": True} if rng.random() < 0.1 else {}
    if kind == K.HEADING:
        text = rng.choice(["Intro", "CHAPTER IV: THE CELL", random_text(rng)])
        return Block(kind=kind, text=text, level=rng.randint(1, 3))
    if kind == K.TABLE:
        n_rows, n_cols = rng.randint(0, 5), rng.randint(1, 4)
        rows = [[rng.choice(["Name", "Age", "30", "", "Alice", random_text(rng)]) for _ in range(n_cols)]
                for _ in range(n_rows)]
        return Block(kind=kind, text=random_text(rng), rows=rows or None, meta=meta)
    if kind == K.EQUATION:
        equations = ["x^2 + 1", r"\frac{a}{b}", r"\sum_{i=1}^{n} i", "E = mc²", "{}"]
        return Block(kind=kind, text=rng.choice(equations))
    if kind == K.CODE:
        return Block(kind=kind, text=rng.choice(["x = 1\ny = 2", "  # comment\n\nprint(x)", "a" * 600]))
    if kind == K.FIGURE:
        return Block(kind=kind, text=rng.choice(["", "A cell diagram.", random_text(rng)]))
    return Block(kind=kind, text=random_text(rng), meta=meta)


def random_options(rng: random.Random) -> ReadingOptions:
    return ReadingOptions(
        read_headings=rng.random() < 0.8,
        announce_section_title=rng.random() < 0.8,
        read_captions=rng.random() < 0.8,
        read_footnotes=rng.random() < 0.5,
        read_figures=rng.random() < 0.8,
        tables=rng.choice(["skip", "announce", "rows"]),
        equations=rng.choice(["skip", "read"]),
        code=rng.choice(["skip", "announce", "read"]),
        citations=rng.choice(["keep", "remove"]),
        urls=rng.choice(["full", "domain", "skip"]),
        expand_abbreviations=rng.random() < 0.8,
        max_chars=rng.choice([80, 81, 120, 200, 400, 2000]),
    )


def test_plan_invariants_randomized():
    rng = random.Random(2026)
    for _ in range(250):
        blocks = [random_block(rng) for _ in range(rng.randint(0, 8))]
        title = rng.choice(["Cells", "", "INTRODUCTION", random_text(rng)])
        sec = Section(id="s1", title=title, blocks=blocks)
        opts = random_options(rng)
        plan = check_plan(sec, opts)
        kinds = kinds_in(plan)
        if not opts.read_headings:
            assert all(u.block == -1 for u in plan if u.kind == K.HEADING)
        if not opts.read_captions:
            assert K.CAPTION not in kinds
        if not opts.read_footnotes:
            assert K.FOOTNOTE not in kinds
        if not opts.read_figures:
            assert K.FIGURE not in kinds
        if opts.tables == "skip":
            assert K.TABLE not in kinds
        if opts.equations == "skip":
            assert K.EQUATION not in kinds
        if opts.code == "skip":
            assert K.CODE not in kinds


def test_plan_covers_every_word_of_plain_paragraphs():
    rng = random.Random(11)
    vocab = ["cells", "divide", "quickly", "and", "then", "rest", "for", "a", "while", "proteins", "fold"]
    for _ in range(100):
        blocks = []
        for _ in range(rng.randint(1, 5)):
            sents = []
            for _ in range(rng.randint(1, 8)):
                words = [rng.choice(vocab) for _ in range(rng.randint(1, 60))]
                sents.append(" ".join(words).capitalize() + rng.choice([".", "!", "?", ";", ","]))
            blocks.append(Block(kind=K.PARAGRAPH, text=" ".join(sents)))
        sec = Section(id="s1", title="T", blocks=blocks)
        opts = ReadingOptions(max_chars=rng.choice([80, 100, 150, 400]), announce_section_title=False)
        plan = check_plan(sec, opts)
        for bi, block in enumerate(blocks):
            covered = [False] * len(block.text)
            for u in plan:
                if u.block == bi:
                    for i in range(u.start, u.end):
                        assert not covered[i], "overlapping highlights"
                        covered[i] = True
            assert all(c or ch.isspace() for c, ch in zip(covered, block.text, strict=True))
            spoken = " ".join(u.speech for u in plan if u.block == bi)
            assert re.findall(r"\w+", spoken) == re.findall(r"\w+", block.text)


def test_plan_performance_40k_section():
    rng = random.Random(1)
    blocks = [Block(kind=K.HEADING, text="Chapter 1", level=1)]
    total = 0
    while total < 40_000:
        text = " ".join(rng.choice(SENTENCE_BANK[:10]) for _ in range(rng.randint(3, 8)))
        blocks.append(Block(kind=rng.choice([K.PARAGRAPH, K.PARAGRAPH, K.LIST_ITEM, K.QUOTE]), text=text))
        total += len(text)
    sec = Section(id="s1", title="Big", blocks=blocks)
    plan_section(sec)  # warm up regex caches
    best = min(_timed(lambda: plan_section(sec)) for _ in range(3))
    assert best < 0.5, f"planning a 40,000 character section took {best:.3f}s"


def _timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


# --------------------------------------------------------------------------- section_plain_text


def test_section_plain_text_formatting():
    sec = Section(
        id="s1",
        title="Cells",
        blocks=[
            Block(kind=K.HEADING, text="Chapter 1: Cells", level=1),
            Block(kind=K.PARAGRAPH, text="Cells are small (Smith, 2019)."),
            Block(kind=K.HEADING, text="Parts", level=2),
            Block(kind=K.LIST_ITEM, text="• Nucleus"),
            Block(kind=K.LIST_ITEM, text="Membrane"),
            Block(kind=K.QUOTE, text="Life is cells."),
            Block(kind=K.TABLE, text="Name Age", rows=PEOPLE),
            Block(kind=K.CAPTION, text="Table 1.1 People."),
            Block(kind=K.FIGURE, text="A cell"),
            Block(kind=K.EQUATION, text="E = mc^2"),
            Block(kind=K.CODE, text="x = 1\ny = 2\n"),
            Block(kind=K.FOOTNOTE, text="A note."),
            Block(kind=K.PARAGRAPH, text="Say hi.", meta={"speaker_notes": True}),
            Block(kind=K.HEADING, text="Deep", level=9),
        ],
    )
    assert section_plain_text(sec) == (
        "# Chapter 1: Cells\n\n"
        "Cells are small (Smith, 2019).\n\n"
        "## Parts\n\n"
        "- Nucleus\n- Membrane\n\n"
        "> Life is cells.\n\n"
        "| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |\n\n"
        "Caption: Table 1.1 People.\n\n"
        "[Image: A cell]\n\n"
        "Equation: E = mc^2\n\n"
        "```\nx = 1\ny = 2\n```\n\n"
        "Footnote: A note.\n\n"
        "Speaker notes: Say hi.\n\n"
        "###### Deep"
    )


def test_section_plain_text_adds_title_without_heading():
    sec = Section(id="s1", title="Intro", blocks=[Block(kind=K.PARAGRAPH, text="Body.")])
    assert section_plain_text(sec) == "# Intro\n\nBody."
    assert section_plain_text(Section(id="s1", title="", blocks=[])) == ""


def test_section_plain_text_truncation():
    blocks = [Block(kind=K.PARAGRAPH, text=f"Paragraph {i} " + "words " * 50) for i in range(50)]
    sec = Section(id="s1", title="Long", blocks=blocks)
    full = section_plain_text(sec)
    assert section_plain_text(sec, max_chars=len(full)) == full
    cut = section_plain_text(sec, max_chars=2000)
    assert len(cut) <= 2000
    assert cut.endswith("\n[...the rest of this section was cut for length]")
    assert full.startswith(cut[: -len("\n[...the rest of this section was cut for length]")])
    body = cut[: -len("\n[...the rest of this section was cut for length]")]
    assert not body.endswith(" ") and body.split()[-1] in {"words", "Paragraph"} | {str(i) for i in range(50)}
