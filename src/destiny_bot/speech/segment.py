"""Sentence splitting for highlighting, and chunking of over-long sentences.

Spans are ``(start, end)`` character offsets into the original text, trimmed of
surrounding whitespace, so ``text[start:end]`` is exactly what gets highlighted.
The splitter knows the abbreviations textbooks use ("Fig. 3", "e.g. the",
"Smith et al. found", "J. K. Rowling", "U.S. Army") and does not end a sentence
after them.
"""

from __future__ import annotations

import re

__all__ = ["split_sentences", "chunk_span"]

# Sentence-final punctuation, then the closing quotes / brackets / footnote marks that belong to it.
_CLOSERS = "\"'”’»)]}*†‡¹²³⁰⁴⁵⁶⁷⁸⁹"
_TERMINAL = re.compile(r"[.!?‽…]+[" + re.escape(_CLOSERS) + r"]*(?=\s)")
_PARAGRAPH_BREAK = re.compile(r"\n[^\S\n]*\n")
_OPENERS = "\"'“‘«([{¿¡"
# A following character that never starts a new sentence.
_NO_SENTENCE_START = frozenset(",;:.!?)]}…%/-–—")

# Never end a sentence after these (case-sensitive, without the period).
_ALWAYS_ABBREV = frozenset(
    """
    Mr Mrs Ms Mx Dr Drs Prof Profs St Ste Mt Ft Rev Hon Gen Col Capt Lt Sgt Cpl Pvt Adm Maj
    Cmdr Gov Sen Rep Reps Pres Supt Insp Messrs Mmes Mme Mlle Msgr Fr
    Fig Figs fig figs FIG FIGS Eq Eqs eq eqs Eqn Eqns eqn eqns Ch Chs Chap ch chap Sec Secs
    Vol Vols vol vols vs Vs cf Cf viz approx Approx ca al resp esp incl Dept dept Univ Assn Inst
    """.split()
)
# End a sentence after these only when the next word is not a number ("No. 5", "p. 12", "Jan. 3").
_NUMBER_ABBREV = frozenset(
    """
    No no Nos nos Nr nr p pp Art art Ref Refs ref refs Tab Thm Ex ex Pt pt Para para
    Jan Feb Mar Apr Jun Jul Aug Sep Sept Oct Nov Dec c ed Ed Op op Fn fn n nn l ll
    """.split()
)
# Usually mid-sentence ("John Smith Jr. said", "Apple Inc. announced"), but they can end one:
# split only when a very common sentence opener follows.
_WEAK_ABBREV = frozenset("Jr Sr Esq Inc Ltd Co Corp Bros Ave Blvd Rd Ln Hwy Pl Sq".split())
_ALWAYS_DOTTED = frozenset({"e.g", "i.e", "E.g", "I.e", "e.g.,", "i.e.,", "et al", "a.k.a", "A.k.a"})
_DOTTED = re.compile(r"(?:[A-Za-z]{1,3}\.)+[A-Za-z]{1,3}")  # U.S, Ph.D, a.m, B.Sc, U.S.A
_LIST_MARKER = re.compile(r"\(?(?:\d{1,3}|[IVXLCivxlc]{1,6}|[A-Za-z])\)?")

_SENTENCE_STARTERS = frozenset(
    """
    A An The This That These Those It Its He She They We I You Our My His Her Their Your
    There Here In On At For But And So Yet Or If When While After Before As Since Because
    Although Though However Therefore Thus Hence Also Then Now Next Finally First Second Third
    Later Today Moreover Furthermore Instead Still Such One Some Many Most Each Every All Both
    No Not None Nothing What Why How Where Who Which Whose To By With From Of Unlike Like
    During Over Under Despite Without Within Between Among Once Only Even Indeed Meanwhile
    Similarly Likewise Consequently Overall Again Another Other Others Several Few Let Note
    See Do Does Did Is Are Was Were Has Have Had Can Could Should Would Will May Might Must
    Yes Please Sometimes Often Usually Perhaps Maybe Otherwise Whether Unless Until
    """.split()
)


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Split ``text`` into sentence spans ``(start, end)``.

    Spans are in order, trimmed of whitespace, and together cover every
    non-whitespace character of ``text``. Blank lines always end a sentence.
    """
    if not text or text.isspace():
        return []
    n = len(text)
    first = _skip_space(text, 0, n)
    cuts: list[int] = []
    for m in _TERMINAL.finditer(text):
        nxt = _skip_space(text, m.end(), n)
        if nxt >= n:
            break
        if _is_boundary(text, m.start(), m.end(), nxt, first):
            cuts.append(m.end())
    if "\n" in text:
        cuts.extend(m.start() for m in _PARAGRAPH_BREAK.finditer(text))
        cuts.sort()
    spans: list[tuple[int, int]] = []
    pos = 0
    for cut in [*cuts, n]:
        s = _skip_space(text, pos, cut)
        e = _rstrip_index(text, s, cut)
        if s < e:
            spans.append((s, e))
        pos = max(pos, cut)
    return spans


def chunk_span(
    text: str, start: int, end: int, max_chars: int, *, split_words: bool = True
) -> list[tuple[int, int]]:
    """Split ``text[start:end]`` into trimmed pieces of at most ``max_chars`` characters.

    Breaks after ``;`` or ``:``, then after ``,`` or a spaced dash, then at a space,
    preferring places outside brackets. A word is cut in the middle only when it
    alone is longer than ``max_chars``; with ``split_words=False`` such a word is
    returned whole instead (that piece is then longer than ``max_chars``).
    """
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    start = _skip_space(text, max(0, start), min(end, len(text)))
    end = _rstrip_index(text, start, min(end, len(text)))
    pieces: list[tuple[int, int]] = []
    s = start
    while end - s > max_chars:
        cut = _best_cut(text, s, end, max_chars)
        if cut is None:
            if split_words:
                cut = s + max_chars
            else:  # keep the over-long word whole
                cut = s
                while cut < end and not text[cut].isspace():
                    cut += 1
        piece_end = _rstrip_index(text, s, cut)
        pieces.append((s, piece_end))
        s = _skip_space(text, cut, end)
    if s < end:
        pieces.append((s, end))
    return pieces


# --------------------------------------------------------------------------- helpers


def _skip_space(text: str, i: int, limit: int) -> int:
    while i < limit and text[i].isspace():
        i += 1
    return i


def _rstrip_index(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1].isspace():
        end -= 1
    return end


def _word_at(text: str, i: int) -> str:
    """The word starting at ``i``, without leading quotes/brackets or trailing punctuation."""
    j = i
    n = len(text)
    while j < n and not text[j].isspace():
        j += 1
    return text[i:j].lstrip(_OPENERS).rstrip(".,;:!?\"'”’)]}")


def _starts_sentence(text: str, i: int) -> bool:
    return _word_at(text, i) in _SENTENCE_STARTERS


def _is_boundary(text: str, term_start: int, term_end: int, nxt: int, first: int) -> bool:
    """Does the punctuation at ``text[term_start:term_end]`` end a sentence?"""
    following = text[nxt]
    if following.islower() or following in _NO_SENTENCE_START:
        return False
    term = text[term_start:term_end]
    core = term.rstrip(_CLOSERS)
    if any(c in core for c in "!?‽") or "…" in core or len(core) > 1:
        return True  # "?", "!", "...": the lowercase check above already caught "Yahoo! is"
    if len(core) != len(term):
        return True  # '... he said." Then' - the period sits inside closing quotes/brackets

    tok_start = term_start
    while tok_start > 0 and not text[tok_start - 1].isspace():
        tok_start -= 1
    word = text[tok_start:term_start].lstrip(_OPENERS).rstrip(_CLOSERS)
    if not word:
        return True
    if _LIST_MARKER.fullmatch(word) and (
        tok_start == first or text[:tok_start].rstrip().endswith(":")
    ):
        return False  # "1. Introduction", "Steps: 2. Mix"
    if "." in word:
        if word in _ALWAYS_DOTTED:
            return False
        if _DOTTED.fullmatch(word):  # U.S., Ph.D., a.m.
            return _starts_sentence(text, nxt)
        return True
    if word in _ALWAYS_ABBREV:
        return False
    if word in _NUMBER_ABBREV and (following.isdigit() or following == "("):
        return False
    if len(word) == 1 and word.isalpha():
        if word == "v":  # Smith v. Jones
            return False
        return _starts_sentence(text, nxt)  # initials: "J. K. Rowling"
    if word in _WEAK_ABBREV:
        return _starts_sentence(text, nxt)
    return True


def _best_cut(text: str, s: int, end: int, max_chars: int) -> int | None:
    """Where to end the piece that starts at ``s`` (exclusive index), or None for one long word."""
    limit = min(s + max_chars, end)
    min_len = max(1, max_chars // 4)
    strong = medium = space0 = space_any = None
    depth = 0
    for i in range(s, limit + 1):
        c = text[i] if i < end else " "
        if c.isspace():
            if i > s:
                space_any = i
                if depth == 0:
                    space0 = i
            continue
        if i >= limit:
            break
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0 and i + 1 < end and text[i + 1].isspace() and i + 1 - s >= min_len:
            if c in ";:":
                strong = i + 1
            elif c == "," or (c in "—–" and i > s and text[i - 1].isspace()):
                medium = i + 1
    for cut in (strong, medium, space0, space_any):
        if cut is not None and cut > s:
            return cut
    return None
