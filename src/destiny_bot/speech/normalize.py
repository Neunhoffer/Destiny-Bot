"""Speech-only text normalization: what the voice says, never what the screen shows.

:func:`normalize_for_speech` expands abbreviations ("e.g." -> "for example"),
drops citations, speaks links, symbols, units, Greek letters, Roman numerals and
number ranges as words, and tames ALL-CAPS headings so voices don't spell them
out. :func:`verbalize_math` turns LaTeX-ish or linear math into spoken English
for equation blocks.

The rules are precise rather than greedy. Leaving something unchanged is always
better than deleting or changing words a student needs to hear.
"""

from __future__ import annotations

import html
import re
from typing import Callable

from ..models import ReadingOptions

__all__ = ["normalize_for_speech", "verbalize_math"]

_DEFAULT_OPTIONS = ReadingOptions()

# Uppercase letters (ASCII + Latin-1 + Latin Extended-A/B) for name patterns.
_UP = "A-Z" + "".join(ch for ch in map(chr, range(0xC0, 0x250)) if ch.isupper())
_CAPWORD = rf"[{_UP}][^\W\d_]*(?:['’\-][^\W\d_]+)*"
_CAPWORD_RE = re.compile(_CAPWORD)
_SPEAKABLE = re.compile(r"[^\W_]")
_WS = re.compile(r"\s+")


def normalize_for_speech(text: str, options: ReadingOptions | None = None) -> str:
    """Return the text a voice should say for ``text`` ("" if nothing speakable remains)."""
    if not text or text.isspace():
        return ""
    opts = options or _DEFAULT_OPTIONS
    vault = _Vault()
    t = _prepare(text)
    t = _inline_math(t, vault)
    t = _links(t, opts.urls, vault)
    if opts.citations == "remove":
        t = _remove_citations(t)
    if opts.expand_abbreviations:
        t = _expand_abbreviations(t)
    t = _number_ranges(t)
    t = _roman_numerals(t)
    t = _symbols(t)
    t = _fix_caps(t)
    t = vault.restore(t)
    t = _tidy(t)
    return t if _SPEAKABLE.search(t) else ""


# --------------------------------------------------------------------------- plumbing


class _Vault:
    """Finished speech hidden behind private-use placeholder characters.

    Later passes can't touch a placeholder, so a spoken link or verbalized math
    survives the symbol and abbreviation rules untouched.
    """

    BASE = 0xF0000

    def __init__(self) -> None:
        self.items: list[str] = []

    def put(self, speech: str) -> str:
        self.items.append(speech)
        return " " + chr(self.BASE + len(self.items) - 1) + " "

    def restore(self, text: str) -> str:
        if not self.items:
            return text
        return _VAULT_CHARS.sub(lambda m: self.items[ord(m.group()) - self.BASE], text)


_VAULT_CHARS = re.compile("[\U000f0000-\U000ffffd]")
_ENTITY = re.compile(r"&(?:[A-Za-z][A-Za-z0-9]{1,30}|#\d{1,7}|#[xX][0-9A-Fa-f]{1,6});")
_HTML_TAG = re.compile(
    r"</?(?:a|abbr|b|big|blockquote|br|cite|code|del|div|em|font|h[1-6]|hr|i|img|ins|kbd|li|mark|ol|p|pre|"
    r"q|s|small|span|strong|sub|sup|table|td|th|tr|tt|u|ul)\b[^<>]{0,300}?/?>",
    re.IGNORECASE,
)
_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u2060\ufeff"), None)


def _prepare(text: str) -> str:
    if "&" in text and ";" in text:
        text = _ENTITY.sub(lambda m: html.unescape(m.group()), text)
    text = text.translate(_INVISIBLE)
    text = _VAULT_CHARS.sub("", text)
    if "<" in text:
        text = _HTML_TAG.sub(" ", text)
    if "*" in text or "_" in text or "`" in text:
        text = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"\1", text)
        text = re.sub(r"(?<!\w)__(?=\S)(.+?)(?<=\S)__(?!\w)", r"\1", text)
        text = text.replace("`", "")
    return _WS.sub(" ", text).strip()


def _cap_like(source: str, repl: str) -> str:
    """``repl`` with its first letter capitalized when ``source`` starts with a capital."""
    if source[:1].isupper() and repl:
        return repl[0].upper() + repl[1:]
    return repl


def _at_end(m: re.Match[str]) -> bool:
    """True when only closing punctuation follows the match (the period also ends the sentence)."""
    return not m.string[m.end() :].strip(" \"'”’)]")


# --------------------------------------------------------------------------- inline math

_INLINE_MATH = re.compile(
    r"\$\$(.+?)\$\$|\\\((.+?)\\\)|\\\[(.+?)\\\]|(?<![\w$\\])\$(?=[^\s$])([^$]{1,300}?)(?<=[^\s\\])\$(?![\w$])"
)
_MATHY = re.compile(r"[\\^_{}=]|^[A-Za-z]$")


def _inline_math(text: str, vault: _Vault) -> str:
    if "$" not in text and "\\" not in text:
        return text

    def repl(m: re.Match[str]) -> str:
        body = next(g for g in m.groups() if g is not None)
        if m.group(4) is not None and not _MATHY.search(body):
            return m.group()  # "$5 and $10" is money, not math
        spoken = verbalize_math(body)
        return vault.put(spoken) if spoken else " "

    return _INLINE_MATH.sub(repl, text)


# --------------------------------------------------------------------------- links

_TLDS = (
    "com|org|net|edu|gov|mil|int|info|biz|io|ai|app|dev|co|us|uk|ca|au|nz|ie|de|fr|nl|eu|jp|cn|ru|br|"
    "tv|ly|news|online|site|xyz|tech|blog|museum"
)
_EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9][A-Za-z0-9._%+-]{0,63}@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_URL = re.compile(
    r"(?<![\w@/.-])(?:"
    r"(?i:(?:https?|ftp)://|www\d?\.)[^\s<>\"'“”]+"
    rf"|(?:[A-Za-z0-9](?:[A-Za-z0-9-]{{0,61}}[A-Za-z0-9])?\.)+(?:{_TLDS})(?![\w-])(?:/[^\s<>\"'“”]*)?"
    r")"
)


def _strip_url_tail(url: str) -> str:
    while url:
        last = url[-1]
        if last in ".,;:!?'\"”’*":
            url = url[:-1]
        elif last == ")" and url.count("(") < url.count(")"):
            url = url[:-1]
        elif last == "]" and url.count("[") < url.count("]"):
            url = url[:-1]
        else:
            break
    return url


def _spell_url_part(part: str) -> str:
    part = re.sub(r"%[0-9A-Fa-f]{2}", " ", part)
    for sym, word in ((".", " dot "), ("/", " slash "), ("-", " dash "), ("_", " underscore "),
                      ("~", " tilde "), ("+", " plus "), ("@", " at "), ("=", " equals "), ("&", " and ")):
        part = part.replace(sym, word)
    return _WS.sub(" ", part).strip()


def _url_host_path(url: str) -> tuple[str, str]:
    u = re.sub(r"^[A-Za-z][A-Za-z0-9+.-]*://", "", url)
    u = re.sub(r"^www\d?\.", "", u, flags=re.IGNORECASE)
    u = re.split(r"[?#]", u, maxsplit=1)[0].rstrip("/")
    host, _, path = u.partition("/")
    host = host.rsplit("@", 1)[-1].split(":", 1)[0].lower()
    return host, path


def _speak_email(addr: str) -> str:
    local, _, domain = addr.partition("@")
    return f"{_spell_url_part(local)} at {_spell_url_part(domain.lower())}"


def _links(text: str, mode: str, vault: _Vault) -> str:
    if "@" in text:

        def email(m: re.Match[str]) -> str:
            # Emails are short and often matter (an instructor's address), so they are
            # read in full in both "full" and "domain" modes.
            return " " if mode == "skip" else vault.put(_speak_email(m.group()))

        text = _EMAIL.sub(email, text)
    if "." not in text:
        return text

    def url(m: re.Match[str]) -> str:
        raw = m.group()
        core = _strip_url_tail(raw)
        tail = raw[len(core) :]
        if "://" not in core and not core.lower().startswith("www"):
            label = core.split("/", 1)[0].split(".")[0]
            if not re.search(r"[A-Za-z]", label):
                return raw
        host, path = _url_host_path(core)
        if not host:
            return raw
        if mode == "skip":
            return " " + tail
        if mode == "domain":
            return vault.put(f"a link to {host}") + tail
        spoken = _spell_url_part(host)
        if path:
            spoken += " slash " + _spell_url_part(path)
        return vault.put(spoken) + tail

    return _URL.sub(url, text)


# --------------------------------------------------------------------------- citations

_PARTICLE = r"(?:van|von|de|der|den|del|della|di|da|das|dos|du|le|la|ter|ten|bin|ibn|al|el|zu)"
_NAME = rf"(?:{_PARTICLE}\s+)*{_CAPWORD}(?:\s+(?:(?:{_PARTICLE}|of|for|the|on)\s+)*{_CAPWORD}){{0,4}}"
_AUTHORS = rf"{_NAME}(?:\s*,\s*{_NAME})*(?:\s*,?\s*(?:&|and)\s+{_NAME})?(?:,?\s+et\s+al\.?)?"
_YEAR1 = r"(?:1[5-9]\d\d|20\d\d)[a-z]?"
_YEAR = (
    rf"(?:{_YEAR1}(?:\s*/\s*{_YEAR1})?|n\.\s?d\."
    r"|(?i:in\s+press|forthcoming|in\s+preparation|under\s+review|submitted|unpublished))"
)
_YEARS = rf"{_YEAR}(?:\s*,\s*(?:{_YEAR}|[a-h](?![.\w])))*"
_LOCWORD = (
    r"(?i:pp?|paras?|¶|chaps?|ch|chapters?|secs?|sections?|§|figs?|figures?|tables?|tab|vols?|volumes?"
    r"|nn?|notes?|fn|footnotes?|ll?|lines?|cols?|pages?)"
)
_LOCNUM = r"[\dIVXLCivxlc]+[a-z]?(?:\s*[-–—]\s*[\dIVXLCivxlc]+[a-z]?)?"
# Each repetition below starts with a distinct token so the regex never has two ways to read
# the same text (which would make a near-miss take exponential time).
_LOCATOR = (
    rf"(?:\s*[,:]\s*(?:{_LOCWORD}\.?\s*)?{_LOCNUM}|\s*(?:and|&)\s*{_LOCNUM}"
    r"|\s*,\s*(?i:emphasis|italics)\s+(?i:added|in\s+original|mine|ours))*"
)
_PREFIX = (
    r"(?:(?i:see(?:\s+also)?|e\.\s?g\.|i\.\s?e\.|cf\.?|but\s+see|compare|contra|"
    r"for\s+(?:a\s+)?(?:recent\s+)?reviews?,?\s+see|for\s+example|for\s+instance|as\s+cited\s+in|cited\s+in|"
    r"quoted\s+in|reviewed\s+in|adapted\s+from|modified\s+from|reprinted\s+from|redrawn\s+from|"
    r"reproduced\s+from|data\s+from|based\s+on|sources?\s*:)(?:\s*,\s*|\s+))"
)
_ITEM = rf"{_PREFIX}{{0,2}}{_AUTHORS}(?:\s*,\s*|\s+){_YEARS}{_LOCATOR}"
_ITEM_SEP = r"(?:\s*;\s*|\s*,\s*(?=(?i:see|cf|as\s+cited|cited|quoted|reviewed)\b))"
_CITE_CONTENT = re.compile(rf"\s*{_ITEM}(?:{_ITEM_SEP}{_ITEM})*\s*[.;,]?\s*")
_YEARISH = re.compile(r"(?:1[5-9]|20)\d\d|n\.\s?d\.|(?i:in\s+press|forthcoming|preparation|under\s+review|submitted|unpublished)")
_PAREN_GROUP = re.compile(r"\s*\(([^()]{4,400})\)")
_BRACKET_GROUP = re.compile(r"\s*\[([^\[\]]{4,400})\]")
_NARRATIVE_YEAR = re.compile(rf"\s*\(\s*{_YEARS}{_LOCATOR}\s*\)")
_SINGLE_AUTHOR = rf"(?:{_PARTICLE}\s+)*{_CAPWORD}"
_AUTHOR_RUN_END = re.compile(
    rf"(?:{_SINGLE_AUTHOR}(?:\s*,\s*{_SINGLE_AUTHOR})*\s*,?\s*(?:&|and)\s+)?{_SINGLE_AUTHOR}\s*$"
)
_ETAL_END = re.compile(r"\bet\s+al\.?\s*$")
_NUMERIC_CITE = re.compile(
    r"\s*\[\s*(\d{1,4}(?:\s*[-–—]\s*\d{1,4})?(?:\s*[,;]\s*\d{1,4}(?:\s*[-–—]\s*\d{1,4})?)*)\s*\]"
)
_NEEDED_TAG = re.compile(r"\s*\[(?i:[a-z ]{0,30}\bneeded|sic\?)\]")

# Capitalized words that look like a name before a year but are not authors:
# "(Summer 2019)", "(Room 2010)", "the Clean Air Act (1970)", "Hurricane Katrina (2005)".
_NOT_AUTHOR = frozenset(
    """
    January February March April May June July August September October November December
    Jan Feb Mar Apr Jun Jul Aug Sep Sept Oct Nov Dec Spring Summer Fall Autumn Winter
    Monday Tuesday Wednesday Thursday Friday Saturday Sunday Christmas Easter
    In Since Before After Until Till Through From By During Circa Around About Approximately
    Ca Est Established Founded Born Died Published Updated Revised Copyright Released Opened Closed
    Built Completed Started Ended Began Year Years Fiscal FY Class Cohort Graduates Version Edition
    Ed Eds Release Census Vision Windows Office Excel Word Python Java Late Early Mid End Present
    Now Today AD BC CE BCE Room Building Bldg Suite Route Highway Model Type Unit Lot Box Station
    Gate Platform Flight Bus Line Code Form Item Part Step Level Grade Phase Stage Vol Volume Issue
    No Number Page Pages Exhibit Plate Catalog Catalogue Opus Op Article Rule Section Title Act Acts
    Public Law Laws Bill Exercise Problem Question Example Figure Fig Table Chapter Appendix
    Equation Eq Lesson Module Week Day Course Team Engine Prize Award Awards Cup Games Olympics
    Olympic Expo Exposition Exhibition Fair Festival Conference Summit Treaty Agreement Accord
    Accords Convention Protocol Declaration Report Survey Study Trial Revolution War Wars Crisis
    Depression Recession Pandemic Epidemic Outbreak Earthquake Hurricane Storm Tsunami Flood
    Famine Election Elections Constitution Amendment Era Period Age Dynasty Empire Republic
    Movement Plan Program Programme Project Mission Battle Siege Invasion Season Series Episode
    Draft Reprint Session Term Semester Quarter Total Population Estimated Approx Firmware Apollo
    The A An This That These Those His Her Its Their Our My Your Fiscal Budget Bowl Wave Round Tier
    Group Batch Week Weeks Month Months Decade Century Spring Term Fall Term
    London Paris Berlin Rome Vienna Madrid Moscow Tokyo Beijing Peking Boston Chicago Philadelphia
    Geneva Amsterdam Stockholm Copenhagen Oslo Athens Cairo Delhi Mumbai Bombay Shanghai Sydney
    Toronto York Kyoto Prague Budapest Warsaw Lisbon Dublin Edinburgh Brussels Florence Venice Milan
    Munich Hamburg Zurich Barcelona Seoul Jerusalem Istanbul Constantinople Baghdad Tehran Mexico
    Brazil China Japan India France Germany England Britain Italy Spain Russia America Africa
    Europe Asia Canada Australia Egypt Greece Ireland Scotland Wales Texas California Florida
    """.split()
)
_DETERMINERS = frozenset("the a an this that these those his her its their our my your".split())
_REF_LEADS = frozenset(
    "in see from by of cf cf. to and also e.g. e.g., compare following per after like than with".split()
)
_INTERVAL_LEADS = frozenset(
    "interval intervals range ranges domain on over within between matrix vector array list element index".split()
)


def _is_author_year(content: str) -> bool:
    if not _YEARISH.search(content) or not _CITE_CONTENT.fullmatch(content):
        return False
    return not any(w in _NOT_AUTHOR for w in _CAPWORD_RE.findall(content))


def _narrative_citation(text: str, pos: int) -> bool:
    """Is the ``(2019)`` at ``pos`` preceded by an author name ("Smith (2019) showed")?"""
    pre = text[max(0, pos - 200) : pos]
    if _ETAL_END.search(pre):
        return True
    m = _AUTHOR_RUN_END.search(pre)
    if not m:
        return False
    for word in _CAPWORD_RE.findall(m.group()):
        if re.sub(r"['’]s$", "", word) in _NOT_AUTHOR:
            return False
    before = text[: pos - len(pre) + m.start()].rstrip()
    if not before or before[-1] in ".!?:;([“\"'‘,—–":
        return True
    lw = re.search(r"([^\W\d_][\w'’\-]*)\W*$", before)
    if not lw:
        return True
    word = lw.group(1)
    if word.lower() in _DETERMINERS:
        return False  # "the Smith (2019) study", "the Act (1970)"
    if word[0].isupper():
        # Only OK when that capital is the start of the sentence ("As Smith (2019) noted");
        # otherwise the name is part of a longer proper noun ("Clean Air Act (1970)").
        head = before[: lw.start()].rstrip()
        return (not head or head[-1] in ".!?:;([“\"'‘") and word not in _NOT_AUTHOR
    return True


def _ref_words(content: str) -> str:
    parts = [p.strip() for p in re.split(r"[,;]", content) if p.strip()]
    spoken = [re.sub(r"\s*[-–—]\s*", " to ", p) for p in parts]
    plural = len(parts) > 1 or any(" to " in p for p in spoken)
    if len(spoken) > 1:
        listed = ", ".join(spoken[:-1]) + " and " + spoken[-1]
    else:
        listed = spoken[0]
    return ("references " if plural else "reference ") + listed


def _remove_citations(text: str) -> str:
    if "(" in text and _YEARISH.search(text):
        text = _PAREN_GROUP.sub(lambda m: "" if _is_author_year(m.group(1)) else m.group(), text)
        if "(" in text:
            text = _NARRATIVE_YEAR.sub(
                lambda m: "" if _narrative_citation(m.string, m.start()) else m.group(), text
            )
    if "[" in text:
        if _YEARISH.search(text):
            text = _BRACKET_GROUP.sub(lambda m: "" if _is_author_year(m.group(1)) else m.group(), text)
        text = _NEEDED_TAG.sub("", text)
        text = _NUMERIC_CITE.sub(_numeric_citation, text)
    return text


def _numeric_citation(m: re.Match[str]) -> str:
    content = m.group(1)
    if any(n.startswith("0") for n in re.findall(r"\d+", content)):
        return m.group()  # "[0, 1]" is an interval
    before = m.string[: m.start()]
    stripped = before.rstrip()
    if stripped and stripped[-1] in "∈=:<>≤≥(":
        return m.group()
    glued = not m.group()[0].isspace() and before[-1:].isalnum()
    lw = re.search(r"([\w.,]+)$", stripped)
    word = lw.group(1).lower().rstrip(",") if lw else ""
    if word == "in" and re.search(r"(?:^|\s)[A-Za-z]\s+in$", stripped):
        return m.group()  # "x in [1, 10]"
    if glued:
        token = re.search(r"(\w+)$", before)
        if token and (len(token.group(1)) <= 1 or token.group(1).isdigit()):
            return m.group()  # "a[1]", "x[3]": an index, not a citation
    if word in _INTERVAL_LEADS:
        return m.group()
    if not glued and word.rstrip(".") in {"reference", "references", "ref", "refs"}:
        return " " + re.sub(r"\s*[-–—]\s*", " to ", content)
    if not glued and word in _REF_LEADS:
        return " " + _ref_words(content)  # "as shown in [3]" -> "as shown in reference 3"
    return ""


# --------------------------------------------------------------------------- abbreviations

_Rule = tuple["re.Pattern[str]", Callable[["re.Match[str]"], str], str]


def _rule(
    pattern: str, repl: str | Callable[[re.Match[str]], str], needle: str, *, end_period: bool = False
) -> _Rule:
    """An abbreviation rule; it only runs when ``needle`` occurs in the lowercased text."""
    regex = re.compile(pattern)
    if callable(repl):
        return regex, repl, needle

    def fn(m: re.Match[str]) -> str:
        out = _cap_like(m.group(), repl)
        if end_period and m.group().endswith(".") and _at_end(m):
            out += "."
        return out

    return regex, fn, needle


_ORDINAL_EDITION = (
    r"\d+(?:st|nd|rd|th)|[Ff]irst|[Ss]econd|[Tt]hird|[Ff]ourth|[Ff]ifth|[Ss]ixth|[Ss]eventh|[Ee]ighth|"
    r"[Nn]inth|[Tt]enth|[Ee]leventh|[Tt]welfth|[Nn]ew|[Rr]evised|[Ii]nternational|[Gg]lobal|[Ss]tudent|"
    r"[Ii]nstructor'?s|[Tt]eacher'?s|[Aa]nniversary|[Ss]pecial|[Ll]atest|[Cc]urrent|[Uu]pdated"
)

_ABBREVIATIONS: list[_Rule] = [
    _rule(r"(?<![\w.])[Ee]\.\s?g\.(?=[\s,;:)\]]|$)", "for example", "g."),
    _rule(r"(?<![\w.])[Ii]\.\s?e\.(?=[\s,;:)\]]|$)", "that is", "e."),
    _rule(r"\b[Ee]tc\.", "et cetera", "etc.", end_period=True),
    _rule(r"\b[Cc]f\.(?=\s)", "compare", "cf."),
    _rule(r"\b[Vv]s\.?(?=\s)", "versus", "vs"),
    _rule(rf"(\b[{_UP}][\w'’\-]*)\s+v\.\s+(?=[{_UP}])", lambda m: m.group(1) + " versus ", " v. "),
    _rule(r"\b[Vv]iz\.(?=[\s,])", "namely", "viz."),
    _rule(r"\b[Rr]efs\.(?=\s)", "references", "refs."),
    _rule(r"\b[Rr]ef\.(?=\s*\[?\d)", "reference", "ref."),
    _rule(r"\b[Aa]pprox\.(?=\s|$)", "approximately", "approx."),
    _rule(r"\b[Cc]a\.\s?(?=\d)", "circa ", "ca."),
    _rule(r"(?<![\w.])c\.\s?(?=\d{3,4}\b)", "circa ", "c."),
    _rule(r"\bet\s+al\b\.?", "and colleagues", "et", end_period=True),
    _rule(r"\b(?:[Ff]ig|FIG)(?P<s>[sS]?)\.\s*(?=\d|[IVX]+\b|\(|[A-Z]\d)",
          lambda m: _cap_like(m.group(), "Figure" + ("s " if m.group("s") else " ")), "fig"),
    _rule(r"\b[Ee]q(?:n)?(?P<s>s?)\.\s*(?=[\d(\[])",
          lambda m: _cap_like(m.group(), "Equation" + ("s " if m.group("s") else " ")), "eq"),
    _rule(r"\b[Cc]hs\.\s*(?=\d|[IVXLC]+\b)", "chapters ", "chs."),
    _rule(r"\b(?:Ch|Chap|ch|chap)\.\s*(?=\d|[IVXLC]+\b)", "chapter ", "ch"),
    _rule(r"\bSec(?P<s>s?)\.\s*(?=\d|[IVXLC]+\b)", lambda m: "Sections " if m.group("s") else "Section ", "sec"),
    _rule(r"§§\s*", "Sections ", "§§"),
    _rule(r"§\s*", "Section ", "§"),
    _rule(r"(?<![\w.])pp\.\s*(?=\d|[ivxlc]+\b)", "pages ", "pp."),
    _rule(r"(?<![\w.])p\.\s*(?=\d|[ivxlc]+\b)", "page ", "p."),
    _rule(r"\b[Vv]ol(?P<s>s?)\.\s*(?=\d|[IVXLC]+\b)",
          lambda m: _cap_like(m.group(), "volumes " if m.group("s") else "volume "), "vol"),
    _rule(r"\b[Nn]o(?P<s>s?)\.\s*(?=\d)",
          lambda m: _cap_like(m.group(), "numbers " if m.group("s") else "number "), "no"),
    _rule(rf"\b({_ORDINAL_EDITION})\s+[Ee]d\.", lambda m: m.group(1) + " edition", "ed."),
    _rule(r"\(\s*[Ee]d\.\s*\)", "(editor)", "ed."),
    _rule(r"\(\s*[Ee]ds\.\s*\)", "(editors)", "eds."),
    _rule(r"\b[Ee]ds\.(?=[\s,;)]|$)", "editors", "eds."),
    _rule(r"(?<![\w/])[Ww]/(?:[Oo]|out)(?![\w/])", "without", "w/"),
    _rule(r"(?<![\w/])[Ww]/(?=\s|[A-Za-z])", "with ", "w/"),
    _rule(r"\b([AaPp])\.\s?[Mm]\.", lambda m: m.group(1).upper() + " M" + ("." if _at_end(m) else ""), ".m."),
    _rule(r"\b((?:1[0-2]|0?[1-9])(?::[0-5]\d)?)([AaPp])[Mm]\b",
          lambda m: f"{m.group(1)} {m.group(2).upper()} M", "m"),
    _rule(r"\b((?:1[0-2]|0?[1-9]):[0-5]\d)\s([AaPp])[Mm]\b",
          lambda m: f"{m.group(1)} {m.group(2).upper()} M", ":"),
    _rule(r"\s*&\s*", " and ", "&"),
]


def _expand_abbreviations(text: str) -> str:
    lowered = text.lower()
    for regex, fn, needle in _ABBREVIATIONS:
        if needle in lowered:
            text = regex.sub(fn, text)
            lowered = text.lower()
    return text


# --------------------------------------------------------------------------- number ranges

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_RANGE = re.compile(rf"(?<![\w.,\-–—−/:+])({_NUM})(\s*)([–—-])(\s*)({_NUM})(?![\w\-–—−/]|[.,:]\d)")
_PHONE_WORDS = re.compile(r"(?i)\b(?:phone|tel|telephone|call|fax|cell|mobile|text|contact|dial|ext)\b")
_DECADE_RANGE = re.compile(r"\b(\d{2,4}s)\s?[–—]\s?(\d{2,4}s)\b")
_TIME_RANGE = re.compile(r"\b(\d{1,2}:\d{2})\s*[–—-]\s*(\d{1,2}:\d{2})\b")
# "Figure 2-3" is figure 3 of chapter 2, not a range.
_NUMBERED_THINGS = frozenset(
    """
    figure fig table tab equation eq eqn exhibit example exercise problem box panel plate chart graph map
    diagram activity lesson unit model case question item appendix theorem lemma definition formula
    step rule listing algorithm section sec chapter ch program code form
    """.split()
)


def _range_repl(m: re.Match[str]) -> str:
    a, sp1, dash, sp2, b = m.groups()
    if dash == "-" and (sp1 or sp2):
        return m.group()  # "10 - 2" is subtraction
    before = m.string[: m.start()]
    if dash == "-" and len(a) == 3 and len(b) == 4 and (
        re.search(r"\(\d{3}\)\s*$", before) or _PHONE_WORDS.search(before[-30:])
    ):
        return m.group()  # "(555) 123-4567"
    prev = re.search(r"([A-Za-z]+)\.?\s*$", before)
    if prev and prev.group(1).lower() in _NUMBERED_THINGS:
        return m.group()
    if len(b) > 1 and b[0] == "0" and b[1] != ".":
        return m.group()  # "2019-05" is a date
    va, vb = float(a.replace(",", "")), float(b.replace(",", ""))
    ascending = vb > va
    shortened = (
        len(b) < len(a) and a.isdigit() and b.isdigit() and int(b) > int(a[-len(b) :])
    )  # "1990-95"
    if not (ascending or shortened):
        return m.group()  # "won 3-2"
    joiner = "and" if prev and prev.group(1).lower() == "between" else "to"
    return f"{a} {joiner} {b}"


def _number_ranges(text: str) -> str:
    if not re.search(r"\d\w?\s*[–—-]\s*\d", text):
        return text
    text = _TIME_RANGE.sub(r"\1 to \2", text)
    text = _DECADE_RANGE.sub(r"\1 to \2", text)
    return _RANGE.sub(_range_repl, text)


# --------------------------------------------------------------------------- Roman numerals

_ROMAN_RE = re.compile(r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})")
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_ROMAN_KEYWORDS = (
    "chapter|part|section|unit|book|volume|vol\\.?|appendix|act|scene|phase|type|class|stage|article|"
    "title|level|grade|tier|canto|lesson|module|round|plate|psalm|division|category"
)
_ROMAN_AFTER_KEYWORD = re.compile(rf"\b((?i:{_ROMAN_KEYWORDS}))(\s+)([IVXLCDM]+)\b(?!['’]\w)")
_ROMAN_LIST = re.compile(
    rf"\b((?i:(?:{_ROMAN_KEYWORDS})s))(\s+)([IVXLCDM]+(?:\s*(?:,|and|&|or|to|through|–|-)\s*[IVXLCDM]+)+)\b"
)
_WORLD_WAR = re.compile(r"\b(World\s+War|WORLD\s+WAR)\s+(I{1,3}|IV)\b(?!['’]\w)")
# Words after "Chapter I" that show "I" is the pronoun ("In this Chapter I argue").
_PRONOUN_FOLLOWERS = frozenset(
    """
    am will would have had was think believe argue describe discuss show want can could should must do
    did hope say said use used found find present propose consider review explain examine focus look
    also then first now shall may might mean need try tried wrote write suggest note begin began turn
    """.split()
)
_REGNAL_NAMES = (
    "Henry|Edward|Richard|George|William|James|Charles|John|Elizabeth|Mary|Anne|Victoria|Louis|Philip|"
    "Philippe|Peter|Catherine|Napoleon|Alexander|Nicholas|Frederick|Ferdinand|Leopold|Francis|Franz|Pius|"
    "Leo|Gregory|Innocent|Benedict|Clement|Boniface|Urban|Paul|Wilhelm|Otto|Rudolf|Ivan|Constantine|"
    "Justinian|Ramesses|Rameses|Ramses|Thutmose|Amenhotep|Ptolemy|Seleucus|Antiochus|Darius|Xerxes|"
    "Cyrus|Artaxerxes|Mehmed|Suleiman|Selim|Murad|Ahmed|Carlos|Juan|Felipe|Alfonso|Fernando|Sancho|"
    "Pedro|Manuel|Afonso|Joseph|Maximilian|Sigismund|Casimir|Stanislaus|Gustav|Gustavus|Christian|"
    "Haakon|Olaf|Harald|Magnus|Eric|Erik|Valdemar|Canute|Malcolm|David|Robert|Duncan|Kenneth|"
    "Stephen|Matilda|Harold|Alfred|Edmund|Ethelred|Athelstan|Edgar|Llywelyn|Hugh|Lothair|Pepin|Childeric|"
    "Clovis|Conrad|Lothar|Sixtus|Julius|Adrian|Hadrian|Celestine|Martin|Nicolas|Eugene|Eugenius|"
    "Sergius|Theodore|Basil|Michael|Andronikos|Manuel|Alexios|Isaac|Romanos|Leopold|Albert|Baudouin|"
    "Willem|Juliana|Beatrix|Margrethe|Carl|Akihito|Rama|Kamehameha|Mohammed|Muhammad|Hussein|Abdullah|"
    "Faisal|Farouk|Fuad|Reza|Ludwig|Augustus|Tutankhamun|Sobekhotep|Senusret|Amenemhat|Sesostris|"
    "Nebuchadnezzar|Ashurbanipal|Sargon|Shalmaneser|Tiglath-Pileser|Sennacherib|Esarhaddon|Antiochus"
)
_REGNAL = re.compile(
    rf"\b((?i:{_REGNAL_NAMES}))\s+([IVXL]+)(['’]s)?\b(?!['’]\w)"
)
_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORD_IRREGULAR = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
                  "nine": "ninth", "twelve": "twelfth"}


def _roman_value(s: str) -> int | None:
    s = s.upper()
    if not s or not _ROMAN_RE.fullmatch(s):
        return None
    total = 0
    for i, ch in enumerate(s):
        v = _ROMAN_VALUES[ch]
        total += -v if i + 1 < len(s) and _ROMAN_VALUES[s[i + 1]] > v else v
    return total


def _cardinal_word(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + ("-" + _ONES[ones] if ones else "")


def _ordinal_word(n: int) -> str:
    words = _cardinal_word(n)
    head, sep, last = words.rpartition("-")
    if last in _ORD_IRREGULAR:
        last = _ORD_IRREGULAR[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return head + sep + last


def _keyword_roman(m: re.Match[str]) -> str:
    keyword, space, numeral = m.groups()
    if not keyword[0].isupper():
        return m.group()  # "the third act I saw"
    if len(numeral) == 1 and numeral in "CDLM":
        return m.group()  # "Class C", "Appendix D" are letters
    if keyword.lower() == "appendix" and len(numeral) == 1:
        return m.group()
    if numeral == "I":
        after = re.match(r"\s+([A-Za-z']+)", m.string[m.end() :])
        if after and after.group(1).lower() in _PRONOUN_FOLLOWERS:
            return m.group()  # "In this Chapter I argue"
    value = _roman_value(numeral)
    return f"{keyword}{space}{value}" if value else m.group()


def _roman_list(m: re.Match[str]) -> str:
    if not m.group(1)[0].isupper():
        return m.group()

    def one(r: re.Match[str]) -> str:
        value = _roman_value(r.group())
        return str(value) if value else r.group()

    numerals = re.sub(r"[IVXLCDM]+", one, m.group(3))
    numerals = re.sub(r"\s*[–-]\s*", " to ", numerals)
    return m.group(1) + m.group(2) + numerals


def _regnal(m: re.Match[str]) -> str:
    name, numeral, possessive = m.group(1), m.group(2), m.group(3) or ""
    if not name[0].isupper():
        return m.group()
    value = _roman_value(numeral)
    if not value or value > 99:
        return m.group()
    if numeral == "I":
        after = re.match(r"\s+([A-Za-z']+)", m.string[m.end() :])
        if after and after.group(1).lower() in _PRONOUN_FOLLOWERS:
            return m.group()  # "Then Mary I think..."
    return f"{name} the {_ordinal_word(value).capitalize()}{possessive}"


def _roman_numerals(text: str) -> str:
    if not re.search(r"\b[IVXLCDM]+\b", text):
        return text
    text = _WORLD_WAR.sub(lambda m: f"{m.group(1)} {_roman_value(m.group(2))}", text)
    text = _ROMAN_LIST.sub(_roman_list, text)
    text = _ROMAN_AFTER_KEYWORD.sub(_keyword_roman, text)
    return _REGNAL.sub(_regnal, text)


# --------------------------------------------------------------------------- symbols

_GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ϵ": "epsilon", "ζ": "zeta",
    "η": "eta", "θ": "theta", "ϑ": "theta", "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu", "ν": "nu",
    "ξ": "xi", "ο": "omicron", "π": "pi", "ϖ": "pi", "ρ": "rho", "ϱ": "rho", "σ": "sigma", "ς": "sigma",
    "τ": "tau", "υ": "upsilon", "φ": "phi", "ϕ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
    "Α": "alpha", "Β": "beta", "Γ": "gamma", "Δ": "delta", "∆": "delta", "Ε": "epsilon", "Ζ": "zeta",
    "Η": "eta", "Θ": "theta", "Ι": "iota", "Κ": "kappa", "Λ": "lambda", "Μ": "mu", "Ν": "nu", "Ξ": "xi",
    "Ο": "omicron", "Π": "pi", "Ρ": "rho", "Σ": "sigma", "Τ": "tau", "Υ": "upsilon", "Φ": "phi", "Χ": "chi",
    "Ψ": "psi", "Ω": "omega",
}
_GREEK_RE = re.compile("[" + "".join(_GREEK) + "]")
_GREEK_NAMES = frozenset(_GREEK.values())

_FRACTIONS = {
    "½": ("one half", "a half"), "⅓": ("one third", "a third"), "⅔": ("two thirds", "two thirds"),
    "¼": ("one quarter", "a quarter"), "¾": ("three quarters", "three quarters"),
    "⅕": ("one fifth", "a fifth"), "⅖": ("two fifths", "two fifths"), "⅗": ("three fifths", "three fifths"),
    "⅘": ("four fifths", "four fifths"), "⅙": ("one sixth", "a sixth"), "⅚": ("five sixths", "five sixths"),
    "⅐": ("one seventh", "a seventh"), "⅛": ("one eighth", "an eighth"), "⅜": ("three eighths", "three eighths"),
    "⅝": ("five eighths", "five eighths"), "⅞": ("seven eighths", "seven eighths"),
    "⅑": ("one ninth", "a ninth"), "⅒": ("one tenth", "a tenth"),
}
_FRACTION_RE = re.compile(r"(\d)?\s?([" + "".join(_FRACTIONS) + "])")
_FRACTION_SLASH = re.compile(r"(?<!\d)(\d{1,2})\u2044(\d{1,2})(?!\d)")

_SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱˣʸᵀ"
_SUP_MAP = str.maketrans(_SUPERSCRIPTS, "0123456789+-=()nixyT")
_SUBSCRIPTS = "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₕₖₗₘₙₚₛₜᵢⱼ"
_SUB_MAP = str.maketrans(_SUBSCRIPTS, "0123456789+-=()aeoxhklmnpstij")
_SUP_ORDINAL = re.compile(r"(\d)(ˢᵗ|ⁿᵈ|ʳᵈ|ᵗʰ)")
_FORMULA = re.compile(
    r"(?<![\w])((?:[A-Z][a-z]?|[()\[\]])(?:[A-Z][a-z]?|[()\[\]]|[₀-₉]+)*)([⁰¹²³⁴⁵⁶⁷⁸⁹]*[⁺⁻])?(?![\w])"
)
_SUP_RUN = re.compile("[" + _SUPERSCRIPTS + "]+")
_SUB_RUN = re.compile("[" + _SUBSCRIPTS + "]+")
_PROSE_POWER = re.compile(r"(?<![\w^\\])([A-Za-z]|\d+(?:\.\d+)?|\))\^(\{[^{}]{1,40}\}|[-−+]?[A-Za-z0-9]+)")
_PROSE_LATEX = re.compile(
    r"\\(?:frac|dfrac|tfrac|binom|sqrt)(?:\[[^\]]{1,5}\])?\{[^{}]{1,80}\}(?:\{[^{}]{1,80}\})?"
)
_LATEX_CMD = re.compile(r"\\([A-Za-z]+)")
_LATEX_GREEK = {
    name: name for name in (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau "
        "upsilon phi chi psi omega"
    ).split()
}

# Units after a number. Values are (singular, plural).
_UNITS: dict[str, tuple[str, str]] = {}


def _add_units(spec: str) -> None:
    for line in spec.strip().splitlines():
        keys, _, names = line.partition("=")
        singular, _, plural = names.strip().partition("|")
        for key in keys.split():
            _UNITS[key] = (singular.strip(), (plural or singular + "s").strip())


_add_units(
    """
    nm = nanometer
    µm μm um = micrometer
    mm = millimeter
    cm = centimeter
    dm = decimeter
    km = kilometer
    m = meter
    mg = milligram
    µg μg mcg ug = microgram
    ng = nanogram
    pg = picogram
    kg = kilogram
    g = gram
    mL ml = milliliter
    µL μL uL µl μl = microliter
    dL dl = deciliter
    cL cl = centiliter
    kL = kiloliter
    L = liter
    mol = mole
    mmol = millimole
    µmol μmol = micromole
    nmol = nanomole
    M = molar|molar
    mM = millimolar|millimolar
    µM μM uM = micromolar|micromolar
    nM = nanomolar|nanomolar
    Hz = hertz|hertz
    kHz = kilohertz|kilohertz
    MHz = megahertz|megahertz
    GHz = gigahertz|gigahertz
    J = joule
    kJ = kilojoule
    MJ = megajoule
    cal = calorie
    kcal = kilocalorie
    eV = electron volt
    keV = kilo electron volt
    MeV = mega electron volt
    GeV = giga electron volt
    W = watt
    mW = milliwatt
    kW = kilowatt
    MW = megawatt
    GW = gigawatt
    Wh = watt hour
    kWh = kilowatt hour
    V = volt
    mV = millivolt
    µV μV = microvolt
    kV = kilovolt
    mA = milliamp
    µA μA = microamp
    Ω = ohm
    kΩ = kilohm
    MΩ = megohm
    N = newton
    kN = kilonewton
    Pa = pascal
    kPa = kilopascal
    MPa = megapascal
    GPa = gigapascal
    hPa = hectopascal
    atm = atmosphere
    mmHg = millimeter of mercury|millimeters of mercury
    psi = P S I|P S I
    ms = millisecond
    µs μs = microsecond
    ns = nanosecond
    min = minute
    hr hrs = hour
    yr yrs = year
    wk wks = week
    K = kelvin|kelvin
    Da = dalton
    kDa = kilodalton
    bp = base pair
    kb kbp = kilobase
    Mb Mbp = megabase
    kB KB = kilobyte
    MB = megabyte
    GB = gigabyte
    TB = terabyte
    ft = foot|feet
    mi = mile
    yd = yard
    lb lbs = pound
    oz = ounce
    gal = gallon
    qt = quart
    mph = mile per hour|miles per hour
    kph = kilometer per hour|kilometers per hour
    rpm = revolution per minute|revolutions per minute
    bpm = beat per minute|beats per minute
    dB = decibel
    ha = hectare
    cc = cubic centimeter
    Å = angstrom
    lx = lux|lux
    Gy = gray
    Sv = sievert
    mSv = millisievert
    Bq = becquerel
    """
)
# The unit after a "/" ("m/s", "mg/kg") is read in the singular.
_PER_UNITS: dict[str, str] = {k: v[0] for k, v in _UNITS.items()}
_PER_UNITS.update({"s": "second", "sec": "second", "h": "hour", "d": "day", "day": "day", "min": "minute",
                   "yr": "year", "wk": "week", "mo": "month", "dL": "deciliter", "cell": "cell"})
# Units that are also common abbreviations after money ("$5 m", "$300K").
_MONEY_AMBIGUOUS = frozenset({"m", "M", "K", "B", "g"})


def _unit_alternation(keys: list[str]) -> str:
    return "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))


_UNIT_AFTER_NUMBER = re.compile(
    rf"(?<![\w.])([$£€¥]?)(\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)\s?"
    rf"({_unit_alternation(list(_UNITS))})"
    rf"(?:/({_unit_alternation(list(_PER_UNITS))}))?([²³])?(?![\w/])"
)
_UNIT_AFTER_PER = re.compile(
    rf"(?:(?<=\bper )|(?<=[a-z]{{3}})/)({_unit_alternation([k for k in _UNITS if len(k) > 1])})([²³])?(?![\w/])"
)
_SPECIAL_UNITS = re.compile(
    r"(?<![\w])(" + "|".join(re.escape(k) for k in sorted(_UNITS, key=len, reverse=True) if re.search("[µμ]", k))
    + r")(?![\w])"
)
_DEGREES = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s?(?:°\s?([CFK])\b|(℃)|(℉)|°)")


def _unit_repl(m: re.Match[str]) -> str:
    money, number, unit, per, power = m.groups()
    if money and unit in _MONEY_AMBIGUOUS:
        return m.group()
    if unit in ("K", "M") and m.group(3) and not re.search(r"\d\s", m.group()):
        return m.group()  # "10K run", "5M users"
    singular, plural = _UNITS[unit]
    one = number in ("1", "1.0")
    name = singular if one else plural
    if power and not per:
        name = ("square " if power == "²" else "cubic ") + name
    out = f"{money}{number} {name}"
    if per:
        out += f" per {_PER_UNITS[per]}"
        if power:
            out += " squared" if power == "²" else " cubed"
    return out


def _per_unit_repl(m: re.Match[str]) -> str:
    unit, power = m.groups()
    name = _UNITS[unit][0]
    if power:
        name = ("square " if power == "²" else "cubic ") + name
    prefix = "" if m.string[: m.start()].endswith("per ") else " per "
    return prefix + name


def _degrees_repl(m: re.Match[str]) -> str:
    number, scale, celsius, fahrenheit = m.groups()
    unit = "degree" if number in ("1", "-1", "1.0") else "degrees"
    if scale == "K":
        return f"{number} kelvin"
    if scale == "C" or celsius:
        return f"{number} {unit} Celsius"
    if scale == "F" or fahrenheit:
        return f"{number} {unit} Fahrenheit"
    return f"{number} {unit}"


def _formula_repl(m: re.Match[str]) -> str:
    body, charge = m.group(1), m.group(2)
    if not charge and not re.search("[₀-₉]", body):
        return m.group()
    parts = re.findall(r"[A-Z][a-z]?|[₀-₉]+", body)
    words = [p.translate(_SUB_MAP) for p in parts]
    if charge:
        digits = charge[:-1].translate(_SUP_MAP)
        sign = "plus" if charge[-1] == "⁺" else "minus"
        words.append(f"{digits} {sign}".strip())
    return " ".join(words)


def _superscript_repl(m: re.Match[str]) -> str:
    run = m.group().translate(_SUP_MAP)
    before = m.string[: m.start()]
    prev = before[-1:]
    if not prev or prev.isspace() or prev in ".,;:!?\"'”’»":
        return "" if re.fullmatch(r"[\d,\-]+", run) else " " + run  # a footnote marker
    if prev.isalpha():
        word = re.search(r"([^\W\d_]+)$", before)
        if word and len(word.group(1)) >= 3 and word.group(1) not in _UNITS and re.fullmatch(r"[\d,]+", run):
            return ""  # "cells¹": footnote marker after a word
    if run == "2":
        return " squared"
    if run == "3":
        return " cubed"
    if run == "T":
        return " transpose"
    if run in ("+", "-"):
        return " plus" if run == "+" else " minus"
    spoken = run.replace("-", " minus ").replace("+", " plus ").replace("=", " equals ")
    return " to the power of " + _WS.sub(" ", spoken).strip()


def _greek_repl(m: re.Match[str]) -> str:
    name = _GREEK[m.group()]
    s, i, j = m.string, m.start(), m.end()
    left = " " if i > 0 and s[i - 1].isalnum() else ""
    right = " " if j < len(s) and s[j].isalnum() else ""
    return left + name + right


def _fraction_repl(m: re.Match[str]) -> str:
    whole, frac = m.groups()
    standalone, after_whole = _FRACTIONS[frac]
    return f"{whole} and {after_whole}" if whole else f" {standalone}"


_DENOMINATORS = {2: ("half", "halves"), 4: ("quarter", "quarters")}


def _fraction_slash_repl(m: re.Match[str]) -> str:
    num, den = int(m.group(1)), int(m.group(2))
    if den < 2:
        return m.group()
    singular, plural = _DENOMINATORS.get(den, (_ordinal_word(den) if den < 100 else str(den), ""))
    plural = plural or singular + "s"
    return f"{_cardinal_word(num) if num < 100 else num} {singular if num == 1 else plural}"


_NUMBER_TOKEN = re.compile(r"^[-−]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)%?$")


def _is_operand(token: str) -> bool:
    """A number, a single-letter variable, a Greek-letter name, or a bracketed expression."""
    token = token.strip("$£€")
    if not token:
        return False
    if _NUMBER_TOKEN.match(token):
        return True
    core = token.strip("'′")
    return (
        (len(core) == 1 and core.isalpha())
        or core in _GREEK_NAMES
        or core in ("squared", "cubed")
        or token[-1] in ")]"
        or token[0] in "(["
    )


def _left_token(s: str, i: int) -> str:
    m = re.search(r"(\S+)\s*$", s[:i])
    return m.group(1) if m else ""


def _right_token(s: str, j: int) -> str:
    m = re.match(r"\s*([^\s,;]+)", s[j:])
    return m.group(1).rstrip(".") if m else ""


def _comparison(words_with_left: str, words_without: str, *, strict: bool) -> Callable[[re.Match[str]], str]:
    """Replacement for <, >, ≤, ≥: phrased by whether there is a left operand.

    With ``strict``, the symbol is only spoken between operands (ASCII < and >, which
    are also used as arrows and quote marks); otherwise it is dropped.
    """

    def fn(m: re.Match[str]) -> str:
        left = _is_operand(_left_token(m.string, m.start()))
        right = _is_operand(_right_token(m.string, m.end()))
        if left and (right or not strict):
            return f" {words_with_left} "
        if right:
            return f" {words_without} "
        return " " if strict else f" {words_with_left} "

    return fn


def _between_operands(spoken: str, otherwise: str) -> Callable[[re.Match[str]], str]:
    def fn(m: re.Match[str]) -> str:
        if _is_operand(_left_token(m.string, m.start())) and _is_operand(_right_token(m.string, m.end())):
            return f" {spoken} "
        return otherwise

    return fn


def _spaced_hyphen(m: re.Match[str]) -> str:
    """" - " is minus between variables, a range between rising numbers, else a dash pause."""
    left = _left_token(m.string, m.start())
    right = _right_token(m.string, m.end())
    left_num, right_num = _NUMBER_TOKEN.match(left), _NUMBER_TOKEN.match(right)
    if left_num and right_num:
        if "=" in m.string or " equals " in m.string:
            return " minus "
        try:
            rising = float(left.replace(",", "").rstrip("%")) < float(right.replace(",", "").rstrip("%"))
        except ValueError:
            rising = False
        return " to " if rising else " - "
    if _is_operand(left) and _is_operand(right):
        return " minus "
    if left[-1:].isalnum() and right[:1].isalnum():
        return ", "  # "The cell - the basic unit - is"
    return " - "


def _latex_word(name: str) -> str:
    if name in _MATH_GREEK:
        return _MATH_GREEK[name]
    if name in _MATH_SYMBOLS:
        return _MATH_SYMBOLS[name][1]
    return _MATH_FUNCS.get(name, name)


def _times_repl(m: re.Match[str]) -> str:
    s, i, j = m.string, m.start(), m.end()
    if 0 < i and j < len(s) and s[i - 1].isdigit() and s[j].isdigit():
        return " by "  # "3×3 grid"
    return " times "


def _approx_repl(m: re.Match[str]) -> str:
    if _is_operand(_left_token(m.string, m.start())):
        return " is approximately equal to "
    return " approximately "


def _tilde_repl(m: re.Match[str]) -> str:
    right = _right_token(m.string, m.end())
    if _NUMBER_TOKEN.match(right.strip("$£€")):
        return " about "
    if _is_operand(_left_token(m.string, m.start())) and _is_operand(right):
        return " is similar to "
    return " "


def _minus_repl(m: re.Match[str]) -> str:
    left = _is_operand(_left_token(m.string, m.start()))
    if not left and m.end() < len(m.string) and m.string[m.end()].isdigit():
        return "-"  # "−5": voices read "-5" as "minus five"
    return " minus "


def _sqrt_repl(m: re.Match[str]) -> str:
    kind, arg = m.group(1), m.group(2).strip()
    root = {"√": "the square root of", "∛": "the cube root of", "∜": "the fourth root of"}[kind]
    if arg.startswith("(") and arg.endswith(")") and len(arg) > 2:
        arg = arg[1:-1]
    before = m.string[: m.start()]
    times = "times " if before[-1:].isalnum() else ""
    return f" {times}{root} {arg} "


_SIMPLE_SYMBOLS = [
    ("±", " plus or minus "), ("∓", " minus or plus "), ("÷", " divided by "), ("≠", " is not equal to "),
    ("≡", " is equivalent to "), ("≅", " is congruent to "), ("∝", " is proportional to "),
    ("⇌", " is in equilibrium with "), ("⇄", " is in equilibrium with "), ("⟶", " leads to "),
    ("→", " leads to "), ("⟵", " comes from "), ("←", " comes from "), ("↔", " corresponds to "),
    ("⟷", " corresponds to "), ("⇒", " implies "), ("⟹", " implies "), ("⇐", " is implied by "),
    ("⇔", " if and only if "), ("⟺", " if and only if "), ("↑", " up "), ("↓", " down "),
    ("∞", " infinity "), ("∑", " sum of "), ("∏", " product of "), ("∫", " integral of "), ("∬", " double integral of "),
    ("∮", " contour integral of "), ("∂", " partial "), ("∇", " del "), ("∈", " is in "), ("∉", " is not in "),
    ("⊂", " is a subset of "), ("⊆", " is a subset of or equal to "), ("⊃", " is a superset of "),
    ("⊇", " is a superset of or equal to "), ("∪", " union "), ("∩", " intersection "), ("∅", " the empty set "),
    ("∀", " for all "), ("∃", " there exists "), ("¬", " not "), ("∧", " and "), ("∨", " or "),
    ("∴", " therefore "), ("∵", " because "), ("∠", " angle "), ("△", " triangle "), ("⊥", " is perpendicular to "),
    ("∥", " is parallel to "), ("∘", " composed with "), ("‰", " per mille "), ("‱", " per ten thousand "),
    ("©", " copyright "), ("¶", " paragraph "), ("§", " section "), ("ℝ", " the real numbers "),
    ("ℤ", " the integers "), ("ℕ", " the natural numbers "), ("ℚ", " the rational numbers "),
    ("ℂ", " the complex numbers "), ("ħ", " h bar "), ("ℓ", " l "), ("µ", " micro "),
]
_BULLETS = re.compile("[•▪◦■□►▸▶▷➢➤➔✓✔✗✘☐☑☒●○◆◇★☆♦❖⁃∙‣⦿⦾]")
_DROP_CHARS = re.compile("[®™℠†‡※]")
_LEADING_DOT = re.compile(r"(?<![\w.])\.(\d)")
_ARROW_ASCII = re.compile(r"(?<=[\w\s)])\s*(?:-{1,2}>|=>)\s*(?=[\w\s(])")


_Step = tuple["re.Pattern[str]", "str | Callable[[re.Match[str]], str]", str]


def _steps(*steps: tuple[str, str | Callable[[re.Match[str]], str], str]) -> list[_Step]:
    """Compile (pattern, replacement, needle) steps; a step only runs when its needle is present."""
    return [(re.compile(pattern), repl, needle) for pattern, repl, needle in steps]


# Unicode symbols (only run on non-ASCII text).
_UNICODE_STEPS = _steps(
    (r"[≈≃]", _approx_repl, ""),
    (r"≤|⩽", _comparison("is less than or equal to", "at most", strict=False), ""),
    (r"≥|⩾", _comparison("is greater than or equal to", "at least", strict=False), ""),
    (r"×", _times_repl, "×"),
    (r"[·⋅]", _between_operands("times", ", "), ""),
    (r"−", _minus_repl, "−"),
    (r"∼", _tilde_repl, "∼"),
    (r"\s*(?:—|―|--)\s*", ", ", ""),
    (r"\s+–\s+", ", ", "–"),
    (r"–", "-", "–"),
    (r"…", "...", "…"),
    (r"(?<=\d)[′″]", "", ""),
    (r"″", " double prime ", "″"),
    (r"′", " prime ", "′"),
)
# ASCII operators and leftovers, in order.
_ASCII_STEPS = _steps(
    (r"\s*--\s*", ", ", "--"),
    (r"\.\s\.\s\.", "...", ". ."),
    (r"%", " percent ", "%"),
    (r"\b[Aa]nd\s?/\s?[Oo]r\b", lambda m: _cap_like(m.group(), "and or"), "/"),
    (r"\b([Hh]e|[Hh]is|[Hh]im)/(she|her)\b", lambda m: f"{m.group(1)} or {m.group(2)}", "/"),
    (_ARROW_ASCII.pattern, lambda m: " implies " if "=" in m.group() else " leads to ", ">"),
    (r"<=|=<", _comparison("is less than or equal to", "at most", strict=True), "="),
    (r">=|=>", _comparison("is greater than or equal to", "at least", strict=True), "="),
    (r"!=", _between_operands("is not equal to", " "), "!="),
    (r"={3,}", " ", "==="),
    (r"==?", " equals ", "="),
    (r"<", _comparison("is less than", "less than", strict=True), "<"),
    (r">", _comparison("is greater than", "more than", strict=True), ">"),
    (r"~", _tilde_repl, "~"),
    (r"(?<=\s)-(?=\s)", _spaced_hyphen, " - "),
    (r"\*", _between_operands("times", " "), "*"),
    (r"\+", _between_operands("plus", "+"), "+"),
    (r"\b([CF])#", r"\1 sharp", "#"),
    (r"#\s?(?=\d)", "number ", "#"),
    (r"(?<!\w)@(?=\w)", "at ", "@"),
    (r"[#|_\\{}^`<>~]", " ", ""),
)
_ASCII_TRIGGER = re.compile(r"[\d%=<>~*#^@|_\\{}/+-]")
_SQRT = re.compile(r"([√∛∜])\s?(\([^()]{1,60}\)|[\w.]+)")
_DEGREE_C = re.compile(r"°\s?C\b")
_DEGREE_F = re.compile(r"°\s?F\b")
_ANGSTROM = re.compile(r"(?<![\w])Å(?![\w])")
_SUP_ORDINAL_MAP = str.maketrans("ˢᵗⁿᵈʳʰ", "stndrh")


def _run_steps(text: str, steps: list[_Step]) -> str:
    for regex, repl, needle in steps:
        if not needle or needle in text:
            text = regex.sub(repl, text)
    return text


def _symbols(text: str) -> str:
    ascii_only = text.isascii()
    if ascii_only and not _ASCII_TRIGGER.search(text):
        return text
    if "\\" in text:
        text = _PROSE_LATEX.sub(lambda m: " " + verbalize_math(m.group()) + " ", text)
        text = _LATEX_CMD.sub(lambda m: f" {_latex_word(m.group(1))} ", text)
    if "^" in text:
        text = _PROSE_POWER.sub(lambda m: " " + verbalize_math(m.group()) + " ", text)
    if "." in text:
        text = _LEADING_DOT.sub(r"0.\1", text)  # "p < .05": voices say "dot zero five"
    if not ascii_only:
        text = _SUP_ORDINAL.sub(lambda m: m.group(1) + m.group(2).translate(_SUP_ORDINAL_MAP), text)
        if "°" in text or "℃" in text or "℉" in text:
            text = _DEGREES.sub(_degrees_repl, text)
            text = text.replace("℃", " degrees Celsius ").replace("℉", " degrees Fahrenheit ")
            text = _DEGREE_C.sub(" degrees Celsius ", text)
            text = _DEGREE_F.sub(" degrees Fahrenheit ", text)
            text = text.replace("°", " degrees ")
        text = _FRACTION_SLASH.sub(_fraction_slash_repl, text)
        text = _FRACTION_RE.sub(_fraction_repl, text)
        text = _SQRT.sub(_sqrt_repl, text)
    if any(c.isdigit() for c in text):
        text = _UNIT_AFTER_NUMBER.sub(_unit_repl, text)
    if "per " in text or "/" in text:
        text = _UNIT_AFTER_PER.sub(_per_unit_repl, text)
    if not ascii_only:
        text = _ANGSTROM.sub(" angstroms ", text)
        text = _SPECIAL_UNITS.sub(lambda m: " " + _UNITS[m.group()][1] + " ", text)
        text = _FORMULA.sub(_formula_repl, text)
        text = _SUP_RUN.sub(_superscript_repl, text)
        text = _SUB_RUN.sub(lambda m: " " + " ".join(m.group().translate(_SUB_MAP)) + " ", text)
        text = _GREEK_RE.sub(_greek_repl, text)
        for sym, words in _SIMPLE_SYMBOLS:
            if sym in text:
                text = text.replace(sym, words)
        text = _run_steps(text, _UNICODE_STEPS)
        text = _BULLETS.sub(" ", text)
        text = _DROP_CHARS.sub("", text)
    return _run_steps(text, _ASCII_STEPS)


# --------------------------------------------------------------------------- ALL CAPS

# Abbreviations that must stay spelled out even inside an ALL-CAPS heading.
_KEEP_CAPS = frozenset(
    """
    DNA RNA MRNA TRNA RRNA CDNA ATP ADP AMP GTP GDP NAD NADH NADP NADPH FAD FADH PCR HIV AIDS USA UK EU UN
    USSR NASA FBI CIA NSA CDC NIH FDA EPA IMF GNP CPI CEO CFO PHD MD MBA IQ ID TV PC CPU GPU RAM ROM USB URL
    HTML CSS HTTP HTTPS API SQL JSON XML PDF AI ML UV IR LED LCD DC AC MRI CT EEG ECG EKG ICU BMI BP ADHD OCD
    PTSD ASD HPV HBV HCV TB STD STI CRISPR SNP GMO IVF ESL GPA NCAA NFL NBA MLB NHL OK CO UNESCO UNICEF NAACP
    IUPAC HIPAA FERPA NIOSH OSHA MSDS SDS BCE CE BC GMT UTC EST PST CST DIY FAQ ISBN DOI ASCII LLC IEEE
    ACM APA MLA RSVP AMA NIMH WWII WWI
    """.split()
)
_CAPS_WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*")
_SHORT_ROMAN = re.compile(r"[IVX]{1,4}")


def _fix_caps(text: str) -> str:
    """Title-case shouted words ("CHAPTER ONE: THE CELL" -> "Chapter One: The Cell")."""
    if not re.search(r"[A-Z]{2}", text):
        return text
    words = list(_CAPS_WORD.finditer(text))
    out: list[str] = []
    pos = 0
    i = 0
    while i < len(words):
        if not _is_caps(words[i].group()):
            i += 1
            continue
        j = i
        while (
            j + 1 < len(words)
            and _is_caps(words[j + 1].group())
            and not re.search(r"[a-z]", text[words[j].end() : words[j + 1].start()])
        ):
            j += 1
        run = words[i : j + 1]
        plain = [w for w in run if _is_plain_word(w)]
        if len(plain) >= 2 or any(len(w.group()) > 4 for w in plain):
            for w in plain:
                out.append(text[pos : w.start()])
                out.append(_title(w.group()))
                pos = w.end()
        i = j + 1
    out.append(text[pos:])
    return "".join(out)


def _is_caps(word: str) -> bool:
    return word.isupper()


def _is_plain_word(m: re.Match[str]) -> bool:
    """An ALL-CAPS word that is an ordinary word rather than an acronym or formula."""
    word = m.group()
    s = m.string
    if len(word) < 2 or word.replace("’", "'").upper() in _KEEP_CAPS:
        return False
    if (m.start() > 0 and s[m.start() - 1].isdigit()) or (m.end() < len(s) and s[m.end()].isdigit()):
        return False  # "CO2", "H2O"
    if not re.search("[AEIOUY]", word) or _SHORT_ROMAN.fullmatch(word):
        return False  # "HTTP", "LDL", "II"
    return True


def _title(word: str) -> str:
    """"DON'T" -> "Don't", "WELL-KNOWN" -> "Well-Known"."""
    return "-".join(part[:1] + part[1:].lower() for part in word.split("-"))


# --------------------------------------------------------------------------- tidy


_TIDY_STEPS = _steps(
    (r"\(\s*(?:(?i:see|at|via|from|available at|link)\s*)?[,;:.]*\s*\)", "", "("),
    (r"\[\s*[,;:.]*\s*\]", "", "["),
    (r"\s+([,.;:!?)\]}])", r"\1", ""),
    (r"([(\[{])\s+", r"\1", ""),
    (r"([,;:])(?:\s*[,;:])+", r"\1", ""),
    (r"[,;:]\s*([.!?])", r"\1", ""),
    (r",(?=\S)(?!\d)", ", ", ","),
    (r"^[\s,;:.)\]}]+", "", ""),
)


def _tidy(text: str) -> str:
    return _WS.sub(" ", _run_steps(_WS.sub(" ", text), _TIDY_STEPS)).strip()


# --------------------------------------------------------------------------- math

_MATH_GREEK = {
    **{n: n for n in _LATEX_GREEK},
    "varepsilon": "epsilon", "vartheta": "theta", "varphi": "phi", "varrho": "rho", "varsigma": "sigma",
    "varpi": "pi", "varkappa": "kappa", "Gamma": "gamma", "Delta": "delta", "Theta": "theta",
    "Lambda": "lambda", "Xi": "xi", "Pi": "pi", "Sigma": "sigma", "Upsilon": "upsilon", "Phi": "phi",
    "Psi": "psi", "Omega": "omega",
}
# Command -> (kind, words). Kinds: "op" joins two operands, "rel" is a relation, "atom" is a value.
_MATH_SYMBOLS: dict[str, tuple[str, str]] = {
    "cdot": ("op", "times"), "times": ("op", "times"), "ast": ("op", "times"), "div": ("op", "divided by"),
    "pm": ("op", "plus or minus"), "mp": ("op", "minus or plus"), "le": ("rel", "is less than or equal to"),
    "leq": ("rel", "is less than or equal to"), "leqslant": ("rel", "is less than or equal to"),
    "ge": ("rel", "is greater than or equal to"), "geq": ("rel", "is greater than or equal to"),
    "geqslant": ("rel", "is greater than or equal to"), "neq": ("rel", "is not equal to"),
    "ne": ("rel", "is not equal to"), "approx": ("rel", "is approximately equal to"),
    "simeq": ("rel", "is approximately equal to"), "sim": ("rel", "is similar to"),
    "equiv": ("rel", "is equivalent to"), "cong": ("rel", "is congruent to"),
    "propto": ("rel", "is proportional to"), "lt": ("rel", "is less than"), "gt": ("rel", "is greater than"),
    "ll": ("rel", "is much less than"), "gg": ("rel", "is much greater than"), "to": ("rel", "goes to"),
    "rightarrow": ("rel", "goes to"), "longrightarrow": ("rel", "goes to"), "Rightarrow": ("rel", "implies"),
    "Longrightarrow": ("rel", "implies"), "implies": ("rel", "implies"), "Leftarrow": ("rel", "is implied by"),
    "leftarrow": ("rel", "comes from"), "gets": ("rel", "comes from"), "iff": ("rel", "if and only if"),
    "Leftrightarrow": ("rel", "if and only if"), "Longleftrightarrow": ("rel", "if and only if"),
    "leftrightarrow": ("rel", "corresponds to"), "rightleftharpoons": ("rel", "is in equilibrium with"),
    "mapsto": ("rel", "maps to"), "in": ("rel", "in"), "notin": ("rel", "not in"),
    "subset": ("rel", "is a subset of"), "subseteq": ("rel", "is a subset of or equal to"),
    "supset": ("rel", "is a superset of"), "supseteq": ("rel", "is a superset of or equal to"),
    "cup": ("op", "union"), "cap": ("op", "intersection"), "setminus": ("op", "minus"),
    "emptyset": ("atom", "the empty set"), "varnothing": ("atom", "the empty set"),
    "forall": ("rel", "for all"), "exists": ("rel", "there exists"), "neg": ("rel", "not"),
    "lnot": ("rel", "not"), "land": ("op", "and"), "wedge": ("op", "and"), "lor": ("op", "or"),
    "vee": ("op", "or"), "therefore": ("rel", "therefore"), "because": ("rel", "because"),
    "angle": ("atom", "angle"), "triangle": ("atom", "triangle"), "perp": ("rel", "is perpendicular to"),
    "parallel": ("rel", "is parallel to"), "circ": ("op", "composed with"), "infty": ("atom", "infinity"),
    "partial": ("atom", "partial"), "nabla": ("atom", "del"), "hbar": ("atom", "h bar"), "ell": ("atom", "l"),
    "ldots": ("atom", "and so on"), "cdots": ("atom", "and so on"), "dots": ("atom", "and so on"),
    "vdots": ("atom", "and so on"), "ddots": ("atom", "and so on"), "prime": ("atom", "prime"),
    "mid": ("rel", "given"), "deg": ("atom", "degrees"), "degree": ("atom", "degrees"),
    "%": ("atom", "percent"), "&": ("op", "and"), "$": ("atom", "dollars"), "#": ("atom", "number"),
    "_": ("atom", "underscore"), "bmod": ("op", "mod"), "pmod": ("op", "mod"), "mod": ("op", "mod"), "Re": ("atom", "the real part of"), "Im": ("atom", "the imaginary part of"),
}
_MATH_FUNCS = {
    "sin": "sine", "cos": "cosine", "tan": "tangent", "cot": "cotangent", "sec": "secant", "csc": "cosecant",
    "arcsin": "arc sine", "arccos": "arc cosine", "arctan": "arc tangent", "sinh": "hyperbolic sine",
    "cosh": "hyperbolic cosine", "tanh": "hyperbolic tangent", "log": "log", "ln": "natural log",
    "lg": "log", "exp": "exponential", "det": "determinant", "dim": "dimension", "ker": "kernel",
    "gcd": "greatest common divisor", "lcm": "least common multiple", "Pr": "probability", "arg": "argument",
    "var": "variance", "Var": "variance", "cov": "covariance", "Cov": "covariance",
    "tr": "trace", "rank": "rank", "sgn": "sign",
}
_MATH_BIGOPS = {
    "sum": "sum", "prod": "product", "coprod": "coproduct", "int": "integral", "iint": "double integral",
    "iiint": "triple integral", "oint": "contour integral", "bigcup": "union", "bigcap": "intersection",
    "lim": "limit", "limsup": "limit superior", "liminf": "limit inferior", "max": "maximum",
    "min": "minimum", "sup": "supremum", "inf": "infimum", "argmax": "arg max", "argmin": "arg min",
}
_MATH_ACCENTS = {
    "hat": "{} hat", "widehat": "{} hat", "bar": "{} bar", "overline": "{} bar", "tilde": "{} tilde",
    "widetilde": "{} tilde", "dot": "{} dot", "ddot": "{} double dot", "vec": "vector {}",
    "overrightarrow": "vector {}", "underline": "{}", "boxed": "{}", "cancel": "{}",
}
_MATH_TEXT = {"text", "textrm", "textnormal", "mbox", "textit", "textbf", "texttt", "textsf", "emph", "operatorname"}
_MATH_FONTS = {
    "mathbf", "mathrm", "mathit", "mathsf", "mathtt", "mathcal", "mathscr", "mathfrak", "boldsymbol", "bm",
    "pmb", "mathnormal", "mathop", "displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle",
}
_MATH_SETS = {"R": "the real numbers", "Z": "the integers", "N": "the natural numbers",
              "Q": "the rational numbers", "C": "the complex numbers"}
_MATH_SKIP = {
    "left", "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr", "Biggl",
    "Biggr", "middle", "quad", "qquad", "nonumber", "notag", "limits", "nolimits", ",", ";", ":", "!", " ",
    "displaystyle", "label", "tag", "hline", "noindent", "centering",
}
_MATH_DROP_ARG = {"label", "tag", "ref", "eqref", "hspace", "vspace", "phantom", "color", "textcolor"}
_MATH_ROOTS = {"2": "square", "3": "cube", "4": "fourth", "5": "fifth", "n": "n-th"}
_FUNC_LETTERS = frozenset("fghFGHPEL")
_LINEAR_FUNCS = re.compile(
    r"(?<![\\A-Za-z])(arcsin|arccos|arctan|sinh|cosh|tanh|sin|cos|tan|cot|sec|csc|log|ln|exp|sqrt|lim|det|gcd)(?![A-Za-z])"
)
_MATH_UNICODE = {
    "×": r"\times ", "·": r"\cdot ", "⋅": r"\cdot ", "÷": r"\div ", "±": r"\pm ", "∓": r"\mp ",
    "≤": r"\le ", "⩽": r"\le ", "≥": r"\ge ", "⩾": r"\ge ", "≠": r"\neq ", "≈": r"\approx ", "≃": r"\simeq ",
    "≡": r"\equiv ", "≅": r"\cong ", "∝": r"\propto ", "∞": r"\infty ", "∂": r"\partial ", "∇": r"\nabla ",
    "∑": r"\sum ", "∏": r"\prod ", "∫": r"\int ", "∬": r"\iint ", "∮": r"\oint ", "√": r"\sqrt ",
    "∛": r"\sqrt[3]", "∜": r"\sqrt[4]", "→": r"\to ", "⟶": r"\to ", "←": r"\leftarrow ", "⇒": r"\Rightarrow ",
    "⟹": r"\Rightarrow ", "⇔": r"\iff ", "⟺": r"\iff ", "↔": r"\leftrightarrow ", "⇌": r"\rightleftharpoons ",
    "∈": r"\in ", "∉": r"\notin ", "⊂": r"\subset ", "⊆": r"\subseteq ", "⊃": r"\supset ", "⊇": r"\supseteq ",
    "∪": r"\cup ", "∩": r"\cap ", "∅": r"\emptyset ", "∀": r"\forall ", "∃": r"\exists ", "¬": r"\neg ",
    "∧": r"\wedge ", "∨": r"\vee ", "∴": r"\therefore ", "∵": r"\because ", "∠": r"\angle ", "⊥": r"\perp ",
    "∥": r"\parallel ", "∘": r"\circ ", "°": r"^{\circ}", "′": "'", "″": "''", "−": "-", "–": "-", "—": "-",
    "…": r"\ldots ", "⋯": r"\cdots ", "ℝ": r"\mathbb{R}", "ℤ": r"\mathbb{Z}", "ℕ": r"\mathbb{N}",
    "ℚ": r"\mathbb{Q}", "ℂ": r"\mathbb{C}", "ħ": r"\hbar ", "ℓ": r"\ell ", "⟨": "", "⟩": "", "‖": r"\|",
    "µ": r"\mu ", "∣": r"\mid ", "∼": r"\sim ",
}
_GREEK_TO_CMD = {ch: "\\" + name + " " for ch, name in _GREEK.items()}
_MATH_TOKEN = re.compile(r"\\(?:[A-Za-z]+|.)|\d+(?:\.\d+)?|\.\d+|[A-Za-z]+|\s+|.", re.DOTALL)
_SHORT_WORDS = frozenset({"and", "or", "if", "for", "not", "the", "mod", "pH", "max", "min", "sum"})


class _MathItem:
    __slots__ = ("words", "kind", "complex", "func", "head")

    def __init__(
        self, words: str, kind: str = "atom", complex_: bool = False, func: bool = False, head: str = ""
    ) -> None:
        self.words = words
        self.kind = kind  # "atom", "op", "rel", "sep", "paren", "bigop"
        self.complex = complex_
        self.func = func  # a function name waiting for its argument: "sine" + "of x"
        self.head = head  # the single letter this atom is built on ("f" in "f prime"), for "f(x)"

    def but(self, words: str, complex_: bool | None = None) -> _MathItem:
        """A copy with new words (postfix operators keep the function-ness and head letter)."""
        return _MathItem(words, "atom", self.complex if complex_ is None else complex_, self.func, self.head)


class _MathParser:
    MAX_DEPTH = 40  # deeper nesting is read flat (and keeps us far from Python's recursion limit)

    def __init__(self, source: str) -> None:
        self.toks = _MATH_TOKEN.findall(source)
        self.i = 0
        self.depth = 0

    # -- token helpers
    def peek(self, skip_space: bool = True) -> str | None:
        j = self.i
        while skip_space and j < len(self.toks) and self.toks[j].isspace():
            j += 1
        return self.toks[j] if j < len(self.toks) else None

    def take(self) -> str | None:
        while self.i < len(self.toks) and self.toks[self.i].isspace():
            self.i += 1
        if self.i >= len(self.toks):
            return None
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def raw_group(self) -> str:
        """The literal text of a ``{...}`` argument (for \\text)."""
        if self.peek() != "{":
            tok = self.take()
            return tok or ""
        self.take()
        depth, out = 1, []
        while self.i < len(self.toks):
            tok = self.toks[self.i]
            self.i += 1
            if tok == "{":
                depth += 1
            elif tok == "}":
                depth -= 1
                if depth == 0:
                    break
            out.append(tok)
        return "".join(out)

    def optional_bracket(self) -> str | None:
        if self.peek() != "[":
            return None
        self.take()
        out = []
        while self.i < len(self.toks) and self.toks[self.i] != "]":
            out.append(self.toks[self.i])
            self.i += 1
        self.i += 1
        return "".join(out).strip()

    # -- grammar
    def parse(self, stops: frozenset[str] = frozenset()) -> tuple[str, bool]:
        items: list[_MathItem] = []
        self.depth += 1
        try:
            while True:
                tok = self.peek()
                if tok is None or tok in stops:
                    break
                item = self.item()
                if item is not None:
                    items.append(item)
        finally:
            self.depth -= 1
        return _join_math(items)

    @property
    def too_deep(self) -> bool:
        return self.depth > self.MAX_DEPTH

    def arg(self) -> tuple[str, bool]:
        """One argument: a {group}, a (group), or a single (possibly signed) token."""
        tok = self.peek()
        if tok == "{":
            self.take()
            words, cx = self.parse(frozenset({"}"}))
            self.take()
            return words, cx
        if tok in ("(", "["):
            item = self.item()
            return (item.words, item.complex) if item else ("", False)
        if tok in ("-", "+"):
            self.take()
            if self.too_deep:
                return "", False
            self.depth += 1
            try:
                words, _ = self.arg()
            finally:
                self.depth -= 1
            return ("minus " if tok == "-" else "plus ") + words, True
        item = self.item(postfix=False)
        if item is None:
            return "", False
        return item.words, item.complex or item.kind != "atom"

    def item(self, postfix: bool = True) -> _MathItem | None:
        tok = self.take()
        if tok is None:
            return None
        item = self.primary(tok)
        if item is None:
            return None
        if postfix:
            item = self.postfix(item)
        return item

    def primary(self, tok: str) -> _MathItem | None:
        if self.too_deep and tok in ("{", "(", "[", "|", "\\{", "\\lbrace", "\\|"):
            return None
        if tok == "{":
            words, cx = self.parse(frozenset({"}"}))
            self.take()
            return _MathItem(words, "atom", cx)
        if tok in ("(", "[") or tok == "\\{" or tok == "\\lbrace":
            close = {"(": ")", "[": "]"}.get(tok, "\\}")
            words, cx = self.parse(frozenset({close, "\\rbrace"}))
            self.take()
            return _MathItem(words, "paren", cx)
        if tok in (")", "]", "}", "\\}"):
            return None
        if tok in ("|", "\\|", "\\lvert", "\\rvert", "\\vert", "\\lVert", "\\rVert", "\\Vert"):
            close = {"|", "\\rvert", "\\vert"} if tok in ("|", "\\lvert", "\\vert") else {"\\|", "\\rVert", "\\Vert"}
            words, _ = self.parse(frozenset(close))
            self.take()
            what = "the absolute value of" if tok in ("|", "\\lvert", "\\vert") else "the norm of"
            return _MathItem(f"{what} {words},", "atom", True)
        if tok[0].isdigit() or (tok[0] == "." and len(tok) > 1):
            return _MathItem(tok if tok[0] != "." else "0" + tok)
        if tok.isalpha():
            if tok in ("max", "min") and self.peek(skip_space=False) in ("(", "_"):
                return self.bigop(tok)
            if tok in _MATH_FUNCS:
                return self.function(_MATH_FUNCS[tok])
            if tok == "sqrt":
                return self.command("sqrt")
            if tok == "mod":
                return _MathItem("mod", "op")
            if len(tok) >= 4 or tok in _SHORT_WORDS:
                return _MathItem(tok)  # a word: "mass", "rate", "max"
            if len(tok) == 1:
                return _MathItem(tok, head=tok)
            self.toks[self.i : self.i] = list(tok[1:])  # "xy" is x times y: read letter by letter
            return _MathItem(tok[0], head=tok[0])
        if tok.startswith("\\"):
            return self.command(tok[1:])
        return self.symbol(tok)

    def symbol(self, tok: str) -> _MathItem | None:
        table = {
            "+": ("op", "plus"), "-": ("op", "minus"), "*": ("op", "times"), "/": ("op", "over"),
            "=": ("rel", "equals"), "<": ("rel", "is less than"), ">": ("rel", "is greater than"),
            ",": ("sep", ","), ";": ("sep", ";"), "&": ("sep", ","), ":": ("sep", ","), "%": ("atom", "percent"),
            "!": ("atom", "factorial"), "'": ("atom", "prime"), "~": ("rel", "is similar to"),
        }
        if tok in table:
            kind, words = table[tok]
            if tok in "<>=" and self.peek(skip_space=False) == "=":
                self.take()
                words = {"<": "is less than or equal to", ">": "is greater than or equal to", "=": "equals"}[tok]
            return _MathItem(words, kind)
        if tok in ("$", ".", "\\", "^", "_", "?", '"', "`"):
            return None
        if tok in _GREEK:
            return _MathItem(_GREEK[tok])
        return _MathItem(tok) if tok.isalnum() else None

    def command(self, name: str) -> _MathItem | None:
        if name in _MATH_SKIP:
            if name in _MATH_DROP_ARG:
                self.raw_group()
            return None
        if name in _MATH_DROP_ARG:
            self.raw_group()
            return None
        if name == "\\":
            return _MathItem(";", "sep")
        if name in ("begin", "end"):
            env = self.raw_group().strip("*")
            if name == "begin" and "matrix" in env:
                return _MathItem("the matrix", "atom")
            if name == "begin" and env == "cases":
                return _MathItem("cases:", "sep")
            return _MathItem(",", "sep") if name == "end" else None
        if name in _MATH_GREEK:
            return _MathItem(_MATH_GREEK[name])
        if name in ("frac", "dfrac", "tfrac", "cfrac"):
            num, ncx = self.arg()
            den, dcx = self.arg()
            ncx = ncx and not re.fullmatch(r"(?:d|partial) ?\w*", num)
            dcx = dcx and not re.fullmatch(r"(?:d|partial) ?\w*", den)
            if ncx or dcx:
                return _MathItem(f"the fraction {num}, over {den},", "atom", True)
            return _MathItem(f"{num} over {den}", "atom", True)
        if name == "sqrt":
            index = self.optional_bracket()
            body, cx = self.arg()
            if index is None or index in _MATH_ROOTS:
                root = _MATH_ROOTS.get(index or "2", "square")
            else:
                index_words, _ = _MathParser(index).parse()
                root = f"{index_words}-th" if re.fullmatch(r"\w+", index_words) else "nth"
            return _MathItem(f"the {root} root of {body}" + ("," if cx else ""), "atom", True)
        if name == "binom":
            n, _ = self.arg()
            k, _ = self.arg()
            return _MathItem(f"{n} choose {k}", "atom", True)
        if name in _MATH_TEXT:
            return _MathItem(re.sub(r"\\[A-Za-z]*|[{}$^_]", " ", self.raw_group()).strip())
        if name == "mathbb":
            raw = re.sub(r"[\\{}]", "", self.raw_group()).strip()
            return _MathItem(_MATH_SETS.get(raw, raw))
        if name in _MATH_FONTS:
            if self.peek() == "{":
                words, cx = self.arg()
                return _MathItem(words, "atom", cx)
            return None
        if name in _MATH_ACCENTS:
            body, _ = self.arg()
            return _MathItem(_MATH_ACCENTS[name].format(body).strip())
        if name in _MATH_FUNCS:
            return self.function(_MATH_FUNCS[name])
        if name in _MATH_BIGOPS:
            return self.bigop(name)
        if name in _MATH_SYMBOLS:
            kind, words = _MATH_SYMBOLS[name]
            return _MathItem(words, kind)
        if len(name) == 1 and not name.isalpha():
            return None  # "\," "\;" spacing and other escapes
        return _MathItem(name)  # unknown command: drop the backslash

    def function(self, words: str) -> _MathItem:
        if self.peek() == "_":  # \log_2 x
            self.take()
            base, _ = self.arg()
            words = f"{words} base {base}"
        if self.peek() == "^":  # \sin^2 x
            self.take()
            power, _ = self.arg()
            words += {"2": " squared", "3": " cubed", "minus 1": " inverse"}.get(power, f" to the power {power}")
        return _MathItem(words, "atom", False, func=True)

    def bigop(self, name: str) -> _MathItem:
        lower = upper = None
        for _ in range(2):
            nxt = self.peek()
            if nxt == "_":
                self.take()
                lower, _cx = self.arg()
            elif nxt == "^":
                self.take()
                upper, _cx = self.arg()
            elif nxt == "\\limits" or nxt == "\\nolimits":
                self.take()
        noun = _MATH_BIGOPS[name]
        if name == "lim" or name in ("limsup", "liminf"):
            if lower:
                cond = lower.replace("goes to", "approaches")
                return _MathItem(f"the {noun} as {cond} of", "bigop")
            return _MathItem(f"the {noun} of", "bigop")
        if lower and upper:
            return _MathItem(f"the {noun} from {lower} to {upper} of", "bigop")
        if lower:
            return _MathItem(f"the {noun} over {lower} of", "bigop")
        if upper:
            return _MathItem(f"the {noun} up to {upper} of", "bigop")
        return _MathItem(f"the {noun} of", "bigop")

    def postfix(self, item: _MathItem) -> _MathItem:
        while True:
            nxt = self.peek()
            if nxt == "_":
                self.take()
                sub, _ = self.arg()
                if re.fullmatch(r"[A-Za-z]{2,3}", sub) and sub.lower() not in {"max", "min", "eff", "avg", "tot"}:
                    sub = " ".join(sub)
                item = item.but(f"{item.words} sub {sub}", True)
            elif nxt == "^":
                self.take()
                if self.peek() in ("\\circ",) or (self.peek() == "{" and self._group_is("\\circ")):
                    self.arg()
                    nxt2 = self.peek()
                    if nxt2 in ("C", "F"):
                        self.take()
                        item = _MathItem(f"{item.words} degrees {'Celsius' if nxt2 == 'C' else 'Fahrenheit'}")
                    else:
                        item = _MathItem(f"{item.words} degrees")
                    continue
                power, cx = self.arg()
                if power in ("2", "3"):
                    phrase = " squared" if power == "2" else " cubed"
                elif power in ("T", "top", "intercal"):
                    phrase = " transpose"
                elif power in ("prime", "'"):
                    phrase = " prime"
                elif power in ("*", "times", "ast"):
                    phrase = " star"
                else:
                    phrase = f" to the power of {power}" + ("," if cx else "")
                base = f"the quantity {item.words}," if item.kind == "paren" and item.complex else item.words
                item = item.but(base + phrase, True)
            elif nxt == "'":
                self.take()
                item = item.but(item.words + " prime")
                item.func = item.func or item.head in _FUNC_LETTERS
            elif nxt == "!" and self.peek(skip_space=False) == "!":
                self.take()
                item = item.but(item.words + " factorial")
            else:
                return item

    def _group_is(self, text: str) -> bool:
        j = self.i
        while j < len(self.toks) and self.toks[j].isspace():
            j += 1
        k = j + 1
        while k < len(self.toks) and self.toks[k].isspace():
            k += 1
        return k < len(self.toks) and self.toks[k] == text


def _join_math(items: list[_MathItem]) -> tuple[str, bool]:
    """Join parsed items into words; also report whether the result is more than one simple term."""
    out: list[str] = []
    atoms = 0
    has_op = any(item.complex for item in items) and len(items) == 1
    prev: _MathItem | None = None
    for item in items:
        words = item.words
        if item.kind == "paren":
            if prev is not None and prev.kind == "bigop":
                words += "," if item.complex else ""  # max(a, b)
            elif prev is not None and (
                prev.func or (prev.head and (prev.head in _FUNC_LETTERS or not item.complex))
            ):
                words = f"of {words}" + ("," if item.complex else "")  # f(x), sin(x), x(t)
            elif item.complex:
                lead = "times " if prev is not None and prev.kind in ("atom", "paren") else ""
                words = f"{lead}the quantity {words},"  # 2(x + 1)
            atoms += 1
        elif item.kind == "atom":
            if prev is not None and prev.func:
                words = "of " + words  # sin x
            atoms += 1
        elif item.kind in ("op", "rel", "bigop"):
            has_op = True
        out.append(words)
        prev = item
    text = _WS.sub(" ", " ".join(w for w in out if w)).strip()
    text = re.sub(r"\s+([,;])", r"\1", text)
    text = re.sub(r"([,;])(?:\s*[,;])+", r"\1", text)
    return text, has_op or atoms > 1


def verbalize_math(text: str) -> str:
    """Spoken English for LaTeX-ish or linear math ("x^2" -> "x squared")."""
    if not text or text.isspace():
        return ""
    src = text.strip()
    src = re.sub(r"^\s*\$+|\$+\s*$", "", src)
    src = re.sub(r"^\\[\[(]|\\[\])]$", "", src).strip()
    src = re.sub(r"\s*\((\d+(?:\.\d+)*[a-z]?)\)\s*$", "", src)  # trailing equation number "(3.2)"
    if not src.isascii():
        src = _SUP_RUN.sub(lambda m: "^{" + m.group().translate(_SUP_MAP) + "}", src)
        src = _SUB_RUN.sub(lambda m: "_{" + m.group().translate(_SUB_MAP) + "}", src)
        src = "".join(_MATH_UNICODE.get(ch) or _GREEK_TO_CMD.get(ch) or ch for ch in src)
    src = _LINEAR_FUNCS.sub(r"\\\1 ", src)
    try:
        words, _ = _MathParser(src).parse()
    except RecursionError:  # pathological nesting: read the symbols that are there
        words = re.sub(r"[\\{}^_]", " ", src)
    words = _WS.sub(" ", words)
    words = re.sub(r"\s+([,;.])", r"\1", words)
    words = re.sub(r"([,;])(?:\s*[,;])+", r"\1", words)
    words = re.sub(r"^[\s,;]+|[\s,;]+$", "", words.replace("\\", " "))
    return _WS.sub(" ", words).strip() if _SPEAKABLE.search(words) else ""
