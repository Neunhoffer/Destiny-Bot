"""The speech layer: display text -> what the voice says, and sections -> reading plans."""

from __future__ import annotations

from .normalize import normalize_for_speech, verbalize_math
from .plan import plan_section, section_plain_text
from .segment import chunk_span, split_sentences

__all__ = [
    "chunk_span",
    "normalize_for_speech",
    "plan_section",
    "section_plain_text",
    "split_sentences",
    "verbalize_math",
]
