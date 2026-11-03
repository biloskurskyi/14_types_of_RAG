"""Language-agnostic text helpers shared by lexical retrieval, the hashing embedder and judges.

The corpus mixes English and Ukrainian, so tokenisation is Unicode-aware and stemming is a
crude prefix stemmer that behaves the same for both languages. It is good enough for a lab and,
more importantly, fully deterministic.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[\w.%$-]*[\w%]", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")

STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or of to in on for with by at from is are was were be been it its this that
    these those what which who whom how when where why do does did can could should would will
    i you we they my our your me us about as if than then so not no yes there here any all per
    much many each into after before over under up down out get got have has had
    і й та а але в у на до з із за по про що як який яка яке які чи це той ті є бути від для
    при через або мені нам вам його її їх ми ви вони він вона воно скільки коли де чому
    """.split()
)

STEM_PREFIX = 6


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens; keeps decimals, percents and codes like ``cb-2025-1103``."""
    return [t.strip(".-") for t in _TOKEN_RE.findall(text.lower()) if t.strip(".-")]


def stem(token: str) -> str:
    """Plural-strip + prefix stemmer.

    ``plans``/``plan`` and ``повернення``/``поверненню`` collapse to the same stem.
    """
    if any(ch.isdigit() for ch in token):
        return token  # ids, prices and codes must match exactly
    if token.isascii() and token.isalpha() and len(token) > 3 and token.endswith("s"):
        token = token[:-1] if not token.endswith("ss") else token
    return token[:STEM_PREFIX] if len(token) > STEM_PREFIX else token


def content_terms(text: str) -> list[str]:
    """Stemmed tokens without stop words (order preserved, duplicates kept)."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation and line breaks, dropping empty pieces."""
    return [s.strip() for s in _SENTENCE_RE.split(text) if s and s.strip()]


def overlap(query_terms: set[str], text: str) -> float:
    """Share of ``query_terms`` that occur in ``text`` (0..1)."""
    if not query_terms:
        return 0.0
    return len(query_terms & set(content_terms(text))) / len(query_terms)
