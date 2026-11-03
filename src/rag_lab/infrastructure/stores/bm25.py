"""Okapi BM25 in pure Python (Robertson & Zaragoza, 2009).

score(q, d) = sum_t idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
with the non-negative idf variant ln(1 + (N - df + 0.5) / (df + 0.5)).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from rag_lab.domain.models import Chunk, Metadata, RetrievedChunk
from rag_lab.domain.text import content_terms
from rag_lab.infrastructure.stores.memory import matches


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._tfs: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._df: Counter[str] = Counter()

    def add(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            terms = content_terms(chunk.text)
            self._chunks.append(chunk)
            self._tfs.append(Counter(terms))
            self._lengths.append(len(terms))
            self._df.update(set(terms))

    def idf(self, term: str) -> float:
        n = len(self._chunks)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, text: str, k: int, filters: Metadata | None = None) -> list[RetrievedChunk]:
        terms = content_terms(text)
        if not terms or not self._chunks:
            return []
        avgdl = sum(self._lengths) / len(self._lengths) or 1.0
        results: list[RetrievedChunk] = []
        for chunk, tf, length in zip(self._chunks, self._tfs, self._lengths, strict=True):
            if not matches(chunk.metadata, filters):
                continue
            score = 0.0
            for term in set(terms):
                freq = tf.get(term, 0)
                if freq:
                    norm = freq + self.k1 * (1 - self.b + self.b * length / avgdl)
                    score += self.idf(term) * freq * (self.k1 + 1) / norm
            if score > 0:
                results.append(RetrievedChunk(chunk, score, "bm25"))
        results.sort(key=lambda r: (-r.score, r.chunk.id))
        return results[:k]
