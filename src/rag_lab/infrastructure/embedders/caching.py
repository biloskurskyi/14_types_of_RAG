"""Decorator that memoises embeddings by text, wrapping any ``Embedder``.

Query-transform strategies (multi-query, HyDE, corrective retries) embed many near-identical
strings; the cache keeps ``compare`` and ``eval`` fast and its hit counter shows up in traces.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

from rag_lab.domain.ports import Embedder


class CachingEmbedder:
    def __init__(self, inner: Embedder, max_size: int = 4096) -> None:
        self.inner = inner
        self.max_size = max_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def dim(self) -> int:
        return self.inner.dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        self.misses += len(missing)
        self.hits += len(texts) - len(missing)
        if missing:
            for text, vector in zip(missing, self.inner.embed(missing), strict=True):
                self._cache[text] = vector
                if len(self._cache) > self.max_size:
                    self._cache.popitem(last=False)
        out = []
        for text in texts:
            self._cache.move_to_end(text)
            out.append(self._cache[text])
        return out
