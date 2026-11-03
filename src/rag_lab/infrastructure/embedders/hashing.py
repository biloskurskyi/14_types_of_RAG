"""Deterministic feature-hashing embedder.

Each text becomes a bag of stemmed words plus character trigrams, hashed into a fixed-size,
signed, L2-normalised vector (the "hashing trick"). There is no semantics beyond surface form,
which is exactly what makes it useful: tests are reproducible, CI needs no model download, and
the gap to a real embedder is visible in ``rag-lab eval``.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from rag_lab.domain.text import STOPWORDS, stem, tokenize


class HashingEmbedder:
    def __init__(self, dim: int = 384, ngram: int = 3, char_weight: float = 0.5) -> None:
        self._dim = dim
        self.ngram = ngram
        self.char_weight = char_weight

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _features(self, text: str) -> list[tuple[str, float]]:
        feats: list[tuple[str, float]] = []
        for token in tokenize(text):
            if token in STOPWORDS:
                continue
            feats.append((f"w:{stem(token)}", 1.0))
            padded = f"#{token}#"
            feats.extend(
                (f"c:{padded[i : i + self.ngram]}", self.char_weight)
                for i in range(max(len(padded) - self.ngram + 1, 1))
            )
        return feats

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for feature, weight in self._features(text):
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            sign = 1.0 if value & 1 else -1.0
            vec[(value >> 1) % self._dim] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec
