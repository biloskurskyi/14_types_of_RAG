"""Rerankers: a deterministic lexical scorer and an optional cross-encoder.

The lexical reranker approximates what a cross-encoder rewards (query terms present in the
passage, adjacent query bigrams, coverage over raw similarity) so the rerank strategy can be
tested offline. ``CrossEncoderReranker`` is the real thing via ``sentence-transformers``.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

from rag_lab.domain.models import RetrievedChunk
from rag_lab.domain.text import content_terms


class LexicalReranker:
    def __init__(self, prior_weight: float = 0.2) -> None:
        self.prior_weight = prior_weight

    def score(self, query: str, text: str) -> float:
        q_terms = content_terms(query)
        if not q_terms:
            return 0.0
        d_terms = content_terms(text)
        d_set = set(d_terms)
        coverage = len(set(q_terms) & d_set) / len(set(q_terms))
        q_bigrams = set(itertools.pairwise(q_terms))
        d_bigrams = set(itertools.pairwise(d_terms))
        bigram = len(q_bigrams & d_bigrams) / len(q_bigrams) if q_bigrams else 0.0
        return 0.75 * coverage + 0.25 * bigram

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        top = max(c.score for c in candidates) or 1.0
        rescored = [
            c.rescored(
                (1 - self.prior_weight) * self.score(query, c.chunk.text)
                + self.prior_weight * (c.score / top),
                "rerank",
            )
            for c in candidates
        ]
        rescored.sort(key=lambda r: (-r.score, r.chunk.id))
        return rescored[:k]


class CrossEncoderReranker:
    """Wraps ``sentence_transformers.CrossEncoder`` (optional ``embeddings`` extra)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pip install 'rag-lab[embeddings]' for the cross-encoder") from exc
        self._model: Any = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], k: int
    ) -> list[RetrievedChunk]:  # pragma: no cover - needs a model download
        if not candidates:
            return []
        scores = self._model.predict([(query, c.chunk.text) for c in candidates])
        rescored = [
            c.rescored(float(s), "cross-encoder") for c, s in zip(candidates, scores, strict=False)
        ]
        rescored.sort(key=lambda r: -r.score)
        return rescored[:k]
