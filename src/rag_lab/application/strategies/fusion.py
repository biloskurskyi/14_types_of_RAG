"""Rank fusion helpers used by hybrid and multi-query strategies."""

from __future__ import annotations

from collections.abc import Sequence

from rag_lab.domain.models import RetrievedChunk


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]], k: int = 60, source: str = "rrf"
) -> list[RetrievedChunk]:
    """RRF (Cormack et al., 2009): score(d) = sum_r 1 / (k + rank_r(d)), ranks start at 1.

    Uses only ranks, so it needs no score calibration between retrievers.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(hit.chunk.id, hit)
    fused = [chunks[cid].rescored(score, source) for cid, score in scores.items()]
    fused.sort(key=lambda r: (-r.score, r.chunk.id))
    return fused


def min_max(results: Sequence[RetrievedChunk]) -> dict[str, float]:
    if not results:
        return {}
    lo = min(r.score for r in results)
    hi = max(r.score for r in results)
    span = hi - lo
    return {r.chunk.id: (r.score - lo) / span if span else 1.0 for r in results}


def weighted_fusion(
    dense: Sequence[RetrievedChunk],
    sparse: Sequence[RetrievedChunk],
    alpha: float = 0.5,
    source: str = "weighted",
) -> list[RetrievedChunk]:
    """Convex combination of min-max normalised scores: alpha * dense + (1 - alpha) * sparse."""
    nd, ns = min_max(dense), min_max(sparse)
    chunks = {r.chunk.id: r for r in [*dense, *sparse]}
    fused = [
        chunks[cid].rescored(alpha * nd.get(cid, 0.0) + (1 - alpha) * ns.get(cid, 0.0), source)
        for cid in chunks
    ]
    fused.sort(key=lambda r: (-r.score, r.chunk.id))
    return fused
