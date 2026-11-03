"""In-memory vector store (exact cosine search) and a parent-chunk document store."""

from __future__ import annotations

from collections.abc import Sequence

from rag_lab.domain.models import Chunk, Metadata, RetrievedChunk


def matches(metadata: Metadata, filters: Metadata | None) -> bool:
    """Exact-match metadata filter; a list value means "any of"."""
    if not filters:
        return True
    for key, expected in filters.items():
        value = metadata.get(key)
        if isinstance(expected, list | tuple | set):
            if value not in expected:
                return False
        elif value != expected:
            return False
    return True


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class InMemoryVectorStore:
    """Brute-force cosine search over normalised vectors. Fine for thousands of chunks."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = list(vector)

    def search(
        self, vector: Sequence[float], k: int, filters: Metadata | None = None
    ) -> list[RetrievedChunk]:
        scored = [
            RetrievedChunk(chunk, dot(vector, self._vectors[cid]), "dense")
            for cid, chunk in self._chunks.items()
            if matches(chunk.metadata, filters)
        ]
        scored.sort(key=lambda r: (-r.score, r.chunk.id))
        return scored[:k]

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def vectors_for(self, chunk_ids: Sequence[str]) -> dict[str, list[float]]:
        return {cid: self._vectors[cid] for cid in chunk_ids if cid in self._vectors}


class InMemoryDocStore:
    """Key-value store for parent chunks used by small-to-big retrieval."""

    def __init__(self) -> None:
        self._items: dict[str, Chunk] = {}

    def put(self, chunks: Sequence[Chunk]) -> None:
        self._items.update({c.id: c for c in chunks})

    def get(self, chunk_id: str) -> Chunk | None:
        return self._items.get(chunk_id)
