"""Ports: the interfaces the application layer depends on.

Infrastructure provides adapters (in-memory, pgvector, OpenAI-compatible, ...). Using
``typing.Protocol`` keeps adapters free of inheritance from the domain.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable

from rag_lab.domain.models import (
    ChatMessage,
    ChatResponse,
    Chunk,
    Metadata,
    RetrievedChunk,
    ToolSpec,
)

Vector = list[float]


@runtime_checkable
class Embedder(Protocol):
    """Turns text into dense vectors of a fixed dimension."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Stores chunk embeddings and answers nearest-neighbour queries (cosine similarity)."""

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None: ...

    def search(
        self, vector: Vector, k: int, filters: Metadata | None = None
    ) -> list[RetrievedChunk]: ...

    def get(self, chunk_id: str) -> Chunk | None: ...

    def all_chunks(self) -> list[Chunk]: ...

    def vectors_for(self, chunk_ids: Sequence[str]) -> dict[str, Vector]: ...


@runtime_checkable
class KeywordIndex(Protocol):
    """Lexical (sparse) retrieval: BM25, Postgres full-text search, ..."""

    def add(self, chunks: Sequence[Chunk]) -> None: ...

    def search(
        self, text: str, k: int, filters: Metadata | None = None
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class LLM(Protocol):
    """Text generation with optional tool calling."""

    @property
    def name(self) -> str: ...

    def generate(self, prompt: str, *, system: str | None = None) -> str: ...

    def chat(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> ChatResponse: ...


@runtime_checkable
class Reranker(Protocol):
    """Re-scores a candidate list against the query (cross-encoder style)."""

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], k: int
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class Tracer(Protocol):
    """Records nested, timed spans of a request so it can be dumped to JSON."""

    def span(self, name: str, **attrs: Any) -> AbstractContextManager[dict[str, Any]]: ...

    def export(self) -> list[dict[str, Any]]: ...

    def reset(self) -> None: ...


def iter_batches(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    """Yield ``items`` in slices of ``size`` (helper shared by adapters)."""
    for start in range(0, len(items), size):
        yield items[start : start + size]
