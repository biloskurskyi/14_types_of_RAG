"""Strategy pattern core: the ``RetrievalStrategy`` base class, its context and a registry.

``RetrievalStrategy.retrieve`` is a template method: it opens a trace span, delegates to
``_retrieve`` (implemented by each strategy), then de-duplicates and truncates the result.
Strategies self-register with ``@register("name")`` and are built by :func:`create`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

from rag_lab.domain.models import Query, RetrievedChunk
from rag_lab.domain.ports import (
    LLM,
    DocStore,
    Embedder,
    KeywordIndex,
    Reranker,
    Tracer,
    VectorStore,
)


@dataclass
class StrategyContext:
    """Everything a strategy may need. Strategies use only the ports they care about."""

    embedder: Embedder
    vector_store: VectorStore
    keyword_index: KeywordIndex
    llm: LLM
    reranker: Reranker
    tracer: Tracer
    doc_store: DocStore
    doc_types: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)


class RetrievalStrategy(ABC):
    """Base class for every retrieval approach."""

    name: ClassVar[str] = "base"
    summary: ClassVar[str] = ""

    def __init__(self, ctx: StrategyContext, **params: Any) -> None:
        self.ctx = ctx
        self.params = params

    def retrieve(self, query: Query) -> list[RetrievedChunk]:
        with self.ctx.tracer.span(f"strategy.{self.name}", query=query.text, k=query.k) as attrs:
            results = dedupe(self._retrieve(query))[: query.k]
            attrs["results"] = [{"id": r.chunk.id, "score": round(r.score, 4)} for r in results]
            return results

    @abstractmethod
    def _retrieve(self, query: Query) -> list[RetrievedChunk]: ...

    # Helpers shared by many strategies -------------------------------------------------
    def dense(
        self, query: Query, k: int | None = None, text: str | None = None
    ) -> list[RetrievedChunk]:
        with self.ctx.tracer.span("retriever.dense", text=text or query.text):
            vector = self.ctx.embedder.embed([text or query.text])[0]
            hits = self.ctx.vector_store.search(vector, k or query.k, query.filters or None)
            return [h.rescored(h.score, "dense") for h in hits]

    def sparse(
        self, query: Query, k: int | None = None, text: str | None = None
    ) -> list[RetrievedChunk]:
        with self.ctx.tracer.span("retriever.sparse", text=text or query.text):
            hits = self.ctx.keyword_index.search(
                text or query.text, k or query.k, query.filters or None
            )
            return [h.rescored(h.score, "sparse") for h in hits]


def dedupe(results: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep the first (best) occurrence of each chunk id, preserving order."""
    seen: set[str] = set()
    out = []
    for r in results:
        if r.chunk.id not in seen:
            seen.add(r.chunk.id)
            out.append(r)
    return out


S = TypeVar("S", bound=type[RetrievalStrategy])
_REGISTRY: dict[str, type[RetrievalStrategy]] = {}


def register(name: str) -> Callable[[S], S]:
    def decorator(cls: S) -> S:
        if name in _REGISTRY:
            raise ValueError(f"strategy {name!r} registered twice")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def available() -> list[str]:
    return list(_REGISTRY)


def get_class(name: str) -> type[RetrievalStrategy]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown strategy {name!r}; available: {', '.join(_REGISTRY)}") from exc


def create(name: str, ctx: StrategyContext, **params: Any) -> RetrievalStrategy:
    """Factory: build a registered strategy by name."""
    return get_class(name)(ctx, **params)
