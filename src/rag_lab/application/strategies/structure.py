"""Small-to-big (parent document) retrieval."""

from __future__ import annotations

from rag_lab.application.strategies.base import RetrievalStrategy, register
from rag_lab.domain.models import Query, RetrievedChunk


@register("parent_document")
class ParentDocumentStrategy(RetrievalStrategy):
    """Match on small child chunks (precise), return their larger parents (enough context).

    Parent score is the best child score; children without a stored parent pass through.
    """

    summary = "Search small chunks, return their parents"

    def _retrieve(self, query: Query) -> list[RetrievedChunk]:
        children = self.dense(query, 3 * query.k)
        best: dict[str, RetrievedChunk] = {}
        for hit in children:
            parent = self.ctx.doc_store.get(hit.chunk.parent_id) if hit.chunk.parent_id else None
            item = RetrievedChunk(parent, hit.score, "parent") if parent else hit
            if item.chunk.id not in best or best[item.chunk.id].score < item.score:
                best[item.chunk.id] = item
        with self.ctx.tracer.span("parent.expand", children=len(children), parents=len(best)):
            return sorted(best.values(), key=lambda r: (-r.score, r.chunk.id))
