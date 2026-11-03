"""Graph-lite (entity) RAG: a chunk–entity co-occurrence graph with one-hop expansion.

A tiny take on GraphRAG without an LLM extraction step: entities are capitalised terms
(plan names, acronyms such as SLA/VAT/SSO), codes and numbers with units. Retrieval seeds on
chunks that share entities with the query, then spreads to neighbours that share entities with
the seeds. The final score mixes entity evidence with dense similarity.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from rag_lab.application.strategies.base import RetrievalStrategy, register
from rag_lab.domain.models import Chunk, Query, RetrievedChunk

_ENTITY_RE = re.compile(
    r"[A-Za-z]{1,4}-\d{2,}(?:-\d+)*"  # codes: CB-2025-1103
    r"|\d+(?:\.\d+)?\s?(?:%|GB|TB|USD|days?|hours?|днів|годин\w*)"  # quantities
    r"|\b[A-Z][A-Za-z0-9]+\b"  # plan names, acronyms, product terms
)
_IGNORED = {"CloudBox", "EN", "UA", "How", "What", "Which", "Is", "Can", "The", "Do", "Does"}


def extract_entities(text: str) -> set[str]:
    return {m.lower() for m in _ENTITY_RE.findall(text) if m not in _IGNORED}


class EntityGraph:
    """Bipartite chunk <-> entity index built lazily from the vector store contents."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = {c.id: c for c in chunks}
        self.entities_of: dict[str, set[str]] = {}
        self.chunks_of: dict[str, set[str]] = defaultdict(set)
        for chunk in chunks:
            entities = extract_entities(chunk.text)
            self.entities_of[chunk.id] = entities
            for entity in entities:
                self.chunks_of[entity].add(chunk.id)

    def idf(self, entity: str) -> float:
        return math.log(1 + len(self.chunks) / (1 + len(self.chunks_of.get(entity, ()))))


@register("graph")
class GraphStrategy(RetrievalStrategy):
    summary = "Entity co-occurrence graph, 1-hop expansion"

    def graph(self) -> EntityGraph:
        cached = self.ctx.extras.get("entity_graph")
        chunks = self.ctx.vector_store.all_chunks()
        if not isinstance(cached, EntityGraph) or len(cached.chunks) != len(chunks):
            cached = EntityGraph(chunks)
            self.ctx.extras["entity_graph"] = cached
        return cached

    def _retrieve(self, query: Query) -> list[RetrievedChunk]:
        graph = self.graph()
        q_entities = extract_entities(query.text) & set(graph.chunks_of)
        scores: dict[str, float] = defaultdict(float)
        for entity in q_entities:
            for cid in graph.chunks_of[entity]:
                scores[cid] += graph.idf(entity)
        seeds = sorted(scores, key=lambda c: (-scores[c], c))[: query.k]
        hop_weight = float(self.params.get("hop_weight", 0.3))
        for seed in seeds:
            for entity in graph.entities_of[seed] - q_entities:
                for cid in graph.chunks_of[entity]:
                    scores[cid] += (
                        hop_weight * graph.idf(entity) / max(len(graph.chunks_of[entity]), 1)
                    )
        dense = {h.chunk.id: h.score for h in self.dense(query, 3 * query.k)}
        top = max(scores.values(), default=1.0) or 1.0
        combined = {
            cid: 0.6 * scores.get(cid, 0) / top + 0.4 * dense.get(cid, 0.0)
            for cid in {*scores, *dense}
        }
        with self.ctx.tracer.span("graph.expand", entities=sorted(q_entities), seeds=seeds):
            results = [
                RetrievedChunk(graph.chunks[cid], score, "graph")
                for cid, score in combined.items()
                if cid in graph.chunks and _passes(graph.chunks[cid], query)
            ]
        return sorted(results, key=lambda r: (-r.score, r.chunk.id))


def _passes(chunk: Chunk, query: Query) -> bool:
    return all(chunk.metadata.get(k) == v for k, v in query.filters.items())
