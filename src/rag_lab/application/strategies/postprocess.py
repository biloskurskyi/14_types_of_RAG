"""Post-retrieval strategies: MMR diversification, reranking and contextual compression."""

from __future__ import annotations

from rag_lab.application.prompts import render
from rag_lab.application.strategies.base import RetrievalStrategy, register
from rag_lab.application.strategies.fusion import reciprocal_rank_fusion
from rag_lab.domain.models import Query, RetrievedChunk


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@register("mmr")
class MMRStrategy(RetrievalStrategy):
    """Maximal Marginal Relevance: trade relevance against redundancy among picked chunks.

    MMR = argmax_d [ lambda * sim(q, d) - (1 - lambda) * max_{s in S} sim(d, s) ]
    """

    summary = "Relevance minus redundancy (lambda)"

    def _retrieve(self, query: Query) -> list[RetrievedChunk]:
        lam = float(self.params.get("lambda_", 0.6))
        candidates = self.dense(query, int(self.params.get("fan_out", 4)) * query.k)
        vectors = self.ctx.vector_store.vectors_for([c.chunk.id for c in candidates])
        selected: list[RetrievedChunk] = []
        pool = [c for c in candidates if c.chunk.id in vectors]
        with self.ctx.tracer.span("mmr.select", lambda_=lam, pool=len(pool)):
            while pool and len(selected) < query.k:

                def mmr(c: RetrievedChunk) -> float:
                    redundancy = max(
                        (_dot(vectors[c.chunk.id], vectors[s.chunk.id]) for s in selected),
                        default=0.0,
                    )
                    return lam * c.score - (1 - lam) * redundancy

                best = max(pool, key=mmr)
                selected.append(best.rescored(mmr(best), "mmr"))
                pool.remove(best)
        return selected


@register("rerank")
class RerankStrategy(RetrievalStrategy):
    """Retrieve a wide hybrid candidate set, then let a (cross-encoder style) reranker pick."""

    summary = "Wide hybrid recall, reranker precision"

    def _retrieve(self, query: Query) -> list[RetrievedChunk]:
        fan_out = int(self.params.get("fan_out", 5)) * query.k
        candidates = reciprocal_rank_fusion(
            [self.dense(query, fan_out), self.sparse(query, fan_out)]
        )
        with self.ctx.tracer.span("rerank", candidates=len(candidates)):
            return self.ctx.reranker.rerank(query.text, candidates, query.k)


@register("compression")
class ContextualCompressionStrategy(RetrievalStrategy):
    """Retrieve, then have the LLM keep only the sentences relevant to the question.

    Chunks that compress to nothing are dropped, so the generator sees less noise.
    """

    summary = "LLM keeps only relevant sentences"

    def _retrieve(self, query: Query) -> list[RetrievedChunk]:
        hits = reciprocal_rank_fusion(
            [self.dense(query, 2 * query.k), self.sparse(query, 2 * query.k)]
        )
        out: list[RetrievedChunk] = []
        before = after = 0
        for hit in hits:
            prompt = render("compress", {"question": query.text, "document": hit.chunk.text})
            text = self.ctx.llm.generate(prompt).strip()
            before += len(hit.chunk.text)
            if text and text.upper() != "NONE":
                after += len(text)
                out.append(RetrievedChunk(hit.chunk.with_text(text), hit.score, "compression"))
            if len(out) == query.k:
                break
        with self.ctx.tracer.span("compression.stats", chars_before=before, chars_after=after):
            return out
