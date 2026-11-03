"""Query-side strategies: multi-query (RAG-Fusion) and HyDE."""

from __future__ import annotations

from rag_lab.application.prompts import render
from rag_lab.application.strategies.base import RetrievalStrategy, register
from rag_lab.application.strategies.fusion import reciprocal_rank_fusion
from rag_lab.domain.models import Query, RetrievedChunk


def rewrite_queries(strategy: RetrievalStrategy, question: str, n: int = 3) -> list[str]:
    prompt = render("rewrite_queries", {"question": question, "n": str(n)}, n=str(n))
    lines = [
        line.strip(" -*0123456789.") for line in strategy.ctx.llm.generate(prompt).splitlines()
    ]
    return [q for q in dict.fromkeys(lines) if q][:n]


@register("multi_query")
class MultiQueryStrategy(RetrievalStrategy):
    """LLM writes query variants; each is searched (hybrid) and the rankings are RRF-fused."""

    summary = "LLM query variants, hybrid search each, RRF"

    def _retrieve(self, query: Query) -> list[RetrievedChunk]:
        variants = [query.text, *rewrite_queries(self, query.text, int(self.params.get("n", 3)))]
        with self.ctx.tracer.span("multi_query.variants", variants=variants):
            rankings = []
            for text in variants:
                rankings.append(self.dense(query, 2 * query.k, text))
                rankings.append(self.sparse(query, 2 * query.k, text))
        return reciprocal_rank_fusion(rankings, source="multi_query")


@register("hyde")
class HyDEStrategy(RetrievalStrategy):
    """Hypothetical Document Embeddings: embed an LLM-written fake answer instead of the query."""

    summary = "Embed an LLM-written hypothetical answer"

    def _retrieve(self, query: Query) -> list[RetrievedChunk]:
        passage = self.ctx.llm.generate(render("hyde", {"question": query.text}))
        with self.ctx.tracer.span("hyde.passage", passage=passage):
            vector_text = f"{query.text}\n{passage}"
        return [h.rescored(h.score, "hyde") for h in self.dense(query, query.k, vector_text)]
