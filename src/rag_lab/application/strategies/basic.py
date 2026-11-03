"""Baselines: dense, sparse (BM25 / Postgres FTS) and two flavours of hybrid search."""

from __future__ import annotations

from rag_lab.application.strategies.base import RetrievalStrategy, register
from rag_lab.application.strategies.fusion import reciprocal_rank_fusion, weighted_fusion
from rag_lab.domain.models import Query, RetrievedChunk


@register("dense")
class DenseStrategy(RetrievalStrategy):
    """Naive RAG: embed the question, return the k nearest chunks by cosine similarity."""

    summary = "Embed query, k-NN over chunk vectors"

    def _retrieve(self, query: Query) -> list[RetrievedChunk]:
        return self.dense(query)


@register("bm25")
class SparseStrategy(RetrievalStrategy):
    """Lexical retrieval: exact terms, ids and numbers win; synonyms and paraphrases lose."""

    summary = "Keyword index (BM25 or Postgres FTS)"

    def _retrieve(self, query: Query) -> list[RetrievedChunk]:
        return self.sparse(query)


@register("hybrid_rrf")
class HybridRRFStrategy(RetrievalStrategy):
    """Dense + sparse candidates fused with Reciprocal Rank Fusion."""

    summary = "Dense + sparse, Reciprocal Rank Fusion"

    def _retrieve(self, query: Query) -> list[RetrievedChunk]:
        fan_out = int(self.params.get("fan_out", 3)) * query.k
        with self.ctx.tracer.span("fusion.rrf"):
            return reciprocal_rank_fusion(
                [self.dense(query, fan_out), self.sparse(query, fan_out)],
                k=int(self.params.get("rrf_k", 60)),
            )


@register("hybrid_weighted")
class HybridWeightedStrategy(RetrievalStrategy):
    """Dense + sparse with min-max normalised, alpha-weighted scores."""

    summary = "Dense + sparse, alpha-weighted normalised scores"

    def _retrieve(self, query: Query) -> list[RetrievedChunk]:
        fan_out = int(self.params.get("fan_out", 3)) * query.k
        alpha = float(self.params.get("alpha", 0.5))
        with self.ctx.tracer.span("fusion.weighted", alpha=alpha):
            return weighted_fusion(self.dense(query, fan_out), self.sparse(query, fan_out), alpha)
