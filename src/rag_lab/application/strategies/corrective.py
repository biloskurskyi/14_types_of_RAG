"""Corrective RAG (CRAG-style): grade retrieved chunks, and if they are weak, correct course.

Correction is a Chain of Responsibility: each handler gets the query and the graded results and
either returns an improved result list or passes to the next handler.

1. ``KeepRelevant``     – enough chunks pass the grader: keep them, done.
2. ``RewriteAndRetry``  – rewrite the query with the LLM (incl. translation) and search again.
3. ``KeywordFallback``  – fall back to pure lexical search on the rewritten terms.
4. ``DropFilters``      – last resort: retry without metadata filters (stand-in for web search).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from rag_lab.application.prompts import render
from rag_lab.application.strategies.base import RetrievalStrategy, StrategyContext, register
from rag_lab.application.strategies.fusion import reciprocal_rank_fusion
from rag_lab.application.strategies.query_transform import rewrite_queries
from rag_lab.domain.models import Query, RetrievedChunk


@dataclass
class Graded:
    hit: RetrievedChunk
    grade: float


class CorrectionHandler(ABC):
    def __init__(self, successor: CorrectionHandler | None = None) -> None:
        self.successor = successor

    def handle(
        self, strategy: CorrectiveStrategy, query: Query, graded: list[Graded]
    ) -> list[RetrievedChunk]:
        result = self.try_handle(strategy, query, graded)
        if result is not None:
            name = type(self).__name__
            with strategy.ctx.tracer.span("corrective.handled_by", handler=name, n=len(result)):
                return result
        return self.successor.handle(strategy, query, graded) if self.successor else []

    @abstractmethod
    def try_handle(
        self, strategy: CorrectiveStrategy, query: Query, graded: list[Graded]
    ) -> list[RetrievedChunk] | None: ...


class KeepRelevant(CorrectionHandler):
    def try_handle(
        self, strategy: CorrectiveStrategy, query: Query, graded: list[Graded]
    ) -> list[RetrievedChunk] | None:
        relevant = [g for g in graded if g.grade >= strategy.threshold]
        if len(relevant) >= strategy.min_relevant:
            return [g.hit.rescored(g.grade, "corrective") for g in relevant]
        return None


class RewriteAndRetry(CorrectionHandler):
    def try_handle(
        self, strategy: CorrectiveStrategy, query: Query, graded: list[Graded]
    ) -> list[RetrievedChunk] | None:
        variants = rewrite_queries(strategy, query.text, n=2)
        rankings = []
        for text in variants:
            rankings.append(strategy.dense(query, 2 * query.k, text))
            rankings.append(strategy.sparse(query, 2 * query.k, text))
        regraded = strategy.grade(query, reciprocal_rank_fusion(rankings))
        relevant = [g for g in regraded if g.grade >= strategy.threshold]
        strategy.last_variants = variants
        return [g.hit.rescored(g.grade, "corrective") for g in relevant] or None


class KeywordFallback(CorrectionHandler):
    def try_handle(
        self, strategy: CorrectiveStrategy, query: Query, graded: list[Graded]
    ) -> list[RetrievedChunk] | None:
        text = " ".join([query.text, *strategy.last_variants])
        hits = strategy.sparse(query, query.k, text)
        return [h.rescored(h.score, "corrective") for h in hits] or None


class DropFilters(CorrectionHandler):
    def try_handle(
        self, strategy: CorrectiveStrategy, query: Query, graded: list[Graded]
    ) -> list[RetrievedChunk] | None:
        if not query.filters:
            return None
        return strategy.dense(query.with_filters({}), query.k) or None


@register("corrective")
class CorrectiveStrategy(RetrievalStrategy):
    summary = "Grade chunks; rewrite / fall back when weak"

    def __init__(self, ctx: StrategyContext, **params: Any) -> None:
        super().__init__(ctx, **params)
        self.threshold = float(params.get("threshold", 0.5))
        self.min_relevant = int(params.get("min_relevant", 1))
        self.last_variants: list[str] = []
        self.chain = KeepRelevant(RewriteAndRetry(KeywordFallback(DropFilters())))

    def grade(self, query: Query, hits: list[RetrievedChunk]) -> list[Graded]:
        graded = []
        for hit in hits:
            prompt = render("grade_relevance", {"question": query.text, "document": hit.chunk.text})
            graded.append(Graded(hit, _parse_score(self.ctx.llm.generate(prompt))))
        return sorted(graded, key=lambda g: (-g.grade, -g.hit.score))

    def _retrieve(self, query: Query) -> list[RetrievedChunk]:
        hits = reciprocal_rank_fusion(
            [self.dense(query, 2 * query.k), self.sparse(query, 2 * query.k)]
        )
        graded = self.grade(query, hits[: 2 * query.k])
        with self.ctx.tracer.span(
            "corrective.grades", grades={g.hit.chunk.id: g.grade for g in graded}
        ):
            return self.chain.handle(self, query, graded)


def _parse_score(raw: str) -> float:
    try:
        return float(json.loads(raw[raw.index("{") : raw.rindex("}") + 1])["score"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return 0.0
