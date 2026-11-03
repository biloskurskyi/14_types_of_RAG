"""Adaptive routing: pick a strategy per query with a Chain of Responsibility of rules.

Inspired by Adaptive-RAG (route by query complexity). Rules are cheap and deterministic, so
routing adds no LLM call; each rule either claims the query or passes it on.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from rag_lab.application.strategies.base import RetrievalStrategy, create, register
from rag_lab.domain.models import Query, RetrievedChunk

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")


class RouteRule(ABC):
    def __init__(self, successor: RouteRule | None = None) -> None:
        self.successor = successor

    def route(self, query: Query) -> tuple[str, str]:
        target = self.match(query)
        if target:
            return target, type(self).__name__
        return self.successor.route(query) if self.successor else ("hybrid_rrf", "Default")

    @abstractmethod
    def match(self, query: Query) -> str | None: ...


class ExactIdentifierRule(RouteRule):
    """Codes, invoice numbers, prices: lexical search is the right tool."""

    pattern = re.compile(r"\b[A-Za-z]{1,4}-\d{2,}|\$\d|\d+(\.\d+)?%|HTTP \d{3}")

    def match(self, query: Query) -> str | None:
        return "bm25" if self.pattern.search(query.text) else None


class MetadataHintRule(RouteRule):
    """Explicit language or document constraints: let self-query build filters."""

    hints = ("ukrainian", "english", "in the contract", "the invoice", "refund policy")

    def match(self, query: Query) -> str | None:
        lowered = query.text.lower()
        return "self_query" if any(h in lowered for h in self.hints) else None


class MultiHopRule(RouteRule):
    """Comparisons and multi-part questions: let the agent run several searches."""

    markers = (" and ", " compare", " difference", " both ", " versus ", " vs ")

    def match(self, query: Query) -> str | None:
        lowered = f" {query.text.lower()} "
        return "agentic" if sum(m in lowered for m in self.markers) >= 1 else None


class CrossLingualRule(RouteRule):
    """Latin-script questions over a bilingual corpus benefit from translated query variants."""

    def match(self, query: Query) -> str | None:
        return None if _CYRILLIC.search(query.text) else "multi_query"


def default_router() -> RouteRule:
    return ExactIdentifierRule(MetadataHintRule(MultiHopRule(CrossLingualRule())))


@register("adaptive")
class AdaptiveStrategy(RetrievalStrategy):
    summary = "Rule chain routes each query to a strategy"

    def _retrieve(self, query: Query) -> list[RetrievedChunk]:
        target, rule = default_router().route(query)
        with self.ctx.tracer.span("adaptive.route", target=target, rule=rule):
            return create(target, self.ctx).retrieve(query)
