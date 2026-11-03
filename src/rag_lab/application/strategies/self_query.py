"""Self-query retrieval: the LLM turns natural-language constraints into metadata filters."""

from __future__ import annotations

import json

from rag_lab.application.prompts import render
from rag_lab.application.strategies.base import RetrievalStrategy, register
from rag_lab.application.strategies.fusion import reciprocal_rank_fusion
from rag_lab.domain.models import Query, RetrievedChunk

ALLOWED_KEYS = {"lang", "doc_type"}


def parse_self_query(
    raw: str, fallback: str, doc_types: tuple[str, ...]
) -> tuple[str, dict[str, str]]:
    """Validate the LLM's JSON; unknown keys or values are dropped instead of trusted."""
    try:
        data = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return fallback, {}
    filters = {
        k: str(v)
        for k, v in (data.get("filters") or {}).items()
        if k in ALLOWED_KEYS and v not in (None, "")
    }
    if filters.get("lang") not in (None, "en", "ua"):
        filters.pop("lang")
    if "doc_type" in filters and doc_types and filters["doc_type"] not in doc_types:
        filters.pop("doc_type")
    return str(data.get("query") or fallback), filters


@register("self_query")
class SelfQueryStrategy(RetrievalStrategy):
    """E.g. "Ukrainian docs: how long are files kept?" -> filter lang=ua, query "files kept"."""

    summary = "LLM extracts metadata filters, then hybrid search"

    def _retrieve(self, query: Query) -> list[RetrievedChunk]:
        prompt = render(
            "self_query",
            {"question": query.text, "doc_types": ", ".join(self.ctx.doc_types)},
            doc_types=", ".join(self.ctx.doc_types),
        )
        text, filters = parse_self_query(
            self.ctx.llm.generate(prompt), query.text, self.ctx.doc_types
        )
        filtered = query.with_text(text).with_filters({**query.filters, **filters})
        with self.ctx.tracer.span("self_query.parsed", query=text, filters=filters):
            hits = reciprocal_rank_fusion(
                [self.dense(filtered, 3 * query.k), self.sparse(filtered, 3 * query.k)],
                source="self_query",
            )
        if not hits and filters:  # over-restrictive filter: fall back to unfiltered search
            hits = reciprocal_rank_fusion(
                [self.dense(query, 3 * query.k), self.sparse(query, 3 * query.k)]
            )
        return hits
