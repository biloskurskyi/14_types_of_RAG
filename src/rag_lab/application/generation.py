"""Answer generation with numbered context and citation parsing."""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag_lab.application.prompts import number_context, render
from rag_lab.domain.models import Answer, Citation, RetrievedChunk
from rag_lab.domain.ports import LLM, Tracer

_MARKER = re.compile(r"\[(\d+)\]")


def parse_citations(text: str, context: Sequence[RetrievedChunk]) -> tuple[Citation, ...]:
    """Map ``[n]`` markers to chunks; markers outside the context are ignored (hallucinated)."""
    seen: dict[int, Citation] = {}
    for match in _MARKER.finditer(text):
        n = int(match.group(1))
        if 1 <= n <= len(context) and n not in seen:
            chunk = context[n - 1].chunk
            seen[n] = Citation(marker=n, chunk_id=chunk.id, doc_id=chunk.doc_id)
    return tuple(seen.values())


class AnswerGenerator:
    def __init__(self, llm: LLM, tracer: Tracer) -> None:
        self.llm = llm
        self.tracer = tracer

    def generate(self, question: str, context: Sequence[RetrievedChunk], strategy: str) -> Answer:
        with self.tracer.span("generate", chunks=len(context)) as attrs:
            prompt = render("answer", {"question": question, "context": number_context(context)})
            text = self.llm.generate(prompt).strip()
            citations = parse_citations(text, context)
            attrs["citations"] = [c.chunk_id for c in citations]
        return Answer(question, text, strategy, citations, tuple(context))
