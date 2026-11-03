"""``RagLab`` facade: one object the CLI (or an API) talks to."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rag_lab.application import strategies
from rag_lab.application.generation import AnswerGenerator
from rag_lab.application.ingestion import IngestionPipeline, IngestReport
from rag_lab.application.strategies import StrategyContext
from rag_lab.domain.models import Answer, Document, Query, RetrievedChunk


@dataclass(frozen=True)
class StrategyRun:
    strategy: str
    hits: list[RetrievedChunk]
    latency_ms: float
    llm_calls: int


class RagLab:
    def __init__(
        self,
        ctx: StrategyContext,
        pipeline: IngestionPipeline,
        generator: AnswerGenerator,
        default_k: int = 4,
    ) -> None:
        self.ctx = ctx
        self.pipeline = pipeline
        self.generator = generator
        self.default_k = default_k

    def ingest(self, documents: Sequence[Document]) -> IngestReport:
        self.ctx.doc_types = tuple(sorted({d.metadata.get("doc_type", "") for d in documents}))
        return self.pipeline.run(documents)

    def strategy(self, name: str, **params: Any) -> strategies.RetrievalStrategy:
        return strategies.create(name, self.ctx, **params)

    def retrieve(
        self, name: str, question: str, k: int | None = None, **params: Any
    ) -> StrategyRun:
        llm_calls_before = getattr(self.ctx.llm, "call_count", 0)
        start = time.perf_counter()
        hits = self.strategy(name, **params).retrieve(Query(question, k=k or self.default_k))
        latency = (time.perf_counter() - start) * 1000
        llm_calls = getattr(self.ctx.llm, "call_count", 0) - llm_calls_before
        return StrategyRun(name, hits, latency, llm_calls)

    def ask(self, name: str, question: str, k: int | None = None) -> Answer:
        run = self.retrieve(name, question, k)
        return self.generator.generate(question, run.hits, name)

    def compare(
        self, question: str, names: Sequence[str] | None = None, k: int | None = None
    ) -> list[StrategyRun]:
        return [self.retrieve(n, question, k) for n in (names or strategies.available())]
