"""Composition root: builds adapters from :class:`~rag_lab.config.Settings`."""

from __future__ import annotations

from rag_lab.application.chunking import make_chunker
from rag_lab.application.generation import AnswerGenerator
from rag_lab.application.ingestion import IngestionPipeline
from rag_lab.application.service import RagLab
from rag_lab.application.strategies import StrategyContext
from rag_lab.config import Settings
from rag_lab.domain.ports import (
    LLM,
    DocStore,
    Embedder,
    KeywordIndex,
    Reranker,
    Tracer,
    VectorStore,
)
from rag_lab.infrastructure.embedders.caching import CachingEmbedder
from rag_lab.infrastructure.embedders.hashing import HashingEmbedder
from rag_lab.infrastructure.llm.fake import FakeLLM
from rag_lab.infrastructure.llm.tracing import TracingLLM
from rag_lab.infrastructure.rerankers import LexicalReranker
from rag_lab.infrastructure.stores.bm25 import BM25Index
from rag_lab.infrastructure.stores.memory import InMemoryDocStore, InMemoryVectorStore
from rag_lab.infrastructure.tracing import JsonTracer


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "sentence-transformers":
        from rag_lab.infrastructure.embedders.sentence_transformers import (  # noqa: PLC0415
            SentenceTransformerEmbedder,
        )

        return CachingEmbedder(SentenceTransformerEmbedder(settings.st_model))
    return CachingEmbedder(HashingEmbedder(settings.embedding_dim))


def build_llm(settings: Settings) -> LLM:
    if settings.llm == "openai":
        from rag_lab.infrastructure.llm.http import OpenAICompatibleLLM  # noqa: PLC0415

        return OpenAICompatibleLLM(
            settings.openai_base_url,
            settings.openai_api_key.get_secret_value(),
            settings.openai_model,
        )
    if settings.llm == "ollama":
        from rag_lab.infrastructure.llm.http import OllamaLLM  # noqa: PLC0415

        return OllamaLLM(settings.ollama_url, settings.ollama_model)
    return FakeLLM()


def build_reranker(settings: Settings) -> Reranker:
    if settings.reranker == "cross-encoder":
        from rag_lab.infrastructure.rerankers import CrossEncoderReranker  # noqa: PLC0415

        return CrossEncoderReranker(settings.cross_encoder_model)
    return LexicalReranker()


def build_stores(
    settings: Settings, embedder: Embedder
) -> tuple[VectorStore, KeywordIndex, DocStore]:
    if settings.backend == "postgres":
        from rag_lab.infrastructure.stores.pgvector import (  # noqa: PLC0415
            PgDocStore,
            PgFullTextIndex,
            PgVectorStore,
        )

        store = PgVectorStore(settings.database_url, embedder.dim)
        return store, PgFullTextIndex(store), PgDocStore(store)
    return InMemoryVectorStore(), BM25Index(), InMemoryDocStore()


def build_lab(
    settings: Settings | None = None, tracer: Tracer | None = None, llm: LLM | None = None
) -> RagLab:
    settings = settings or Settings()
    tracer = tracer or JsonTracer()
    embedder = build_embedder(settings)
    vector_store, keyword_index, doc_store = build_stores(settings, embedder)
    raw_llm = llm or build_llm(settings)
    traced_llm = TracingLLM(raw_llm, tracer)
    ctx = StrategyContext(
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        llm=traced_llm,
        reranker=build_reranker(settings),
        tracer=tracer,
        doc_store=doc_store,
    )
    pipeline = IngestionPipeline(
        make_chunker(settings.chunker, settings.chunk_size, settings.chunk_overlap),
        embedder,
        vector_store,
        keyword_index,
        doc_store,
        tracer,
    )
    return RagLab(ctx, pipeline, AnswerGenerator(traced_llm, tracer), settings.top_k)
