"""pgvector + Postgres FTS adapters against a real database.

Run locally with ``docker compose up -d db`` and
``RAG_LAB_TEST_DATABASE_URL=postgresql://rag:rag@localhost:5442/rag_lab pytest -m pgvector``.
"""

from __future__ import annotations

import os

import pytest

from rag_lab.application.service import RagLab
from rag_lab.bootstrap import build_lab
from rag_lab.config import Settings
from rag_lab.domain.models import Document, Query

DSN = os.environ.get("RAG_LAB_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.pgvector,
    pytest.mark.skipif(not DSN, reason="RAG_LAB_TEST_DATABASE_URL is not set"),
]


@pytest.fixture
def pg_lab(documents: list[Document]) -> RagLab:
    settings = Settings(_env_file=None, backend="postgres", database_url=DSN, embedding_dim=128)
    lab = build_lab(settings)
    store = lab.ctx.vector_store
    store.reset()
    lab.ingest(documents)
    yield lab
    store.conn.close()


def test_schema_uses_hnsw_index(pg_lab: RagLab) -> None:
    store = pg_lab.ctx.vector_store
    vector = pg_lab.ctx.embedder.embed(["refund"])[0]
    plan = store.explain_search(vector)
    assert "rag_chunks_embedding_hnsw" in plan
    indexes = {
        r[0]
        for r in store.conn.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename = 'rag_chunks'"
        ).fetchall()
    }
    assert {"rag_chunks_embedding_hnsw", "rag_chunks_tsv_gin"} <= indexes


def test_dense_and_fulltext_search(pg_lab: RagLab) -> None:
    dense = pg_lab.strategy("dense").retrieve(Query("API rate limit for Pro", k=3))
    assert "07_api_limits_en" in [h.doc_id for h in dense]
    fts = pg_lab.ctx.keyword_index.search("CB-2025-1103", k=2)
    assert fts[0].doc_id == "03_invoice_example_ua"


def test_metadata_filter_and_parents(pg_lab: RagLab) -> None:
    hits = pg_lab.ctx.keyword_index.search("повернення", k=5, filters={"lang": "ua"})
    assert hits
    assert all(h.chunk.metadata["lang"] == "ua" for h in hits)
    parents = pg_lab.strategy("parent_document").retrieve(Query("SSO SAML", k=2))
    assert all("#p" in h.chunk.id for h in parents)


@pytest.mark.parametrize(
    "name", ["hybrid_rrf", "mmr", "rerank", "multi_query", "corrective", "agentic", "graph"]
)
def test_strategies_run_on_postgres(pg_lab: RagLab, name: str) -> None:
    assert pg_lab.strategy(name).retrieve(Query("Team plan phone support", k=3))
