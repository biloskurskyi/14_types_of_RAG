"""Every registered strategy runs offline (FakeLLM + hashing embedder) and behaves sanely."""

import pytest

from rag_lab.application import strategies
from rag_lab.application.service import RagLab
from rag_lab.application.strategies.fusion import reciprocal_rank_fusion, weighted_fusion
from rag_lab.application.strategies.self_query import parse_self_query
from rag_lab.domain.models import Chunk, Query, RetrievedChunk
from rag_lab.infrastructure.tracing import span_names

ALL = strategies.available()


def docs(hits: list[RetrievedChunk]) -> list[str]:
    return [h.chunk.doc_id for h in hits]


def test_registry_has_core_strategies() -> None:
    expected = {"dense", "bm25", "hybrid_rrf", "hybrid_weighted", "multi_query", "hyde"}
    expected |= {"parent_document", "mmr", "rerank", "compression", "self_query"}
    assert expected <= set(ALL)
    with pytest.raises(KeyError):
        strategies.get_class("nope")


@pytest.mark.parametrize("name", ALL)
def test_strategy_contract(lab: RagLab, name: str) -> None:
    hits = lab.strategy(name).retrieve(Query("What is the Pro plan API rate limit?", k=3))
    assert 0 < len(hits) <= 3
    assert len({h.chunk.id for h in hits}) == len(hits)
    assert "07_api_limits_en" in docs(hits)
    assert f"strategy.{name}" in span_names(lab.ctx.tracer.export())


@pytest.mark.parametrize("name", ALL)
def test_strategy_respects_k(lab: RagLab, name: str) -> None:
    assert len(lab.strategy(name).retrieve(Query("CloudBox pricing", k=1))) <= 1


def test_bm25_wins_on_exact_identifier(lab: RagLab) -> None:
    hits = lab.strategy("bm25").retrieve(Query("CB-2025-1103", k=2))
    assert docs(hits)[0] == "03_invoice_example_ua"


def test_multi_query_bridges_languages(lab: RagLab) -> None:
    question = Query("How does the refund work for annual payments?", k=4)
    assert "05_refunds_policy_ua" in docs(lab.strategy("multi_query").retrieve(question))


def test_hyde_uses_hypothetical_passage(lab: RagLab) -> None:
    hits = lab.strategy("hyde").retrieve(Query("How long are files kept after termination?", k=4))
    assert "08_data_retention_ua" in docs(hits)
    assert "hyde.passage" in span_names(lab.ctx.tracer.export())


def test_self_query_applies_language_filter(lab: RagLab) -> None:
    hits = lab.strategy("self_query").retrieve(Query("Ukrainian docs: trash кошик", k=3))
    assert hits
    assert all(h.chunk.metadata["lang"] == "ua" for h in hits)


def test_parse_self_query_rejects_unknown_values() -> None:
    raw = '{"query": "q", "filters": {"lang": "de", "doc_type": "x", "evil": 1}}'
    assert parse_self_query(raw, "fallback", ("pricing",)) == ("q", {})
    assert parse_self_query("not json", "fallback", ()) == ("fallback", {})


def test_parent_document_returns_parents(lab: RagLab) -> None:
    hits = lab.strategy("parent_document").retrieve(Query("SSO SAML Team plan", k=2))
    assert all("#p" in h.chunk.id and ".c" not in h.chunk.id for h in hits)
    assert "06_security_en" in docs(hits)


def test_mmr_diversifies_documents(lab: RagLab) -> None:
    query = Query("Team plan", k=4)
    dense_docs = docs(lab.strategy("dense").retrieve(query))
    mmr_docs = docs(lab.strategy("mmr", lambda_=0.3).retrieve(query))
    assert len(set(mmr_docs)) >= len(set(dense_docs))


def test_compression_shortens_chunks(lab: RagLab) -> None:
    hits = lab.strategy("compression").retrieve(Query("What is the annual discount?", k=2))
    originals = {c.id: c for c in lab.ctx.vector_store.all_chunks()}
    assert hits
    assert all(len(h.chunk.text) <= len(originals[h.chunk.id].text) for h in hits)
    assert any("15%" in h.chunk.text for h in hits)


def test_rerank_uses_reranker_source(lab: RagLab) -> None:
    hits = lab.strategy("rerank").retrieve(Query("phone support Team", k=2))
    assert {h.source for h in hits} == {"rerank"}


def _hit(cid: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(Chunk(cid, cid, cid), score)


def test_rrf_formula() -> None:
    fused = reciprocal_rank_fusion([[_hit("a", 9), _hit("b", 8)], [_hit("b", 1)]], k=60)
    assert [f.chunk.id for f in fused] == ["b", "a"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_weighted_fusion_alpha() -> None:
    dense = [_hit("a", 0.9), _hit("b", 0.1)]
    sparse = [_hit("b", 5.0), _hit("a", 1.0)]
    assert weighted_fusion(dense, sparse, alpha=1.0)[0].chunk.id == "a"
    assert weighted_fusion(dense, sparse, alpha=0.0)[0].chunk.id == "b"


def test_compare_reports_latency_and_llm_calls(lab: RagLab) -> None:
    runs = {r.strategy: r for r in lab.compare("refund", names=["dense", "multi_query"])}
    assert runs["dense"].llm_calls == 0
    assert runs["multi_query"].llm_calls == 1
    assert runs["dense"].latency_ms >= 0
