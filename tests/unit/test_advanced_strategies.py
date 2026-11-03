import pytest

from rag_lab.application.service import RagLab
from rag_lab.application.strategies.adaptive import default_router
from rag_lab.application.strategies.corrective import CorrectiveStrategy, _parse_score
from rag_lab.application.strategies.graph import extract_entities
from rag_lab.domain.models import ChatResponse, Query, ToolCall
from rag_lab.infrastructure.tracing import span_names


def names(lab: RagLab) -> list[str]:
    return span_names(lab.ctx.tracer.export())


def find_span(spans: list[dict], name: str) -> dict:
    for span in spans:
        if span["name"] == name:
            return span
        found = find_span(span["children"], name)
        if found:
            return found
    return {}


def test_corrective_keeps_relevant_chunks(lab: RagLab) -> None:
    hits = lab.strategy("corrective").retrieve(Query("API rate limit Pro plan", k=3))
    assert hits[0].doc_id == "07_api_limits_en"
    assert "corrective.grades" in names(lab)
    handled = find_span(lab.ctx.tracer.export(), "corrective.handled_by")
    assert handled["attrs"]["handler"] == "KeepRelevant"


def test_corrective_rewrites_when_grades_are_low(lab: RagLab) -> None:
    strategy = lab.strategy("corrective", threshold=0.99, min_relevant=4)
    assert isinstance(strategy, CorrectiveStrategy)
    hits = strategy.retrieve(Query("How are annual payments refunded?", k=3))
    assert hits
    assert strategy.last_variants  # RewriteAndRetry ran
    assert "05_refunds_policy_ua" in [h.doc_id for h in hits]


def test_parse_score_is_defensive() -> None:
    assert _parse_score('noise {"score": 0.7} noise') == 0.7
    assert _parse_score("no json") == 0.0


def test_agentic_loop_calls_tools_and_stops(lab: RagLab) -> None:
    hits = lab.strategy("agentic").retrieve(
        Query("How long are files kept after termination?", k=4)
    )
    tool_spans = [n for n in names(lab) if n == "agent.tool"]
    assert len(tool_spans) == 2
    assert "08_data_retention_ua" in [h.doc_id for h in hits]


class LoopingLLM:
    """Always asks for another tool call: the loop must stop at max_steps."""

    name = "looping"

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return ""

    def chat(self, messages, tools=()):  # type: ignore[no-untyped-def]
        return ChatResponse(tool_calls=(ToolCall("c", "vector_search", {"query": "plan"}),))


def test_agentic_respects_max_steps(lab: RagLab) -> None:
    lab.ctx.llm = LoopingLLM()
    hits = lab.strategy("agentic", max_steps=3).retrieve(Query("plan", k=2))
    assert hits
    assert names(lab).count("agent.tool") == 3
    assert "agent.max_steps_reached" in names(lab)


def test_agentic_read_document_tool(lab: RagLab) -> None:
    chunk = lab.ctx.vector_store.all_chunks()[0]
    strategy = lab.strategy("agentic")
    parents = strategy._read_document({"chunk_id": chunk.id})
    assert parents[0].chunk.id == chunk.parent_id
    assert strategy._read_document({"chunk_id": "missing"}) == []


def test_entity_extraction() -> None:
    entities = extract_entities(
        "Pro plan: 60 requests, SSO via SAML, invoice CB-2025-1103, 99.9% uptime"
    )
    assert {"pro", "sso", "saml", "cb-2025-1103", "99.9%"} <= entities


def test_graph_strategy_uses_entities(lab: RagLab) -> None:
    hits = lab.strategy("graph").retrieve(Query("Does the Team plan include SSO?", k=3))
    assert hits[0].doc_id == "06_security_en"
    graph_span = find_span(lab.ctx.tracer.export(), "graph.expand")
    assert "sso" in graph_span["attrs"]["entities"]


@pytest.mark.parametrize(
    ("question", "target"),
    [
        ("Total on invoice CB-2025-1103", "bm25"),
        ("Ukrainian docs: trash retention", "self_query"),
        ("Compare Pro and Team support", "agentic"),
        ("How are refunds calculated?", "multi_query"),
        ("Скільки днів тестовий період?", "hybrid_rrf"),
    ],
)
def test_router_rules(question: str, target: str) -> None:
    assert default_router().route(Query(question))[0] == target


def test_adaptive_delegates_and_traces_route(lab: RagLab) -> None:
    hits = lab.strategy("adaptive").retrieve(Query("CB-2025-1103", k=2))
    assert hits[0].doc_id == "03_invoice_example_ua"
    assert {"adaptive.route", "strategy.bm25"} <= set(names(lab))
