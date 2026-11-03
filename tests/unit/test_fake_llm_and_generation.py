import json

from rag_lab.application.generation import AnswerGenerator, parse_citations
from rag_lab.application.prompts import parse_sections, render
from rag_lab.domain.models import ChatMessage, Chunk, RetrievedChunk, ToolSpec
from rag_lab.infrastructure.llm.fake import FakeLLM
from rag_lab.infrastructure.llm.tracing import TracingLLM
from rag_lab.infrastructure.tracing import JsonTracer, NullTracer, span_names

CONTEXT = [
    RetrievedChunk(
        Chunk("x#0", "x", "Annual discount: 15% (prepaid). Overage: $0.02 per GB."), 1.0
    ),
    RetrievedChunk(Chunk("y#0", "y", "Uptime target: 99.9% monthly."), 0.5),
]


def test_render_and_parse_roundtrip() -> None:
    prompt = render("hyde", {"question": "What is the SLA?"})
    task, sections = parse_sections(prompt)
    assert task == "hyde"
    assert sections["question"] == "What is the SLA?"


def test_rewrite_queries_translates_terms() -> None:
    llm = FakeLLM()
    out = llm.generate(render("rewrite_queries", {"question": "refund policy", "n": "3"}, n="3"))
    assert "повернення" in out
    assert len(out.splitlines()) <= 3


def test_grade_and_self_query_return_json() -> None:
    llm = FakeLLM()
    grade = json.loads(
        llm.generate(
            render(
                "grade_relevance",
                {"question": "annual discount", "document": "Annual discount 15%"},
            )
        )
    )
    assert grade["score"] == 1.0
    parsed = json.loads(
        llm.generate(
            render(
                "self_query",
                {"question": "In Ukrainian docs: trash retention", "doc_types": "pricing"},
                doc_types="pricing",
            )
        )
    )
    assert parsed["filters"] == {"lang": "ua"}


def test_answer_is_extractive_with_citations() -> None:
    tracer = JsonTracer()
    llm = TracingLLM(FakeLLM(), tracer)
    answer = AnswerGenerator(llm, tracer).generate("What is the annual discount?", CONTEXT, "dense")
    assert "15%" in answer.text
    assert answer.citations[0].chunk_id == "x#0"
    assert "llm.generate" in span_names(tracer.export())
    assert llm.call_count == 1


def test_unknown_context_gives_dont_know() -> None:
    answer = AnswerGenerator(FakeLLM(), NullTracer()).generate("Who is the CEO?", CONTEXT, "dense")
    assert "don't know" in answer.text
    assert answer.citations == ()


def test_hallucinated_citation_markers_are_ignored() -> None:
    citations = parse_citations("Yes [1]. Also [7]. Again [1].", CONTEXT)
    assert [c.marker for c in citations] == [1]


def test_fake_llm_tool_policy() -> None:
    tools = [ToolSpec("vector_search", "", {}), ToolSpec("keyword_search", "", {})]
    llm = FakeLLM()
    first = llm.chat([ChatMessage("user", "refund for annual plan")], tools)
    assert first.tool_calls[0].name == "vector_search"
    second = llm.chat(
        [ChatMessage("user", "refund"), ChatMessage("tool", "...", tool_call_id="call_0")], tools
    )
    assert second.tool_calls[0].name == "keyword_search"
    done = llm.chat([ChatMessage("user", "q"), *[ChatMessage("tool", "r")] * 2], tools)
    assert not done.wants_tools
