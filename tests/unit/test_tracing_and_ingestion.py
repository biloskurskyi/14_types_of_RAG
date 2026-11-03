import json

import pytest

from rag_lab.application.service import RagLab
from rag_lab.domain.text import content_terms
from rag_lab.infrastructure.tracing import JsonTracer, span_names


def test_json_tracer_nests_and_records_errors(tmp_path) -> None:
    tracer = JsonTracer()
    with tracer.span("outer", a=1) as attrs:
        attrs["b"] = 2
        with tracer.span("inner"):
            pass
    with pytest.raises(RuntimeError), tracer.span("boom"):
        raise RuntimeError("x")
    spans = tracer.export()
    assert span_names(spans) == ["outer", "inner", "boom"]
    assert spans[0]["attrs"] == {"a": 1, "b": 2}
    assert "RuntimeError" in spans[1]["error"]
    path = tracer.dump(tmp_path / "t.json")
    assert json.loads(path.read_text())[0]["name"] == "outer"


def test_ingestion_builds_parent_child_hierarchy(lab: RagLab) -> None:
    chunks = lab.ctx.vector_store.all_chunks()
    assert len(chunks) >= 10
    for chunk in chunks:
        assert chunk.parent_id is not None
        parent = lab.ctx.doc_store.get(chunk.parent_id)
        assert parent is not None
        assert set(content_terms(chunk.text)) <= set(content_terms(parent.text))
        assert chunk.metadata["lang"] in {"en", "ua"}
    assert "refunds_policy" in lab.ctx.doc_types
