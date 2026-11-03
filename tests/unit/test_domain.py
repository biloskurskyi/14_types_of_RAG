import dataclasses

import pytest

from rag_lab.config import Settings
from rag_lab.domain import Chunk, Query, RetrievedChunk


def test_chunk_is_immutable() -> None:
    chunk = Chunk(id="c1", doc_id="d1", text="hello")
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.text = "changed"  # type: ignore[misc]
    assert chunk.with_text("short").text == "short"
    assert chunk.text == "hello"


def test_query_copies() -> None:
    query = Query("refund?", k=3)
    filtered = query.with_filters({"lang": "ua"})
    assert filtered.filters == {"lang": "ua"}
    assert query.filters == {}
    assert filtered.with_k(7).k == 7


def test_retrieved_chunk_rescored_keeps_chunk() -> None:
    hit = RetrievedChunk(Chunk(id="c1", doc_id="d1", text="x"), 0.5, "dense")
    again = hit.rescored(0.9)
    assert again.chunk is hit.chunk
    assert again.score == 0.9
    assert again.source == "dense"
    assert again.doc_id == "d1"


def test_settings_default_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_LAB_LLM", raising=False)
    settings = Settings(_env_file=None)
    assert settings.backend == "memory"
    assert settings.llm == "fake"
    assert settings.embedder == "hashing"
    assert settings.openai_api_key.get_secret_value() == ""
