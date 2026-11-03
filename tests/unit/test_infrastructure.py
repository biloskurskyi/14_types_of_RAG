import math

from rag_lab.domain.models import Chunk, RetrievedChunk
from rag_lab.domain.ports import Embedder, KeywordIndex, VectorStore
from rag_lab.infrastructure.embedders.caching import CachingEmbedder
from rag_lab.infrastructure.embedders.hashing import HashingEmbedder
from rag_lab.infrastructure.loaders import metadata_from_name
from rag_lab.infrastructure.rerankers import LexicalReranker
from rag_lab.infrastructure.stores.bm25 import BM25Index
from rag_lab.infrastructure.stores.memory import InMemoryVectorStore, dot, matches

CHUNKS = [
    Chunk("a", "d1", "Refunds for annual plans are prorated", metadata={"lang": "en"}),
    Chunk("b", "d2", "Pro plan costs 19.90 USD per month", metadata={"lang": "en"}),
    Chunk("c", "d3", "Повернення коштів за річні платежі", metadata={"lang": "ua"}),
]


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    emb = HashingEmbedder(dim=64)
    v1, v2 = emb.embed(["refund policy", "refund policy"])
    assert v1 == v2
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0, rel_tol=1e-9)
    near, far = emb.embed(["refunds policies", "data center location"])
    assert dot(v1, near) > dot(v1, far)
    assert isinstance(emb, Embedder)


def test_caching_embedder_counts_hits() -> None:
    cached = CachingEmbedder(HashingEmbedder(dim=32))
    cached.embed(["a", "b"])
    cached.embed(["a", "c"])
    assert (cached.hits, cached.misses) == (1, 3)
    assert cached.dim == 32


def test_in_memory_store_search_and_filters() -> None:
    emb = HashingEmbedder(dim=128)
    store = InMemoryVectorStore()
    store.add(CHUNKS, emb.embed([c.text for c in CHUNKS]))
    assert isinstance(store, VectorStore)
    top = store.search(emb.embed(["pro plan cost"])[0], k=1)
    assert top[0].chunk.id == "b"
    ua = store.search(emb.embed(["refund"])[0], k=5, filters={"lang": "ua"})
    assert [h.chunk.id for h in ua] == ["c"]
    assert set(store.vectors_for(["a", "zzz"])) == {"a"}


def test_matches_supports_any_of() -> None:
    assert matches({"lang": "ua"}, {"lang": ["en", "ua"]})
    assert not matches({"lang": "ua"}, {"lang": "en"})
    assert matches({}, None)


def test_bm25_ranks_exact_terms() -> None:
    index = BM25Index()
    index.add(CHUNKS)
    assert isinstance(index, KeywordIndex)
    hits = index.search("pro plan price", k=3)
    assert hits[0].chunk.id == "b"
    assert index.search("nothing matches here", k=3) == []
    assert index.search("повернення", k=3, filters={"lang": "en"}) == []
    assert index.idf("pro") > index.idf("plan")


def test_lexical_reranker_prefers_coverage() -> None:
    candidates = [RetrievedChunk(CHUNKS[0], 0.9), RetrievedChunk(CHUNKS[1], 0.1)]
    reranked = LexicalReranker().rerank("pro plan costs per month", candidates, k=2)
    assert reranked[0].chunk.id == "b"
    assert reranked[0].source == "rerank"


def test_metadata_from_file_name(tmp_path) -> None:
    meta = metadata_from_name(tmp_path / "05_refunds_policy_ua.txt")
    assert meta == {
        "doc_type": "refunds_policy",
        "lang": "ua",
        "source": "05_refunds_policy_ua.txt",
    }
