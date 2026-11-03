import pytest

from rag_lab.application.chunking import (
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
    make_chunker,
)
from rag_lab.domain.models import Document
from rag_lab.domain.text import content_terms, split_sentences, stem, tokenize

TEXT = (
    "First paragraph has two sentences. It talks about refunds.\n\n"
    "Second paragraph is about pricing. Pro costs $19.90 per month.\n"
    "Third line mentions invoice CB-2025-1103 and VAT 20%."
)


def test_tokenize_keeps_codes_numbers_and_cyrillic() -> None:
    tokens = tokenize("Invoice CB-2025-1103 costs $19.90, ПДВ 20%.")
    assert "cb-2025-1103" in tokens
    assert "$19.90" in tokens
    assert "пдв" in tokens
    assert "20%" in tokens


def test_stem_and_content_terms() -> None:
    assert stem("refunds") == stem("refund")
    assert stem("повернення") == stem("поверненню")
    assert "the" not in content_terms("the refund")


def test_split_sentences() -> None:
    assert split_sentences("One. Two!\nThree") == ["One.", "Two!", "Three"]


@pytest.mark.parametrize("cls", [FixedSizeChunker, RecursiveChunker, SentenceChunker])
def test_chunkers_cover_text_with_stable_ids(cls: type) -> None:
    doc = Document(id="d", text=TEXT, metadata={"lang": "en"})
    chunks = cls(size=80, overlap=10).split(doc)
    assert len(chunks) >= 2
    assert [c.id for c in chunks] == [f"d#{i}" for i in range(len(chunks))]
    assert all(c.metadata == {"lang": "en"} for c in chunks)
    assert "CB-2025-1103" in " ".join(c.text for c in chunks)


def test_recursive_respects_size_and_paragraphs() -> None:
    chunks = RecursiveChunker(size=80, overlap=0).split_text(TEXT)
    assert all(len(c) <= 80 for c in chunks)
    assert chunks[0].startswith("First paragraph")


def test_sentence_chunker_never_cuts_sentences() -> None:
    sentences = set(split_sentences(TEXT))
    for chunk in SentenceChunker(size=60, overlap=0).split_text(TEXT):
        assert all(part in sentences for part in split_sentences(chunk))


def test_fixed_chunker_overlap() -> None:
    chunks = FixedSizeChunker(size=50, overlap=10).split_text("x" * 120)
    assert len(chunks) == 3


def test_invalid_chunker_config() -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(size=10, overlap=10)
    with pytest.raises(ValueError):
        make_chunker("nope", 100, 10)
