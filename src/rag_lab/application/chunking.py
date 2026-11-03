"""Chunking strategies: fixed window, recursive separators and sentence packing.

All chunkers are deterministic and produce stable ids ``<doc_id>#<n>`` so that evaluation
results and traces are comparable between runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from rag_lab.domain.models import Chunk, Document
from rag_lab.domain.text import split_sentences


class Chunker(ABC):
    """Splits a document into chunks no longer than ``size`` characters (best effort)."""

    def __init__(self, size: int = 320, overlap: int = 40) -> None:
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size = size
        self.overlap = overlap

    @abstractmethod
    def split_text(self, text: str) -> list[str]: ...

    def split(self, doc: Document, *, id_prefix: str | None = None) -> list[Chunk]:
        prefix = id_prefix if id_prefix is not None else f"{doc.id}#"
        return [
            Chunk(
                id=f"{prefix}{i}",
                doc_id=doc.id,
                text=piece,
                position=i,
                metadata=dict(doc.metadata),
            )
            for i, piece in enumerate(p for p in self.split_text(doc.text) if p.strip())
        ]


class FixedSizeChunker(Chunker):
    """Sliding character window with overlap. Simple, but cuts words and sentences."""

    def split_text(self, text: str) -> list[str]:
        step = self.size - self.overlap
        return [text[i : i + self.size].strip() for i in range(0, max(len(text), 1), step)]


class RecursiveChunker(Chunker):
    """Split on the coarsest separator that fits (paragraph, line, sentence, word)."""

    separators: Sequence[str] = ("\n\n", "\n", ". ", " ")

    def split_text(self, text: str) -> list[str]:
        return self._merge(self._split(text.strip(), 0))

    def _split(self, text: str, level: int) -> list[str]:
        if len(text) <= self.size or level >= len(self.separators):
            return [text]
        sep = self.separators[level]
        parts = [p for p in text.split(sep) if p.strip()]
        out: list[str] = []
        for part in parts:
            out.extend(self._split(part.strip(), level + 1))
        return out

    def _merge(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current}\n{piece}" if current else piece
            if len(candidate) <= self.size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            tail = current[-self.overlap :] if self.overlap and current else ""
            tail = tail.split(" ", 1)[-1] if " " in tail else ""
            current = f"{tail} {piece}".strip() if tail else piece
        if current:
            chunks.append(current)
        return chunks


class SentenceChunker(Chunker):
    """Pack whole sentences until ``size``; overlap is measured in sentences (1 if > 0)."""

    def split_text(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        chunks: list[str] = []
        current: list[str] = []
        for sentence in sentences:
            if current and len(" ".join([*current, sentence])) > self.size:
                chunks.append(" ".join(current))
                current = current[-1:] if self.overlap else []
            current.append(sentence)
        if current:
            chunks.append(" ".join(current))
        return chunks


CHUNKERS: dict[str, type[Chunker]] = {
    "fixed": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "sentence": SentenceChunker,
}


def make_chunker(name: str, size: int, overlap: int) -> Chunker:
    try:
        return CHUNKERS[name](size=size, overlap=overlap)
    except KeyError as exc:
        raise ValueError(f"unknown chunker {name!r}; choose from {sorted(CHUNKERS)}") from exc
