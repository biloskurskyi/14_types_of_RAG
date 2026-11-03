"""Core value objects shared by every layer.

They are frozen dataclasses on purpose: strategies pass them around freely and must never
mutate a chunk that another strategy (or the index) still holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

Metadata = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Document:
    """A source document before chunking."""

    id: str
    text: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit of text.

    ``parent_id`` links a small child chunk to its larger parent (small-to-big retrieval).
    """

    id: str
    doc_id: str
    text: str
    position: int = 0
    metadata: Metadata = field(default_factory=dict)
    parent_id: str | None = None

    def with_text(self, text: str) -> Chunk:
        """Return a copy with replaced text (used by contextual compression)."""
        return replace(self, text=text)


@dataclass(frozen=True, slots=True)
class Query:
    """A user question plus retrieval options.

    ``filters`` is an exact-match metadata filter, e.g. ``{"lang": "ua"}``.
    """

    text: str
    k: int = 4
    filters: Metadata = field(default_factory=dict)

    def with_text(self, text: str) -> Query:
        return replace(self, text=text)

    def with_filters(self, filters: Metadata) -> Query:
        return replace(self, filters=dict(filters))

    def with_k(self, k: int) -> Query:
        return replace(self, k=k)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk with the score a retriever assigned to it and the retriever's name."""

    chunk: Chunk
    score: float
    source: str = ""

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    def rescored(self, score: float, source: str | None = None) -> RetrievedChunk:
        return RetrievedChunk(self.chunk, score, source if source is not None else self.source)


@dataclass(frozen=True, slots=True)
class Citation:
    """A reference from an answer back to the chunk that supports it."""

    marker: int
    chunk_id: str
    doc_id: str


@dataclass(frozen=True, slots=True)
class Answer:
    """Generated answer with citations and the context it was grounded on."""

    question: str
    text: str
    strategy: str
    citations: tuple[Citation, ...] = ()
    context: tuple[RetrievedChunk, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool an LLM may call, described by a JSON schema for its arguments."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message in a chat transcript (``role`` is system/user/assistant/tool)."""

    role: str
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """LLM reply: either final ``content`` or a request to run ``tool_calls``."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)
