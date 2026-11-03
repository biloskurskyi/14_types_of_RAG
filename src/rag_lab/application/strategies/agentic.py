"""Agentic RAG: a tool-calling loop where the LLM decides which searches to run.

The loop is bounded by ``max_steps``; every tool call is traced. Tools are thin wrappers over
other strategies, so the agent composes existing retrievers instead of re-implementing them.
With ``FakeLLM`` the policy is fixed (vector search, then translated keyword search); with a
real model the choice of tools and queries is up to the model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rag_lab.application.prompts import SYSTEM_AGENT
from rag_lab.application.strategies.base import RetrievalStrategy, register
from rag_lab.application.strategies.fusion import reciprocal_rank_fusion
from rag_lab.domain.models import ChatMessage, Query, RetrievedChunk, ToolSpec

_QUERY_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "search query"}},
    "required": ["query"],
}

TOOLS = (
    ToolSpec("vector_search", "Semantic search over CloudBox documents.", _QUERY_SCHEMA),
    ToolSpec("keyword_search", "Exact keyword search (ids, numbers, rare terms).", _QUERY_SCHEMA),
    ToolSpec(
        "read_document",
        "Return the full parent section for a chunk id seen in earlier results.",
        {
            "type": "object",
            "properties": {"chunk_id": {"type": "string"}},
            "required": ["chunk_id"],
        },
    ),
)


@register("agentic")
class AgenticStrategy(RetrievalStrategy):
    summary = "LLM tool loop over retrievers (bounded steps)"

    def _retrieve(self, query: Query) -> list[RetrievedChunk]:
        max_steps = int(self.params.get("max_steps", 4))
        tools: dict[str, Callable[[dict[str, Any]], list[RetrievedChunk]]] = {
            "vector_search": lambda a: self.dense(
                query, query.k, str(a.get("query") or query.text)
            ),
            "keyword_search": lambda a: self.sparse(
                query, query.k, str(a.get("query") or query.text)
            ),
            "read_document": self._read_document,
        }
        messages = [ChatMessage("system", SYSTEM_AGENT), ChatMessage("user", query.text)]
        rankings: list[list[RetrievedChunk]] = []
        for step in range(max_steps):
            response = self.ctx.llm.chat(messages, TOOLS)
            if not response.wants_tools:
                break
            messages.append(ChatMessage("assistant", response.content, response.tool_calls))
            for call in response.tool_calls:
                with self.ctx.tracer.span(
                    "agent.tool", step=step, tool=call.name, args=call.arguments
                ):
                    handler = tools.get(call.name)
                    hits = handler(call.arguments) if handler else []
                rankings.append(hits)
                body = "\n".join(f"[{h.chunk.id}] {h.chunk.text}" for h in hits) or "NO_RESULTS"
                messages.append(ChatMessage("tool", body, tool_call_id=call.id))
        else:
            with self.ctx.tracer.span("agent.max_steps_reached", max_steps=max_steps):
                pass
        return reciprocal_rank_fusion(rankings, source="agentic")

    def _read_document(self, args: dict[str, Any]) -> list[RetrievedChunk]:
        chunk = self.ctx.vector_store.get(str(args.get("chunk_id", "")))
        parent = self.ctx.doc_store.get(chunk.parent_id) if chunk and chunk.parent_id else None
        return [RetrievedChunk(parent, 1.0, "agentic")] if parent else []
