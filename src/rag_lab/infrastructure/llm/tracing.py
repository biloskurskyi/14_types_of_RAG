"""Decorator that records every LLM call as a trace span (task name, latency, sizes)."""

from __future__ import annotations

from collections.abc import Sequence

from rag_lab.application.prompts import parse_sections
from rag_lab.domain.models import ChatMessage, ChatResponse, ToolSpec
from rag_lab.domain.ports import LLM, Tracer


class TracingLLM:
    def __init__(self, inner: LLM, tracer: Tracer) -> None:
        self.inner = inner
        self.tracer = tracer
        self.call_count = 0

    @property
    def name(self) -> str:
        return self.inner.name

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        task, _ = parse_sections(prompt)
        self.call_count += 1
        with self.tracer.span("llm.generate", model=self.name, task=task or "free") as attrs:
            out = self.inner.generate(prompt, system=system)
            attrs.update(prompt_chars=len(prompt), output_chars=len(out))
            return out

    def chat(self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()) -> ChatResponse:
        self.call_count += 1
        with self.tracer.span("llm.chat", model=self.name, messages=len(messages)) as attrs:
            response = self.inner.chat(messages, tools)
            attrs["tool_calls"] = [c.name for c in response.tool_calls]
            return response
