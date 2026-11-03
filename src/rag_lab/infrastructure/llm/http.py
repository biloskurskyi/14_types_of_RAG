"""Optional HTTP LLM adapters (stdlib only): OpenAI-compatible ``/chat/completions`` and Ollama.

They are never used by tests or CI. Configure them through ``RAG_LAB_LLM`` and related env vars;
the API key is read from settings and never logged.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Sequence
from typing import Any

from rag_lab.domain.models import ChatMessage, ChatResponse, ToolCall, ToolSpec


def _post(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _to_openai_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        item: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            item["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in m.tool_calls
            ]
        if m.tool_call_id:
            item["tool_call_id"] = m.tool_call_id
        out.append(item)
    return out


def _tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


class OpenAICompatibleLLM:
    """Works with OpenAI, Azure-style gateways, vLLM, LM Studio, llama.cpp server, etc."""

    def __init__(
        self, base_url: str, api_key: str, model: str, timeout: float = 60.0, temperature: float = 0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        messages = [ChatMessage("system", system)] if system else []
        return self.chat([*messages, ChatMessage("user", prompt)]).content

    def chat(self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai_messages(messages),
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = _tools(tools)
        data = _post(f"{self.base_url}/chat/completions", payload, self._headers(), self.timeout)
        message = data["choices"][0]["message"]
        calls = tuple(
            ToolCall(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"] or "{}"))
            for c in message.get("tool_calls") or ()
        )
        return ChatResponse(content=message.get("content") or "", tool_calls=calls)


class OllamaLLM:
    """Local models through Ollama's ``/api/chat`` (tool calling supported by recent models)."""

    def __init__(self, url: str, model: str, timeout: float = 120.0) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        messages = [ChatMessage("system", system)] if system else []
        return self.chat([*messages, ChatMessage("user", prompt)]).content

    def chat(self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    **(
                        {
                            "tool_calls": [
                                {"function": {"name": c.name, "arguments": c.arguments}}
                                for c in m.tool_calls
                            ]
                        }
                        if m.tool_calls
                        else {}
                    ),
                }
                for m in messages
            ],
        }
        if tools:
            payload["tools"] = _tools(tools)
        data = _post(f"{self.url}/api/chat", payload, {}, self.timeout)
        message = data.get("message", {})
        calls = tuple(
            ToolCall(f"call_{i}", c["function"]["name"], dict(c["function"].get("arguments", {})))
            for i, c in enumerate(message.get("tool_calls") or ())
        )
        return ChatResponse(content=message.get("content") or "", tool_calls=calls)
