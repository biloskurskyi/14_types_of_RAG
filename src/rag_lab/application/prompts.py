"""Prompt templates.

Every prompt starts with ``### task: <name>`` and uses ``### <section>`` headers. A real LLM
reads them as structured instructions; the offline ``FakeLLM`` dispatches on the task name and
parses the sections, which keeps every strategy testable without network access.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_lab.domain.models import RetrievedChunk

INSTRUCTIONS: dict[str, str] = {
    "rewrite_queries": (
        "Rewrite the question into {n} alternative search queries, one per line. Use synonyms "
        "and translate key terms between English and Ukrainian. Output only the queries."
    ),
    "hyde": (
        "Write a short passage (2-3 sentences) that would answer the question, as if taken from "
        "the company's documentation. Invent plausible details; it is only used for search."
    ),
    "grade_relevance": (
        'Rate how relevant the document is to the question. Reply with JSON {{"score": <0..1>}}.'
    ),
    "compress": (
        "Copy only the sentences from the document that help answer the question. "
        "Reply NONE if nothing is relevant."
    ),
    "self_query": (
        "Extract metadata filters from the question. Allowed keys: lang (en|ua), doc_type "
        "({doc_types}). Reply with JSON "
        '{{"query": "<question without filter words>", "filters": {{...}}}}.'
    ),
    "answer": (
        "Answer the question using only the numbered context. Cite sources as [n] after each "
        "sentence. If the context does not contain the answer, say you don't know."
    ),
    "judge_faithfulness": (
        "Score how well the answer is supported by the context. Reply with JSON "
        '{{"score": <0..1>, "reason": "..."}}.'
    ),
}

SYSTEM_AGENT = (
    "You are a retrieval agent. Use the tools to collect passages that answer the user's "
    "question. Call several tools if the first results look weak, then reply 'DONE'."
)


def render(task: str, sections: dict[str, str], **fmt: str) -> str:
    lines = [f"### task: {task}", INSTRUCTIONS[task].format(**fmt)]
    for name, body in sections.items():
        lines.extend((f"### {name}", body.strip()))
    return "\n".join(lines)


def number_context(chunks: Sequence[RetrievedChunk]) -> str:
    return "\n".join(f"[{i}] {c.chunk.text}" for i, c in enumerate(chunks, start=1))


def parse_sections(prompt: str) -> tuple[str, dict[str, str]]:
    """Inverse of :func:`render`: returns ``(task, {section: body})``."""
    task = ""
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in prompt.splitlines():
        if line.startswith("### task:"):
            task = line.split(":", 1)[1].strip()
            current = None
        elif line.startswith("### "):
            current = sections.setdefault(line[4:].strip(), [])
        elif current is not None:
            current.append(line)
    return task, {k: "\n".join(v).strip() for k, v in sections.items()}
