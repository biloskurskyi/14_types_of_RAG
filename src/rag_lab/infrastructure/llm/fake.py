"""``FakeLLM``: a deterministic, extractive stand-in for a chat model.

It understands the task prompts from :mod:`rag_lab.application.prompts` and answers them
with simple heuristics (keyword overlap, the bilingual glossary, sentence extraction). It also
plays a fixed tool-calling policy so the agentic loop runs offline:

1. call ``vector_search`` with the question;
2. call ``keyword_search`` with the question translated through the glossary;
3. reply ``DONE``.

The goal is not to be smart but to be reproducible: every strategy gets exercised end-to-end
in unit tests and CI without a key, a model download or network access.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence

from rag_lab.application.prompts import parse_sections
from rag_lab.domain.models import ChatMessage, ChatResponse, ToolCall, ToolSpec
from rag_lab.domain.text import STOPWORDS, content_terms, split_sentences, stem, tokenize
from rag_lab.infrastructure.llm.glossary import synonyms, translate

_CONTEXT_LINE = re.compile(r"^\[(\d+)\]\s*(.*)$")
_LANG_HINTS = {"ua": ("ukrainian", "українськ", "in ua"), "en": ("english", "англійськ")}
_DOC_HINTS = {
    "invoice_example": ("the invoice", "invoice cb-", "у рахунку"),
    "contract": ("according to the contract", "in the contract", "у договорі"),
    "refunds_policy": ("refund policy", "політика повернення", "політиці повернення"),
    "sla_support": ("the sla", "in the sla"),
    "team_admin_faq": ("the faq", "admin faq"),
}


def keywords(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


def expanded_terms(text: str) -> set[str]:
    """Content stems of ``text`` plus stems of all glossary synonyms."""
    terms = set(content_terms(text))
    for token in keywords(text):
        for syn in synonyms(token):
            terms.update(content_terms(syn))
    return terms


def _context_blocks(context: str) -> list[tuple[int, str]]:
    """Parse ``[n] text`` blocks; lines without a marker continue the previous block."""
    blocks: list[tuple[int, str]] = []
    for line in context.splitlines():
        match = _CONTEXT_LINE.match(line)
        if match:
            blocks.append((int(match.group(1)), match.group(2)))
        elif blocks:
            blocks[-1] = (blocks[-1][0], f"{blocks[-1][1]}\n{line}")
    return blocks


def _other_script(token: str) -> list[str]:
    is_cyrillic = any("\u0400" <= ch <= "\u04ff" for ch in token)
    return [w for w in synonyms(token) if any("\u0400" <= c <= "\u04ff" for c in w) != is_cyrillic]


class FakeLLM:
    def __init__(self, max_answer_sentences: int = 2) -> None:
        self.max_answer_sentences = max_answer_sentences
        self.calls: list[str] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def name(self) -> str:
        return "fake"

    # ------------------------------------------------------------------ generate
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        task, sec = parse_sections(prompt)
        self.calls.append(task)
        handler = getattr(self, f"_task_{task}", None)
        if handler is None:
            return sec.get("question", prompt)[:200]
        return handler(sec)

    def _task_rewrite_queries(self, sec: dict[str, str]) -> str:
        question = sec.get("question", "")
        words = keywords(question)
        n = int(sec.get("n", "3") or 3)
        translated = [" ".join(_other_script(w)) or w for w in words]
        expanded = words + [s for w in words for s in synonyms(w)]
        variants = [" ".join(translated), " ".join(dict.fromkeys(expanded)), " ".join(words)]
        unique = [v for v in dict.fromkeys(variants) if v and v != question.lower()]
        return "\n".join(unique[:n])

    def _task_hyde(self, sec: dict[str, str]) -> str:
        words = keywords(sec.get("question", ""))
        translated = [t for w in words if (t := translate(w))]
        return (
            f"CloudBox documentation: {' '.join(words)}. "
            f"{' '.join(translated)}. The policy describes {', '.join(words[:3])}."
        )

    def _task_grade_relevance(self, sec: dict[str, str]) -> str:
        q_terms = set(content_terms(sec.get("question", "")))
        d_terms = set(content_terms(sec.get("document", "")))
        if not q_terms:
            return json.dumps({"score": 0.0})
        direct = len(q_terms & d_terms)
        cross = len((expanded_terms(sec.get("question", "")) - q_terms) & d_terms)
        score = min(1.0, (direct + 0.5 * cross) / len(q_terms))
        return json.dumps({"score": round(score, 3)})

    def _task_compress(self, sec: dict[str, str]) -> str:
        terms = expanded_terms(sec.get("question", ""))
        kept = [
            s for s in split_sentences(sec.get("document", "")) if terms & set(content_terms(s))
        ]
        return "\n".join(kept) if kept else "NONE"

    def _task_self_query(self, sec: dict[str, str]) -> str:
        question = sec.get("question", "")
        lowered = question.lower()
        filters: dict[str, str] = {}
        cleaned = question
        for lang, hints in _LANG_HINTS.items():
            for hint in hints:
                if hint in lowered:
                    filters["lang"] = lang
                    cleaned = re.sub(rf"\b{re.escape(hint)}\w*", "", cleaned, flags=re.I)
        allowed = set(sec.get("doc_types", "").split(", "))
        for doc_type, hints in _DOC_HINTS.items():
            if doc_type in allowed and any(h in lowered for h in hints):
                filters["doc_type"] = doc_type
                break
        return json.dumps({"query": " ".join(cleaned.split()), "filters": filters})

    def _task_answer(self, sec: dict[str, str]) -> str:
        terms = expanded_terms(sec.get("question", ""))
        sentences: list[tuple[int, int, str, set[str]]] = []
        for order, (marker, text) in enumerate(_context_blocks(sec.get("context", ""))):
            for sentence in split_sentences(text):
                sentences.append((order, marker, sentence, set(content_terms(sentence))))
        df: dict[str, int] = {}
        for *_, sent_terms in sentences:
            for term in sent_terms:
                df[term] = df.get(term, 0) + 1
        n = len(sentences) or 1
        candidates: list[tuple[float, int, int, str]] = []
        for order, marker, sentence, sent_terms in sentences:
            hit = terms & sent_terms
            if not hit or len(sent_terms) < 3:
                continue
            score = sum(math.log(1 + n / df[t]) for t in hit)
            score += 0.5 if any(ch.isdigit() for ch in sentence) else 0.0
            candidates.append((score, order, marker, sentence))
        if not candidates:
            return "I don't know based on the provided context."
        candidates.sort(key=lambda c: (-c[0], c[1]))
        picked: list[str] = []
        for _, _, marker, sentence in candidates:
            text = f"{sentence.rstrip('.')} [{marker}]."
            if text not in picked:
                picked.append(text)
            if len(picked) == self.max_answer_sentences:
                break
        return " ".join(picked)

    def _task_judge_faithfulness(self, sec: dict[str, str]) -> str:
        context_terms = set(content_terms(sec.get("context", "")))
        sentences = [re.sub(r"\[\d+\]", "", s) for s in split_sentences(sec.get("answer", "")) if s]
        if not sentences:
            return json.dumps({"score": 0.0, "reason": "empty answer"})
        supported = 0
        for sentence in sentences:
            terms = set(content_terms(sentence))
            if terms and len(terms & context_terms) / len(terms) >= 0.6:
                supported += 1
        score = supported / len(sentences)
        return json.dumps({"score": round(score, 3), "reason": f"{supported}/{len(sentences)}"})

    # ---------------------------------------------------------------------- chat
    def chat(self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()) -> ChatResponse:
        self.calls.append("chat")
        question = next((m.content for m in messages if m.role == "user"), "")
        tool_names = {t.name for t in tools}
        used = [m for m in messages if m.role == "tool"]
        step = len(used)
        if step == 0 and "vector_search" in tool_names:
            return self._call("vector_search", question, step)
        if step == 1 and "keyword_search" in tool_names:
            words = keywords(question)
            rewritten = " ".join(dict.fromkeys(words + [t for w in words if (t := translate(w))]))
            return self._call("keyword_search", rewritten, step)
        return ChatResponse(content="DONE")

    @staticmethod
    def _call(name: str, query: str, step: int) -> ChatResponse:
        return ChatResponse(tool_calls=(ToolCall(f"call_{step}", name, {"query": query}),))


__all__ = ["FakeLLM", "expanded_terms", "keywords", "stem"]
