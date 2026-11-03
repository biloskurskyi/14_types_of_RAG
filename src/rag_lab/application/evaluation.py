"""Offline evaluation: retrieval metrics, a faithfulness judge and a runner over golden Q&A.

Retrieval is scored at document level (a strategy gets credit for any chunk of a relevant
document), which keeps metrics comparable across chunkers:

* ``hit@k``  – at least one relevant document in the top k;
* ``MRR``    – 1 / rank of the first relevant document;
* ``nDCG@k`` – binary-relevance nDCG over the de-duplicated document ranking.

Answer quality uses two cheap signals: ``answer_recall`` (expected facts appear in the answer)
and ``faithfulness`` (answer sentences are supported by the retrieved context), judged by a
deterministic overlap judge or, optionally, an LLM-as-judge.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rag_lab.application.prompts import number_context, render
from rag_lab.application.service import RagLab
from rag_lab.domain.models import Answer
from rag_lab.domain.ports import LLM


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    relevant: tuple[str, ...]
    expected: tuple[str, ...] = ()


def load_golden(path: Path) -> list[GoldenItem]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [
        GoldenItem(
            id=str(item["id"]),
            question=str(item["question"]),
            relevant=tuple(item["relevant"]),
            expected=tuple(item.get("expected", ())),
        )
        for item in raw
    ]


def doc_ranking(doc_ids: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(doc_ids))


def hit_at_k(ranking: Sequence[str], relevant: Sequence[str], k: int) -> float:
    return 1.0 if set(ranking[:k]) & set(relevant) else 0.0


def reciprocal_rank(ranking: Sequence[str], relevant: Sequence[str]) -> float:
    for i, doc in enumerate(ranking, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranking: Sequence[str], relevant: Sequence[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranking[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def answer_recall(answer: str, expected: Sequence[str]) -> float:
    if not expected:
        return 1.0
    lowered = answer.lower()
    return sum(e.lower() in lowered for e in expected) / len(expected)


class OverlapJudge:
    """Deterministic faithfulness: share of answer sentences mostly covered by the context."""

    name = "overlap"

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def faithfulness(self, answer: Answer) -> float:
        if "don't know" in answer.text.lower():
            return 1.0  # abstaining is faithful
        prompt = render(
            "judge_faithfulness",
            {"answer": answer.text, "context": number_context(answer.context)},
        )
        try:
            raw = self.llm.generate(prompt)
            return float(json.loads(raw[raw.index("{") : raw.rindex("}") + 1])["score"])
        except (ValueError, KeyError, json.JSONDecodeError):
            return 0.0


class LLMJudge(OverlapJudge):
    """Same prompt, sent to a real model (LLM-as-judge). Opt-in via ``rag-lab eval --judge llm``."""

    name = "llm"


@dataclass
class StrategyScore:
    strategy: str
    hit_at_k: float
    mrr: float
    ndcg: float
    answer_recall: float
    faithfulness: float
    latency_ms: float
    llm_calls: float
    per_question: list[dict[str, object]] = field(default_factory=list)

    def row(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "hit@k": round(self.hit_at_k, 3),
            "mrr": round(self.mrr, 3),
            "ndcg@k": round(self.ndcg, 3),
            "ans_recall": round(self.answer_recall, 3),
            "faithful": round(self.faithfulness, 3),
            "p50_ms": round(self.latency_ms, 2),
            "llm_calls": round(self.llm_calls, 1),
        }


class EvalRunner:
    def __init__(self, lab: RagLab, judge: OverlapJudge, k: int = 4) -> None:
        self.lab = lab
        self.judge = judge
        self.k = k

    def run_strategy(self, name: str, items: Sequence[GoldenItem]) -> StrategyScore:
        rows: list[dict[str, object]] = []
        for item in items:
            run = self.lab.retrieve(name, item.question, self.k)
            answer = self.lab.generator.generate(item.question, run.hits, name)
            ranking = doc_ranking([h.doc_id for h in run.hits])
            rows.append(
                {
                    "id": item.id,
                    "hit": hit_at_k(ranking, item.relevant, self.k),
                    "rr": reciprocal_rank(ranking, item.relevant),
                    "ndcg": ndcg_at_k(ranking, item.relevant, self.k),
                    "recall": answer_recall(answer.text, item.expected),
                    "faithful": self.judge.faithfulness(answer),
                    "latency_ms": run.latency_ms,
                    "llm_calls": run.llm_calls,
                    "docs": ranking,
                }
            )

        def mean(key: str) -> float:
            return statistics.fmean(float(r[key]) for r in rows) if rows else 0.0  # type: ignore[arg-type]

        latency = statistics.median(float(r["latency_ms"]) for r in rows) if rows else 0.0  # type: ignore[arg-type]
        return StrategyScore(
            name,
            mean("hit"),
            mean("rr"),
            mean("ndcg"),
            mean("recall"),
            mean("faithful"),
            latency,
            mean("llm_calls"),
            rows,
        )

    def run(self, names: Sequence[str], items: Sequence[GoldenItem]) -> list[StrategyScore]:
        return [self.run_strategy(n, items) for n in names]
