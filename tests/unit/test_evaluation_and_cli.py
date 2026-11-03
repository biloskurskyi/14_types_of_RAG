import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_lab.application.evaluation import (
    EvalRunner,
    OverlapJudge,
    answer_recall,
    doc_ranking,
    hit_at_k,
    load_golden,
    ndcg_at_k,
    reciprocal_rank,
)
from rag_lab.application.service import RagLab
from rag_lab.domain.models import Answer, Chunk, RetrievedChunk
from rag_lab.infrastructure.llm.fake import FakeLLM
from rag_lab.interfaces.cli import app

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "eval" / "golden.yaml"


def test_ranking_metrics() -> None:
    ranking = doc_ranking(["a", "a", "b", "c"])
    assert ranking == ["a", "b", "c"]
    assert hit_at_k(ranking, ["c"], 2) == 0.0
    assert hit_at_k(ranking, ["c"], 3) == 1.0
    assert reciprocal_rank(ranking, ["b"]) == 0.5
    assert reciprocal_rank(ranking, ["z"]) == 0.0
    assert ndcg_at_k(["a"], ["a"], 4) == 1.0
    assert ndcg_at_k(["x", "a"], ["a"], 4) == pytest.approx(1 / 1.5849625)


def test_answer_recall() -> None:
    assert answer_recall("Pro costs 19.90 USD", ["19.90"]) == 1.0
    assert answer_recall("no idea", ["19.90", "USD"]) == 0.0
    assert answer_recall("anything", []) == 1.0


def test_overlap_judge_flags_unsupported_answers() -> None:
    context = (RetrievedChunk(Chunk("c", "d", "Uptime target is 99.9% monthly."), 1.0),)
    judge = OverlapJudge(FakeLLM())
    good = Answer("q", "Uptime target is 99.9% monthly [1].", "dense", context=context)
    bad = Answer("q", "Bananas are shipped from Ecuador weekly.", "dense", context=context)
    assert judge.faithfulness(good) == 1.0
    assert judge.faithfulness(bad) == 0.0


def test_golden_set_loads() -> None:
    items = load_golden(GOLDEN)
    assert len(items) >= 15
    corpus = {p.stem for p in (ROOT / "corpus").glob("*.txt")}
    assert all(set(i.relevant) <= corpus for i in items)


def test_eval_runner_scores_strategies(lab: RagLab) -> None:
    items = load_golden(GOLDEN)
    scores = {
        s.strategy: s
        for s in EvalRunner(lab, OverlapJudge(FakeLLM())).run(["bm25", "multi_query"], items)
    }
    for score in scores.values():
        assert 0 <= score.hit_at_k <= 1
        assert len(score.per_question) == len(items)
    # translated query variants must help on the bilingual corpus
    assert scores["multi_query"].ndcg > scores["bm25"].ndcg


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.chdir(ROOT)
    for key in ("BACKEND", "LLM", "EMBEDDER"):
        monkeypatch.delenv(f"RAG_LAB_{key}", raising=False)
    monkeypatch.setenv("RAG_LAB_TRACE_DIR", str(tmp_path / "traces"))
    return tmp_path


def test_cli_strategies_and_ask(cli_env: Path) -> None:
    runner = CliRunner()
    listed = runner.invoke(app, ["strategies"])
    assert listed.exit_code == 0
    assert "corrective" in listed.output
    asked = runner.invoke(
        app, ["ask", "What is the annual discount?", "-s", "hybrid_rrf", "--trace"]
    )
    assert asked.exit_code == 0, asked.output
    assert "15%" in asked.output
    trace_files = list((cli_env / "traces").glob("*.json"))
    assert trace_files
    assert json.loads(trace_files[0].read_text())[0]["name"] == "strategy.hybrid_rrf"


def test_cli_compare_and_eval(cli_env: Path) -> None:
    runner = CliRunner()
    compared = runner.invoke(app, ["compare", "CB-2025-1103", "--only", "dense,bm25"])
    assert compared.exit_code == 0, compared.output
    assert "| bm25" in compared.output
    out = cli_env / "eval.json"
    evaluated = runner.invoke(app, ["eval", "--only", "bm25,hybrid_rrf", "--output", str(out)])
    assert evaluated.exit_code == 0, evaluated.output
    assert "ndcg@k" in evaluated.output
    assert {row["strategy"] for row in json.loads(out.read_text())} == {"bm25", "hybrid_rrf"}
