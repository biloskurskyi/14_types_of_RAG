"""``rag-lab`` command line: ingest, ask, compare, eval, strategies.

With the default in-memory backend every command ingests the corpus first (it takes
milliseconds); with ``RAG_LAB_BACKEND=postgres`` run ``rag-lab ingest`` once and the other
commands reuse the database.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer

from rag_lab.application import strategies
from rag_lab.application.evaluation import EvalRunner, LLMJudge, OverlapJudge, load_golden
from rag_lab.application.service import RagLab
from rag_lab.bootstrap import build_lab
from rag_lab.config import Settings
from rag_lab.infrastructure.llm.fake import FakeLLM
from rag_lab.infrastructure.loaders import load_directory
from rag_lab.infrastructure.tracing import JsonTracer
from rag_lab.interfaces.table import render_table

app = typer.Typer(
    help="RAG Retrieval Lab: compare retrieval strategies side by side.", no_args_is_help=True
)

StrategyOpt = Annotated[str, typer.Option("--strategy", "-s", help="Strategy name")]
KOpt = Annotated[int | None, typer.Option("--k", "-k", help="Number of chunks")]
TraceOpt = Annotated[bool, typer.Option("--trace", help="Write a JSON trace to the trace dir")]


def _lab(settings: Settings, tracer: JsonTracer, *, ingest: bool | None = None) -> RagLab:
    lab = build_lab(settings, tracer=tracer)
    docs = load_directory(settings.corpus_dir)
    if ingest if ingest is not None else settings.backend == "memory":
        lab.ingest(docs)
    else:
        lab.ctx.doc_types = tuple(sorted({d.metadata["doc_type"] for d in docs}))
    tracer.reset()
    return lab


def _dump_trace(settings: Settings, tracer: JsonTracer, label: str) -> Path:
    path = settings.trace_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}.json"
    return tracer.dump(path)


@app.command()
def ingest() -> None:
    """Chunk, embed and index the corpus (persists when the backend is postgres)."""
    settings = Settings()
    tracer = JsonTracer()
    lab = build_lab(settings, tracer=tracer)
    if settings.backend == "postgres":
        lab.ctx.vector_store.reset()  # type: ignore[attr-defined]
    report = lab.ingest(load_directory(settings.corpus_dir))
    typer.echo(
        f"backend={settings.backend} documents={report.documents} "
        f"parents={report.parents} chunks={report.chunks}"
    )


@app.command("strategies")
def list_strategies() -> None:
    """List registered strategies."""
    rows = [{"name": n, "summary": strategies.get_class(n).summary} for n in strategies.available()]
    typer.echo(render_table(rows))


@app.command()
def ask(
    question: str,
    strategy: StrategyOpt = "hybrid_rrf",
    k: KOpt = None,
    trace: TraceOpt = False,
) -> None:
    """Retrieve with one strategy and print a cited answer."""
    settings = Settings()
    tracer = JsonTracer()
    lab = _lab(settings, tracer)
    answer = lab.ask(strategy, question, k)
    typer.echo(answer.text)
    typer.echo("")
    for i, hit in enumerate(answer.context, start=1):
        cited = "*" if any(c.marker == i for c in answer.citations) else " "
        typer.echo(f"{cited}[{i}] {hit.chunk.id}  score={hit.score:.4f}")
    if trace:
        typer.echo(f"trace: {_dump_trace(settings, tracer, strategy)}")


@app.command()
def compare(
    question: str,
    k: KOpt = None,
    only: Annotated[str | None, typer.Option(help="Comma-separated strategy names")] = None,
    trace: TraceOpt = False,
) -> None:
    """Run every strategy on one question and print a latency / top-documents table."""
    settings = Settings()
    tracer = JsonTracer()
    lab = _lab(settings, tracer)
    names = only.split(",") if only else None
    rows = []
    for run in lab.compare(question, names, k):
        top = run.hits[0] if run.hits else None
        rows.append(
            {
                "strategy": run.strategy,
                "ms": f"{run.latency_ms:.2f}",
                "llm": run.llm_calls,
                "top_score": f"{top.score:.4f}" if top else "-",
                "top_docs": ", ".join(dict.fromkeys(h.doc_id for h in run.hits)),
            }
        )
    typer.echo(render_table(rows))
    if trace:
        typer.echo(f"trace: {_dump_trace(settings, tracer, 'compare')}")


@app.command("eval")
def evaluate(
    golden: Annotated[Path, typer.Option(help="Golden Q&A YAML")] = Path("eval/golden.yaml"),
    k: Annotated[int, typer.Option("--k", "-k")] = 4,
    only: Annotated[str | None, typer.Option(help="Comma-separated strategy names")] = None,
    judge: Annotated[str, typer.Option(help="overlap (deterministic) or llm")] = "overlap",
    output: Annotated[Path | None, typer.Option(help="Write per-question JSON here")] = None,
) -> None:
    """Score strategies on the golden set: hit@k, MRR, nDCG@k, answer recall, faithfulness."""
    settings = Settings()
    tracer = JsonTracer()
    lab = _lab(settings, tracer)
    chosen = LLMJudge(lab.ctx.llm) if judge == "llm" else OverlapJudge(FakeLLM())
    runner = EvalRunner(lab, chosen, k=k)
    names = only.split(",") if only else strategies.available()
    scores = runner.run(names, load_golden(golden))
    scores.sort(key=lambda s: (-s.ndcg, -s.mrr, s.latency_ms))
    typer.echo(render_table([s.row() for s in scores]))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = [{**s.row(), "questions": s.per_question} for s in scores]
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"wrote {output}")


if __name__ == "__main__":  # pragma: no cover
    app()
