# RAG Retrieval Lab

[![CI](https://github.com/biloskurskyi/14_types_of_RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/biloskurskyi/14_types_of_RAG/actions/workflows/ci.yml)

A learning project that puts **15 retrieval-augmented generation (RAG) strategies behind one
interface** so they can be compared on the same corpus, the same questions and the same metrics.
Everything runs **offline by default**: a deterministic hashing embedder and an extractive
`FakeLLM` replace paid APIs, so tests, CI and the eval suite are reproducible and free.
Real backends plug in through adapters: Postgres + **pgvector (HNSW)** with **full-text search
(`tsvector`)**, `sentence-transformers`, any OpenAI-compatible endpoint or Ollama.

What is inside:

- dense, BM25 and **hybrid search** (Reciprocal Rank Fusion and weighted fusion);
- query transforms (multi-query / RAG-Fusion, HyDE), small-to-big, MMR, reranking,
  contextual compression, self-query metadata filters;
- **corrective RAG** (relevance grading plus fallback chain), an **agentic tool-calling loop**
  with a step limit, a graph-lite entity strategy and **adaptive routing**;
- an **eval suite**: golden Q&A in YAML, hit@k, MRR, nDCG@k, answer recall and a faithfulness
  judge (deterministic by default, **LLM-as-judge** optional);
- **tracing**: every strategy, retriever and LLM call becomes a nested, timed span in JSON;
- three chunkers (fixed, recursive, sentence) and parent/child ingestion;
- pytest suite (100+ tests) with pgvector integration tests against a real Postgres in CI.

The corpus is a synthetic, bilingual (English + Ukrainian) knowledge base of a fictional
cloud storage company, CloudBox: pricing, contract, invoice, SLA, refunds, security, API limits,
data retention, admin FAQ and support escalation. English questions over Ukrainian documents
make the differences between strategies easy to see.

## Architecture

```mermaid
flowchart LR
    subgraph interfaces
        CLI["rag-lab CLI<br/>ingest · ask · compare · eval"]
    end
    subgraph application
        SVC[RagLab facade]
        ING["IngestionPipeline<br/>(template method)"]
        REG["Strategy registry<br/>+ factory"]
        STRAT["15 RetrievalStrategy<br/>classes"]
        GEN[AnswerGenerator<br/>citations]
        EVAL["EvalRunner<br/>metrics + judges"]
    end
    subgraph domain
        MODELS["Document · Chunk · Query<br/>RetrievedChunk · Answer"]
        PORTS["Ports: Embedder · VectorStore<br/>KeywordIndex · LLM · Reranker<br/>Tracer · DocStore"]
    end
    subgraph infrastructure
        EMB["HashingEmbedder<br/>SentenceTransformer<br/>+ CachingEmbedder"]
        VS["InMemoryVectorStore<br/>PgVectorStore (HNSW)"]
        KW["BM25Index<br/>PgFullTextIndex (tsvector)"]
        LLMS["FakeLLM · OpenAI-compatible<br/>Ollama + TracingLLM"]
        RR["LexicalReranker<br/>CrossEncoderReranker"]
        TR[JsonTracer]
    end
    CLI --> SVC
    SVC --> ING & REG & GEN
    CLI --> EVAL --> SVC
    REG --> STRAT
    STRAT --> PORTS
    ING --> PORTS
    GEN --> PORTS
    PORTS -.implemented by.-> EMB & VS & KW & LLMS & RR & TR
```

Dependencies point inwards: `domain` imports nothing from other layers, `application` depends
only on domain ports, `infrastructure` implements them, and `bootstrap.py` is the composition
root that wires adapters from `pydantic-settings` configuration.

Patterns used where they fit naturally:

| Pattern | Where |
|---|---|
| Strategy + Registry/Factory | `application/strategies/base.py` (`@register`, `create`) |
| Template Method | `RetrievalStrategy.retrieve` (trace, delegate, dedupe), `IngestionPipeline.run` |
| Adapter | every class in `infrastructure/` behind a `typing.Protocol` port |
| Decorator | `CachingEmbedder`, `TracingLLM` |
| Chain of Responsibility | corrective fallbacks, adaptive routing rules |
| Facade | `RagLab` in `application/service.py` |

## Strategies

| Name | Idea | When to use | Trade-offs | File |
|---|---|---|---|---|
| `dense` | k-NN over chunk embeddings | Paraphrased questions, good embedder | Misses exact ids and rare terms | `strategies/basic.py` |
| `bm25` | Okapi BM25 / Postgres FTS | Codes, numbers, jargon | No synonyms or cross-lingual matching | `strategies/basic.py` |
| `hybrid_rrf` | Dense + sparse, Reciprocal Rank Fusion | Sensible production default | Two retrievers per query | `strategies/basic.py` |
| `hybrid_weighted` | Min-max normalised, alpha-weighted scores | When you can tune alpha on eval data | Sensitive to score distributions | `strategies/basic.py` |
| `multi_query` | LLM writes query variants, RRF over all | Vague or cross-lingual questions | 1 LLM call and N× retrieval | `strategies/query_transform.py` |
| `hyde` | Embed a hypothetical answer | Short questions vs long passages | Hallucinated passage can mislead | `strategies/query_transform.py` |
| `parent_document` | Match small chunks, return parents | Precise matching, rich context | Larger prompts | `strategies/structure.py` |
| `mmr` | Relevance minus redundancy | Many near-duplicate chunks | May drop a relevant duplicate | `strategies/postprocess.py` |
| `rerank` | Wide hybrid recall, reranker picks top-k | Precision at small k | Cross-encoder latency | `strategies/postprocess.py` |
| `compression` | LLM keeps only relevant sentences | Long noisy chunks, tight context | 1 LLM call per chunk | `strategies/postprocess.py` |
| `self_query` | LLM extracts metadata filters | "in Ukrainian docs", "in the invoice" | Wrong filter hides answers (validated + fallback) | `strategies/self_query.py` |
| `corrective` | Grade chunks; rewrite or fall back | Unreliable first-pass retrieval | Most LLM calls | `strategies/corrective.py` |
| `agentic` | Tool-calling loop over retrievers, step limit | Multi-part questions | Non-deterministic with real LLMs | `strategies/agentic.py` |
| `graph` | Entity co-occurrence graph, 1-hop expansion | Questions about named entities | Entity extraction quality | `strategies/graph.py` |
| `adaptive` | Rule chain routes each query | Mixed traffic | Rules need maintenance | `strategies/adaptive.py` |

Background, references and implementation notes: [`docs/strategies.md`](docs/strategies.md).

## Quickstart

### Offline (no keys, no database)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

rag-lab strategies
rag-lab ask "How much is the total due on invoice CB-2025-1103?" --strategy adaptive --trace
rag-lab compare "How are annual payments refunded after cancellation?"
rag-lab eval --output traces/eval.json
```

`--trace` writes a JSON span tree to `traces/` (strategy → retrievers → LLM calls, with
timings, queries, scores and routing decisions).

### With Postgres + pgvector

```bash
docker compose up -d db                     # pgvector/pgvector:pg16 on localhost:5442
export RAG_LAB_BACKEND=postgres
rag-lab ingest                              # creates tables, HNSW + GIN indexes, loads corpus
rag-lab compare "What is the API rate limit on the Pro plan?"
docker compose down
```

### Real models (optional)

```bash
pip install -e ".[embeddings]"              # sentence-transformers + cross-encoder
export RAG_LAB_EMBEDDER=sentence-transformers RAG_LAB_RERANKER=cross-encoder
export RAG_LAB_LLM=ollama                   # or: openai + RAG_LAB_OPENAI_API_KEY / BASE_URL
rag-lab eval --judge llm
```

All settings live in [`.env.example`](.env.example).

## Tests and evaluation

```bash
ruff check . && ruff format --check .
pytest                                      # unit tests, fully offline

docker compose up -d db
RAG_LAB_TEST_DATABASE_URL=postgresql://rag:rag@localhost:5442/rag_lab pytest -m pgvector
```

CI (`.github/workflows/ci.yml`) runs ruff and the whole suite, including the pgvector tests,
against a `pgvector/pgvector:pg16` service container.

`rag-lab eval` scores every strategy on [`eval/golden.yaml`](eval/golden.yaml) (16 questions,
English and Ukrainian). Retrieval metrics are computed at document level so they stay
comparable across chunkers. Output with the default offline stack:

```text
| strategy        | hit@k | mrr   | ndcg@k | ans_recall | faithful | p50_ms | llm_calls |
|-----------------|-------|-------|--------|------------|----------|--------|-----------|
| corrective      | 0.938 | 0.906 | 0.89   | 0.875      | 1.0      | 0.95   | 12.2      |
| adaptive        | 0.938 | 0.833 | 0.855  | 0.875      | 1.0      | 1.5    | 1.4       |
| agentic         | 1.0   | 0.812 | 0.855  | 0.938      | 1.0      | 0.53   | 3.0       |
| multi_query     | 0.938 | 0.786 | 0.814  | 0.875      | 1.0      | 2.09   | 1.0       |
| hyde            | 0.875 | 0.76  | 0.765  | 0.875      | 1.0      | 0.6    | 1.0       |
| mmr             | 0.812 | 0.708 | 0.711  | 0.688      | 1.0      | 1.55   | 0.0       |
| parent_document | 0.812 | 0.677 | 0.686  | 0.812      | 1.0      | 0.45   | 0.0       |
| dense           | 0.75  | 0.661 | 0.659  | 0.688      | 1.0      | 0.48   | 0.0       |
| hybrid_rrf      | 0.688 | 0.656 | 0.64   | 0.688      | 1.0      | 0.48   | 0.0       |
| ...             |       |       |        |            |          |        |           |
| bm25            | 0.688 | 0.615 | 0.609  | 0.625      | 1.0      | 0.04   | 0.0       |
```

How to read it: the hashing embedder only sees surface forms, so English questions over
Ukrainian documents defeat `dense`, `bm25` and plain hybrid. Strategies that rewrite or
translate the query (`multi_query`, `corrective`, `agentic`, `adaptive`) recover most of the gap,
and the `llm_calls` column shows what that costs. Faithfulness is always 1.0 here because
`FakeLLM` is extractive. Switch to a real LLM and `--judge llm` to make that column informative.

### `compare` example

```text
$ rag-lab compare "How are annual payments refunded after cancellation?"
| strategy        | ms   | llm | top_score | top_docs                                                    |
|-----------------|------|-----|-----------|-------------------------------------------------------------|
| dense           | 0.66 | 0   | 0.2442    | 06_security_en, 01_pricing_en, 07_api_limits_en             |
| bm25            | 0.05 | 0   | 2.4275    | 01_pricing_en                                               |
| hybrid_rrf      | 0.75 | 0   | 0.0323    | 01_pricing_en, 06_security_en, 07_api_limits_en             |
| multi_query     | 2.31 | 1   | 0.0792    | 01_pricing_en, 02_contract_ua, 05_refunds_policy_ua         |
| hyde            | 0.64 | 1   | 0.2952    | 06_security_en, 01_pricing_en, 05_refunds_policy_ua         |
| corrective      | 3.50 | 22  | 7.7105    | 02_contract_ua, 05_refunds_policy_ua, 03_invoice_example_ua |
| agentic         | 0.56 | 3   | 0.0164    | 02_contract_ua, 06_security_en, 05_refunds_policy_ua        |
| adaptive        | 2.18 | 1   | 0.0792    | 01_pricing_en, 02_contract_ua, 05_refunds_policy_ua         |
```

(Rows trimmed.) Scores are not comparable across strategies: RRF scores are about 1/60,
BM25 scores are unbounded, and corrective reports grader output. Compare ranks and documents
instead.

## Project layout

```text
src/rag_lab/
  domain/          models.py, ports.py, text.py
  application/     chunking, ingestion, prompts, generation, service, evaluation
    strategies/    base (registry), fusion, basic, query_transform, structure,
                   postprocess, self_query, corrective, agentic, graph, adaptive
  infrastructure/  embedders/, stores/ (memory, bm25, pgvector), llm/ (fake, http,
                   tracing), rerankers, tracing, loaders
  interfaces/      cli.py (Typer), table.py
  bootstrap.py     composition root
  config.py        pydantic-settings
corpus/            synthetic CloudBox documents (EN + UA)
eval/golden.yaml   golden questions
tests/             unit/ and integration/ (pgvector)
```

## Limitations

- The offline stack shows the *mechanics* of each strategy. Absolute numbers depend on the
  embedder and LLM, so rerun `rag-lab eval` with real models before drawing conclusions.
- `FakeLLM` uses a small bilingual glossary for rewrites and translations. That gives
  query-transform strategies an honest but tiny advantage, and it is documented in
  `infrastructure/llm/glossary.py`.
- The corpus is small. HNSW only starts to matter with thousands of chunks, but the schema,
  index and `ef_search` setting are the ones you would use at scale.

## License

MIT
