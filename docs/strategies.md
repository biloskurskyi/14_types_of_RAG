# Retrieval strategies: notes and references

Every strategy subclasses `RetrievalStrategy` (`src/rag_lab/application/strategies/base.py`)
and implements `_retrieve(query) -> list[RetrievedChunk]`. The base class traces the call,
removes duplicate chunks and cuts the list to `k`. Strategies only use the ports they need
(`Embedder`, `VectorStore`, `KeywordIndex`, `LLM`, `Reranker`, `DocStore`, `Tracer`), so each
one runs on the in-memory or the Postgres backend without changes.

The taxonomy follows recent surveys, which group RAG methods by where they act in the pipeline:
before retrieval (query transforms), during retrieval (indexes and fusion), after retrieval
(reranking, filtering, compression), and around retrieval (feedback loops, agents, routing).
See Gao et al., *Retrieval-Augmented Generation for Large Language Models: A Survey* (2023,
[arXiv:2312.10997](https://arxiv.org/abs/2312.10997)) and the 2025 architecture survey
[arXiv:2506.00054](https://arxiv.org/abs/2506.00054). The original RAG formulation is Lewis et
al. (2020, [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)).

---

## Retrieval core

### `dense`: naive dense retrieval
Embed the question and return the k nearest chunks by cosine similarity. On Postgres this is
`ORDER BY embedding <=> $1` on an HNSW index (`vector_cosine_ops`, `m=16`,
`ef_construction=64`, `hnsw.ef_search` set per query).
- Dense Passage Retrieval: Karpukhin et al., 2020, [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
- HNSW: Malkov and Yashunin, 2016, [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)
- pgvector: <https://github.com/pgvector/pgvector>

### `bm25`: sparse lexical retrieval
Okapi BM25 (`k1=1.5`, `b=0.75`, non-negative idf) in pure Python, or Postgres full-text search
over a generated `tsvector` column ranked with `ts_rank_cd`. It handles invoice numbers, prices
and rare terms well. It has no notion of synonyms or translation.
- Robertson and Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009
- Postgres text search: <https://www.postgresql.org/docs/current/textsearch.html>

### `hybrid_rrf` and `hybrid_weighted`: hybrid search
Run dense and sparse retrieval with a wider fan-out (3·k each) and fuse the results.
**RRF** scores `Σ 1/(60 + rank)` and needs no score calibration. **Weighted** fusion min-max
normalises both lists and mixes them as `α·dense + (1-α)·sparse`. It can beat RRF once α is
tuned on eval data, and it is sensitive to outliers.
- Cormack, Clarke and Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual
  rank learning methods*, SIGIR 2009

## Query transformation (pre-retrieval)

### `multi_query`: multi-query / RAG-Fusion
The LLM writes alternative phrasings (here they include English↔Ukrainian translations). Each
variant is searched with both retrievers, and all rankings are RRF-fused. This is the cheapest
fix for vocabulary mismatch: one LLM call and N× retrieval.
- Rackauckas, *RAG-Fusion*, 2024, [arXiv:2402.03367](https://arxiv.org/abs/2402.03367)
- Ma et al., *Query Rewriting for Retrieval-Augmented LLMs*, 2023,
  [arXiv:2305.14283](https://arxiv.org/abs/2305.14283)

### `hyde`: Hypothetical Document Embeddings
The LLM writes a fake answer passage. The passage, concatenated with the question, is embedded
and searched. It works because answers look more like answers than questions do. The risk is
that a confidently wrong passage pulls retrieval off-topic.
- Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels*, 2022,
  [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)

### `self_query`: metadata filters from natural language
The LLM returns JSON `{"query": ..., "filters": {...}}`. Filters are **validated** against an
allow-list (`lang`, `doc_type` with known values), and an over-restrictive filter that returns
nothing falls back to unfiltered search. Never trust model-generated filters blindly: they are
also an injection surface for SQL or metadata.
- Pattern popularised by LangChain's `SelfQueryRetriever`

## Index structure

### `parent_document`: small-to-big
Ingestion stores **parents** (≈4× chunk size) in a `DocStore` and indexes their **children**.
Retrieval matches on precise child chunks and returns the parent for context. The parent score
is its best child's score.
- Related: Chen et al., *Dense X Retrieval: What Retrieval Granularity Should We Use?*, 2023,
  [arXiv:2312.06648](https://arxiv.org/abs/2312.06648)

### Chunkers (`application/chunking.py`)
`fixed` (character window plus overlap), `recursive` (paragraph → line → sentence → word, then
greedy merge) and `sentence` (pack whole sentences). Set the chunker with `RAG_LAB_CHUNKER`. Eval
metrics are computed per document, so you can compare chunkers with `rag-lab eval`.

## Post-retrieval

### `mmr`: Maximal Marginal Relevance
Greedily pick `argmax λ·sim(q,d) − (1−λ)·max_{s∈S} sim(d,s)` from a 4·k candidate pool. It
reduces near-duplicate context, and with `λ` too low it starts dropping relevant chunks.
- Carbonell and Goldstein, SIGIR 1998

### `rerank`: two-stage retrieval
Wide hybrid recall (5·k), then a reranker picks the top k. `CrossEncoderReranker` wraps a
`sentence-transformers` cross-encoder (optional extra). The default `LexicalReranker` is a
deterministic stand-in: query-term coverage, query bigrams and a small prior from the
first-stage score.
- Nogueira and Cho, *Passage Re-ranking with BERT*, 2019,
  [arXiv:1901.04085](https://arxiv.org/abs/1901.04085)

### `compression`: contextual compression
For each retrieved chunk the LLM copies only the relevant sentences, and chunks that compress to
`NONE` are dropped. This means fewer prompt tokens and less noise, at the cost of one LLM call
per chunk (see the `llm_calls` column).
- Xu et al., *RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective
  Augmentation*, 2023, [arXiv:2310.04408](https://arxiv.org/abs/2310.04408)

## Feedback loops, agents and routing

### `corrective`: Corrective RAG
Retrieve with hybrid search, then grade each chunk with the LLM (`{"score": 0..1}`). The
corrective actions form a **Chain of Responsibility**:
`KeepRelevant → RewriteAndRetry → KeywordFallback → DropFilters`. The handler that resolved
the query is recorded in the trace. In the original paper the last step is web search. Here it
is "drop the metadata filters", because the lab never calls external services.
- Yan et al., *Corrective Retrieval Augmented Generation*, 2024,
  [arXiv:2401.15884](https://arxiv.org/abs/2401.15884)
- Related: Asai et al., *Self-RAG*, 2023, [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)

### `agentic`: tool-calling agent loop
The LLM gets three tools (`vector_search`, `keyword_search`, `read_document`) and decides what
to call. The loop stops when the model answers without tool calls or after `max_steps`
(default 4, traced as `agent.max_steps_reached`). Results from all tool calls are RRF-fused.
`FakeLLM` plays a fixed policy: vector search, then translated keyword search, then stop. Real
models go through the OpenAI-compatible or Ollama adapters with native tool calling.
- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, 2022,
  [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)

### `graph`: graph-lite entity retrieval
A bipartite chunk↔entity graph is built from capitalised terms, acronyms, codes and quantities.
Query entities seed chunks (idf-weighted), one hop spreads to chunks that share the seeds'
other entities, and the final score mixes entity evidence (0.6) with dense similarity (0.4).
This is deliberately much smaller than GraphRAG: it has no LLM extraction and no community
summaries, but the retrieval idea is the same.
- Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*,
  2024, [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)

### `adaptive`: query routing
A rule chain picks a strategy per query:
- exact identifiers or prices go to `bm25`;
- explicit language or document hints go to `self_query`;
- comparisons and multi-part questions go to `agentic`;
- Latin-script questions over the bilingual corpus go to `multi_query`;
- everything else goes to `hybrid_rrf`.

Routing costs no LLM call, and the decision is traced (`adaptive.route`).
- Jeong et al., *Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question
  Complexity*, 2024, [arXiv:2403.14403](https://arxiv.org/abs/2403.14403)

## Evaluation

`application/evaluation.py` computes **hit@k**, **MRR** and binary **nDCG@k** over the
de-duplicated document ranking. It also computes **answer recall** (expected facts appear in
the answer) and **faithfulness** (share of answer sentences supported by the retrieved
context). The default judge is deterministic, and `--judge llm` sends the same prompt to the
configured model (LLM-as-judge).
- Järvelin and Kekäläinen, *Cumulated gain-based evaluation of IR techniques*, TOIS 2002
- Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation*, 2023,
  [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, 2023,
  [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)

## Not implemented (yet)

- RAPTOR-style recursive summarisation trees
  ([arXiv:2401.18059](https://arxiv.org/abs/2401.18059)): needs a real LLM to be meaningful.
- Late-interaction retrieval (ColBERT): needs model weights.
- Learned sparse retrieval (SPLADE): same reason.
