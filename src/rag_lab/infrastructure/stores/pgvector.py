"""Postgres adapters: pgvector (HNSW, cosine) for dense search and ``tsvector`` full-text search.

Schema (created idempotently by :meth:`PgVectorStore.create_schema`)::

    rag_chunks(id, doc_id, parent_id, position, text, metadata jsonb,
               embedding vector(dim),
               tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED)
    HNSW index on embedding (vector_cosine_ops), GIN index on tsv and metadata
    rag_parents(id, doc_id, position, text, metadata jsonb)

The ``simple`` text-search configuration is used because the corpus is bilingual and Postgres
ships no Ukrainian stemmer; query terms are OR-ed with prefix matching instead.
Requires the ``postgres`` extra (psycopg 3 + pgvector).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from rag_lab.domain.models import Chunk, Metadata, RetrievedChunk
from rag_lab.domain.text import content_terms

_SAFE_TERM = re.compile(r"[^\w]", re.UNICODE)


def _connect(dsn: str) -> Any:
    try:
        import psycopg  # noqa: PLC0415
        from pgvector.psycopg import register_vector  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pip install 'rag-lab[postgres]' to use the pgvector backend") from exc
    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def _row_to_chunk(row: Sequence[Any]) -> Chunk:
    cid, doc_id, parent_id, position, text, metadata = row[:6]
    return Chunk(
        id=cid,
        doc_id=doc_id,
        text=text,
        position=position,
        metadata=dict(metadata or {}),
        parent_id=parent_id,
    )


def _as_list(value: Any) -> list[float]:
    """pgvector returns ``Vector`` objects (or numpy arrays when numpy is installed)."""
    raw = value.to_list() if hasattr(value, "to_list") else value
    return [float(x) for x in raw]


_COLUMNS = "id, doc_id, parent_id, position, text, metadata"


class PgVectorStore:
    """Dense search with an HNSW index. ``ef_search`` trades recall for latency."""

    def __init__(self, dsn: str, dim: int, ef_search: int = 40, conn: Any = None) -> None:
        self.dim = dim
        self.ef_search = ef_search
        self.conn = conn or _connect(dsn)
        self.create_schema()

    def create_schema(self) -> None:
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id text PRIMARY KEY,
                doc_id text NOT NULL,
                parent_id text,
                position int NOT NULL DEFAULT 0,
                text text NOT NULL,
                metadata jsonb NOT NULL DEFAULT '{{}}',
                embedding vector({self.dim}) NOT NULL,
                tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED
            );
            CREATE INDEX IF NOT EXISTS rag_chunks_embedding_hnsw
                ON rag_chunks USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            CREATE INDEX IF NOT EXISTS rag_chunks_tsv_gin ON rag_chunks USING gin (tsv);
            CREATE INDEX IF NOT EXISTS rag_chunks_metadata_gin ON rag_chunks USING gin (metadata);
            CREATE TABLE IF NOT EXISTS rag_parents (
                id text PRIMARY KEY,
                doc_id text NOT NULL,
                position int NOT NULL DEFAULT 0,
                text text NOT NULL,
                metadata jsonb NOT NULL DEFAULT '{{}}'
            );
            """
        )

    def reset(self) -> None:
        """Drop and recreate the tables (needed when the embedding dimension changes)."""
        self.conn.execute("DROP TABLE IF EXISTS rag_chunks, rag_parents")
        self.create_schema()

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        from pgvector import Vector  # noqa: PLC0415

        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO rag_chunks (id, doc_id, parent_id, position, text, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding,
                    parent_id = EXCLUDED.parent_id, position = EXCLUDED.position
                """,
                [
                    (
                        c.id,
                        c.doc_id,
                        c.parent_id,
                        c.position,
                        c.text,
                        json.dumps(c.metadata),
                        Vector(list(v)),
                    )
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )

    def search(
        self, vector: Sequence[float], k: int, filters: Metadata | None = None
    ) -> list[RetrievedChunk]:
        from pgvector import Vector  # noqa: PLC0415

        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(f"SET LOCAL hnsw.ef_search = {int(self.ef_search)}")
            cur.execute(
                f"""
                SELECT {_COLUMNS}, 1 - (embedding <=> %(v)s) AS score
                FROM rag_chunks
                WHERE metadata @> %(f)s::jsonb
                ORDER BY embedding <=> %(v)s
                LIMIT %(k)s
                """,
                {"v": Vector(list(vector)), "f": json.dumps(filters or {}), "k": k},
            )
            rows = cur.fetchall()
        return [RetrievedChunk(_row_to_chunk(r), float(r[6]), "pgvector") for r in rows]

    def get(self, chunk_id: str) -> Chunk | None:
        row = self.conn.execute(
            f"SELECT {_COLUMNS} FROM rag_chunks WHERE id = %s", (chunk_id,)
        ).fetchone()
        return _row_to_chunk(row) if row else None

    def all_chunks(self) -> list[Chunk]:
        rows = self.conn.execute(f"SELECT {_COLUMNS} FROM rag_chunks ORDER BY id").fetchall()
        return [_row_to_chunk(r) for r in rows]

    def vectors_for(self, chunk_ids: Sequence[str]) -> dict[str, list[float]]:
        rows = self.conn.execute(
            "SELECT id, embedding FROM rag_chunks WHERE id = ANY(%s)", (list(chunk_ids),)
        ).fetchall()
        return {r[0]: _as_list(r[1]) for r in rows}

    def explain_search(self, vector: Sequence[float], k: int = 4) -> str:
        """Return the query plan, handy to confirm the HNSW index is used."""
        from pgvector import Vector  # noqa: PLC0415

        with self.conn.transaction():
            self.conn.execute("SET LOCAL enable_seqscan = off")
            rows = self.conn.execute(
                "EXPLAIN SELECT id FROM rag_chunks ORDER BY embedding <=> %s LIMIT %s",
                (Vector(list(vector)), k),
            ).fetchall()
        return "\n".join(r[0] for r in rows)


class PgFullTextIndex:
    """Keyword search over the generated ``tsv`` column, ranked with ``ts_rank_cd``.

    ``add`` is a no-op: rows are written by :class:`PgVectorStore` and the ``tsvector`` column
    is maintained by Postgres itself.
    """

    def __init__(self, store: PgVectorStore) -> None:
        self.conn = store.conn

    def add(self, chunks: Sequence[Chunk]) -> None:
        return None

    @staticmethod
    def to_tsquery(text: str) -> str:
        parts = (p for t in content_terms(text) for p in _SAFE_TERM.split(t))
        terms = dict.fromkeys(p for p in parts if p)
        return " | ".join(f"{t}:*" for t in terms)

    def search(self, text: str, k: int, filters: Metadata | None = None) -> list[RetrievedChunk]:
        tsquery = self.to_tsquery(text)
        if not tsquery:
            return []
        rows = self.conn.execute(
            f"""
            SELECT {_COLUMNS}, ts_rank_cd(tsv, q) AS score
            FROM rag_chunks, to_tsquery('simple', %(q)s) AS q
            WHERE tsv @@ q AND metadata @> %(f)s::jsonb
            ORDER BY score DESC, id
            LIMIT %(k)s
            """,
            {"q": tsquery, "f": json.dumps(filters or {}), "k": k},
        ).fetchall()
        return [RetrievedChunk(_row_to_chunk(r), float(r[6]), "pg_fts") for r in rows]


class PgDocStore:
    """Parent chunks for small-to-big retrieval, stored next to the child chunks."""

    def __init__(self, store: PgVectorStore) -> None:
        self.conn = store.conn

    def put(self, chunks: Sequence[Chunk]) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO rag_parents (id, doc_id, position, text, metadata)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, metadata = EXCLUDED.metadata
                """,
                [(c.id, c.doc_id, c.position, c.text, json.dumps(c.metadata)) for c in chunks],
            )

    def get(self, chunk_id: str) -> Chunk | None:
        row = self.conn.execute(
            "SELECT id, doc_id, NULL, position, text, metadata FROM rag_parents WHERE id = %s",
            (chunk_id,),
        ).fetchone()
        return _row_to_chunk(row) if row else None
