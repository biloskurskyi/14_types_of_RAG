"""Ingestion pipeline (Template Method).

``IngestionPipeline.run`` fixes the order of steps — split into parents, split parents into
children, enrich, embed, index — while subclasses may override individual hooks. Parents go to
the ``DocStore`` (for small-to-big retrieval); children are what every index searches.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from rag_lab.application.chunking import Chunker, RecursiveChunker
from rag_lab.domain.models import Chunk, Document
from rag_lab.domain.ports import DocStore, Embedder, KeywordIndex, Tracer, VectorStore, iter_batches


@dataclass(frozen=True)
class IngestReport:
    documents: int
    parents: int
    chunks: int


class IngestionPipeline:
    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        doc_store: DocStore,
        tracer: Tracer,
        parent_factor: int = 4,
        batch_size: int = 64,
    ) -> None:
        self.chunker = chunker
        self.parent_chunker = RecursiveChunker(size=chunker.size * parent_factor, overlap=0)
        self.embedder = embedder
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.doc_store = doc_store
        self.tracer = tracer
        self.batch_size = batch_size

    # Template method ----------------------------------------------------------------
    def run(self, documents: Sequence[Document]) -> IngestReport:
        with self.tracer.span("ingest", documents=len(documents)) as attrs:
            parents: list[Chunk] = []
            children: list[Chunk] = []
            for doc in documents:
                doc_parents = self.split_parents(doc)
                parents.extend(doc_parents)
                for parent in doc_parents:
                    children.extend(self.enrich(c) for c in self.split_children(doc, parent))
            self.store_parents(parents)
            self.index(children)
            attrs.update(parents=len(parents), chunks=len(children))
        return IngestReport(len(documents), len(parents), len(children))

    # Hooks ----------------------------------------------------------------------------
    def split_parents(self, doc: Document) -> list[Chunk]:
        return self.parent_chunker.split(doc, id_prefix=f"{doc.id}#p")

    def split_children(self, doc: Document, parent: Chunk) -> list[Chunk]:
        sub = Document(id=doc.id, text=parent.text, metadata=doc.metadata)
        return [
            replace(c, parent_id=parent.id)
            for c in self.chunker.split(sub, id_prefix=f"{parent.id}.c")
        ]

    def enrich(self, chunk: Chunk) -> Chunk:
        return replace(chunk, metadata={**chunk.metadata, "chars": len(chunk.text)})

    def store_parents(self, parents: list[Chunk]) -> None:
        self.doc_store.put(parents)

    def index(self, chunks: list[Chunk]) -> None:
        for batch in iter_batches(chunks, self.batch_size):
            with self.tracer.span("ingest.embed", batch=len(batch)):
                vectors = self.embedder.embed([c.text for c in batch])
            self.vector_store.add(batch, vectors)
        self.keyword_index.add(chunks)
