"""Optional adapter for ``sentence-transformers`` models (install the ``embeddings`` extra).

The import is lazy so the rest of the package never depends on torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pip install 'rag-lab[embeddings]' to use this embedder") from exc
        self._model: Any = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]
