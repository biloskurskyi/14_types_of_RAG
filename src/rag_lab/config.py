"""Application settings loaded from environment variables (prefix ``RAG_LAB_``) or ``.env``.

Defaults are fully offline: in-memory store, hashing embedder and the extractive FakeLLM.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_LAB_", env_file=".env", extra="ignore")

    backend: Literal["memory", "postgres"] = "memory"
    database_url: str = "postgresql://rag:rag@localhost:5442/rag_lab"

    embedder: Literal["hashing", "sentence-transformers"] = "hashing"
    embedding_dim: int = Field(default=384, ge=16, le=4096)
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm: Literal["fake", "openai", "ollama"] = "fake"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = "gpt-4o-mini"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    reranker: Literal["lexical", "cross-encoder"] = "lexical"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunker: Literal["recursive", "fixed", "sentence"] = "recursive"
    chunk_size: int = Field(default=200, ge=32)
    chunk_overlap: int = Field(default=30, ge=0)

    corpus_dir: Path = Path("corpus")
    trace_dir: Path = Path("traces")
    top_k: int = Field(default=4, ge=1, le=50)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
