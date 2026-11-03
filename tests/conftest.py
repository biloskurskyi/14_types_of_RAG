from __future__ import annotations

from pathlib import Path

import pytest

from rag_lab.application.service import RagLab
from rag_lab.bootstrap import build_lab
from rag_lab.config import Settings
from rag_lab.domain.models import Document
from rag_lab.infrastructure.loaders import load_directory
from rag_lab.infrastructure.tracing import JsonTracer

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"


@pytest.fixture(scope="session")
def documents() -> list[Document]:
    return load_directory(CORPUS)


@pytest.fixture
def offline_settings() -> Settings:
    return Settings(_env_file=None, backend="memory", llm="fake", embedder="hashing")


@pytest.fixture
def lab(offline_settings: Settings, documents: list[Document]) -> RagLab:
    instance = build_lab(offline_settings, tracer=JsonTracer())
    instance.ingest(documents)
    return instance
