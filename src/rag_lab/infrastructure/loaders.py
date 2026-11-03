"""Load ``.txt`` documents from a directory.

File names follow ``NN_<doc_type>_<lang>.txt`` (e.g. ``05_refunds_policy_ua.txt``); the type,
language and first-line title become metadata that self-query retrieval can filter on.
"""

from __future__ import annotations

from pathlib import Path

from rag_lab.domain.models import Document

LANGS = {"en", "ua"}


def metadata_from_name(path: Path) -> dict[str, str]:
    parts = path.stem.split("_")
    lang = parts[-1] if parts[-1] in LANGS else "en"
    core = parts[1:-1] if parts[-1] in LANGS else parts[1:]
    return {"doc_type": "_".join(core) or path.stem, "lang": lang, "source": path.name}


def load_directory(directory: Path) -> list[Document]:
    if not directory.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {directory}")
    docs = []
    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        meta = metadata_from_name(path)
        meta["title"] = text.splitlines()[0].strip() if text else path.stem
        docs.append(Document(id=path.stem, text=text, metadata=meta))
    return docs
