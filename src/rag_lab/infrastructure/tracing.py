"""Tracers: a JSON span recorder and a no-op tracer.

Spans nest through a stack, so a strategy that calls another strategy (adaptive, corrective)
shows up as a tree. ``JsonTracer.export`` returns plain dicts ready for ``json.dump``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class JsonTracer:
    def __init__(self) -> None:
        self._roots: list[dict[str, Any]] = []
        self._stack: list[dict[str, Any]] = []

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        node: dict[str, Any] = {"name": name, "attrs": dict(attrs), "children": []}
        (self._stack[-1]["children"] if self._stack else self._roots).append(node)
        self._stack.append(node)
        start = time.perf_counter()
        try:
            yield node["attrs"]
        except Exception as exc:
            node["error"] = repr(exc)
            raise
        finally:
            node["duration_ms"] = round((time.perf_counter() - start) * 1000, 3)
            self._stack.pop()

    def export(self) -> list[dict[str, Any]]:
        return json.loads(json.dumps(self._roots, default=str))

    def reset(self) -> None:
        self._roots.clear()
        self._stack.clear()

    def dump(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.export(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path


class NullTracer:
    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        yield {}

    def export(self) -> list[dict[str, Any]]:
        return []

    def reset(self) -> None:
        return None


def span_names(spans: list[dict[str, Any]]) -> list[str]:
    """Flatten a span tree into names (depth-first), handy in tests."""
    names: list[str] = []
    for span in spans:
        names.append(span["name"])
        names.extend(span_names(span["children"]))
    return names
