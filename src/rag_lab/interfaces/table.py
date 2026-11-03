"""Minimal plain-text table renderer (no extra dependency, copy-paste friendly in READMEs)."""

from __future__ import annotations

from collections.abc import Sequence


def render_table(rows: Sequence[dict[str, object]]) -> str:
    if not rows:
        return "(no rows)"
    headers = list(rows[0])
    cells = [[str(r.get(h, "")) for h in headers] for r in rows]
    widths = [max(len(h), *(len(c[i]) for c in cells)) for i, h in enumerate(headers)]

    def line(values: Sequence[str]) -> str:
        return "| " + " | ".join(v.ljust(w) for v, w in zip(values, widths, strict=True)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(headers), sep, *(line(c) for c in cells)])
