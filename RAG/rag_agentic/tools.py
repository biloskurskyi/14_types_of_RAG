# tools.py
from typing import Any, Dict, List

class ToolResult:
    def __init__(self, content: str, meta: Dict[str, Any] | None = None):
        self.content = content
        self.meta = meta or {}

class VectorSearchTool:
    def __init__(self, retriever):
        self.retriever = retriever
        self.name = "vector_search"

    def call(self, query: str, top_k: int = 3) -> ToolResult:
        hits = self.retriever.retrieve(query, top_k=top_k)
        text = "\n\n".join([h["text"] for h in hits])
        ids = [h["id"] for h in hits]
        return ToolResult(content=text, meta={"retrieved_ids": ids})

class CalculatorTool:
    def __init__(self):
        self.name = "calculator"

    def call(self, expression: str) -> ToolResult:
        allowed = "0123456789.+-*/()% "
        expr = "".join(ch for ch in expression if ch in allowed)
        try:
            val = eval(expr, {"__builtins__": {}}, {})
        except Exception:
            val = "NaN"
        return ToolResult(content=str(val))
