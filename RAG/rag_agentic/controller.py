# controller.py
from typing import Any, Dict, List
from tools import VectorSearchTool, CalculatorTool, ToolResult
from llm_wrapper import simple_llm_generate

class AgentController:
    def __init__(self, vector_tool: VectorSearchTool, calc_tool: CalculatorTool, memory=None, max_steps: int = 3):
        self.vector_tool = vector_tool
        self.calc_tool = calc_tool
        self.memory = memory
        self.max_steps = max_steps

    def _need_calc(self, q: str) -> bool:
        keys = ["порах", "%", "*", "+", "-", "/", "sum", "average", "середн"]
        return any(k in q.lower() for k in keys)

    def _plan_tools(self, q: str) -> List[Dict[str, Any]]:
        plan = [{"name": "vector_search", "args": {"query": q, "top_k": 3}}]
        if self._need_calc(q):
            plan.append({"name": "calculator", "args": {"expression": q}})
        return plan

    def run(self, question: str) -> Dict[str, Any]:
        mem_ctx = self.memory.context(800) if self.memory else ""
        steps: List[Dict[str, Any]] = []
        ctx_parts: List[str] = []
        retrieved_ids: List[str] = []

        plan = self._plan_tools(question)
        for i, step in enumerate(plan[: self.max_steps]):
            if step["name"] == "vector_search":
                res: ToolResult = self.vector_tool.call(**step["args"])
                ctx_parts.append(res.content)
                retrieved_ids.extend(res.meta.get("retrieved_ids", []))
                steps.append({"tool": "vector_search", "meta": res.meta})
            elif step["name"] == "calculator":
                res: ToolResult = self.calc_tool.call(**step["args"])
                ctx_parts.append(f"Calc: {res.content}")
                steps.append({"tool": "calculator", "meta": {}})

        combined_context = ""
        if mem_ctx:
            combined_context += f"Previous Interactions:\n{mem_ctx}\n\n"
        if ctx_parts:
            combined_context += f"Context:\n{'\n\n'.join(ctx_parts)}"

        answer = simple_llm_generate(context=combined_context, question=question)
        return {"answer": answer, "retrieved_ids": retrieved_ids}
