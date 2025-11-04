# agent_pipeline.py
import os
from retriever import SimpleRetriever, JsonIndexRetriever
from document_loader import load_documents
from memory import MemoryStore
from tools import VectorSearchTool, CalculatorTool
from controller import AgentController

class AgenticRAG:
    def __init__(self, docs_folder="data/documents", use_json_index=True, memory: MemoryStore | None = None):
        self.memory = memory
        if use_json_index and os.path.exists(os.path.join("data", "index.json")):
            retriever = JsonIndexRetriever(os.path.join("data", "index.json"))
        else:
            docs = load_documents(docs_folder)
            retriever = SimpleRetriever(docs)
        self.controller = AgentController(
            vector_tool=VectorSearchTool(retriever),
            calc_tool=CalculatorTool(),
            memory=self.memory,
            max_steps=3,
        )

    def answer(self, question: str) -> str:
        out = self.controller.run(question)
        if self.memory:
            self.memory.append(question, out.get("retrieved_ids", []), out["answer"])
        return out["answer"]
