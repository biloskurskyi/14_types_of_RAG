# pipeline.py
import os
from document_loader import load_documents
from retriever import SimpleRetriever, JsonIndexRetriever
from llm_wrapper import simple_llm_generate

class SimpleRAGPipeline:
    def __init__(self, docs_folder: str, use_json_index: bool = False):
        if use_json_index and os.path.exists(os.path.join("data", "index.json")):
            self.retriever = JsonIndexRetriever(os.path.join("data", "index.json"))
            self.documents = None
        else:
            self.documents = load_documents(docs_folder)
            self.retriever = SimpleRetriever(self.documents)

    def answer(self, question: str):
        retrieved = self.retriever.retrieve(question)
        context = "\n\n".join([d["text"] for d in retrieved])
        return simple_llm_generate(context=context, question=question)
