# main.py
from pipeline import SimpleRAGPipeline

if __name__ == "__main__":
    # use_json_index=True щоб читати data/index.json, False — щоб читати .txt напряму
    pipeline = SimpleRAGPipeline(docs_folder="data/documents", use_json_index=True)
    while True:
        q = input("Your question (or 'exit'): ")
        if q.lower() in ("exit", "quit"):
            break
        ans = pipeline.answer(q)
        print("Answer:", ans)
