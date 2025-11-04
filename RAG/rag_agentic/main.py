# main.py
from agent_pipeline import AgenticRAG
from memory import MemoryStore

if __name__ == "__main__":
    mem = MemoryStore(path="data/memory.json", session_id="default", max_events=12)
    agent = AgenticRAG(docs_folder="data/documents", use_json_index=True, memory=mem)

    print("AgenticRAG. Commands: /history, /reset, exit")
    while True:
        q = input("Your question (or 'exit'): ")
        if q.lower() in ("exit", "quit"):
            break
        if q.strip() == "/history":
            for e in mem.recent(8):
                print(f"- {e['q']} → {e['answer_brief']}")
            continue
        if q.strip() == "/reset":
            mem.clear()
            print("Memory cleared.")
            continue
        ans = agent.answer(q)
        print("Answer:", ans)
