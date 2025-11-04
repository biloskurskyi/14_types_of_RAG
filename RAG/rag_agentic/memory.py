# memory.py
import os, json, datetime

class MemoryStore:
    def __init__(self, path="data/memory.json", session_id="default", max_events=12):
        self.path = path
        self.session_id = session_id
        self.max_events = max_events
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load(self):
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get(self.session_id, {"events": []})

    def _save(self, state):
        with open(self.path, encoding="utf-8") as f:
            all_data = json.load(f)
        all_data[self.session_id] = state
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False)

    def append(self, question, retrieved_ids, answer):
        state = self._load()
        brief = (answer or "").strip()
        if len(brief) > 400:
            brief = brief[:400] + "…"
        state["events"].append({
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "q": question,
            "retrieved_ids": retrieved_ids,
            "answer_brief": brief
        })
        state["events"] = state["events"][-self.max_events:]
        self._save(state)

    def recent(self, n=5):
        return self._load()["events"][-n:]

    def clear(self):
        self._save({"events": []})

    def context(self, max_chars=800):
        """Компакний Previous Interactions для підказки LLM."""
        evs = self._load()["events"]
        lines = []
        for e in evs[-self.max_events:]:
            lines.append(f"Q: {e['q']}\nA: {e['answer_brief']}")
        text = "\n---\n".join(lines).strip()
        if len(text) > max_chars:
            text = text[-max_chars:]
        return text
