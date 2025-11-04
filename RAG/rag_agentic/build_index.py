# build_index.py
import os, json

def load_documents(folder):
    docs = []
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            p = os.path.join(folder, fname)
            with open(p, encoding="utf-8") as f:
                docs.append({"id": fname, "text": f.read()})
    return docs

def vectorize(text):
    words = text.lower().split()
    uniq = list(set(words))
    vec = [words.count(w) for w in uniq]
    return {"vocab": uniq, "vec": vec}

if __name__ == "__main__":
    docs = load_documents(os.path.join("rag_simple", "data", "documents"))
    doc_entries = []
    for d in docs:
        v = vectorize(d["text"])
        doc_entries.append({
            "id": d["id"],
            "vocab": v["vocab"],
            "vec": v["vec"],
            "text": d["text"]
        })
    out = {
        "version": 1,
        "documents": doc_entries
    }
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "index.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("index.json written to data/index.json")
