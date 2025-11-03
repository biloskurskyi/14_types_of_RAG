# document_loader.py
import os, unicodedata

def _clean(s: str) -> str:
    # прибрати сурогати та нормалізувати
    s = s.encode("utf-8", "replace").decode("utf-8")
    return unicodedata.normalize("NFC", s)

def load_documents(folder_path: str):
    docs = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            with open(os.path.join(folder_path, fname), encoding="utf-8", errors="replace") as f:
                text = _clean(f.read())
            docs.append({"id": fname, "text": text})
    return docs
