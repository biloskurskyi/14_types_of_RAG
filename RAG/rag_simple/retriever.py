# retriever.py
import json, os
import numpy as np

class SimpleRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.doc_vectors = [self._vectorize(d["text"]) for d in documents]

    def _vectorize(self, text):
        words = text.lower().split()
        uniq = list(set(words))
        return np.array([words.count(w) for w in uniq], dtype=float)

    def _cosine(self, a, b):
        if np.linalg.norm(a)==0 or np.linalg.norm(b)==0:
            return 0.0
        return float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))

    def retrieve(self, query, top_k=3):
        qv = self._vectorize(query)
        sims = [self._cosine(qv, dv) for dv in self.doc_vectors]
        idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return [self.documents[i] for i in idxs]

class JsonIndexRetriever:
    def __init__(self, index_path):
        with open(index_path, encoding="utf-8") as f:
            idx = json.load(f)
        self.docs = idx["documents"]
        self.doc_arrays = [np.array(d["vec"], dtype=float) for d in self.docs]
        self.doc_vocabs = [d["vocab"] for d in self.docs]

    def _qvec(self, query):
        words = query.lower().split()
        uniq = list(set(words))
        return uniq, np.array([words.count(w) for w in uniq], dtype=float)

    def _align_and_cosine(self, q_vocab, q_vec, d_vocab, d_vec):
        all_terms = list(set(q_vocab) | set(d_vocab))
        q_map = {t:i for i,t in enumerate(q_vocab)}
        d_map = {t:i for i,t in enumerate(d_vocab)}
        q_full = np.array([q_vec[q_map[t]] if t in q_map else 0.0 for t in all_terms], dtype=float)
        d_full = np.array([d_vec[d_map[t]] if t in d_map else 0.0 for t in all_terms], dtype=float)
        if np.linalg.norm(q_full)==0 or np.linalg.norm(d_full)==0:
            return 0.0
        return float(np.dot(q_full, d_full)/(np.linalg.norm(q_full)*np.linalg.norm(d_full)))

    def retrieve(self, query, top_k=3):
        q_vocab, q_vec = self._qvec(query)
        sims = []
        for i, d in enumerate(self.docs):
            sim = self._align_and_cosine(q_vocab, q_vec, self.doc_vocabs[i], self.doc_arrays[i])
            sims.append(sim)
        idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return [{"id": self.docs[i]["id"], "text": self.docs[i]["text"]} for i in idxs]
