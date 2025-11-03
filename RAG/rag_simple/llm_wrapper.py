# llm_wrapper.py
from openai import OpenAI
import unicodedata

_API_KEY = ""
_MODEL = ""


_client = OpenAI(api_key=_API_KEY)

def _clean(s: str) -> str:
    s = s.encode("utf-8", "replace").decode("utf-8")
    return unicodedata.normalize("NFC", s)

def simple_llm_generate(context: str, question: str) -> str:
    prompt = f"Use the context to answer succinctly. If math is involved, compute explicitly.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    prompt = _clean(prompt)
    resp = _client.responses.create(model=_MODEL, input=prompt)
    return resp.output_text or ""
