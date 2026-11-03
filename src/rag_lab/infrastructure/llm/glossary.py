"""Tiny bilingual synonym glossary: the only "world knowledge" the FakeLLM has.

It lets query-transform strategies (multi-query, HyDE, corrective rewrite) bridge English
questions and Ukrainian documents, which a surface-form embedder cannot do on its own.
"""

from __future__ import annotations

from rag_lab.domain.text import stem

SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("refund", "refunds", "money back", "повернення", "повернути"),
    ("price", "pricing", "cost", "fee", "ціна", "вартість"),
    ("contract", "agreement", "договір", "договору"),
    ("invoice", "bill", "рахунок-фактура", "рахунок"),
    ("discount", "знижка", "знижкою"),
    ("vat", "tax", "пдв"),
    ("annual", "yearly", "річна", "щорічні", "щорічно"),
    ("monthly", "місячні", "щомісяця"),
    ("trial", "тестовий", "період"),
    ("data", "files", "дані", "файли"),
    ("deleted", "delete", "видаляються", "видалені"),
    ("termination", "terminate", "cancel", "cancellation", "розірвання", "скасування"),
    ("jurisdiction", "law", "юрисдикція", "право", "арбітраж"),
    ("support", "підтримка", "підтримку"),
    ("phone", "телефонна"),
    ("versions", "version", "версій", "версії"),
    ("trash", "bin", "кошик"),
    ("escalation", "escalate", "ескалація", "ескалюють"),
    ("response", "відповіді"),
    ("retention", "kept", "stored", "зберігання", "зберігаються"),
    ("users", "user", "members", "користувачів"),
    ("payment", "pay", "due", "оплата", "сплати"),
    ("total", "amount", "сума"),
    ("priority", "пріоритети", "пріоритет"),
    ("days", "днів"),
    ("hours", "годин", "години"),
)

_INDEX: dict[str, tuple[str, ...]] = {}
for _group in SYNONYM_GROUPS:
    for _word in _group:
        _INDEX.setdefault(stem(_word), _group)


def synonyms(token: str) -> tuple[str, ...]:
    """Other surface forms for ``token`` (empty tuple if unknown)."""
    group = _INDEX.get(stem(token.lower()), ())
    return tuple(w for w in group if stem(w) != stem(token.lower()))


def translate(token: str) -> str | None:
    """Best-effort cross-lingual counterpart: first synonym in the other script."""
    is_cyrillic = any("Ѐ" <= ch <= "ӿ" for ch in token)
    for word in synonyms(token):
        if any("Ѐ" <= ch <= "ӿ" for ch in word) != is_cyrillic:
            return word
    return None
