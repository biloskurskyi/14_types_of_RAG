"""Retrieval strategies. Importing this package registers every built-in strategy."""

from rag_lab.application.strategies import (  # noqa: F401  (import for registration side effects)
    basic,
    postprocess,
    query_transform,
    self_query,
    structure,
)
from rag_lab.application.strategies.base import (
    RetrievalStrategy,
    StrategyContext,
    available,
    create,
    get_class,
    register,
)

__all__ = ["RetrievalStrategy", "StrategyContext", "available", "create", "get_class", "register"]
