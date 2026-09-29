"""Memory-strategy registry -- mirrors MODEL_REGISTRY's/tool-factories'
shape: add a strategy, touch nothing else in ``agents/topology.py``.
"""

from collections.abc import Callable

from mas_deepr.memory.base import (
    BranchTrajectory,
    CompressFn,
    MainThreadState,
    MemoryStrategy,
    PassRecord,
)
from mas_deepr.memory.baseline import StatelessStrategy
from mas_deepr.memory.folding import ContextFoldingStrategy
from mas_deepr.memory.retrac import TrajectoryCompressionStrategy
from mas_deepr.memory.state import StructuredState

MEMORY_REGISTRY: dict[str, Callable[[], MemoryStrategy]] = {
    "stateless": StatelessStrategy,
    "folding": ContextFoldingStrategy,
    "retrac": TrajectoryCompressionStrategy,
}


def get_memory_strategy(key: str) -> MemoryStrategy:
    try:
        return MEMORY_REGISTRY[key]()
    except KeyError as e:
        raise KeyError(
            f"Unknown memory strategy '{key}'. Known: {sorted(MEMORY_REGISTRY)}"
        ) from e


__all__ = [
    "MEMORY_REGISTRY",
    "BranchTrajectory",
    "CompressFn",
    "ContextFoldingStrategy",
    "MainThreadState",
    "MemoryStrategy",
    "PassRecord",
    "StatelessStrategy",
    "StructuredState",
    "TrajectoryCompressionStrategy",
    "get_memory_strategy",
]
