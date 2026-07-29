"""
HyperDecept-WB: White-box explainability module.

Provides structured evidence packets, framework-neutral adapters, append-only
real-time traces, additive risk decomposition, and geometry-faithful graph
explanation components.
"""

__version__ = "1.2"

from .adapters import (
    AdapterRegistry,
    ExplanationFragment,
    ExplanationOrchestrator,
    ExplanationRequest,
)
from .schemas import ExplanationPacket

__all__ = [
    "AdapterRegistry",
    "ExplanationFragment",
    "ExplanationOrchestrator",
    "ExplanationPacket",
    "ExplanationRequest",
]
