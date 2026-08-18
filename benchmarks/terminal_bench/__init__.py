"""Terminal-Bench integration for Sylliptor."""

from typing import Any

__all__ = ["SylliptorAgent", "SylliptorSimpleAgent"]


def __getattr__(name: str) -> Any:
    if name == "SylliptorAgent":
        from .harbor_agent import SylliptorAgent

        return SylliptorAgent
    if name == "SylliptorSimpleAgent":
        from .sylliptor_agent import SylliptorSimpleAgent

        return SylliptorSimpleAgent
    raise AttributeError(name)
