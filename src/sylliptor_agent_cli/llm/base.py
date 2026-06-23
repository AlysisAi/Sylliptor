from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from .types import LLMResponse


@runtime_checkable
class ChatClient(Protocol):
    """Protocol-neutral chat client surface used by the agent runtime."""

    base_url: str
    model: str

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
        on_text_delta: Callable[[str], None] | None = None,
        on_reasoning_delta: Callable[[str], None] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancellation_token: Any | None = None,
    ) -> LLMResponse: ...
