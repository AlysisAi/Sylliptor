from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    provider_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cached_prompt_tokens: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    content: str
    tool_calls: list[ToolCall]
    raw: dict[str, Any]
    response_model: str | None = None
    usage: LLMUsage | None = None
    provider_metadata: dict[str, Any] | None = None
