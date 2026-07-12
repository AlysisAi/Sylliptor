from __future__ import annotations

from typing import Any

from sylliptor_agent_cli.agent.routing import (
    _main_agent_chat,
    _safe_forced_tool_choice_for_recovery,
)
from sylliptor_agent_cli.llm.openai_compat import LLMResponse


def _tool_schema(name: str = "diagnostic_echo") -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object", "properties": {}}},
        }
    ]


class _RecordingClient:
    supports_tool_calling = False
    supports_forced_tool_choice = True

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(dict(kwargs))
        return LLMResponse(content="ok", tool_calls=[], raw={})


def test_main_agent_chat_omits_tools_after_dynamic_tool_calling_rejection() -> None:
    client = _RecordingClient()

    response = _main_agent_chat(
        client=client,
        messages=[{"role": "user", "content": "hi"}],
        tools=_tool_schema(),
        stream=False,
        on_text_delta=None,
        tool_choice={"type": "function", "function": {"name": "diagnostic_echo"}},
    )

    assert response.content == "ok"
    assert client.calls[0]["tools"] is None
    assert "tool_choice" not in client.calls[0]


def test_forced_tool_choice_requires_tool_calling_and_no_active_reasoning() -> None:
    client = type(
        "Client",
        (),
        {
            "supports_tool_calling": True,
            "supports_forced_tool_choice": True,
            "enable_thinking": True,
        },
    )()

    assert (
        _safe_forced_tool_choice_for_recovery(
            client=client,
            tools=_tool_schema(),
            preferred_tool_names=("diagnostic_echo",),
        )
        is None
    )

    client.enable_thinking = False
    assert _safe_forced_tool_choice_for_recovery(
        client=client,
        tools=_tool_schema(),
        preferred_tool_names=("diagnostic_echo",),
    ) == {"type": "function", "function": {"name": "diagnostic_echo"}}
