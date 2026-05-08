from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sylliptor_agent_cli.agent_loop import create_session
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.llm.openai_compat import PROVIDER_METADATA_KEY, LLMResponse, ToolCall
from sylliptor_agent_cli.session_store import read_session_events


class _ScriptedClient:
    def __init__(self, *, responses: list[LLMResponse]) -> None:
        self.model = "test-model"
        self.temperature = 0.2
        self._responses = responses
        self._calls = 0
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        on_text_delta=None,  # type: ignore[no-untyped-def]
        temperature: float | None = None,
    ) -> LLMResponse:
        _ = stream, on_text_delta, temperature
        self.calls.append({"messages": list(messages), "tools": tools})
        response = self._responses[self._calls]
        self._calls += 1
        return response


def _cfg() -> AppConfig:
    cfg = AppConfig(model="test-model", routing_mode="code_only", stream=False, max_steps=4)
    cfg.extra_fields = {
        "model_metadata_overrides": {
            "models": {
                "test-model": {"context_window_tokens": 4096, "max_output_tokens": 512},
            },
            "default": {"context_window_tokens": 4096, "max_output_tokens": 512},
        },
        "compaction": {
            "enabled": True,
            "offload_tool_outputs": True,
            "tool_output_offload_threshold_chars": 2500,
            "tool_output_preview_chars": 400,
            "summarize_conversation": False,
        },
    }
    return cfg


def test_run_turn_shapes_medium_tool_output_but_keeps_tool_transcript_valid(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("A" * 1800, encoding="utf-8")

    session = create_session(
        cfg=_cfg(),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
        session_id_override="tool-shaping",
        enable_compaction=False,
        enable_tool_output_offload=True,
        enable_conversation_summarization=False,
    )
    client = _ScriptedClient(
        responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc-read",
                        name="fs_read",
                        arguments={"path": "sample.txt", "max_bytes": 1800},
                    )
                ],
                raw={},
            ),
            LLMResponse(content="Done.", tool_calls=[], raw={}),
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Read the file and summarize it.")
        log_path = session.store.path
        persisted_messages = list(session.messages)
    finally:
        session.close()

    assert exit_code == 0
    assert len(client.calls) == 2

    followup_messages = client.calls[1]["messages"]
    assistant_call = next(
        message
        for message in reversed(followup_messages)
        if str(message.get("role")) == "assistant" and message.get("tool_calls")
    )
    tool_message = next(
        message for message in reversed(followup_messages) if str(message.get("role")) == "tool"
    )

    assert tool_message["tool_call_id"] == "tc-read"
    assert assistant_call["tool_calls"][0]["id"] == "tc-read"

    shaped = json.loads(str(tool_message.get("content") or "{}"))
    assert shaped["transcript_shaped"] is True
    assert shaped["tool"] == "fs_read"
    assert shaped["tool_call_id"] == "tc-read"
    assert shaped["summary"] == 'Loaded "sample.txt" (1800 chars).'
    assert shaped["preview_chars"] == 400
    assert shaped["original_chars"] > 1800
    assert shaped["content_truncated"] is True
    assert len(shaped["preview"]) <= 400 + len("...(truncated)")
    assert "A" * 800 not in str(tool_message.get("content") or "")

    persisted_tool_message = next(
        message for message in reversed(persisted_messages) if str(message.get("role")) == "tool"
    )
    assert persisted_tool_message["tool_call_id"] == "tc-read"
    assert (
        json.loads(str(persisted_tool_message.get("content") or "{}"))["tool_call_id"] == "tc-read"
    )

    tool_result_events = [
        dict(event.get("payload") or {})
        for event in read_session_events(log_path)
        if event.get("type") == "tool_result"
    ]
    assert tool_result_events
    assert tool_result_events[-1]["name"] == "fs_read"
    assert len(str((tool_result_events[-1]["result"] or {}).get("content") or "")) == 1800


def test_run_turn_preserves_deepseek_reasoning_metadata_for_tool_followup(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")

    session = create_session(
        cfg=_cfg(),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
        session_id_override="deepseek-reasoning-tool-loop",
        enable_compaction=False,
        enable_tool_output_offload=True,
        enable_conversation_summarization=False,
    )
    client = _ScriptedClient(
        responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc-read",
                        name="fs_read",
                        arguments={"path": "sample.txt"},
                    )
                ],
                raw={},
                provider_metadata={"deepseek": {"reasoning_content": "hidden reasoning state"}},
            ),
            LLMResponse(content="Done.", tool_calls=[], raw={}),
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Read the file and summarize it.")
    finally:
        session.close()

    assert exit_code == 0
    followup_messages = client.calls[1]["messages"]
    assistant_call = next(
        message
        for message in reversed(followup_messages)
        if str(message.get("role")) == "assistant" and message.get("tool_calls")
    )
    assert assistant_call[PROVIDER_METADATA_KEY] == {
        "deepseek": {"reasoning_content": "hidden reasoning state"}
    }
    assert "reasoning_content" not in assistant_call
