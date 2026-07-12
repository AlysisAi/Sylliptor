from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sylliptor_agent_cli.agent_loop import create_session
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.llm.metadata import (
    ANTHROPIC_MESSAGES_PROVIDER_METADATA_KEY,
    GEMINI_GENERATE_CONTENT_PROVIDER_METADATA_KEY,
    OPENAI_RESPONSES_PROVIDER_METADATA_KEY,
    PROVIDER_METADATA_KEY,
    strip_provider_metadata_from_message,
)
from sylliptor_agent_cli.llm.types import LLMResponse, ToolCall
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
            "tool_output_offload_threshold_chars": 6000,
            "tool_output_preview_chars": 2000,
            "summarize_conversation": False,
        },
    }
    return cfg


def test_run_turn_keeps_medium_tool_output_full_and_tool_transcript_valid(
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

    content = json.loads(str(tool_message.get("content") or "{}"))
    assert content["path"] == "sample.txt"
    assert len(str(content["content"])) == 1800
    assert "transcript_shaped" not in content
    assert "A" * 800 in str(tool_message.get("content") or "")

    persisted_tool_message = next(
        message for message in reversed(persisted_messages) if str(message.get("role")) == "tool"
    )
    assert persisted_tool_message["tool_call_id"] == "tc-read"
    assert json.loads(str(persisted_tool_message.get("content") or "{}"))["path"] == "sample.txt"

    tool_result_events = [
        dict(event.get("payload") or {})
        for event in read_session_events(log_path)
        if event.get("type") == "tool_result"
    ]
    assert tool_result_events
    assert tool_result_events[-1]["name"] == "fs_read"
    assert len(str((tool_result_events[-1]["result"] or {}).get("content") or "")) == 1800
    logged_content = json.loads(str(tool_result_events[-1].get("content") or "{}"))
    assert logged_content["path"] == "sample.txt"
    assert "transcript_shaped" not in logged_content
    assert "A" * 800 in str(tool_result_events[-1].get("content") or "")


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


@pytest.mark.parametrize(
    ("provider_key", "metadata_payload", "final_text"),
    [
        (
            OPENAI_RESPONSES_PROVIDER_METADATA_KEY,
            {
                "response_id": "resp_web_text",
                "output_items": [
                    {
                        "type": "web_search_call",
                        "id": "ws_1",
                        "status": "completed",
                        "action": {"type": "search", "query": "Sylliptor metadata"},
                    },
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Hosted search found the release notes.",
                            }
                        ],
                    },
                ],
                "web_search_calls": [
                    {
                        "type": "web_search_call",
                        "id": "ws_1",
                        "status": "completed",
                    }
                ],
            },
            "Hosted search found the release notes.",
        ),
        (
            ANTHROPIC_MESSAGES_PROVIDER_METADATA_KEY,
            {
                "message_id": "msg_search_text",
                "content_blocks": [
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_1",
                        "name": "web_search",
                        "input": {"query": "Sylliptor metadata"},
                    },
                    {
                        "type": "text",
                        "text": "Anthropic search found the release notes.",
                        "citations": [
                            {
                                "type": "web_search_result_location",
                                "url": "https://example.test/release",
                                "title": "Release notes",
                            }
                        ],
                    },
                ],
                "server_tool_uses": [
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_1",
                        "name": "web_search",
                    }
                ],
                "citations": [
                    {
                        "type": "web_search_result_location",
                        "url": "https://example.test/release",
                        "title": "Release notes",
                    }
                ],
            },
            "Anthropic search found the release notes.",
        ),
        (
            GEMINI_GENERATE_CONTENT_PROVIDER_METADATA_KEY,
            {
                "response_id": "resp_grounded_text",
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": "Gemini grounding found the release notes.",
                            "thoughtSignature": "thought-text-only",
                        }
                    ],
                },
                "groundingMetadata": {
                    "webSearchQueries": ["Sylliptor metadata"],
                    "groundingChunks": [
                        {
                            "web": {
                                "uri": "https://example.test/release",
                                "title": "Release notes",
                            }
                        }
                    ],
                },
                "search_queries": ["Sylliptor metadata"],
            },
            "Gemini grounding found the release notes.",
        ),
    ],
)
def test_run_turn_preserves_text_only_native_provider_metadata_for_followup(
    tmp_path: Path,
    provider_key: str,
    metadata_payload: dict[str, Any],
    final_text: str,
) -> None:
    session = create_session(
        cfg=_cfg(),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
        session_id_override=f"text-only-{provider_key}",
        enable_compaction=False,
        enable_tool_output_offload=True,
        enable_conversation_summarization=False,
    )
    client = _ScriptedClient(
        responses=[
            LLMResponse(
                content=final_text,
                tool_calls=[],
                raw={},
                provider_metadata={provider_key: metadata_payload},
            ),
            LLMResponse(content="Follow-up complete.", tool_calls=[], raw={}),
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        assert session.run_turn("Use native hosted search and answer.") == 0
        messages_after_first_turn = list(session.messages)
        assert session.run_turn("Use the same provider context.") == 0
        log_path = session.store.path
    finally:
        session.close()

    assistant_message = next(
        message
        for message in reversed(messages_after_first_turn)
        if str(message.get("role")) == "assistant" and message.get("content") == final_text
    )
    assert assistant_message[PROVIDER_METADATA_KEY][provider_key] == metadata_payload
    assert strip_provider_metadata_from_message(assistant_message) == {
        "role": "assistant",
        "content": final_text,
    }

    followup_messages = client.calls[1]["messages"]
    followup_assistant = next(
        message
        for message in followup_messages
        if str(message.get("role")) == "assistant" and message.get("content") == final_text
    )
    assert followup_assistant[PROVIDER_METADATA_KEY][provider_key] == metadata_payload

    assistant_events = [
        dict(event.get("payload") or {})
        for event in read_session_events(log_path)
        if event.get("type") == "assistant_message"
    ]
    first_event = next(
        payload for payload in assistant_events if payload.get("content") == final_text
    )
    assert first_event["message"][PROVIDER_METADATA_KEY][provider_key] == metadata_payload
    assert PROVIDER_METADATA_KEY not in str(first_event.get("content") or "")


def test_run_turn_persists_duplicate_final_text_native_provider_metadata(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("release notes", encoding="utf-8")
    repeated_text = "I found the release notes."
    metadata_payload = {
        "response_id": "resp_duplicate_final_text",
        "output_items": [
            {
                "type": "web_search_call",
                "id": "ws_duplicate_final",
                "status": "completed",
                "action": {"type": "search", "query": "Sylliptor release notes"},
            },
            {
                "type": "message",
                "id": "msg_duplicate_final",
                "role": "assistant",
                "content": [{"type": "output_text", "text": repeated_text}],
            },
        ],
    }
    session = create_session(
        cfg=_cfg(),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
        session_id_override="duplicate-final-native-metadata",
        enable_compaction=False,
        enable_tool_output_offload=True,
        enable_conversation_summarization=False,
    )
    client = _ScriptedClient(
        responses=[
            LLMResponse(
                content=repeated_text,
                tool_calls=[
                    ToolCall(
                        id="tc-read",
                        name="fs_read",
                        arguments={"path": "sample.txt"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content=repeated_text,
                tool_calls=[],
                raw={},
                provider_metadata={OPENAI_RESPONSES_PROVIDER_METADATA_KEY: metadata_payload},
            ),
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Read the file and answer with hosted search context.")
        log_path = session.store.path
        persisted_messages = list(session.messages)
    finally:
        session.close()

    assert exit_code == 0
    final_assistant_message = persisted_messages[-1]
    assert final_assistant_message["content"] == repeated_text
    assert (
        final_assistant_message[PROVIDER_METADATA_KEY][OPENAI_RESPONSES_PROVIDER_METADATA_KEY]
        == metadata_payload
    )

    assistant_events = [
        dict(event.get("payload") or {})
        for event in read_session_events(log_path)
        if event.get("type") == "assistant_message"
    ]
    final_metadata_event = next(
        payload
        for payload in assistant_events
        if (
            isinstance(payload.get("message"), dict)
            and payload["message"]
            .get(PROVIDER_METADATA_KEY, {})
            .get(OPENAI_RESPONSES_PROVIDER_METADATA_KEY, {})
            .get("response_id")
            == "resp_duplicate_final_text"
        )
    )
    assert final_metadata_event["content"] == repeated_text
    assert "web_search_call" not in final_metadata_event["content"]
