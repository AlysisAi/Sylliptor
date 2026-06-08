from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from sylliptor_agent_cli import agent_loop as agent_loop_mod
from sylliptor_agent_cli.agent_loop import create_session
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.custom_tools.discovery import discover_custom_tools
from sylliptor_agent_cli.custom_tools.trust import trust_project_tool
from sylliptor_agent_cli.llm.metadata import (
    GEMINI_GENERATE_CONTENT_PROVIDER_METADATA_KEY,
    PROVIDER_METADATA_KEY,
)
from sylliptor_agent_cli.llm.openai_compat import LLMError, LLMResponse, ToolCall
from sylliptor_agent_cli.request_estimation import estimate_message_tokens
from sylliptor_agent_cli.runtime_kind import RuntimeKind
from sylliptor_agent_cli.session_store import read_session_events
from sylliptor_agent_cli.turn_intent import looks_like_implicit_repo_improvement_request


class _FailClient:
    model = "test-model"
    temperature = 0.2

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        on_text_delta=None,  # type: ignore[no-untyped-def]
        temperature: float | None = None,
    ) -> LLMResponse:
        _ = messages, tools, stream, on_text_delta, temperature
        raise AssertionError("Repo agent client should not be called for this turn.")


class _RouterStubClient:
    model = "test-model"
    temperature = 0.0

    def __init__(
        self,
        *,
        route: str,
        route_reply: str = "",
        response_reply: str = "",
        execution_posture: str | None = None,
        omit_execution_posture: bool = False,
        confidence: float = 0.99,
        language: str = "",
        script: str = "",
        explicit_language_override: bool = False,
        response_provider_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.route = route
        self.route_reply = route_reply
        self.response_reply = response_reply
        self.execution_posture = execution_posture
        self.omit_execution_posture = omit_execution_posture
        self.confidence = confidence
        self.language = language
        self.script = script
        self.explicit_language_override = explicit_language_override
        self.response_provider_metadata = response_provider_metadata
        self.calls = 0
        self.route_calls = 0
        self.response_calls = 0
        self.last_messages: list[dict[str, Any]] = []
        self.last_route_messages: list[dict[str, Any]] = []
        self.last_tools: list[dict[str, Any]] | None = None
        self.last_temperature: float | None = None
        self.last_route_temperature: float | None = None
        self.last_response_temperature: float | None = None

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        on_text_delta=None,  # type: ignore[no-untyped-def]
        temperature: float | None = None,
    ) -> LLMResponse:
        _ = stream, on_text_delta
        self.calls += 1
        self.last_messages = list(messages)
        self.last_tools = tools
        self.last_temperature = temperature
        first_system = ""
        if messages:
            first = messages[0]
            if isinstance(first, dict):
                first_system = str(first.get("content") or "")

        if first_system == agent_loop_mod._ROUTER_SYSTEM_PROMPT:
            self.route_calls += 1
            self.last_route_messages = list(messages)
            self.last_route_temperature = temperature
            execution_posture = self.execution_posture
            if not execution_posture:
                execution_posture = "execute" if self.route == "repo" else "advisory_non_execution"
            payload = {
                "route": self.route,
                "confidence": self.confidence,
                "reply": self.route_reply,
                "language": self.language,
                "script": self.script,
                "explicit_language_override": self.explicit_language_override,
            }
            if not self.omit_execution_posture:
                payload["execution_posture"] = execution_posture
            return LLMResponse(content=json.dumps(payload), tool_calls=[], raw={})

        self.response_calls += 1
        self.last_response_temperature = temperature
        return LLMResponse(
            content=self.response_reply,
            tool_calls=[],
            raw={},
            provider_metadata=self.response_provider_metadata,
        )


class _SequentialRouterClient:
    model = "test-model"
    temperature = 0.0

    def __init__(
        self,
        *,
        routes: list[str],
        response_replies: list[str],
        route_replies: list[str] | None = None,
        response_tool_calls: list[list[ToolCall]] | None = None,
        execution_postures: list[str] | None = None,
        confidence: float = 0.99,
        language: str = "",
        script: str = "",
        explicit_language_override: bool = False,
        tool_family: str = "none",
        tool_candidates: list[str] | None = None,
    ) -> None:
        self.routes = list(routes)
        self.response_replies = list(response_replies)
        self.route_replies = list(route_replies or [])
        self.response_tool_calls = list(response_tool_calls or [])
        self.execution_postures = list(execution_postures or [])
        self.confidence = confidence
        self.language = language
        self.script = script
        self.explicit_language_override = explicit_language_override
        self.tool_family = tool_family
        self.tool_candidates = list(tool_candidates or [])
        self.calls = 0
        self.route_calls = 0
        self.response_calls = 0
        self.last_messages: list[dict[str, Any]] = []
        self.last_route_messages: list[dict[str, Any]] = []
        self.last_tools: list[dict[str, Any]] | None = None
        self.last_temperature: float | None = None
        self.last_route_temperature: float | None = None
        self.last_response_temperature: float | None = None

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        on_text_delta=None,  # type: ignore[no-untyped-def]
        temperature: float | None = None,
    ) -> LLMResponse:
        _ = stream, on_text_delta
        self.calls += 1
        self.last_messages = list(messages)
        self.last_tools = tools
        self.last_temperature = temperature
        first_system = ""
        if messages:
            first = messages[0]
            if isinstance(first, dict):
                first_system = str(first.get("content") or "")

        if first_system == agent_loop_mod._ROUTER_SYSTEM_PROMPT:
            self.route_calls += 1
            self.last_route_messages = list(messages)
            self.last_route_temperature = temperature
            route_index = min(self.route_calls - 1, len(self.routes) - 1)
            route = self.routes[route_index]
            route_reply = ""
            if self.route_replies:
                reply_index = min(self.route_calls - 1, len(self.route_replies) - 1)
                route_reply = self.route_replies[reply_index]
            execution_posture = ""
            if self.execution_postures:
                posture_index = min(self.route_calls - 1, len(self.execution_postures) - 1)
                execution_posture = self.execution_postures[posture_index]
            if not execution_posture:
                execution_posture = "execute" if route == "repo" else "advisory_non_execution"
            payload = {
                "route": route,
                "execution_posture": execution_posture,
                "confidence": self.confidence,
                "reply": route_reply,
                "language": self.language,
                "script": self.script,
                "explicit_language_override": self.explicit_language_override,
                "tool_family": self.tool_family,
                "tool_candidates": list(self.tool_candidates),
            }
            return LLMResponse(content=json.dumps(payload), tool_calls=[], raw={})

        self.response_calls += 1
        self.last_response_temperature = temperature
        response_index = min(self.response_calls - 1, len(self.response_replies) - 1)
        tool_calls: list[ToolCall] = []
        if self.response_tool_calls:
            tool_index = min(self.response_calls - 1, len(self.response_tool_calls) - 1)
            tool_calls = list(self.response_tool_calls[tool_index])
        return LLMResponse(
            content=self.response_replies[response_index],
            tool_calls=tool_calls,
            raw={},
        )


class _RouterExceptionClient:
    model = "test-model"
    temperature = 0.0

    def __init__(self) -> None:
        self.calls = 0
        self.route_calls = 0
        self.response_calls = 0
        self.last_route_messages: list[dict[str, Any]] = []

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        on_text_delta=None,  # type: ignore[no-untyped-def]
        temperature: float | None = None,
    ) -> LLMResponse:
        _ = tools, stream, on_text_delta, temperature
        self.calls += 1
        first_system = ""
        if messages:
            first = messages[0]
            if isinstance(first, dict):
                first_system = str(first.get("content") or "")
        if first_system == agent_loop_mod._ROUTER_SYSTEM_PROMPT:
            self.route_calls += 1
            self.last_route_messages = list(messages)
            raise RuntimeError("router unavailable")
        self.response_calls += 1
        raise AssertionError("Fallback router client should not handle repo or non-repo responses.")


class _MalformedRouterClient:
    model = "test-model"
    temperature = 0.0

    def __init__(self, content: str = "not json") -> None:
        self.content = content
        self.calls = 0
        self.route_calls = 0
        self.response_calls = 0
        self.last_route_messages: list[dict[str, Any]] = []

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        on_text_delta=None,  # type: ignore[no-untyped-def]
        temperature: float | None = None,
    ) -> LLMResponse:
        _ = tools, stream, on_text_delta, temperature
        self.calls += 1
        first_system = ""
        if messages:
            first = messages[0]
            if isinstance(first, dict):
                first_system = str(first.get("content") or "")
        if first_system == agent_loop_mod._ROUTER_SYSTEM_PROMPT:
            self.route_calls += 1
            self.last_route_messages = list(messages)
            return LLMResponse(content=self.content, tool_calls=[], raw={})
        self.response_calls += 1
        raise AssertionError(
            "Malformed router client should not handle repo or non-repo responses."
        )


def _invalid_api_key_error() -> LLMError:
    return LLMError(
        'LLM error 401: {"error":{"message":"Incorrect API key provided.","type":"invalid_request_error","param":null,"code":"invalid_api_key"}}'
    )


class _AuthRejectingClient:
    model = "test-model"
    temperature = 0.0

    def __init__(
        self,
        *,
        route: str = "general",
        route_error: bool = False,
        response_error: bool = False,
        response_reply: str = "",
    ) -> None:
        self.route = route
        self.route_error = route_error
        self.response_error = response_error
        self.response_reply = response_reply
        self.calls = 0
        self.route_calls = 0
        self.response_calls = 0

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        on_text_delta=None,  # type: ignore[no-untyped-def]
        temperature: float | None = None,
    ) -> LLMResponse:
        _ = tools, stream, on_text_delta, temperature
        self.calls += 1
        first_system = ""
        if messages:
            first = messages[0]
            if isinstance(first, dict):
                first_system = str(first.get("content") or "")

        if first_system == agent_loop_mod._ROUTER_SYSTEM_PROMPT:
            self.route_calls += 1
            if self.route_error:
                raise _invalid_api_key_error()
            payload = {
                "route": self.route,
                "execution_posture": (
                    "execute" if self.route == "repo" else "advisory_non_execution"
                ),
                "confidence": 0.99,
                "reply": "",
                "language": "English",
                "script": "Latin",
                "explicit_language_override": False,
            }
            return LLMResponse(content=json.dumps(payload), tool_calls=[], raw={})

        self.response_calls += 1
        if self.response_error:
            raise _invalid_api_key_error()
        return LLMResponse(content=self.response_reply, tool_calls=[], raw={})


class _RepoCaptureClient:
    model = "test-model"
    temperature = 0.2

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0
        self.last_messages: list[dict[str, Any]] = []
        self.last_tools: list[dict[str, Any]] | None = None

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
        self.calls += 1
        self.last_messages = list(messages)
        self.last_tools = tools
        return LLMResponse(content=self.reply, tool_calls=[], raw={})


class _ScriptedClient:
    model = "test-model"
    temperature = 0.2

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self.calls = 0
        self.call_records: list[dict[str, Any]] = []

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        on_text_delta=None,  # type: ignore[no-untyped-def]
        temperature: float | None = None,
    ) -> LLMResponse:
        _ = on_text_delta, temperature
        self.call_records.append(
            {
                "messages": list(messages),
                "tools": tools,
                "stream": stream,
            }
        )
        response = self._responses[self.calls]
        self.calls += 1
        return response


def _init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    (root / "README.md").write_text("repo\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"], cwd=root, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_empty_git_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def _write_project_custom_tool(
    root: Path,
    *,
    name: str,
    body: str,
) -> None:
    tool_path = root / ".sylliptor" / "tools" / f"{name}.py"
    tool_path.parent.mkdir(parents=True, exist_ok=True)
    tool_path.write_text(
        "\n".join(
            [
                "TOOL = {",
                f'    "name": "{name}",',
                '    "description": "Project custom tool",',
                '    "input_schema": {"type": "object", "properties": {}, "required": []},',
                "}",
                "",
                "def run(args):",
                *[f"    {line}" for line in body.splitlines()],
                "",
            ]
        ),
        encoding="utf-8",
    )


def _trust_project_custom_tool(root: Path, *, name: str) -> None:
    discovery = discover_custom_tools(
        workspace_root=root,
        built_in_tool_names=set(),
    )
    specs = {spec.name: spec for spec in discovery.project_tools}
    trust_project_tool(specs[name])


def _route_context_payload(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in messages:
        if str(message.get("role") or "") != "system":
            continue
        content = str(message.get("content") or "")
        if not content.startswith(agent_loop_mod._ROUTE_CONTEXT_MARKER):
            continue
        _marker, _newline, payload_raw = content.partition("\n")
        return json.loads(payload_raw)
    return None


def _session_event_payload(path: Path, event_type: str) -> dict[str, Any]:
    for event in read_session_events(path):
        if str(event.get("type") or "") == event_type:
            payload = event.get("payload")
            if isinstance(payload, dict):
                return payload
    raise AssertionError(f"event not found: {event_type}")


def test_how_are_you_routes_to_chat_without_repo_agent_call(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.73)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
    )
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterStubClient(
        route="chat",
        route_reply="",
        response_reply="I'm doing well, thanks. How are you?",
        language="English",
        script="Latin",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("How are you?")
    finally:
        session.close()

    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 1
    assert router.last_tools is None
    assert router.last_route_temperature == 0.0
    assert router.last_response_temperature == 0.73
    assert session.messages[-1]["role"] == "assistant"
    content = str(session.messages[-1]["content"])
    assert content == "I'm doing well, thanks. How are you?"
    assert "repo" not in content.lower()
    assert "repository" not in content.lower()
    assert "workspace" not in content.lower()


def test_non_repo_text_response_preserves_native_provider_metadata(tmp_path: Path) -> None:
    metadata_payload = {
        "response_id": "resp_non_repo_grounded",
        "content": {
            "role": "model",
            "parts": [
                {
                    "text": "Gemini grounding answered the chat turn.",
                    "thoughtSignature": "non-repo-thought",
                }
            ],
        },
        "groundingMetadata": {
            "webSearchQueries": ["Sylliptor native providers"],
        },
    }
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.73)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterStubClient(
        route="chat",
        route_reply="",
        response_reply="Gemini grounding answered the chat turn.",
        language="English",
        script="Latin",
        response_provider_metadata={
            GEMINI_GENERATE_CONTENT_PROVIDER_METADATA_KEY: metadata_payload,
        },
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("Can you answer from native search context?")
        log_path = session.store.path
        assistant_message = dict(session.messages[-1])
    finally:
        session.close()

    assert exit_code == 0
    assert (
        assistant_message[PROVIDER_METADATA_KEY][GEMINI_GENERATE_CONTENT_PROVIDER_METADATA_KEY]
        == metadata_payload
    )
    assistant_events = [
        dict(event.get("payload") or {})
        for event in read_session_events(log_path)
        if event.get("type") == "assistant_message"
    ]
    final_event = assistant_events[-1]
    assert final_event["content"] == "Gemini grounding answered the chat turn."
    assert (
        final_event["message"][PROVIDER_METADATA_KEY][GEMINI_GENERATE_CONTENT_PROVIDER_METADATA_KEY]
        == metadata_payload
    )


def test_non_repo_fast_path_uses_router_reply_without_second_llm_call(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.73)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterStubClient(
        route="chat",
        route_reply="Hi. What can I help with?",
        response_reply="This should not be used.",
        language="English",
        script="Latin",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("Hi")
    finally:
        session.close()

    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 0
    assert session.messages[-1]["role"] == "assistant"
    assert session.messages[-1]["content"] == "Hi. What can I help with?"
    event_types = [event.get("type") for event in read_session_events(event_path)]
    assert "non_repo_router_reply_used" in event_types


def test_non_repo_chat_follow_up_uses_recent_history_not_router_reply(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.73)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
    )
    session.client = _FailClient()  # type: ignore[assignment]
    router = _SequentialRouterClient(
        routes=["chat", "chat", "chat", "chat"],
        route_replies=[
            "Got it.",
            "Wrong router shortcut.",
            "Wrong router shortcut.",
            "Wrong router shortcut.",
        ],
        response_replies=[
            "I will remember aurora.",
            "Here is a short joke.",
            "Your code word was aurora.",
        ],
    )
    session.router_client = router

    try:
        assert session.run_turn("My code word is aurora.") == 0
        assert session.run_turn("Please remember it for later.") == 0
        assert session.run_turn("Now tell me a short joke.") == 0
        assert session.run_turn("What was my code word?") == 0
        final_message = str(session.messages[-1].get("content") or "")
    finally:
        session.close()

    assert final_message == "Your code word was aurora."
    assert router.route_calls == 4
    assert router.response_calls == 3
    assert "Wrong router shortcut." not in final_message

    response_history_text = "\n".join(
        str(message.get("content") or "")
        for message in router.last_messages[:-1]
        if message.get("role") in {"user", "assistant"}
    )
    route_history_text = "\n".join(
        str(message.get("content") or "")
        for message in router.last_route_messages[:-1]
        if message.get("role") in {"user", "assistant"}
    )
    assert "My code word is aurora." in response_history_text
    assert "My code word is aurora." in route_history_text
    assert router.last_route_messages[-1]["content"] == "What was my code word?"


def test_explain_recursion_routes_to_general_without_repo_agent_call(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.61)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
    )
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        route_reply="",
        response_reply="Recursion is when a function calls itself until a base case stops it.",
        language="English",
        script="Latin",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("Explain recursion in Python in two lines.")
    finally:
        session.close()

    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 1
    assert router.last_route_temperature == 0.0
    assert router.last_response_temperature == 0.61
    assert session.messages[-1]["role"] == "assistant"
    assert session.messages[-1]["content"] == (
        "Recursion is when a function calls itself until a base case stops it."
    )


def test_auto_mode_router_auth_error_is_not_masked_as_clarification_fallback(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.7)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
    )
    session.client = _FailClient()  # type: ignore[assignment]
    session.router_client = _AuthRejectingClient(route_error=True)
    baseline_messages = copy.deepcopy(session.messages)

    try:
        with pytest.raises(LLMError, match="invalid_api_key"):
            session.run_turn("How are you?")
        assert session.messages == baseline_messages
        error_payload = _session_event_payload(session.store.path, "error")
        assert "invalid_api_key" in str(error_payload.get("error") or "")
    finally:
        session.close()


def test_auto_mode_non_repo_auth_error_is_not_masked_as_clarification_fallback(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.7)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
    )
    session.client = _FailClient()  # type: ignore[assignment]
    router = _AuthRejectingClient(route="chat", response_error=True)
    session.router_client = router
    baseline_messages = copy.deepcopy(session.messages)

    try:
        with pytest.raises(LLMError, match="invalid_api_key"):
            session.run_turn("How are you?")
        assert router.route_calls == 1
        assert router.response_calls == 1
        assert session.messages == baseline_messages
        error_payload = _session_event_payload(session.store.path, "error")
        assert "invalid_api_key" in str(error_payload.get("error") or "")
    finally:
        session.close()


def test_repo_request_routes_to_repo_and_calls_main_agent_client(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="auto")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
    )
    repo_client = _RepoCaptureClient(reply="I will inspect src/app.py first.")
    session.client = repo_client  # type: ignore[assignment]
    router = _RouterStubClient(
        route="repo",
        route_reply="",
        response_reply="",
        language="English",
        script="Latin",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("Open src/app.py and find why tests fail.")
    finally:
        session.close()

    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 0
    assert repo_client.calls == 1
    assert not any(
        msg.get("role") == "system"
        and "explicitly requested a language/script override" in str(msg.get("content") or "")
        for msg in repo_client.last_messages
    )


def test_router_context_lists_exposed_custom_tools(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="auto")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
    )
    session.client = _FailClient()  # type: ignore[assignment]
    session.tools["repo_summary"] = agent_loop_mod.ToolDef(
        name="repo_summary",
        description="Read the workspace data fixture and summarize it.",
        parameters={
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
        },
        run=lambda args: {"label": args.get("label")},
        metadata={
            "tool_type": "custom_tool",
            "custom_tool": {
                "name": "repo_summary",
                "source_scope": "project",
                "relative_tool_path": ".sylliptor/tools/repo_summary.py",
            },
        },
    )
    session.tool_list = [tool.as_openai_tool() for tool in session.tools.values()]
    router = _RouterStubClient(
        route="general",
        route_reply="",
        response_reply="Router context captured.",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("Please use the custom tool repo_summary.")
    finally:
        session.close()

    route_context = _route_context_payload(router.last_route_messages)

    assert exit_code == 0
    assert route_context is not None
    assert route_context["custom_tools"] == [
        {
            "name": "repo_summary",
            "description": "Read the workspace data fixture and summarize it.",
            "source_scope": "project",
            "relative_tool_path": ".sylliptor/tools/repo_summary.py",
        }
    ]
    assert (
        'classify it as route="repo" with\n  execution_posture="execute"'
        in agent_loop_mod._ROUTER_SYSTEM_PROMPT
    )


def test_general_non_repo_turn_uses_web_tool_assisted_path_when_available(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        model="qwen3.5-plus",
        base_url="https://coding-intl.dashscope.aliyuncs.com/v1",
        web_search_mode="auto",
        routing_mode="auto",
    )
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        response_reply="Yes, live web search is available.",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("μπορεις να ψαξεις στο ιντερνετ ;")
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    tool_names = {
        str(tool.get("function", {}).get("name") or "") for tool in (router.last_tools or [])
    }
    system_text = "\n".join(
        str(message.get("content") or "")
        for message in router.last_messages
        if message.get("role") == "system"
    )

    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 1
    assert tool_names == {"web_fetch", "web_search"}
    assert route_payload["route"] == "general"
    assert route_payload["original_route"] == "general"
    assert route_payload["route_selection_source"] == "router"
    assert route_payload["route_override_reason"] is None
    assert route_payload["execution_posture"] == "advisory_non_execution"
    assert "Available tools for this turn: web_fetch, web_search" in system_text
    assert "use the available web tool instead of answering from memory" in system_text
    assert "canned/random example" in system_text
    assert "do not claim browsing is unavailable" in system_text


def test_non_repo_tool_prompt_does_not_advertise_search_when_unregistered(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        model="qwen3.5-plus",
        base_url="https://coding-intl.dashscope.aliyuncs.com/v1",
        web_search_mode="off",
        routing_mode="auto",
    )
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        response_reply="Query-based web discovery is not available in this session.",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("Can you search the internet?")
    finally:
        session.close()

    tool_names = {
        str(tool.get("function", {}).get("name") or "") for tool in (router.last_tools or [])
    }
    system_text = "\n".join(
        str(message.get("content") or "")
        for message in router.last_messages
        if message.get("role") == "system"
    )

    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 1
    assert tool_names == {"web_fetch"}
    assert "Available tools for this turn: web_fetch" in system_text
    assert "use web_fetch instead of answering from memory" in system_text
    assert "canned/random example" in system_text
    assert "query-based web discovery is not available" in system_text
    assert "use web_search before answering" not in system_text


def test_non_repo_tool_prompt_filters_stale_unregistered_web_search_schema(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        model="qwen3.5-plus",
        base_url="https://coding-intl.dashscope.aliyuncs.com/v1",
        web_search_mode="auto",
        routing_mode="auto",
    )
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    assert "web_search" in session.tools
    session.tools.pop("web_search")
    # Leave session.tool_list intentionally stale to simulate a long-running
    # session whose runtime tool registry changed after startup.
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        response_reply="Query-based web discovery is not available in this session.",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("Can you search the internet?")
    finally:
        session.close()

    tool_names = {
        str(tool.get("function", {}).get("name") or "") for tool in (router.last_tools or [])
    }
    system_text = "\n".join(
        str(message.get("content") or "")
        for message in router.last_messages
        if message.get("role") == "system"
    )

    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 1
    assert "web_fetch" in tool_names
    assert "web_search" not in tool_names
    assert "Available tools for this turn: web_fetch" in system_text
    assert "use web_search before answering" not in system_text


def test_non_repo_tool_assisted_path_ignores_router_reply_for_general_turn(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        model="qwen3.5-plus",
        base_url="https://coding-intl.dashscope.aliyuncs.com/v1",
        web_search_mode="auto",
        routing_mode="auto",
    )
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        route_reply="I cannot browse.",
        response_reply="I will use web_search before answering.",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("Ψάξε στο internet για την τελευταία έκδοση της Python.")
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    system_text = "\n".join(
        str(message.get("content") or "")
        for message in router.last_messages
        if message.get("role") == "system"
    )

    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 1
    assert session.messages[-1]["content"] == "I will use web_search before answering."
    assert route_payload["route"] == "general"
    assert route_payload["original_route"] == "general"
    assert route_payload["route_selection_source"] == "router"
    assert route_payload["route_override_reason"] is None
    assert "If the user needs live, current, or external information" in system_text
    assert "use web_search before answering" in system_text


def test_tool_route_exposes_mcp_tools_in_non_repo_chat(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.5)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    mcp_calls: list[dict[str, Any]] = []
    session.tools["mcp__alpha__echo_tool"] = agent_loop_mod.ToolDef(
        name="mcp__alpha__echo_tool",
        description="Echo a short message through the alpha MCP server.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        run=lambda args: mcp_calls.append(dict(args)) or {"marker": "alpha-echo-ok"},
    )
    session.tool_list = [tool.as_openai_tool() for tool in session.tools.values()]
    session.client = _FailClient()  # type: ignore[assignment]
    router = _SequentialRouterClient(
        routes=["tool"],
        execution_postures=["execute"],
        response_replies=["", "The MCP echo returned alpha-echo-ok."],
        response_tool_calls=[
            [
                ToolCall(
                    id="tc1",
                    name="mcp__alpha__echo_tool",
                    arguments={"text": "hello"},
                )
            ],
            [],
        ],
        tool_family="mcp",
        tool_candidates=["mcp__alpha__echo_tool"],
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("Use the alpha MCP echo tool with text hello.")
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    route_context = _route_context_payload(router.last_route_messages)
    tool_names = {
        str(tool.get("function", {}).get("name") or "") for tool in (router.last_tools or [])
    }
    tool_call_names = [
        str(event.get("payload", {}).get("name") or "")
        for event in read_session_events(event_path)
        if event.get("type") == "tool_call"
    ]
    system_text = "\n".join(
        str(message.get("content") or "")
        for message in router.last_messages
        if message.get("role") == "system"
    )

    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 2
    assert mcp_calls == [{"text": "hello"}]
    assert route_payload["route"] == "tool"
    assert route_payload["tool_family"] == "mcp"
    assert route_payload["tool_candidates"] == ["mcp__alpha__echo_tool"]
    assert route_context is not None
    assert "mcp__alpha__echo_tool" in {tool["name"] for tool in route_context["non_repo_tools"]}
    assert tool_names == {"mcp__alpha__echo_tool"}
    assert tool_call_names == ["mcp__alpha__echo_tool"]
    assert "MCP tools/resources are available" in system_text
    assert "filesystem, shell, test, or edit actions" in system_text


def test_forge_exec_tool_route_is_promoted_to_repo_execution_for_mcp_write_task(
    tmp_path: Path,
) -> None:
    _init_git_repo(tmp_path)
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.5)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
        runtime_kind=RuntimeKind.FORGE_EXEC,
        non_interactive=True,
        allow_write_globs=["reports/tool_error_result.txt"],
    )
    event_path = session.store.path
    mcp_calls: list[dict[str, Any]] = []
    session.tools["mcp__alpha__fragile_tool"] = agent_loop_mod.ToolDef(
        name="mcp__alpha__fragile_tool",
        description="Controlled MCP diagnostic tool.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        run=lambda args: (
            mcp_calls.append(dict(args))
            or {
                "is_error": True,
                "structured_content": {"marker": "FORGE_TOOL_ERROR_SEEN"},
                "text": "FORGE_TOOL_ERROR_SEEN: controlled fixture error",
            }
        ),
    )
    session.tool_list = [tool.as_openai_tool() for tool in session.tools.values()]
    repo_client = _ScriptedClient(
        [
            LLMResponse(
                content="I will call the MCP diagnostic and write the captured marker.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="mcp__alpha__fragile_tool",
                        arguments={"text": "diagnostic"},
                    ),
                    ToolCall(
                        id="tc2",
                        name="fs_write",
                        arguments={
                            "path": "reports/tool_error_result.txt",
                            "content": "FORGE_TOOL_ERROR_SEEN: controlled fixture error\n",
                        },
                    ),
                ],
                raw={},
            ),
            LLMResponse(content="Done.", tool_calls=[], raw={}),
        ]
    )
    session.client = repo_client  # type: ignore[assignment]
    router = _SequentialRouterClient(
        routes=["tool"],
        execution_postures=["execute"],
        response_replies=["This non-repo response path should not be used."],
        response_tool_calls=[
            [
                ToolCall(
                    id="bad",
                    name="mcp__alpha__fragile_tool",
                    arguments={"text": "write: reports/tool_error_result.txt"},
                )
            ]
        ],
        tool_family="mcp",
        tool_candidates=["mcp__alpha__fragile_tool"],
    )
    session.router_client = router

    try:
        exit_code = session.run_turn(
            "# Task Context Pack\n\n"
            "## MCP Execution Context\n"
            "- Task MCP Scope: allowed live MCP tools: alpha/fragile-tool\n\n"
            "## Task Specification\n"
            "- Description: Call the fragile MCP diagnostic and write the marker to "
            "reports/tool_error_result.txt.\n"
            "- Write Scope: reports/tool_error_result.txt\n"
        )
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    repo_tool_names = {
        str(tool.get("function", {}).get("name") or "")
        for tool in (repo_client.call_records[0]["tools"] or [])
    }

    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 0
    assert repo_client.calls == 2
    assert route_payload["original_route"] == "tool"
    assert route_payload["route"] == "repo"
    assert (
        route_payload["route_override_reason"] == "forge_exec_managed_task_requires_repo_execution"
    )
    assert {"mcp__alpha__fragile_tool", "fs_write"} <= repo_tool_names
    assert mcp_calls == [{"text": "diagnostic"}]
    assert (tmp_path / "reports" / "tool_error_result.txt").read_text(encoding="utf-8") == (
        "FORGE_TOOL_ERROR_SEEN: controlled fixture error\n"
    )


def test_forge_custom_tool_streams_use_session_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _init_git_repo(tmp_path)
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path.parent / "config"))
    _write_project_custom_tool(
        tmp_path,
        name="stream_tool",
        body=(
            "import sys\n"
            "print('CUSTOM_STDOUT_SENTINEL')\n"
            "print('CUSTOM_STDERR_SENTINEL', file=sys.stderr)\n"
            "return {'ok': True}"
        ),
    )
    _trust_project_custom_tool(tmp_path, name="stream_tool")
    sessions_dir = tmp_path.parent / "runtime-sessions"
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.5)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        session_id_override="T01",
        runtime_kind=RuntimeKind.FORGE_EXEC,
        non_interactive=True,
    )
    event_path = session.store.path
    repo_client = _ScriptedClient(
        [
            LLMResponse(
                content="Calling the custom tool.",
                tool_calls=[ToolCall(id="tc1", name="stream_tool", arguments={})],
                raw={},
            ),
            LLMResponse(content="Done.", tool_calls=[], raw={}),
        ]
    )
    session.client = repo_client  # type: ignore[assignment]
    session.router_client = _SequentialRouterClient(
        routes=["repo"],
        execution_postures=["execute"],
        response_replies=["This non-repo response path should not be used."],
    )

    try:
        exit_code = session.run_turn("Use the stream_tool custom tool.")
    finally:
        session.close()

    tool_results = [
        event.get("payload", {}).get("result", {})
        for event in read_session_events(event_path)
        if event.get("type") == "tool_result"
        and event.get("payload", {}).get("name") == "stream_tool"
    ]
    assert exit_code == 0
    assert len(tool_results) == 1
    result = tool_results[0]
    assert result["success"] is True
    assert result["stdout_artifact_path"].startswith("session_artifacts/tool_logs/")
    assert result["stderr_artifact_path"].startswith("session_artifacts/tool_logs/")
    assert not (tmp_path / ".sylliptor" / "runs" / "T01" / "tool_logs").exists()
    stdout_artifacts = sorted((sessions_dir / "T01" / "tool_logs").glob("*.stdout.log"))
    stderr_artifacts = sorted((sessions_dir / "T01" / "tool_logs").glob("*.stderr.log"))
    assert len(stdout_artifacts) == 1
    assert len(stderr_artifacts) == 1
    assert stdout_artifacts[0].read_text(encoding="utf-8") == "CUSTOM_STDOUT_SENTINEL\n"
    assert stderr_artifacts[0].read_text(encoding="utf-8") == "CUSTOM_STDERR_SENTINEL\n"


def test_non_repo_web_tool_one_shot_does_not_require_material_repo_edits(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        model="qwen3.5-plus",
        base_url="https://coding-intl.dashscope.aliyuncs.com/v1",
        web_search_mode="auto",
        routing_mode="auto",
    )
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="readonly",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    session.client = _FailClient()  # type: ignore[assignment]
    router = _SequentialRouterClient(
        routes=["general"],
        response_replies=["", "Python 3.14.4 is the latest stable release."],
        response_tool_calls=[
            [
                ToolCall(
                    id="tc1",
                    name="web_search",
                    arguments={"query": "latest stable Python version"},
                )
            ],
            [],
        ],
    )
    session.router_client = router
    web_search_tool = session.tools["web_search"]
    session.tools["web_search"] = agent_loop_mod.ToolDef(
        name=web_search_tool.name,
        description=web_search_tool.description,
        parameters=web_search_tool.parameters,
        metadata=web_search_tool.metadata,
        run=lambda args: {
            "answer": "Python 3.14.4 is the latest stable release.",
            "sources": [{"title": "Python Downloads", "url": "https://www.python.org/"}],
            "backend": "test",
        },
    )
    session.tool_list = [tool.as_openai_tool() for tool in session.tools.values()]

    try:
        exit_code = session.run_turn(
            "Search the internet for the latest stable Python version. Answer briefly."
        )
    finally:
        session.close()

    event_types = [event.get("type") for event in read_session_events(event_path)]
    route_payload = _session_event_payload(event_path, "route_decision")

    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 2
    assert route_payload["route"] == "general"
    assert route_payload["route_override_reason"] is None
    assert route_payload["execution_posture"] == "advisory_non_execution"
    assert "tool_call" in event_types
    assert "one_shot_completion_gate_failed" not in event_types


@pytest.mark.parametrize(
    "instruction",
    [
        "This notes CLI is too limited for the way I actually work.",
        "Αυτο το notes CLI ειναι πολυ περιορισμενο για τον τροπο που δουλευω.",
        "ths notes cli is way too limted for how i actualy work",
    ],
)
def test_repo_backed_vague_first_turn_requests_use_router_context_for_repo_route(
    tmp_path: Path,
    instruction: str,
) -> None:
    _init_git_repo(tmp_path)
    cfg = AppConfig(model="test-model", routing_mode="auto")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    client = _ScriptedClient(
        [
            LLMResponse(
                content="I will inspect README.md first.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_read",
                        arguments={"path": "README.md"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="I inspected the repo grounding and can continue from there.",
                tool_calls=[],
                raw={},
            ),
        ]
    )
    session.client = client  # type: ignore[assignment]
    router = _RouterStubClient(
        route="repo",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn(instruction)
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    route_context = _route_context_payload(router.last_route_messages)

    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 0
    assert client.calls == 2
    assert route_context is not None
    assert route_context["workspace_kind"] == "git_repo"
    assert route_context["stable_grounding_available"] is True
    assert route_context["active_workspace_task"] is False
    assert route_payload["route"] == "repo"
    assert route_payload["execution_posture"] == "execute"
    assert route_payload["router_execution_posture"] == "execute"
    assert route_payload["route_selection_source"] == "router"
    assert route_payload["router_decision_source"] == "router"
    assert route_payload["route_override_reason"] is None
    assert route_payload["route_context"] == route_context
    assert any(
        msg.get("role") == "tool" and "README.md" in str(msg.get("content") or "")
        for msg in client.call_records[1]["messages"]
    )


def test_repo_backed_conceptual_git_question_stays_on_general_fast_path(
    tmp_path: Path,
) -> None:
    _init_git_repo(tmp_path)
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.5)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        response_reply="A git branch is a movable pointer to a commit.",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("What is a git branch in simple terms?")
    finally:
        session.close()

    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 1
    route_payload = _session_event_payload(event_path, "route_decision")
    route_context = _route_context_payload(router.last_route_messages)
    assert route_context is not None
    assert route_context["workspace_kind"] == "git_repo"
    assert route_context["stable_grounding_available"] is True
    assert route_payload["route"] == "general"
    assert route_payload["execution_posture"] == "advisory_non_execution"
    assert route_payload["router_execution_posture"] == "advisory_non_execution"
    assert route_payload["route_selection_source"] == "router"
    assert route_payload["route_override_reason"] is None


def test_vague_bugfix_request_stays_on_general_fast_path_outside_repo_session(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.5)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
    )
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        execution_posture="execute",
        response_reply="Please paste the markdown content you're working with.",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn(
            "this markdown formatter is being annoying. repeated section headers and blank bullets are not acceptable. can you fix it and leave the ordering sane?"
        )
    finally:
        session.close()

    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 1


def test_repo_backed_bugfix_request_respects_general_router_choice(
    tmp_path: Path,
) -> None:
    _init_git_repo(tmp_path)
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.5)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        execution_posture="execute",
        response_reply="Please paste the markdown content you're working with.",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn(
            "this markdown formatter is being annoying. repeated section headers and blank bullets are not acceptable. can you fix it and leave the ordering sane?"
        )
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 1
    assert route_payload["route"] == "general"
    assert route_payload["original_route"] == "general"
    assert route_payload["execution_posture"] == "execute"
    assert route_payload["router_execution_posture"] == "execute"
    assert route_payload["router_decision_source"] == "router"
    assert route_payload["route_selection_source"] == "router"
    assert route_payload["route_override_reason"] is None


def test_router_exception_fallback_keeps_vague_repo_improvement_on_repo_path(
    tmp_path: Path,
) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "README.md").write_text(
        "notes cli\n\nthis repo contains the local notes CLI implementation.\n",
        encoding="utf-8",
    )
    cfg = AppConfig(model="test-model", routing_mode="auto")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=5,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    client = _ScriptedClient(
        [
            LLMResponse(
                content="Χρειαζομαι λιγες ακομα λεπτομερειες πριν συνεχισω.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="I will inspect README.md first.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_read",
                        arguments={"path": "README.md"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="I inspected the repo first and can continue from there.",
                tool_calls=[],
                raw={},
            ),
        ]
    )
    session.client = client  # type: ignore[assignment]
    router = _RouterExceptionClient()
    session.router_client = router

    try:
        exit_code = session.run_turn(
            "Αυτο το notes CLI ειναι πολυ περιορισμενο για τον τροπο που δουλευω."
        )
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    retry_events = [
        event
        for event in read_session_events(event_path)
        if event.get("type") == "normal_chat_first_turn_repo_execute_retry"
    ]
    assert exit_code == 0
    assert router.route_calls == 1
    assert client.calls == 3
    assert route_payload["route"] == "repo"
    assert route_payload["original_route"] == "repo"
    assert route_payload["execution_posture"] == "execute"
    assert route_payload["router_execution_posture"] == "execute"
    assert route_payload["router_decision_source"] == "fallback_contextual"
    assert route_payload["route_selection_source"] == "fallback_contextual"
    assert route_payload["route_override_reason"] is None
    assert len(retry_events) == 1


def test_malformed_router_output_fallback_keeps_typo_heavy_repo_improvement_on_repo_path(
    tmp_path: Path,
) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "README.md").write_text(
        "notes cli\n\nthis repo contains the local notes CLI implementation.\n",
        encoding="utf-8",
    )
    cfg = AppConfig(model="test-model", routing_mode="auto")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=5,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    client = _ScriptedClient(
        [
            LLMResponse(
                content="I need a bit more detail before I continue.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="I will inspect README.md first.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_read",
                        arguments={"path": "README.md"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="I inspected the repo first and can continue from there.",
                tool_calls=[],
                raw={},
            ),
        ]
    )
    session.client = client  # type: ignore[assignment]
    router = _MalformedRouterClient()
    session.router_client = router

    try:
        exit_code = session.run_turn("ths notes cli is way too limted for how i actualy work")
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    retry_events = [
        event
        for event in read_session_events(event_path)
        if event.get("type") == "normal_chat_first_turn_repo_execute_retry"
    ]
    assert exit_code == 0
    assert router.route_calls == 1
    assert client.calls == 3
    assert route_payload["route"] == "repo"
    assert route_payload["original_route"] == "repo"
    assert route_payload["execution_posture"] == "execute"
    assert route_payload["router_execution_posture"] == "execute"
    assert route_payload["router_decision_source"] == "fallback_contextual"
    assert route_payload["route_selection_source"] == "fallback_contextual"
    assert route_payload["route_override_reason"] is None
    assert len(retry_events) == 1


def test_router_failure_fallback_preserves_general_fast_path_for_unrelated_question(
    tmp_path: Path,
) -> None:
    _init_git_repo(tmp_path)
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.5)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterExceptionClient()
    session.router_client = router

    try:
        exit_code = session.run_turn("What does a Python decorator do?")
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    assert exit_code == 0
    assert router.route_calls == 1
    assert route_payload["route"] == "general"
    assert route_payload["original_route"] == "general"
    assert route_payload["execution_posture"] == "advisory_non_execution"
    assert route_payload["router_execution_posture"] == "advisory_non_execution"
    assert route_payload["router_decision_source"] == "fallback"
    assert route_payload["route_selection_source"] == "fallback"
    assert route_payload["route_override_reason"] is None


def test_malformed_router_fallback_preserves_advisory_repo_question_posture(
    tmp_path: Path,
) -> None:
    _init_git_repo(tmp_path)
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.5)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    session.client = _FailClient()  # type: ignore[assignment]
    router = _MalformedRouterClient()
    session.router_client = router

    try:
        exit_code = session.run_turn("How does the current notes CLI work?")
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    assert exit_code == 0
    assert router.route_calls == 1
    assert route_payload["route"] == "general"
    assert route_payload["original_route"] == "general"
    assert route_payload["execution_posture"] == "advisory_non_execution"
    assert route_payload["router_execution_posture"] == "advisory_non_execution"
    assert route_payload["router_decision_source"] == "fallback"
    assert route_payload["route_selection_source"] == "fallback"
    assert route_payload["route_override_reason"] is None
    assert not any(
        event.get("type") == "normal_chat_first_turn_repo_execute_retry"
        for event in read_session_events(event_path)
    )


def test_router_failure_fallback_preserves_summary_follow_up_as_advisory_non_execution(
    tmp_path: Path,
) -> None:
    _init_git_repo(tmp_path)
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.5)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterExceptionClient()
    session.router_client = router

    try:
        exit_code = session.run_turn("summarize what changed")
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    assert exit_code == 0
    assert router.route_calls == 1
    assert route_payload["route"] == "general"
    assert route_payload["original_route"] == "general"
    assert route_payload["execution_posture"] == "advisory_non_execution"
    assert route_payload["router_execution_posture"] == "advisory_non_execution"
    assert route_payload["router_decision_source"] == "fallback"
    assert route_payload["route_selection_source"] == "fallback"
    assert route_payload["route_context"]["active_workspace_task"] is False
    assert route_payload["route_override_reason"] is None


@pytest.mark.parametrize(
    "follow_up_instruction",
    [
        "what did you change",
        "summarize what changed",
        "explain the fix",
    ],
)
def test_router_failure_active_repo_follow_up_stays_repo_advisory(
    tmp_path: Path,
    follow_up_instruction: str,
) -> None:
    _init_git_repo(tmp_path)
    cfg = AppConfig(model="test-model", routing_mode="auto")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    client = _RepoCaptureClient("I updated src/app.py and reran targeted pytest.")
    session.client = client  # type: ignore[assignment]
    router = _RouterExceptionClient()
    session.router_client = router
    session.workspace_touched_paths = {"src/app.py"}

    try:
        exit_code = session.run_turn(follow_up_instruction)
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    assert exit_code == 0
    assert router.route_calls == 1
    assert client.calls == 1
    assert route_payload["route"] == "repo"
    assert route_payload["original_route"] == "repo"
    assert route_payload["execution_posture"] == "advisory_non_execution"
    assert route_payload["router_execution_posture"] == "advisory_non_execution"
    assert route_payload["router_decision_source"] == "fallback_contextual"
    assert route_payload["route_selection_source"] == "fallback_contextual"
    assert route_payload["route_context"]["active_workspace_task"] is True
    assert route_payload["route_override_reason"] is None
    assert not any(
        event.get("type") == "normal_chat_first_turn_repo_execute_retry"
        for event in read_session_events(event_path)
    )


def test_router_failure_active_repo_context_unrelated_question_stays_general(
    tmp_path: Path,
) -> None:
    _init_git_repo(tmp_path)
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.5)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterExceptionClient()
    session.router_client = router
    session.workspace_touched_paths = {"src/app.py"}

    try:
        exit_code = session.run_turn("How do Python list comprehensions work?")
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    assert exit_code == 0
    assert router.route_calls == 1
    assert route_payload["route"] == "general"
    assert route_payload["original_route"] == "general"
    assert route_payload["execution_posture"] == route_payload["router_execution_posture"]
    assert route_payload["router_decision_source"] == "fallback"
    assert route_payload["route_selection_source"] == "fallback"
    assert route_payload["route_context"]["active_workspace_task"] is True
    assert route_payload["route_override_reason"] is None


def test_router_failure_explicit_repo_request_uses_plain_fallback_source(
    tmp_path: Path,
) -> None:
    _init_git_repo(tmp_path)
    cfg = AppConfig(model="test-model", routing_mode="auto")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    session.client = _ScriptedClient(  # type: ignore[assignment]
        [
            LLMResponse(
                content="I will inspect README.md first.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_read",
                        arguments={"path": "README.md"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="I inspected the repo and can continue from there.",
                tool_calls=[],
                raw={},
            ),
        ]
    )
    router = _RouterExceptionClient()
    session.router_client = router

    try:
        exit_code = session.run_turn("Please inspect this repo and fix the notes CLI flow.")
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    assert exit_code == 0
    assert router.route_calls == 1
    assert route_payload["route"] == "repo"
    assert route_payload["original_route"] == "repo"
    assert route_payload["execution_posture"] == "execute"
    assert route_payload["router_execution_posture"] == "execute"
    assert route_payload["router_decision_source"] == "fallback"
    assert route_payload["route_selection_source"] == "fallback"
    assert route_payload["route_override_reason"] is None


def test_router_exception_fallback_keeps_vague_repo_improvement_on_repo_path_without_repo_scan(
    tmp_path: Path,
) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "README.md").write_text(
        "notes cli\n\nthis repo contains the local notes CLI implementation.\n",
        encoding="utf-8",
    )
    cfg = AppConfig(
        model="test-model",
        routing_mode="auto",
        verify_commands=["pytest tests/test_prompt_payload.py -q"],
    )
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=5,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    client = _ScriptedClient(
        [
            LLMResponse(
                content="I need a bit more detail before I continue.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="I will inspect README.md first.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_read",
                        arguments={"path": "README.md"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="I inspected the repo first and can continue from there.",
                tool_calls=[],
                raw={},
            ),
        ]
    )
    session.client = client  # type: ignore[assignment]
    router = _RouterExceptionClient()
    session.router_client = router

    try:
        exit_code = session.run_turn("ths notes cli is way too limted for how i actualy work")
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    retry_events = [
        event
        for event in read_session_events(event_path)
        if event.get("type") == "normal_chat_first_turn_repo_execute_retry"
    ]
    assert exit_code == 0
    assert router.route_calls == 1
    assert client.calls == 3
    assert route_payload["route"] == "repo"
    assert route_payload["original_route"] == "repo"
    assert route_payload["router_decision_source"] == "fallback_contextual"
    assert route_payload["route_selection_source"] == "fallback_contextual"
    assert route_payload["route_context"]["grounding_source"] == "top_level"
    assert route_payload["route_context"]["workspace_hint"] == "notes cli"
    assert len(retry_events) == 1


def test_router_failure_generic_package_hint_does_not_force_repo_route(
    tmp_path: Path,
) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools>=61"]\n',
        encoding="utf-8",
    )
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.5)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterExceptionClient()
    session.router_client = router

    try:
        exit_code = session.run_turn("Write a Python decorator example.")
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    assert exit_code == 0
    assert router.route_calls == 1
    assert route_payload["route"] == "general"
    assert route_payload["original_route"] == "general"
    assert route_payload["router_decision_source"] == "fallback"
    assert route_payload["route_selection_source"] == "fallback"
    assert route_payload["route_context"]["workspace_hint"] == ""
    assert route_payload["route_override_reason"] is None


def test_repo_backed_vague_repo_route_survives_missing_router_execution_posture(
    tmp_path: Path,
) -> None:
    _init_git_repo(tmp_path)
    cfg = AppConfig(model="test-model", routing_mode="auto")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    client = _ScriptedClient(
        [
            LLMResponse(
                content="I will inspect README.md first.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_read",
                        arguments={"path": "README.md"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="I inspected the repo grounding and can continue from there.",
                tool_calls=[],
                raw={},
            ),
        ]
    )
    session.client = client  # type: ignore[assignment]
    router = _RouterStubClient(
        route="repo",
        omit_execution_posture=True,
    )
    session.router_client = router

    try:
        exit_code = session.run_turn(
            "Αυτο το notes CLI ειναι πολυ περιορισμενο για τον τροπο που δουλευω."
        )
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 0
    assert client.calls == 2
    assert route_payload["route"] == "repo"
    assert route_payload["original_route"] == "repo"
    assert route_payload["execution_posture"] == "execute"
    assert route_payload["execution_posture_source"] == "fallback"
    assert route_payload["router_execution_posture"] == "execute"
    assert route_payload["router_execution_posture_source"] == "fallback"
    assert route_payload["router_decision_source"] == "router"
    assert route_payload["route_selection_source"] == "router"
    assert route_payload["route_override_reason"] is None


@pytest.mark.parametrize(
    ("execution_posture", "omit_execution_posture", "session_suffix"),
    [
        ("", True, "omit-execution-posture"),
        ("totally-invalid-posture", False, "malformed-execution-posture"),
    ],
)
def test_repo_summary_follow_up_preserves_advisory_execution_posture_when_router_posture_is_unusable(
    tmp_path: Path,
    execution_posture: str,
    omit_execution_posture: bool,
    session_suffix: str,
) -> None:
    _init_git_repo(tmp_path)
    cfg = AppConfig(model="test-model", routing_mode="auto")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
        session_id_override=f"summary-follow-up-{session_suffix}",
    )
    event_path = session.store.path
    client = _RepoCaptureClient("I updated src/app.py and reran targeted pytest.")
    session.client = client  # type: ignore[assignment]
    router = _RouterStubClient(
        route="repo",
        execution_posture=execution_posture or None,
        omit_execution_posture=omit_execution_posture,
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("what did you change")
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 0
    assert client.calls == 1
    assert route_payload["route"] == "repo"
    assert route_payload["original_route"] == "repo"
    assert route_payload["execution_posture"] == "advisory_non_execution"
    assert route_payload["execution_posture_source"] == "fallback"
    assert route_payload["router_execution_posture"] == "advisory_non_execution"
    assert route_payload["router_execution_posture_source"] == "fallback"
    assert route_payload["router_decision_source"] == "router"
    assert route_payload["route_selection_source"] == "router"
    assert route_payload["route_override_reason"] is None
    assert not any(
        event.get("type") == "normal_chat_first_turn_repo_execute_retry"
        for event in read_session_events(event_path)
    )


def test_first_repo_turn_no_tool_finalization_is_retried_once_with_grounding(
    tmp_path: Path,
) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "README.md").write_text(
        "notes cli\n\nbuild a local release notes helper in this repo.\n",
        encoding="utf-8",
    )
    cfg = AppConfig(model="test-model", routing_mode="auto")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=5,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    client = _ScriptedClient(
        [
            LLMResponse(
                content="Χρειαζομαι λιγες ακομα λεπτομερειες πριν συνεχισω.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="I will inspect README.md first.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_read",
                        arguments={"path": "README.md"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="I inspected README.md and can build from that repo context.",
                tool_calls=[],
                raw={},
            ),
        ]
    )
    session.client = client  # type: ignore[assignment]
    router = _RouterStubClient(
        route="repo",
        response_reply="",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn(
            "I want you to build this little tool from scratch in the current repo."
        )
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 0
    assert client.calls == 3
    assert route_payload["route"] == "repo"
    assert route_payload["execution_posture"] == "execute"
    assert route_payload["router_execution_posture"] == "execute"
    retry_payload = _session_event_payload(event_path, "normal_chat_first_turn_repo_execute_retry")
    retry_messages = client.call_records[1]["messages"]
    assert any(
        msg.get("role") == "system"
        and "Repo-backed normal chat safeguard" in str(msg.get("content") or "")
        for msg in retry_messages
    )
    assert retry_payload["had_tool_calls"] is False
    assert retry_payload["repo_tool_activity_observed"] is False
    assert retry_payload["workspace_grounding"]["stable_grounding_available"] is True
    assert any(
        msg.get("role") == "tool"
        and "build a local release notes helper" in str(msg.get("content") or "")
        for msg in client.call_records[2]["messages"]
    )


@pytest.mark.parametrize(
    "instruction",
    [
        "Αυτο το notes CLI ειναι πολυ περιορισμενο για τον τροπο που δουλευω.",
        "ths notes cli is way too limted for how i actualy work",
    ],
)
def test_first_repo_turn_vague_improvement_no_tool_finalization_retries_once_with_router_posture(
    tmp_path: Path,
    instruction: str,
) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "README.md").write_text(
        "notes cli\n\nthis repo contains the local notes CLI implementation.\n",
        encoding="utf-8",
    )
    cfg = AppConfig(model="test-model", routing_mode="auto")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=5,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    client = _ScriptedClient(
        [
            LLMResponse(
                content="Χρειαζομαι λιγες ακομα λεπτομερειες πριν συνεχισω.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="I will inspect README.md first.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_read",
                        arguments={"path": "README.md"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="I inspected the repo first and can continue from there.",
                tool_calls=[],
                raw={},
            ),
        ]
    )
    session.client = client  # type: ignore[assignment]
    router = _RouterStubClient(route="repo", execution_posture="execute", response_reply="")
    session.router_client = router

    try:
        exit_code = session.run_turn(instruction)
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    retry_events = [
        event
        for event in read_session_events(event_path)
        if event.get("type") == "normal_chat_first_turn_repo_execute_retry"
    ]
    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 0
    assert client.calls == 3
    assert route_payload["route"] == "repo"
    assert route_payload["execution_posture"] == "execute"
    assert route_payload["router_execution_posture"] == "execute"
    assert len(retry_events) == 1
    retry_messages = client.call_records[1]["messages"]
    assert any(
        msg.get("role") == "system"
        and "Repo-backed normal chat safeguard" in str(msg.get("content") or "")
        for msg in retry_messages
    )
    assert any(msg.get("role") == "tool" for msg in client.call_records[2]["messages"])


def test_first_repo_turn_no_tool_finalization_without_stable_grounding_does_not_retry(
    tmp_path: Path,
) -> None:
    _init_empty_git_repo(tmp_path)
    cfg = AppConfig(model="test-model", routing_mode="auto")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
    )
    client = _ScriptedClient(
        [
            LLMResponse(
                content="Need more specifics before I continue.",
                tool_calls=[],
                raw={},
            )
        ]
    )
    session.client = client  # type: ignore[assignment]
    router = _RouterStubClient(route="repo", response_reply="")
    session.router_client = router

    try:
        exit_code = session.run_turn(
            "I want you to build this little tool from scratch in the current repo."
        )
    finally:
        session.close()

    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 0
    assert client.calls == 1
    assert not any(
        msg.get("role") == "system"
        and "Repo-backed normal chat safeguard" in str(msg.get("content") or "")
        for msg in session.messages
    )


@pytest.mark.parametrize(
    "instruction",
    [
        "How does the current notes CLI work?",
        "πώς δουλεύει το notes CLI;",
    ],
)
def test_first_repo_advisory_question_can_finish_without_forced_repo_inspection(
    tmp_path: Path,
    instruction: str,
) -> None:
    _init_git_repo(tmp_path)
    cfg = AppConfig(model="test-model", routing_mode="auto")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    client = _ScriptedClient(
        [
            LLMResponse(
                content="The current notes CLI reads the local repo config and formats terminal output.",
                tool_calls=[],
                raw={},
            )
        ]
    )
    session.client = client  # type: ignore[assignment]
    router = _RouterStubClient(
        route="repo",
        execution_posture="advisory_non_execution",
        response_reply="",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn(instruction)
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 0
    assert client.calls == 1
    assert route_payload["route"] == "repo"
    assert route_payload["execution_posture"] == "advisory_non_execution"
    assert route_payload["router_execution_posture"] == "advisory_non_execution"
    assert not any(
        event.get("type") == "normal_chat_first_turn_repo_execute_retry"
        for event in read_session_events(event_path)
    )
    assert not any(
        msg.get("role") == "system"
        and "Repo-backed normal chat safeguard" in str(msg.get("content") or "")
        for msg in session.messages
    )


def test_explicit_local_build_request_overrides_general_route_in_plain_dir_session(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="auto")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
    )
    repo_client = _RepoCaptureClient(reply="I will create the local script in this folder.")
    session.client = repo_client  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        response_reply="Please share the script you want me to write.",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn(
            "build me a small python script that cleans a csv. i'm just using a plain folder, not a repo."
        )
    finally:
        session.close()

    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 0
    assert repo_client.calls == 1


def test_plain_dir_follow_up_refinement_stays_on_workspace_path_after_write(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="auto")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
    )
    client = _ScriptedClient(
        [
            LLMResponse(
                content="I will create clean_csv.py.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={
                            "path": "clean_csv.py",
                            "content": "print('clean')\n",
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(content="Created clean_csv.py.", tool_calls=[], raw={}),
            LLMResponse(
                content="I will also add a markdown summary file in this folder.",
                tool_calls=[],
                raw={},
            ),
        ]
    )
    session.client = client  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        response_reply="Please share the current code first.",
    )
    session.router_client = router

    try:
        first_exit = session.run_turn("create clean_csv.py here")
        second_exit = session.run_turn("change of mind: also write a markdown summary file")
    finally:
        session.close()

    assert first_exit == 0
    assert second_exit == 0
    assert router.route_calls == 2
    assert router.response_calls == 0
    assert client.calls == 3
    assert any(
        msg.get("role") == "tool" and "clean_csv.py" in str(msg.get("content") or "")
        for msg in client.call_records[-1]["messages"]
    )


def test_plain_dir_explanatory_follow_up_keeps_workspace_continuity_after_write(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="auto")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
    )
    client = _ScriptedClient(
        [
            LLMResponse(
                content="I will create clean_csv.py.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={
                            "path": "clean_csv.py",
                            "content": "print('clean')\n",
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(content="Created clean_csv.py.", tool_calls=[], raw={}),
            LLMResponse(
                content="The current code is in `clean_csv.py` and prints `clean`.",
                tool_calls=[],
                raw={},
            ),
        ]
    )
    session.client = client  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        response_reply="Please share the code first.",
    )
    session.router_client = router

    try:
        first_exit = session.run_turn("create clean_csv.py here")
        second_exit = session.run_turn("Mind sharing the current code?")
    finally:
        session.close()

    assert first_exit == 0
    assert second_exit == 0
    assert router.route_calls == 2
    assert router.response_calls == 0
    assert client.calls == 3
    assert any(
        msg.get("role") == "tool" and "clean_csv.py" in str(msg.get("content") or "")
        for msg in client.call_records[-1]["messages"]
    )


def test_related_anchored_explanatory_follow_up_stays_workspace_aware_in_plain_dir(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="auto")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
    )
    client = _ScriptedClient(
        [
            LLMResponse(
                content="I will create timer.py.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={
                            "path": "timer.py",
                            "content": "print('tick')\n",
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(content="Created timer.py.", tool_calls=[], raw={}),
            LLMResponse(content="`timer.py` currently prints `tick`.", tool_calls=[], raw={}),
        ]
    )
    session.client = client  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        response_reply="Please share the file first.",
    )
    session.router_client = router

    try:
        first_exit = session.run_turn("create timer.py here")
        second_exit = session.run_turn("How does timer.py work?")
    finally:
        session.close()

    assert first_exit == 0
    assert second_exit == 0
    assert router.route_calls == 2
    assert router.response_calls == 0
    assert client.calls == 3
    assert any(
        msg.get("role") == "tool" and "timer.py" in str(msg.get("content") or "")
        for msg in client.call_records[-1]["messages"]
    )


def test_repo_follow_up_continuity_override_keeps_ambiguous_follow_up_on_repo_path(
    tmp_path: Path,
) -> None:
    _init_git_repo(tmp_path)
    cfg = AppConfig(model="test-model", routing_mode="auto")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
    )
    client = _ScriptedClient(
        [
            LLMResponse(
                content="I will create src/widget.py.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={
                            "path": "src/widget.py",
                            "content": "def load_widget_config():\n    return {'mode': 'safe'}\n",
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(content="Created src/widget.py.", tool_calls=[], raw={}),
            LLMResponse(
                content="`src/widget.py` loads the widget config and returns a safe default.",
                tool_calls=[],
                raw={},
            ),
        ]
    )
    session.client = client  # type: ignore[assignment]
    router = _SequentialRouterClient(
        routes=["repo", "general"],
        response_replies=["I don't know which file you mean."],
    )
    session.router_client = router

    try:
        first_exit = session.run_turn("Create src/widget.py with a tiny config loader.")
        second_exit = session.run_turn("How does it work?")
    finally:
        session.close()

    assert first_exit == 0
    assert second_exit == 0
    assert router.route_calls == 2
    assert router.response_calls == 0
    assert client.calls == 3
    assert any(
        msg.get("role") == "tool" and "src/widget.py" in str(msg.get("content") or "")
        for msg in client.call_records[-1]["messages"]
    )


def test_unrelated_anchored_explanatory_question_uses_non_repo_fast_path_after_plain_dir_task(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.5)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
    )
    client = _ScriptedClient(
        [
            LLMResponse(
                content="I will create timer.py.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={
                            "path": "timer.py",
                            "content": "print('tick')\n",
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(content="Created timer.py.", tool_calls=[], raw={}),
        ]
    )
    session.client = client  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        response_reply="`asyncio.gather` runs awaitables concurrently.",
    )
    session.router_client = router

    try:
        first_exit = session.run_turn("create timer.py here")
        session.client = _FailClient()  # type: ignore[assignment]
        second_exit = session.run_turn("Can you explain `asyncio.gather` more?")
        final_message = str(session.messages[-1].get("content") or "")
    finally:
        session.close()

    assert first_exit == 0
    assert second_exit == 0
    assert router.route_calls == 2
    assert router.response_calls == 1
    assert client.calls == 2
    assert final_message == "`asyncio.gather` runs awaitables concurrently."


def test_unrelated_general_question_still_uses_non_repo_fast_path_after_repo_activity(
    tmp_path: Path,
) -> None:
    _init_git_repo(tmp_path)
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.5)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
    )
    client = _ScriptedClient(
        [
            LLMResponse(
                content="I will create src/widget.py.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={
                            "path": "src/widget.py",
                            "content": "def load_widget_config():\n    return {'mode': 'safe'}\n",
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(content="Created src/widget.py.", tool_calls=[], raw={}),
        ]
    )
    session.client = client  # type: ignore[assignment]
    router = _SequentialRouterClient(
        routes=["repo", "general"],
        response_replies=["Recursion is a function that calls itself."],
    )
    session.router_client = router

    try:
        first_exit = session.run_turn("Create src/widget.py with a tiny config loader.")
        session.client = _FailClient()  # type: ignore[assignment]
        second_exit = session.run_turn("Explain recursion in Python in two lines.")
        final_message = str(session.messages[-1].get("content") or "")
    finally:
        session.close()

    assert first_exit == 0
    assert second_exit == 0
    assert router.route_calls == 2
    assert router.response_calls == 1
    assert client.calls == 2
    assert final_message == "Recursion is a function that calls itself."


def test_unrelated_general_question_still_uses_non_repo_fast_path_after_plain_dir_task(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.5)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
    )
    client = _ScriptedClient(
        [
            LLMResponse(
                content="I will create timer.py.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={
                            "path": "timer.py",
                            "content": "print('tick')\n",
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(content="Created timer.py.", tool_calls=[], raw={}),
        ]
    )
    session.client = client  # type: ignore[assignment]
    router = _SequentialRouterClient(
        routes=["repo", "general"],
        response_replies=["Recursion is a function that calls itself."],
    )
    session.router_client = router

    try:
        first_exit = session.run_turn("create timer.py here")
        session.client = _FailClient()  # type: ignore[assignment]
        second_exit = session.run_turn("Explain recursion in Python in two lines.")
        final_message = str(session.messages[-1].get("content") or "")
    finally:
        session.close()

    assert first_exit == 0
    assert second_exit == 0
    assert router.route_calls == 2
    assert router.response_calls == 1
    assert client.calls == 2
    assert final_message == "Recursion is a function that calls itself."


def test_non_repo_fast_path_uses_small_visible_history_only(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.5)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
    )
    long_request = (
        "Create src/widget.py and keep the behavior stable. "
        + "Please preserve the interface and document the helper. " * 20
    )
    long_reply = (
        "Created src/widget.py with a small helper and kept the interface stable. "
        + "The file now loads a safe default config and documents the behavior. " * 20
    )
    client = _ScriptedClient(
        [
            LLMResponse(
                content="I will create src/widget.py.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={
                            "path": "src/widget.py",
                            "content": "def load_widget_config():\n    return {'mode': 'safe'}\n",
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(content=long_reply, tool_calls=[], raw={}),
        ]
    )
    session.client = client  # type: ignore[assignment]
    router = _SequentialRouterClient(
        routes=["repo", "general"],
        response_replies=["Recursion is a function that calls itself."],
    )
    session.router_client = router

    try:
        first_exit = session.run_turn(long_request)
        session.client = _FailClient()  # type: ignore[assignment]
        second_exit = session.run_turn("Explain recursion in Python in two lines.")
        final_message = str(session.messages[-1].get("content") or "")
    finally:
        session.close()

    assert first_exit == 0
    assert second_exit == 0
    assert final_message == "Recursion is a function that calls itself."
    assert router.response_calls == 1

    non_system_history = [
        msg for msg in router.last_messages[:-1] if msg.get("role") in {"user", "assistant"}
    ]
    assert 1 <= len(non_system_history) <= 2
    assert [msg.get("role") for msg in non_system_history] in (["user"], ["user", "assistant"])
    assert all(msg.get("role") != "tool" for msg in router.last_messages)
    assert not any(msg.get("tool_calls") for msg in router.last_messages if isinstance(msg, dict))
    assert not any(
        str(msg.get("content") or "").startswith(
            ("<task_brief>", "<environment_context>", "Repo summary")
        )
        for msg in router.last_messages
        if isinstance(msg, dict)
    )
    assert all(
        len(str(msg.get("content") or ""))
        <= agent_loop_mod._NON_REPO_MAX_RECENT_VISIBLE_HISTORY_CHARS
        for msg in non_system_history
    )
    assert (
        sum(len(str(msg.get("content") or "")) for msg in non_system_history)
        <= agent_loop_mod._NON_REPO_MAX_RECENT_VISIBLE_HISTORY_TOTAL_CHARS
    )
    assert estimate_message_tokens(non_system_history) < 1200


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        (
            "this markdown formatter is being annoying. repeated section headers and blank bullets are not acceptable. can you fix it and leave the ordering sane?",
            True,
        ),
        ("Can you fix the parser bug?", True),
        ("How are you?", False),
        ("Explain recursion in Python in two lines.", False),
        ("Explain the bug, no code changes.", False),
        ("What is the right fix for this parser bug?", False),
        ("This sentence is wrong, can you correct it?", False),
        ("It's wrong, fix it.", False),
    ],
)
def test_implicit_repo_bugfix_request_heuristic(instruction: str, expected: bool) -> None:
    assert agent_loop_mod._looks_like_implicit_repo_bugfix_request(instruction) is expected


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("This notes CLI is too limited for the way I actually work.", True),
        ("Our release notes CLI doesn't support prereleases yet.", True),
        ("CLI tools are too limited in general.", False),
        ("Explain why the notes CLI behaves this way.", False),
    ],
)
def test_implicit_repo_improvement_request_heuristic(instruction: str, expected: bool) -> None:
    assert looks_like_implicit_repo_improvement_request(instruction) is expected


def test_non_repo_response_uses_model_selected_language_directive(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.5)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
    )
    session.client = _FailClient()  # type: ignore[assignment]
    instruction = "geia sou, ti kaneis?"
    router = _RouterStubClient(
        route="chat",
        route_reply="",
        response_reply="I am well, thanks.",
        language="Greek",
        script="Greek",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn(instruction)
    finally:
        session.close()

    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 1
    assert router.last_messages[-1]["role"] == "user"
    assert router.last_messages[-1]["content"] == instruction
    assert "Classification:" not in str(router.last_messages[-1]["content"])
    assert not any(
        msg.get("role") == "system"
        and "explicitly requested a language/script override" in str(msg.get("content") or "")
        for msg in router.last_messages
    )
    assert any(
        msg.get("role") == "system"
        and "selected reply language/script for this turn is model-determined"
        in str(msg.get("content") or "")
        for msg in router.last_messages
    )
    assert any(
        msg.get("role") == "system"
        and str(msg.get("content") or "").startswith("Turn classification:")
        for msg in router.last_messages
    )


def test_code_only_mode_keeps_legacy_non_repo_hint_behavior(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=2,
        no_log=True,
        api_key_override="override-key",
    )
    capture = _RepoCaptureClient(reply="Recursion is a function calling itself.")
    session.client = capture  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Explain recursion in Python in two lines.")
    finally:
        session.close()

    assert exit_code == 0
    assert any(
        msg.get("role") == "system"
        and msg.get("content") == agent_loop_mod._NON_REPO_TURN_SYSTEM_HINT
        for msg in capture.last_messages
    )


def test_code_only_mode_greeting_uses_main_agent_client_without_canned_shortcut(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=2,
        no_log=True,
        api_key_override="override-key",
    )
    reply = "Custom greeting from model."
    capture = _RepoCaptureClient(reply=reply)
    session.client = capture  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("\u03b3\u03b5\u03b9\u03b1")
    finally:
        session.close()

    assert exit_code == 0
    assert capture.calls == 1
    assert any(
        msg.get("role") == "system"
        and msg.get("content") == agent_loop_mod._NON_REPO_TURN_SYSTEM_HINT
        for msg in capture.last_messages
    )


def test_gibberish_non_repo_turn_defaults_to_english_fallback(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.5)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
    )
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        route_reply="Это запасной ответ.",
        response_reply="",
        language="English",
        script="Latin",
        explicit_language_override=False,
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("dsd")
    finally:
        session.close()

    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 1
    assert session.messages[-1]["role"] == "assistant"
    assert session.messages[-1]["content"] == "Could you clarify what you want me to help with?"


def test_code_only_mode_does_not_inject_non_english_without_explicit_override(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=2,
        no_log=True,
        api_key_override="override-key",
    )
    capture = _RepoCaptureClient(reply="Understood.")
    session.client = capture  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("ti kaneis?")
    finally:
        session.close()

    assert exit_code == 0
    assert not any(
        msg.get("role") == "system"
        and "explicitly requested a language/script override" in str(msg.get("content") or "")
        for msg in capture.last_messages
    )


def test_repo_turn_honors_explicit_script_override(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="auto")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
    )
    repo_client = _RepoCaptureClient(reply="I will inspect src/app.py first.")
    session.client = repo_client  # type: ignore[assignment]
    router = _RouterStubClient(
        route="repo",
        language="English",
        script="Greek",
        explicit_language_override=True,
    )
    session.router_client = router

    try:
        exit_code = session.run_turn(
            "Use Greek script for your response, then open src/app.py and find why tests fail."
        )
    finally:
        session.close()

    assert exit_code == 0
    directive = agent_loop_mod._build_turn_language_system_message(
        "English",
        "Greek",
        explicit_language_override=True,
    )
    assert directive is not None
    assert any(
        msg.get("role") == "system" and msg.get("content") == directive
        for msg in repo_client.last_messages
    )


def test_code_only_mode_leaves_explicit_language_choice_to_main_agent(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="code_only")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=2,
        no_log=True,
        api_key_override="override-key",
    )
    capture = _RepoCaptureClient(reply="Εντάξει.")
    session.client = capture  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Reply in Greek: explain recursion in one line.")
    finally:
        session.close()

    assert exit_code == 0
    assert capture.calls == 1
    assert not any(
        msg.get("role") == "system"
        and (
            "explicitly requested a language/script override" in str(msg.get("content") or "")
            or "selected reply language/script for this turn is model-determined"
            in str(msg.get("content") or "")
        )
        for msg in capture.last_messages
    )


def test_non_repo_turn_honors_explicit_language_override(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.5)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
    )
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterStubClient(
        route="chat",
        language="Greek",
        script="Greek",
        explicit_language_override=True,
        response_reply="Είμαι καλά, ευχαριστώ.",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("Please reply in Greek. How are you?")
    finally:
        session.close()

    assert exit_code == 0
    directive = agent_loop_mod._build_turn_language_system_message(
        "Greek",
        "Greek",
        explicit_language_override=True,
    )
    assert directive is not None
    assert any(
        msg.get("role") == "system" and msg.get("content") == directive
        for msg in router.last_messages
    )


def test_create_session_uses_role_temperatures_for_clients(tmp_path: Path) -> None:
    cfg = AppConfig(
        model="test-model",
        routing_mode="auto",
        coding_temperature=0.23,
        compactor_temperature=0.31,
    )
    cfg.extra_fields = {
        "role_models": {
            "router": "router-model",
        },
        "compaction": {
            "enabled": True,
            "summarize_conversation": True,
        },
    }
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=2,
        no_log=False,
        session_log_dir_override=tmp_path / "sessions",
        api_key_override="override-key",
    )

    try:
        assert session.client.temperature == 0.23
        assert session.client.model == "test-model"
        assert session.router_client is not None
        assert session.router_client.model == "router-model"
        assert session.router_client.temperature == 0.0
        session_start_payload = _session_event_payload(session.store.path, "session_start")
        assert session_start_payload["router_model"] == "router-model"
        assert session.conversation_compactor is not None
        assert session.conversation_compactor.compactor_client.temperature == 0.31
    finally:
        session.close()
