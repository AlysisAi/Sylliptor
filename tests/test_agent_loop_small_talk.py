from __future__ import annotations

import copy
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from sylliptor_agent_cli import agent_loop as agent_loop_mod
from sylliptor_agent_cli.agent.routing import _is_fatal_non_repo_llm_error
from sylliptor_agent_cli.agent_loop import create_session
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.custom_tools.discovery import discover_custom_tools
from sylliptor_agent_cli.custom_tools.trust import trust_project_tool
from sylliptor_agent_cli.llm.metadata import (
    GEMINI_GENERATE_CONTENT_PROVIDER_METADATA_KEY,
    PROVIDER_METADATA_KEY,
    endpoint_descriptor,
)
from sylliptor_agent_cli.llm.openai_compat import LLMError, LLMResponse, ToolCall
from sylliptor_agent_cli.llm.protocols import ReasoningTraceCapability
from sylliptor_agent_cli.llm.types import ReasoningOutputKind
from sylliptor_agent_cli.llm_error_display import friendly_llm_error_message
from sylliptor_agent_cli.request_estimation import (
    estimate_message_tokens,
    estimate_request_token_breakdown,
)
from sylliptor_agent_cli.runtime_kind import RuntimeKind
from sylliptor_agent_cli.session_store import read_session_events
from sylliptor_agent_cli.surface.noop_surface import NoopSurface
from sylliptor_agent_cli.tools.availability import WEB_UNAVAILABLE_OBSERVATION
from sylliptor_agent_cli.tools.web_search import WebSearchError
from sylliptor_agent_cli.turn_intent import looks_like_implicit_repo_improvement_request


@pytest.fixture(autouse=True)
def _clear_generic_web_search_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_API_KEY", raising=False)


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
    reasoning_trace_capability = ReasoningTraceCapability(
        adapter="test_safe_summary",
        output_kind=ReasoningOutputKind.SUMMARY,
        supports_streaming=True,
        supports_buffered=True,
        requestable=True,
    )

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
        response_text_deltas: list[str] | None = None,
        response_reasoning_deltas: list[str] | None = None,
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
        self.response_text_deltas = list(response_text_deltas or [])
        self.response_reasoning_deltas = list(response_reasoning_deltas or [])
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
        on_reasoning_delta=None,  # type: ignore[no-untyped-def]
        temperature: float | None = None,
    ) -> LLMResponse:
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
        if stream and on_reasoning_delta is not None:
            for delta in self.response_reasoning_deltas:
                on_reasoning_delta(delta)
        if stream and on_text_delta is not None:
            for delta in self.response_text_deltas:
                on_text_delta(delta)
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


def _trial_quota_exhausted_error() -> LLMError:
    return LLMError(
        'LLM error 402: {"error":{"message":"Free trial tokens used up.",'
        '"type":"insufficient_quota","code":"quota_exhausted"}}'
    )


def _private_provider_url_error() -> LLMError:
    return LLMError(
        "LLM error 401: provider failed at "
        "https://route-user:route-pa'ssword@api.example.test/"
        "secret-route-segment<PRIVATE_BOUNDARY_SENTINEL"
        "?token=PRIVATE_BOUNDARY_SENTINEL#PRIVATE_BOUNDARY_SENTINEL "
        "api_key='PRIVATE_API_KEY_SENTINEL_123456' "
        "Authorization: Bearer PRIVATE_BEARER_SENTINEL_123456+/="
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
        error_factory: Any = _invalid_api_key_error,
    ) -> None:
        self.route = route
        self.route_error = route_error
        self.response_error = response_error
        self.response_reply = response_reply
        self.error_factory = error_factory
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
                raise self.error_factory()
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
            raise self.error_factory()
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
        tool_choice: Any | None = None,
        stream: bool = False,
        on_text_delta=None,  # type: ignore[no-untyped-def]
        temperature: float | None = None,
    ) -> LLMResponse:
        _ = on_text_delta, temperature, tool_choice
        self.call_records.append(
            {
                "messages": list(messages),
                "tools": tools,
                "tool_choice": tool_choice,
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
    cfg = AppConfig(
        model="test-model",
        routing_mode="auto",
        chat_temperature=0.73,
        stream=False,
    )
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
    cfg = AppConfig(
        model="test-model",
        routing_mode="auto",
        chat_temperature=0.73,
        stream=False,
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


def test_streaming_non_repo_turn_bypasses_buffered_router_reply(tmp_path: Path) -> None:
    class _CaptureSurface(NoopSurface):
        def __init__(self) -> None:
            self.text_deltas: list[str] = []
            self.reasoning_deltas: list[str] = []
            self.reasoning_starts: list[str] = []
            self.reasoning_ends: list[str] = []

        def on_assistant_token(self, delta: str) -> None:
            self.text_deltas.append(delta)

        def on_reasoning_token(self, delta: str) -> None:
            self.reasoning_deltas.append(delta)

        def on_reasoning_start(self, block_id: str) -> None:
            self.reasoning_starts.append(block_id)

        def on_reasoning_end(self, block_id: str) -> None:
            self.reasoning_ends.append(block_id)

    surface = _CaptureSurface()
    cfg = AppConfig(model="test-model", routing_mode="auto", stream=True)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
        surface=surface,
    )
    event_path = session.store.path
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterStubClient(
        route="chat",
        route_reply="Buffered router reply.",
        response_reply="Live response.",
        response_reasoning_deltas=["Check ", "the request."],
        response_text_deltas=["Live ", "response."],
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("Hi")
    finally:
        session.close()

    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 1
    assert surface.reasoning_deltas == ["Check ", "the request."]
    assert len(surface.reasoning_starts) == 1
    assert surface.reasoning_ends == surface.reasoning_starts
    assert surface.text_deltas == ["Live ", "response."]
    assert session.messages[-1]["content"] == "Live response."
    event_types = [event.get("type") for event in read_session_events(event_path)]
    assert "non_repo_router_reply_bypassed_for_streaming" in event_types
    assert "non_repo_router_reply_used" not in event_types


def test_streaming_non_repo_turn_records_llm_usage(tmp_path: Path) -> None:
    # Regression guard for the non-repo metering leak: an auto-routed, streamed
    # chat turn must emit "llm_usage" events (for the router classification AND
    # the non-repo answer call) and fold their tokens into the session usage
    # summary. Before the fix these turns recorded nothing, so every token/cost
    # surface read zero for pure-chat sessions.
    from sylliptor_agent_cli.llm.types import LLMUsage

    class _UsageRouterStubClient(_RouterStubClient):
        def chat(self, **kwargs: Any) -> LLMResponse:  # type: ignore[override]
            response = super().chat(**kwargs)
            # Attach real API usage to the non-repo answer call only; the router
            # classification call keeps its no-usage response (recorded as an
            # estimate), mirroring how a real provider reports usage.
            if response.content == self.response_reply and self.response_reply:
                return LLMResponse(
                    content=response.content,
                    tool_calls=response.tool_calls,
                    raw=response.raw,
                    provider_metadata=response.provider_metadata,
                    usage=LLMUsage(
                        prompt_tokens=123,
                        completion_tokens=45,
                        total_tokens=168,
                    ),
                )
            return response

    cfg = AppConfig(model="test-model", routing_mode="auto", stream=True)
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
    router = _UsageRouterStubClient(
        route="chat",
        response_reply="Hi there.",
        response_text_deltas=["Hi ", "there."],
    )
    session.router_client = router

    try:
        assert session.run_turn("Hi") == 0
    finally:
        session.close()

    usage_events = [
        event for event in read_session_events(event_path) if event.get("type") == "llm_usage"
    ]
    assert usage_events, "auto-routed non-repo turn recorded no llm_usage events"
    operations = {
        str((event.get("payload") or {}).get("operation") or "") for event in usage_events
    }
    assert "routing_llm" in operations
    assert "non_repo_answer" in operations

    answer_payloads = [
        event["payload"]
        for event in usage_events
        if (event.get("payload") or {}).get("operation") == "non_repo_answer"
    ]
    assert answer_payloads, "the non-repo answer call was not metered"
    answer = answer_payloads[-1]
    assert answer["usage_source"] == "api"
    assert answer["prompt_tokens"] == 123
    assert answer["completion_tokens"] == 45

    totals = session.usage_summary.totals()
    assert totals["total_tokens"] >= 168
    assert totals["prompt_tokens"] >= 123


def test_usage_record_uses_provider_count_when_response_omits_input_usage(
    tmp_path: Path,
) -> None:
    from sylliptor_agent_cli.llm.types import (
        InputTokenCount,
        LLMUsage,
        UsageConfidence,
        UsageContract,
    )

    class _CountCapableClient:
        model = "test-model"
        usage_contract = UsageContract(
            response_usage_confidence=UsageConfidence.AUTHORITATIVE,
            input_token_count_strategy="test_count",
        )

        def count_input_tokens(self, **_kwargs: Any) -> InputTokenCount:
            return InputTokenCount(input_tokens=77)

    session = create_session(
        cfg=AppConfig(model="test-model"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=2,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    try:
        record = session._record_llm_usage(
            client=_CountCapableClient(),
            response=LLMResponse(
                content="done",
                tool_calls=[],
                raw={},
                usage=LLMUsage(
                    prompt_tokens=None,
                    completion_tokens=5,
                    total_tokens=None,
                ),
            ),
            messages=[{"role": "user", "content": "hello"}],
            tool_list=None,
            operation="test_count_fallback",
        )
    finally:
        session.close()

    assert record is not None
    assert record.prompt_tokens == 77
    assert record.completion_tokens == 5
    assert record.total_tokens == 82
    assert record.usage_source == "api"
    assert record.usage_source_detail == "provider_count"
    assert record.prompt_estimate_error_ratio is not None


def test_main_hud_keeps_local_preflight_prompt_provenance(tmp_path: Path) -> None:
    from sylliptor_agent_cli.llm.types import (
        InputTokenCount,
        LLMUsage,
        UsageConfidence,
        UsageContract,
        UsageSource,
    )

    class _EstimatedCountClient:
        model = "test-model"
        provider_key = "compat-provider"
        protocol = "openai_compat"
        base_url = "https://compat-provider.invalid/v1"
        usage_contract = UsageContract(
            response_usage_confidence=UsageConfidence.REPORTED,
            input_token_count_strategy="openai_compat_provider_payload",
        )

        def count_input_tokens(self, **_kwargs: Any) -> InputTokenCount:
            return InputTokenCount(
                input_tokens=77,
                source=UsageSource.LOCAL_ESTIMATE,
                confidence=UsageConfidence.ESTIMATED,
            )

    session = create_session(
        cfg=AppConfig(model="test-model"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=2,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    try:
        client = _EstimatedCountClient()
        record = session._record_llm_usage(
            client=client,
            response=LLMResponse(
                content="done",
                tool_calls=[],
                raw={},
                usage=LLMUsage(
                    prompt_tokens=None,
                    completion_tokens=5,
                    total_tokens=105,
                    confidence=UsageConfidence.REPORTED,
                ),
            ),
            messages=[{"role": "user", "content": "hello"}],
            tool_list=None,
            operation="main_llm",
        )
        session.client.provider_key = client.provider_key
        session.client.base_url = client.base_url
        session.messages.append({"role": "assistant", "content": "done"})
        ctx = session.context_left()
    finally:
        session.close()

    assert record is not None
    assert record.usage_source_detail == "mixed"
    assert session.request_context_measurement is not None
    assert session.request_context_measurement.input_tokens == 77
    assert session.request_context_measurement.source == "local_estimate"
    assert session.request_context_measurement.confidence == "estimated"
    assert ctx.token_count_source == "local_estimate"
    assert ctx.token_count_confidence == "estimated"
    assert ctx.provider_projection_applied is False


def test_context_left_omits_tool_schemas_for_unsupported_protocol(tmp_path: Path) -> None:
    from sylliptor_agent_cli.llm.types import UsageConfidence, UsageContract

    class _NoToolClient:
        model = "test-model"
        provider_key = "gemini"
        protocol = "gemini_interactions"
        base_url = "https://generativelanguage.googleapis.com/v1beta"
        supports_tool_calling = False
        usage_contract = UsageContract(
            response_usage_confidence=UsageConfidence.AUTHORITATIVE,
            input_token_count_strategy="gemini_count_tokens_projection",
        )

    session = create_session(
        cfg=AppConfig(model="test-model"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=2,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    try:
        session.client = _NoToolClient()  # type: ignore[assignment]
        session.tool_list = [
            {
                "type": "function",
                "function": {
                    "name": "unused_tool",
                    "description": "large unused schema " * 500,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        expected = estimate_request_token_breakdown(
            messages=session.messages,
            tool_list=None,
            pinned_prefix_len=session.pinned_prefix_len,
        ).total_tokens
        with_tools = estimate_request_token_breakdown(
            messages=session.messages,
            tool_list=session.tool_list,
            pinned_prefix_len=session.pinned_prefix_len,
        ).total_tokens

        ctx = session.context_left()
    finally:
        session.close()

    assert with_tools > expected
    assert ctx.local_request_estimate_tokens == expected
    assert ctx.startup_baseline_tokens == expected
    assert ctx.dynamic_context_used_tokens == 0
    assert ctx.dynamic_context_percent_left == 100.0


def test_context_left_rebases_startup_tools_after_runtime_tool_disable(tmp_path: Path) -> None:
    class _MutableToolClient:
        model = "test-model"
        provider_key = "compat-provider"
        protocol = "openai_compat"
        base_url = "https://compat-provider.invalid/v1"
        supports_tool_calling = True

    session = create_session(
        cfg=AppConfig(model="test-model"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=2,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    try:
        client = _MutableToolClient()
        session.client = client  # type: ignore[assignment]
        session.tool_list = [
            {
                "type": "function",
                "function": {
                    "name": "provider_rejected_tool",
                    "description": "large schema removed after rejection " * 500,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        with_tools = session.context_left()
        client.supports_tool_calling = False
        without_tools = session.context_left()
    finally:
        session.close()

    assert with_tools.startup_baseline_tokens > without_tools.startup_baseline_tokens
    assert without_tools.local_request_estimate_tokens == without_tools.startup_baseline_tokens
    assert without_tools.dynamic_context_used_tokens == 0
    assert without_tools.dynamic_context_percent_left == 100.0


def test_main_usage_anchors_hud_to_provider_visible_request(tmp_path: Path) -> None:
    from sylliptor_agent_cli.llm.types import LLMUsage, UsageConfidence, UsageContract

    session = create_session(
        cfg=AppConfig(model="test-model"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=2,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    try:
        session.client.provider_key = "test-provider"
        session.client.base_url = "https://provider.invalid/v1"
        session.client.usage_contract = UsageContract(
            response_usage_confidence=UsageConfidence.AUTHORITATIVE,
        )
        persistent_before = list(session.messages)
        provider_messages = [
            *persistent_before,
            {"role": "system", "content": "ephemeral controller context " * 200},
            {"role": "user", "content": "ephemeral current instruction " * 50},
        ]
        record = session._record_llm_usage(
            client=session.client,
            response=LLMResponse(
                content="done",
                tool_calls=[],
                raw={},
                usage=LLMUsage(
                    prompt_tokens=7000,
                    completion_tokens=5,
                    total_tokens=7005,
                ),
            ),
            messages=provider_messages,
            tool_list=session.tool_list,
            operation="main_llm",
        )
        assert record is not None
        session.messages.append({"role": "assistant", "content": "done"})

        ctx = session.context_left()
    finally:
        session.close()

    assert session.request_context_measurement is not None
    assert session.request_context_measurement.input_tokens == 7000
    assert ctx.used_input_tokens == 7000 + math.ceil(
        (
            ctx.local_request_estimate_tokens
            - session.request_context_measurement.persistent_anchor_estimate_tokens
        )
        * (
            session.request_context_measurement.input_tokens
            / session.request_context_measurement.anchor_estimate_tokens
        )
    )
    assert ctx.used_input_tokens > 7000
    assert ctx.token_count_source == "mixed"
    assert ctx.token_count_confidence == "estimated"
    assert ctx.anchor_token_count_source == "provider_response"


@pytest.mark.parametrize("trace_level", ["off", "compact", "full"])
@pytest.mark.parametrize("stream", [False, True])
def test_trace_level_does_not_change_normal_agent_requests_or_usage(
    tmp_path: Path,
    trace_level: str,
    stream: bool,
) -> None:
    from sylliptor_agent_cli.cli_impl.tui.surface import TuiSurface
    from sylliptor_agent_cli.cli_impl.tui.transcript import TuiTranscript
    from sylliptor_agent_cli.llm.protocols import ReasoningTraceCapability
    from sylliptor_agent_cli.llm.types import LLMUsage, ReasoningOutputKind

    class _FixedUsageClient(_RouterStubClient):
        usage_counts_authoritative = True
        reasoning_trace_capability = ReasoningTraceCapability(
            adapter="test_summary",
            output_kind=ReasoningOutputKind.SUMMARY,
            supports_streaming=True,
            supports_buffered=True,
        )

        def chat(self, **kwargs: Any) -> LLMResponse:  # type: ignore[override]
            is_route_call = bool(kwargs.get("messages")) and (
                str(kwargs["messages"][0].get("content") or "")
                == agent_loop_mod._ROUTER_SYSTEM_PROMPT
            )
            if not is_route_call:
                self.summary_callback_supplied = callable(kwargs.get("on_reasoning_delta"))
                if not kwargs.get("stream") and self.summary_callback_supplied:
                    for delta in self.response_reasoning_deltas:
                        kwargs["on_reasoning_delta"](delta)
            response = super().chat(**kwargs)
            usage = (
                LLMUsage(prompt_tokens=100, completion_tokens=10, total_tokens=110)
                if is_route_call
                else LLMUsage(
                    prompt_tokens=200,
                    completion_tokens=20,
                    total_tokens=220,
                    reasoning_tokens=12,
                )
            )
            return LLMResponse(
                content=response.content,
                tool_calls=response.tool_calls,
                raw=response.raw,
                usage=usage,
            )

    transcript = TuiTranscript()
    surface = TuiSurface(transcript, auto_approve=lambda: True)
    surface.set_trace_level(trace_level)
    cfg = AppConfig(model="test-model", routing_mode="auto", stream=stream)
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
        surface=surface,
    )
    client = _FixedUsageClient(
        route="chat",
        response_reply="Hi there.",
        response_text_deltas=["Hi ", "there."],
        response_reasoning_deltas=["I should answer briefly."],
    )
    session.router_client = client

    try:
        assert session.run_turn("Hi") == 0
        totals = session.usage_summary.totals()
    finally:
        session.close()

    assert client.calls == 2
    assert totals["prompt_tokens"] == 300
    assert totals["completion_tokens"] == 30
    assert totals["total_tokens"] == 330
    assert totals["reasoning_tokens"] == 12
    assert client.summary_callback_supplied is (trace_level != "off")
    reasoning_entries = [entry for entry in transcript.entries if entry[0] == "reasoning"]
    assert bool(reasoning_entries) is (trace_level != "off")


def test_raw_only_provider_capability_never_receives_display_callback(tmp_path: Path) -> None:
    from sylliptor_agent_cli.cli_impl.tui.surface import TuiSurface
    from sylliptor_agent_cli.cli_impl.tui.transcript import TuiTranscript
    from sylliptor_agent_cli.llm.protocols import ReasoningTraceCapability
    from sylliptor_agent_cli.llm.types import ReasoningOutputKind

    class _RawOnlyClient(_RouterStubClient):
        reasoning_trace_capability = ReasoningTraceCapability(
            adapter="raw_only",
            output_kind=ReasoningOutputKind.PROVIDER_REASONING,
            supports_streaming=True,
            supports_buffered=True,
        )

        def chat(self, **kwargs: Any) -> LLMResponse:  # type: ignore[override]
            is_route_call = bool(kwargs.get("messages")) and (
                str(kwargs["messages"][0].get("content") or "")
                == agent_loop_mod._ROUTER_SYSTEM_PROMPT
            )
            if not is_route_call:
                self.summary_callback_supplied = callable(kwargs.get("on_reasoning_delta"))
            return super().chat(**kwargs)

    transcript = TuiTranscript()
    surface = TuiSurface(transcript, auto_approve=lambda: True)
    surface.set_trace_level("full")
    session = create_session(
        cfg=AppConfig(model="test-model", routing_mode="auto", stream=True),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
        surface=surface,
    )
    client = _RawOnlyClient(
        route="chat",
        response_reply="Safe answer.",
        response_reasoning_deltas=["private chain of thought"],
    )
    session.router_client = client

    try:
        assert session.run_turn("Hi") == 0
    finally:
        session.close()

    assert client.summary_callback_supplied is False
    assert not any(role == "reasoning" for role, _text in transcript.entries)


def test_non_repo_chat_follow_up_uses_recent_history_not_router_reply(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        model="test-model",
        routing_mode="auto",
        chat_temperature=0.73,
        stream=False,
    )
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


def test_turn_error_log_and_display_sanitize_unexpected_provider_url(
    tmp_path: Path,
) -> None:
    session = create_session(
        cfg=AppConfig(model="test-model", routing_mode="auto"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
    )
    session.client = _FailClient()  # type: ignore[assignment]
    session.router_client = _AuthRejectingClient(
        route_error=True,
        error_factory=_private_provider_url_error,
    )

    try:
        with pytest.raises(LLMError) as exc_info:
            session.run_turn("How are you?")
        persisted_error = str(
            _session_event_payload(session.store.path, "error").get("error") or ""
        )
        displayed_error = friendly_llm_error_message(exc_info.value)
        assert "api.example.test" in persisted_error
        for rendered in (persisted_error, displayed_error):
            assert "PRIVATE_BOUNDARY_SENTINEL" not in rendered
            assert "PRIVATE_BEARER_SENTINEL" not in rendered
            assert "PRIVATE_API_KEY_SENTINEL" not in rendered
            assert "route-user" not in rendered
            assert "route-pa'ssword" not in rendered
            assert "secret-route-segment" not in rendered
            assert "token=" not in rendered
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


def test_general_non_repo_turn_exposes_web_search_for_model_selection(
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
    web_search_tool = session.tools["web_search"]
    session.tools["web_search"] = agent_loop_mod.ToolDef(
        name=web_search_tool.name,
        description=web_search_tool.description,
        parameters=web_search_tool.parameters,
        metadata=web_search_tool.metadata,
        run=lambda args: {
            "query": args["query"],
            "answer": "Live web search is available.",
            "sources": [{"title": "Python", "url": "https://www.python.org/"}],
            "backend": "test-search",
        },
    )
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


def test_web_search_decision_is_delegated_to_the_model(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        model="qwen3.7-plus",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        web_search_mode="auto",
        web_search_policy="auto",
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
    web_search_tool = session.tools["web_search"]

    def _must_be_model_selected(_args: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("web_search must not run before the model selects it")

    session.tools["web_search"] = agent_loop_mod.ToolDef(
        name=web_search_tool.name,
        description=web_search_tool.description,
        parameters=web_search_tool.parameters,
        metadata=web_search_tool.metadata,
        run=_must_be_model_selected,
    )
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        response_reply="The answer requires external evidence.",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("Assess a claim that depends on current external evidence.")
    finally:
        session.close()

    events = read_session_events(event_path)
    tool_names = {
        str(tool.get("function", {}).get("name") or "") for tool in (router.last_tools or [])
    }

    assert exit_code == 0
    assert "web_search" in tool_names
    assert router.response_calls == 1
    assert not any(
        event.get("type")
        in {
            "web_search_policy_decision",
            "web_search_context_injected",
            "web_search_required_unavailable",
        }
        for event in events
    )


def test_missing_search_backend_does_not_fail_before_model_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "TAVILY_API_KEY",
        "SYLLIPTOR_WEB_SEARCH_API_KEY",
        "SYLLIPTOR_WEB_SEARCH_ADAPTER",
        "SYLLIPTOR_WEB_SEARCH_BASE_URL",
        "SYLLIPTOR_WEB_SEARCH_MODEL",
        "SYLLIPTOR_WEB_SEARCH_PROVIDER",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SYLLIPTOR_WEB_SEARCH_KEYLESS", "0")
    cfg = AppConfig(
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        web_search_mode="auto",
        web_search_policy="auto",
        routing_mode="auto",
    )
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="deepseek-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        response_reply="No configured search backend is available.",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("Assess a claim that requires current external evidence.")
    finally:
        session.close()

    events = read_session_events(event_path)
    tool_names = {
        str(tool.get("function", {}).get("name") or "") for tool in (router.last_tools or [])
    }

    assert exit_code == 0
    assert "web_search" not in tool_names
    assert router.response_calls == 1
    assert not any(event.get("type") == "web_search_required_unavailable" for event in events)


def test_web_research_with_local_output_routes_to_repo_and_keeps_web_tools(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        model="test-model",
        web_search_mode="auto",
        routing_mode="auto",
    )
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        verification_enabled=False,
        session_log_dir_override=tmp_path / "sessions",
    )
    event_path = session.store.path
    router = _RouterStubClient(
        route="general",
        execution_posture="advisory_non_execution",
        response_reply="This non-repo response must not be used.",
    )
    session.router_client = router
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="web_search",
                        arguments={"query": "latest stable Python version"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="fs_write",
                        arguments={"path": "answer.txt", "content": "Python 3.14.4\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(content="Created answer.txt.", tool_calls=[], raw={}),
        ]
    )  # type: ignore[assignment]
    web_search_tool = session.tools["web_search"]
    session.tools["web_search"] = agent_loop_mod.ToolDef(
        name=web_search_tool.name,
        description=web_search_tool.description,
        parameters=web_search_tool.parameters,
        metadata=web_search_tool.metadata,
        run=lambda _args: {
            "answer": "Python 3.14.4",
            "sources": [{"title": "Python", "url": "https://www.python.org/"}],
        },
    )
    session.tool_list = [tool.as_openai_tool() for tool in session.tools.values()]

    try:
        exit_code = session.run_turn(
            "Search the web for the latest stable Python version and save it to answer.txt."
        )
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    first_tools = {
        str(tool.get("function", {}).get("name") or "")
        for tool in (session.client.call_records[0]["tools"] or [])  # type: ignore[attr-defined]
    }

    assert exit_code == 0
    assert router.route_calls == 1
    assert router.response_calls == 0
    assert route_payload["route"] == "repo"
    assert route_payload["original_route"] == "general"
    assert route_payload["execution_posture"] == "execute"
    assert route_payload["route_override_reason"] == "local_materialization_requires_repo_execution"
    assert route_payload["local_materialization_required"] is True
    assert "web_search" in first_tools
    assert "web_fetch" in first_tools
    assert "fs_write" in first_tools
    assert (tmp_path / "answer.txt").read_text(encoding="utf-8") == "Python 3.14.4\n"


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
        exit_code = session.run_turn(
            "Describe the non-repository web capabilities in this session."
        )
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
        exit_code = session.run_turn("Describe the available non-repository tools.")
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
    web_search_tool = session.tools["web_search"]
    session.tools["web_search"] = agent_loop_mod.ToolDef(
        name=web_search_tool.name,
        description=web_search_tool.description,
        parameters=web_search_tool.parameters,
        metadata=web_search_tool.metadata,
        run=lambda args: {
            "query": args["query"],
            "answer": "Python 3.14 is the latest stable release.",
            "sources": [{"title": "Python", "url": "https://www.python.org/downloads/"}],
            "backend": "test-search",
        },
    )
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
    assert "claims depend on unstable/current facts" in system_text
    assert "available web_search tool" in system_text


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


def _tool_budget_session(tmp_path: Path) -> Any:
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
    session.tools["mcp__alpha__echo_tool"] = agent_loop_mod.ToolDef(
        name="mcp__alpha__echo_tool",
        description="Echo a short message through the alpha MCP server.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        run=lambda args: {"marker": f"echo:{args.get('text', '')}"},
    )
    session.tool_list = [tool.as_openai_tool() for tool in session.tools.values()]
    session.client = _FailClient()  # type: ignore[assignment]
    return session


def _echo_probe_call(index: int) -> ToolCall:
    return ToolCall(
        id=f"tc{index}",
        name="mcp__alpha__echo_tool",
        arguments={"text": f"probe {index}"},
    )


def test_tool_route_step_budget_exhaustion_answers_from_gathered_evidence(
    tmp_path: Path,
) -> None:
    # The non-repo tool loop allows 4 tool steps. When the model is still calling
    # tools at the cap, the turn must finalize FROM the gathered tool evidence
    # (one more call, tools disabled, same transcript) — not degrade to the
    # no-context router fallback that can only ask a generic clarifying question.
    session = _tool_budget_session(tmp_path)
    event_path = session.store.path
    router = _SequentialRouterClient(
        routes=["tool"],
        execution_postures=["advisory_non_execution"],
        response_replies=["", "", "", "", "England beat Australia in the final."],
        response_tool_calls=[
            [_echo_probe_call(1)],
            [_echo_probe_call(2)],
            [_echo_probe_call(3)],
            [_echo_probe_call(4)],
            [],
        ],
        tool_family="mcp",
        tool_candidates=["mcp__alpha__echo_tool"],
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("who won the last world cup match?")
    finally:
        session.close()

    events = read_session_events(event_path)
    final_payload = _session_event_payload(event_path, "final")

    assert exit_code == 0
    assert router.route_calls == 1  # no second (fallback) router call
    assert router.response_calls == 5  # 4 tool steps + 1 finalize
    assert router.last_tools is None  # the finalize call must disable tools
    # The finalize request carries the full gathered evidence.
    finalize_tool_messages = [
        message for message in router.last_messages if message.get("role") == "tool"
    ]
    assert len(finalize_tool_messages) == 4
    # The finalize directive must be a USER message: live-verified that some
    # providers (DeepSeek) ignore a trailing system message after tool results.
    assert router.last_messages[-1]["role"] == "user"
    assert "No further tool calls" in router.last_messages[-1]["content"]
    assert any(event.get("type") == "non_repo_tool_budget_finalize" for event in events)
    assert final_payload["content"] == "England beat Australia in the final."


def test_tool_route_step_budget_finalize_empty_answer_falls_back(
    tmp_path: Path,
) -> None:
    # If the finalize call produces nothing twice (one corrective retry), the
    # turn keeps the previous fallback behavior instead of emitting an empty
    # reply.
    session = _tool_budget_session(tmp_path)
    event_path = session.store.path
    router = _SequentialRouterClient(
        routes=["tool"],
        execution_postures=["advisory_non_execution"],
        route_replies=["", "Could you clarify what you need?"],
        response_replies=["", "", "", "", "", ""],
        response_tool_calls=[
            [_echo_probe_call(1)],
            [_echo_probe_call(2)],
            [_echo_probe_call(3)],
            [_echo_probe_call(4)],
            [],
            [],
        ],
        tool_family="mcp",
        tool_candidates=["mcp__alpha__echo_tool"],
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("who won the last world cup match?")
    finally:
        session.close()

    events = read_session_events(event_path)
    final_payload = _session_event_payload(event_path, "final")

    assert exit_code == 0
    assert router.response_calls == 6  # 4 tool steps + finalize + corrective retry
    assert any(event.get("type") == "non_repo_tool_budget_finalize" for event in events)
    discard_warnings = [
        event.get("payload", {})
        for event in events
        if event.get("type") == "warning"
        and event.get("payload", {}).get("warning") == "non_repo_finalize_discarded"
    ]
    assert discard_warnings and discard_warnings[0]["cause"] == "empty"
    assert str(final_payload["content"]).strip()


def test_tool_route_step_budget_finalize_retries_past_tool_call_markup(
    tmp_path: Path,
) -> None:
    # Live-observed with DeepSeek: a model cut off mid-tool-chain answers the
    # finalize request with raw tool-call markup as text. That markup must never
    # surface to the user; one corrective retry recovers the real answer.
    session = _tool_budget_session(tmp_path)
    event_path = session.store.path
    markup_reply = (
        '<|DSML|tool_calls>\n<|DSML|invoke name="mcp__alpha__echo_tool">\n'
        '<|DSML|parameter name="text" string="true">probe 5</|DSML|parameter>\n'
        "</|DSML|invoke>\n</|DSML|tool_calls>"
    )
    router = _SequentialRouterClient(
        routes=["tool"],
        execution_postures=["advisory_non_execution"],
        response_replies=["", "", "", "", markup_reply, "England beat Australia in the final."],
        response_tool_calls=[
            [_echo_probe_call(1)],
            [_echo_probe_call(2)],
            [_echo_probe_call(3)],
            [_echo_probe_call(4)],
            [],
            [],
        ],
        tool_family="mcp",
        tool_candidates=["mcp__alpha__echo_tool"],
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("who won the last world cup match?")
    finally:
        session.close()

    final_payload = _session_event_payload(event_path, "final")

    assert exit_code == 0
    assert router.response_calls == 6
    assert final_payload["content"] == "England beat Australia in the final."
    assert "DSML" not in final_payload["content"]


def test_tool_route_step_budget_finalize_accepts_prose_mentioning_tool_calls(
    tmp_path: Path,
) -> None:
    # The markup veto must not reject a legitimate answer that merely MENTIONS
    # tool-calling terms (e.g. explaining an API): only replies that ARE markup
    # (leading bracket) are discarded.
    session = _tool_budget_session(tmp_path)
    event_path = session.store.path
    prose_answer = (
        "The `tool_calls` field in the response is an array of requested tool "
        "invocations; each entry has an id, a function name, and arguments."
    )
    router = _SequentialRouterClient(
        routes=["tool"],
        execution_postures=["advisory_non_execution"],
        response_replies=["", "", "", "", prose_answer],
        response_tool_calls=[
            [_echo_probe_call(1)],
            [_echo_probe_call(2)],
            [_echo_probe_call(3)],
            [_echo_probe_call(4)],
            [],
        ],
        tool_family="mcp",
        tool_candidates=["mcp__alpha__echo_tool"],
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("what does the tool_calls response field contain?")
    finally:
        session.close()

    final_payload = _session_event_payload(event_path, "final")

    assert exit_code == 0
    assert router.response_calls == 5  # accepted on the first finalize attempt
    assert final_payload["content"] == prose_answer


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


def test_non_repo_web_failure_returns_observation_and_continues_without_failed_tool(
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
        response_replies=["", "Continued without web access."],
        response_tool_calls=[
            [
                ToolCall(
                    id="tc1",
                    name="web_search",
                    arguments={"query": "latest external docs"},
                )
            ],
            [],
        ],
    )
    session.router_client = router
    web_search_tool = session.tools["web_search"]

    def _failed_search(_args: dict[str, Any]) -> dict[str, Any]:
        raise WebSearchError("gateway permission denied")

    session.tools["web_search"] = agent_loop_mod.ToolDef(
        name=web_search_tool.name,
        description=web_search_tool.description,
        parameters=web_search_tool.parameters,
        metadata=web_search_tool.metadata,
        run=_failed_search,
    )
    session.tool_list = [tool.as_openai_tool() for tool in session.tools.values()]

    try:
        exit_code = session.run_turn("Search the internet, then answer briefly.")
    finally:
        session.close()

    events = list(read_session_events(event_path))
    result = next(
        (event.get("payload") or {}).get("result")
        for event in events
        if event.get("type") == "tool_result"
        and (event.get("payload") or {}).get("name") == "web_search"
    )
    final_tool_names = {
        str((tool.get("function") or {}).get("name") or "") for tool in (router.last_tools or [])
    }
    assert exit_code == 0
    assert router.response_calls == 2
    assert isinstance(result, dict)
    assert "error" not in result
    assert result.get("reason") == WEB_UNAVAILABLE_OBSERVATION
    assert "web_search" not in final_tool_names
    assert session.messages[-1]["content"] == "Continued without web access."


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


def test_failed_router_model_recovers_via_main_model(tmp_path: Path) -> None:
    # A dedicated router-role model that fails to produce a decision (e.g. a
    # broken/unavailable model returning a 502 upstream_error) must not silently
    # degrade every turn to the static clarification reply. Routing retries once
    # with the main model, and the non-repo response is served by it too instead
    # of hitting the same dead router model.
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
    main_client = _RouterStubClient(
        route="general",
        response_reply="Recovered via the main model.",
    )
    session.client = main_client  # type: ignore[assignment]
    broken_router = _MalformedRouterClient()
    session.router_client = broken_router

    try:
        exit_code = session.run_turn("Tell me a fun fact about the ocean.")
    finally:
        session.close()

    assert exit_code == 0
    # The broken router model was tried exactly once and never served the response.
    assert broken_router.route_calls == 1
    assert broken_router.response_calls == 0
    # The main model re-ran routing and produced the response.
    assert main_client.route_calls == 1
    assert main_client.response_calls == 1
    final_payload = _session_event_payload(event_path, "final")
    assert final_payload["content"] == "Recovered via the main model."
    fallback_payload = _session_event_payload(event_path, "router_model_fallback_to_main")
    assert fallback_payload["router_decision_source"].startswith("fallback")
    assert fallback_payload["retry_decision_source"] == "router"
    assert fallback_payload["retry_route"] == "general"
    route_payload = _session_event_payload(event_path, "route_decision")
    assert route_payload["route"] == "general"
    assert route_payload["router_decision_source"] == "router"


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

    # Tool results echo OS-native paths (e.g. "src\\widget.py" on Windows), so
    # normalize separators before asserting the repo history carried into turn 2.
    def _mentions_widget_path(content: object) -> bool:
        text = str(content or "").replace("\\\\", "/").replace("\\", "/")
        return "src/widget.py" in text

    assert any(
        msg.get("role") == "tool" and _mentions_widget_path(msg.get("content"))
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


def test_is_fatal_non_repo_llm_error_classifies_trial_proxy_errors() -> None:
    for code in ("trial_expired", "quota_exhausted", "rate_limit_exceeded", "plan_inactive"):
        err = LLMError("LLM error 402: " + json.dumps({"error": {"code": code}}))
        assert _is_fatal_non_repo_llm_error(err) is True, code
    assert (
        _is_fatal_non_repo_llm_error(
            LLMError(
                "LLM error 400: invalid_request_error: Your credit balance is too low; "
                "purchase credits."
            )
        )
        is True
    )
    assert _is_fatal_non_repo_llm_error(LLMError("LLM error 429: rate limit")) is True
    # A generic upstream failure stays non-fatal (handled/retried as before).
    assert _is_fatal_non_repo_llm_error(LLMError("LLM error 500: upstream boom")) is False


def test_auto_mode_non_repo_trial_quota_error_is_not_masked_as_clarification_fallback(
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
    router = _AuthRejectingClient(
        route="chat",
        response_error=True,
        error_factory=_trial_quota_exhausted_error,
    )
    session.router_client = router
    baseline_messages = copy.deepcopy(session.messages)

    try:
        with pytest.raises(LLMError, match="quota_exhausted"):
            session.run_turn("How are you?")
        assert router.route_calls == 1
        assert router.response_calls == 1
        assert session.messages == baseline_messages
        error_payload = _session_event_payload(session.store.path, "error")
        assert "quota_exhausted" in str(error_payload.get("error") or "")
    finally:
        session.close()


def test_non_repo_empty_completion_logs_breadcrumb_before_fallback(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model", routing_mode="auto", chat_temperature=0.5)
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
    router = _RouterStubClient(
        route="general",
        route_reply="",
        response_reply="",
        language="English",
        script="Latin",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn("hello there")
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 0
    assert router.response_calls == 1
    # Legitimate fallback is still produced for an empty completion ...
    assert session.messages[-1]["content"] == "Could you clarify what you want me to help with?"
    # ... but it now leaves a diagnosable breadcrumb in the session log.
    warnings = [
        dict(event.get("payload") or {})
        for event in read_session_events(log_path)
        if event.get("type") == "warning"
    ]
    assert any(w.get("warning") == "non_repo_empty_completion" for w in warnings)


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
    secret_base_url = "https://route-user:route-password@api.example.test/private/token"
    cfg = AppConfig(
        model="test-model",
        base_url=secret_base_url,
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
        assert session_start_payload["base_url_descriptor"] == endpoint_descriptor(secret_base_url)
        assert session_start_payload["provider_base_url_descriptor"] == endpoint_descriptor(
            secret_base_url
        )
        assert "base_url" not in session_start_payload
        assert "provider_base_url" not in session_start_payload
        serialized_session_start = json.dumps(session_start_payload, sort_keys=True)
        assert "route-user" not in serialized_session_start
        assert "route-password" not in serialized_session_start
        assert "private/token" not in serialized_session_start
        assert session.conversation_compactor is not None
        assert session.conversation_compactor.compactor_client.temperature == 0.31
    finally:
        session.close()
