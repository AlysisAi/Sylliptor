from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from rich.console import Console

from sylliptor_agent_cli.agent_loop import AgentSession, ToolDef
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.llm.openai_compat import LLMResponse, ToolCall
from sylliptor_agent_cli.model_registry import ModelMeta
from sylliptor_agent_cli.session_store import SessionStore
from sylliptor_agent_cli.surface.events import (
    Event,
    MessageDelta,
    MessageEnd,
    ToolCallCompleted,
    ToolCallProgress,
    ToolCallStarted,
)
from sylliptor_agent_cli.surface.noop_surface import NoopSurface
from sylliptor_agent_cli.usage_tracker import UsageSummary


class _FakeRegistry:
    def get(self, model_name: str) -> ModelMeta:
        return ModelMeta(
            model_name=model_name,
            context_window_tokens=8192,
            max_output_tokens=2048,
            input_cost_per_token=None,
            output_cost_per_token=None,
            raw_metadata={},
            source="fallback",
        )


class _ScriptedClient:
    model = "test-model"
    temperature = 0.2

    def __init__(
        self,
        responses: list[LLMResponse],
        *,
        stream_chunks: list[list[str]] | None = None,
    ) -> None:
        self._responses = list(responses)
        self._stream_chunks = list(stream_chunks or [])
        self.calls = 0

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        on_text_delta: Any = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        _ = messages, tools, temperature
        call_index = self.calls
        self.calls += 1
        if stream and callable(on_text_delta) and call_index < len(self._stream_chunks):
            for chunk in self._stream_chunks[call_index]:
                on_text_delta(chunk)
        return self._responses[min(call_index, len(self._responses) - 1)]


class _RecordingEventSurface(NoopSurface):
    def __init__(self) -> None:
        self.events: list[Event] = []
        self.legacy_tokens: list[str] = []
        self.legacy_assistant_done: list[str] = []
        self.legacy_tool_starts: list[Any] = []
        self.legacy_tool_outputs: list[Any] = []
        self.legacy_tool_ends: list[Any] = []

    def on_assistant_token(self, delta: str) -> None:
        self.legacy_tokens.append(delta)

    def on_assistant_message_done(self, text: str) -> None:
        self.legacy_assistant_done.append(text)

    def on_tool_start(self, event: Any) -> None:
        self.legacy_tool_starts.append(event)

    def on_tool_output(self, event: Any) -> None:
        self.legacy_tool_outputs.append(event)

    def on_tool_end(self, event: Any) -> None:
        self.legacy_tool_ends.append(event)

    def emit_message_delta(
        self,
        text: str,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        self.events.append(MessageDelta(text=text, worker_id=worker_id, role=role))

    def emit_message_end(
        self,
        text: str = "",
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        self.events.append(MessageEnd(text=text, worker_id=worker_id, role=role))

    def emit_tool_call_started(
        self,
        call_id: str,
        name: str,
        arguments_preview: str,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        self.events.append(
            ToolCallStarted(
                call_id=call_id,
                name=name,
                arguments_preview=arguments_preview,
                worker_id=worker_id,
                role=role,
            )
        )

    def emit_tool_call_progress(
        self,
        call_id: str,
        text: str,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        self.events.append(
            ToolCallProgress(call_id=call_id, text=text, worker_id=worker_id, role=role)
        )

    def emit_tool_call_completed(
        self,
        call_id: str,
        success: bool,
        result_preview: str,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        self.events.append(
            ToolCallCompleted(
                call_id=call_id,
                success=success,
                result_preview=result_preview,
                worker_id=worker_id,
                role=role,
            )
        )


def _make_store(root: Path) -> SessionStore:
    return SessionStore(
        enabled=False,
        sessions_dir=root / "sessions",
        session_id="s1",
        cwd=str(root),
        repo_root=str(root),
    )


def _make_session(
    *,
    root: Path,
    client: _ScriptedClient,
    surface: _RecordingEventSurface,
    stream: bool = False,
    tool: ToolDef | None = None,
) -> AgentSession:
    tools = {} if tool is None else {tool.name: tool}
    tool_list = [] if tool is None else [tool.as_openai_tool()]
    return AgentSession(
        cfg=AppConfig(model="test-model", routing_mode="code_only", stream=stream),
        root=root,
        mode="auto",
        yes=True,
        stream=stream,
        routing_mode="code_only",
        max_steps=4,
        console=Console(file=io.StringIO(), force_terminal=False),
        surface=surface,
        store=_make_store(root),
        client=client,  # type: ignore[arg-type]
        model_registry=_FakeRegistry(),  # type: ignore[arg-type]
        usage_summary=UsageSummary(),
        usage_role="main",
        tool_output_offloader=None,
        conversation_compactor=None,
        tool_output_offload_enabled=False,
        conversation_summarization_enabled=False,
        compaction_profile="chat",
        tools=tools,
        tool_list=tool_list,
        messages=[{"role": "system", "content": "system prompt"}],
        verification_enabled=False,
        skills_enabled=False,
    )


def test_run_turn_streaming_assistant_text_emits_events_and_legacy(tmp_path: Path) -> None:
    surface = _RecordingEventSurface()
    client = _ScriptedClient(
        [LLMResponse(content="Hello world.", tool_calls=[], raw={})],
        stream_chunks=[["Hello ", "world."]],
    )
    session = _make_session(root=tmp_path, client=client, surface=surface, stream=True)

    try:
        exit_code = session.run_turn("say hello")
    finally:
        session.close()

    assert exit_code == 0
    deltas = [event for event in surface.events if isinstance(event, MessageDelta)]
    ends = [event for event in surface.events if isinstance(event, MessageEnd)]
    assert [event.text for event in deltas] == ["Hello ", "world."]
    assert len(ends) == 1
    assert ends[0].text == "Hello world."
    assert "".join(event.text for event in deltas) == surface.legacy_assistant_done[-1]
    assert surface.legacy_tokens == ["Hello ", "world."]


def test_run_turn_tool_success_emits_lifecycle_events_and_legacy(tmp_path: Path) -> None:
    tool = ToolDef(
        name="echo_tool",
        description="echo",
        parameters={"type": "object", "properties": {}, "required": []},
        run=lambda args: {"ok": args["message"]},
    )
    surface = _RecordingEventSurface()
    client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="call_1", name="echo_tool", arguments={"message": "hello"})
                ],
                raw={},
            ),
            LLMResponse(content="done", tool_calls=[], raw={}),
        ]
    )
    session = _make_session(root=tmp_path, client=client, surface=surface, tool=tool)

    try:
        exit_code = session.run_turn("run the tool")
    finally:
        session.close()

    assert exit_code == 0
    tool_events = [
        event
        for event in surface.events
        if isinstance(event, ToolCallStarted | ToolCallProgress | ToolCallCompleted)
    ]
    assert [type(event) for event in tool_events] == [
        ToolCallStarted,
        ToolCallProgress,
        ToolCallCompleted,
    ]
    assert tool_events[0] == ToolCallStarted(
        call_id="call_1",
        name="echo_tool",
        arguments_preview='{"message": "hello"}',
    )
    assert isinstance(tool_events[2], ToolCallCompleted)
    assert tool_events[2].success is True
    assert len(surface.legacy_tool_starts) == 1
    assert len(surface.legacy_tool_outputs) == 1
    assert len(surface.legacy_tool_ends) == 1


def test_run_turn_tool_failure_emits_failed_completion_event(tmp_path: Path) -> None:
    def _fail_tool(args: dict[str, Any]) -> dict[str, Any]:
        _ = args
        raise RuntimeError("boom")

    tool = ToolDef(
        name="explode_tool",
        description="explode",
        parameters={"type": "object", "properties": {}, "required": []},
        run=_fail_tool,
    )
    surface = _RecordingEventSurface()
    client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="call_1", name="explode_tool", arguments={})],
                raw={},
            ),
            LLMResponse(content="done", tool_calls=[], raw={}),
        ]
    )
    session = _make_session(root=tmp_path, client=client, surface=surface, tool=tool)

    try:
        exit_code = session.run_turn("run the failing tool")
    finally:
        session.close()

    assert exit_code == 0
    completed = [event for event in surface.events if isinstance(event, ToolCallCompleted)]
    assert len(completed) == 1
    assert completed[0].success is False
    assert "boom" in completed[0].result_preview
    assert surface.legacy_tool_ends[0].status == "failed"
