from __future__ import annotations

import io
import threading
from pathlib import Path
from time import perf_counter
from typing import Any

from rich.console import Console

from sylliptor_agent_cli.agent_loop import AgentSession, ToolDef
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.llm.metadata import (
    GEMINI_GENERATE_CONTENT_PROVIDER_METADATA_KEY,
    PROVIDER_METADATA_KEY,
)
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
        self.requests: list[dict[str, Any]] = []
        self.call_messages: list[list[dict[str, Any]]] = []

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
        on_text_delta: Any = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        _ = messages, tools, tool_choice, response_format, temperature, max_tokens
        call_index = self.calls
        self.calls += 1
        self.call_messages.append(list(messages))
        self.requests.append({"stream": stream, "has_delta_callback": callable(on_text_delta)})
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


def test_run_turn_streaming_tool_call_executes_after_stream_finishes(tmp_path: Path) -> None:
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
                content="I will call the tool.",
                tool_calls=[
                    ToolCall(id="call_streamed", name="echo_tool", arguments={"message": "hello"})
                ],
                raw={},
            ),
            LLMResponse(content="done", tool_calls=[], raw={}),
        ],
        stream_chunks=[["I will ", "call the tool."]],
    )
    session = _make_session(root=tmp_path, client=client, surface=surface, stream=True, tool=tool)

    try:
        exit_code = session.run_turn("run the tool")
    finally:
        session.close()

    assert exit_code == 0
    assert client.requests[0] == {"stream": True, "has_delta_callback": True}
    assert client.requests[1] == {"stream": True, "has_delta_callback": True}
    event_types = [type(event) for event in surface.events]
    first_tool_index = event_types.index(ToolCallStarted)
    assert event_types[:first_tool_index] == [MessageDelta, MessageDelta, MessageEnd]
    assert [event.text for event in surface.events[:2]] == ["I will ", "call the tool."]


def test_run_turn_streaming_provider_metadata_survives_tool_followup_request(
    tmp_path: Path,
) -> None:
    tool = ToolDef(
        name="echo_tool",
        description="echo",
        parameters={"type": "object", "properties": {}, "required": []},
        run=lambda args: {"ok": args["message"]},
    )
    provider_content = {
        "role": "model",
        "parts": [
            {"text": "I will call the tool."},
            {
                "functionCall": {
                    "id": "call_streamed",
                    "name": "echo_tool",
                    "args": {"message": "hello"},
                },
                "thoughtSignature": "streamed-thought-signature",
            },
        ],
    }
    surface = _RecordingEventSurface()
    client = _ScriptedClient(
        [
            LLMResponse(
                content="I will call the tool.",
                tool_calls=[
                    ToolCall(
                        id="call_streamed",
                        name="echo_tool",
                        arguments={"message": "hello"},
                        provider_metadata={
                            GEMINI_GENERATE_CONTENT_PROVIDER_METADATA_KEY: {
                                "part_index": 1,
                                "thoughtSignature": "streamed-thought-signature",
                            }
                        },
                    )
                ],
                raw={},
                provider_metadata={
                    GEMINI_GENERATE_CONTENT_PROVIDER_METADATA_KEY: {
                        "content": provider_content,
                    }
                },
            ),
            LLMResponse(content="done", tool_calls=[], raw={}),
        ],
        stream_chunks=[["I will ", "call the tool."]],
    )
    session = _make_session(root=tmp_path, client=client, surface=surface, stream=True, tool=tool)

    try:
        exit_code = session.run_turn("run the tool")
    finally:
        session.close()

    assert exit_code == 0
    assert len(client.call_messages) >= 2
    followup_assistant = next(
        message
        for message in client.call_messages[1]
        if str(message.get("role")) == "assistant" and message.get("tool_calls")
    )
    assert (
        followup_assistant[PROVIDER_METADATA_KEY][GEMINI_GENERATE_CONTENT_PROVIDER_METADATA_KEY][
            "content"
        ]
        == provider_content
    )
    assert (
        followup_assistant[PROVIDER_METADATA_KEY]["_tool_calls"][0]["metadata"][
            GEMINI_GENERATE_CONTENT_PROVIDER_METADATA_KEY
        ]["thoughtSignature"]
        == "streamed-thought-signature"
    )


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


def test_run_turn_dispatches_same_batch_subagent_runs_in_parallel(tmp_path: Path) -> None:
    start_times: dict[str, float] = {}
    end_times: dict[str, float] = {}
    lock = threading.Lock()
    both_started = threading.Event()

    def _run_subagent(args: dict[str, Any]) -> dict[str, Any]:
        task = str(args["task"])
        with lock:
            start_times[task] = perf_counter()
            if len(start_times) == 2:
                both_started.set()
        both_started.wait(timeout=1.0)
        with lock:
            end_times[task] = perf_counter()
        return {
            "subagent": "explorer",
            "subagent_session_id": f"sub-{task}",
            "result": f"catalog for {task}",
        }

    tool = ToolDef(
        name="subagent_run",
        description="fake subagent",
        parameters={"type": "object", "properties": {}, "required": []},
        run=_run_subagent,
    )
    surface = _RecordingEventSurface()
    client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_a",
                        name="subagent_run",
                        arguments={"name": "explorer", "task": "alpha"},
                    ),
                    ToolCall(
                        id="call_b",
                        name="subagent_run",
                        arguments={"name": "explorer", "task": "beta"},
                    ),
                ],
                raw={},
            ),
            LLMResponse(content="done", tool_calls=[], raw={}),
        ]
    )
    session = _make_session(root=tmp_path, client=client, surface=surface, tool=tool)

    try:
        exit_code = session.run_turn("catalog two areas")
    finally:
        session.close()

    assert exit_code == 0
    assert set(start_times) == {"alpha", "beta"}
    assert set(end_times) == {"alpha", "beta"}
    assert max(start_times.values()) < min(end_times.values())

    tool_messages = [
        message for message in client.call_messages[1] if message.get("role") == "tool"
    ]
    assert [message["tool_call_id"] for message in tool_messages] == ["call_a", "call_b"]
    assert "catalog for alpha" in tool_messages[0]["content"]
    assert "catalog for beta" in tool_messages[1]["content"]


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
