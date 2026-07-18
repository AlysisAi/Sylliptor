from __future__ import annotations

import io
import threading
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest
from rich.console import Console

from sylliptor_agent_cli import agent_loop
from sylliptor_agent_cli.agent.turn import core as turn_core
from sylliptor_agent_cli.agent_loop import AgentSession, ToolDef, build_tools
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.llm.openai_compat import LLMResponse, ToolCall
from sylliptor_agent_cli.model_registry import ModelMeta
from sylliptor_agent_cli.session_store import SessionStore
from sylliptor_agent_cli.subagents import SubagentDefinition
from sylliptor_agent_cli.surface.noop_surface import NoopSurface
from sylliptor_agent_cli.surface.types import SubagentEndEvent, SubagentStartEvent
from sylliptor_agent_cli.usage_tracker import UsageRecord, UsageSummary


class _CancellationToken:
    def __init__(self) -> None:
        self._cancelled = threading.Event()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()

    def throw_if_cancelled(self, reason: str = "cancelled_by_user") -> None:
        if self.is_cancelled:
            raise KeyboardInterrupt(reason)


class _Registry:
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

    def __init__(self, tool_calls: list[ToolCall]) -> None:
        self._tool_calls = tool_calls
        self.calls = 0

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        stream: bool = False,
        on_text_delta: Any = None,
        temperature: float | None = None,
        cancellation_token: Any | None = None,
    ) -> LLMResponse:
        _ = (
            messages,
            tools,
            tool_choice,
            stream,
            on_text_delta,
            temperature,
            cancellation_token,
        )
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("cancelled parent must not make another model request")
        return LLMResponse(content="", tool_calls=self._tool_calls, raw={})


class _RecordingSurface(NoopSurface):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.starts: list[SubagentStartEvent] = []
        self.ends: list[SubagentEndEvent] = []

    def on_subagent_start(self, event: SubagentStartEvent) -> None:
        with self._lock:
            self.starts.append(event)

    def on_subagent_end(self, event: SubagentEndEvent) -> None:
        with self._lock:
            self.ends.append(event)


class _ChildStore:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

    def events_snapshot(self) -> list[dict[str, Any]]:
        return []


class _BlockingChildSession:
    def __init__(
        self,
        *,
        index: int,
        started_count: list[int],
        started_lock: threading.Lock,
        all_started: threading.Event,
        expected_children: int,
        cleanup_release: threading.Event,
        mutation_path: Path,
    ) -> None:
        self.index = index
        self.store = _ChildStore(f"child-{index}")
        self.tools = {
            "fs_read": ToolDef(
                name="fs_read",
                description="read",
                parameters={"type": "object", "properties": {}, "required": []},
                run=lambda _args: {"ok": True},
            )
        }
        self.tool_list = [tool.as_openai_tool() for tool in self.tools.values()]
        self.messages: list[dict[str, Any]] = []
        self.usage_summary = UsageSummary()
        self.usage_summary.add_record(
            UsageRecord(
                timestamp="2026-07-17T00:00:00+00:00",
                role="main:subagent:explorer",
                requested_model="test-model",
                response_model="test-model",
                prompt_tokens=3,
                completion_tokens=2,
                total_tokens=5,
                input_cost_per_token=None,
                output_cost_per_token=None,
                cost_usd=None,
                usage_source="api",
            )
        )
        self._started_count = started_count
        self._started_lock = started_lock
        self._all_started = all_started
        self._expected_children = expected_children
        self._cleanup_release = cleanup_release
        self._mutation_path = mutation_path
        self.received_tokens: list[Any] = []
        self.close_calls = 0

    def run_turn(self, task: str, *, cancellation_token: Any | None = None) -> int:
        _ = task
        self.received_tokens.append(cancellation_token)
        with self._started_lock:
            self._started_count[0] += 1
            if self._started_count[0] == self._expected_children:
                self._all_started.set()

        while cancellation_token is None or not cancellation_token.is_cancelled:
            if self._cleanup_release.wait(timeout=0.01):
                self._mutation_path.write_text("late mutation\n", encoding="utf-8")
                return 0

        cancellation_token.throw_if_cancelled()
        self._mutation_path.write_text("mutation after cancellation\n", encoding="utf-8")
        return 0

    def close(self) -> None:
        self.close_calls += 1


def _session_store(tmp_path: Path) -> SessionStore:
    return SessionStore(
        enabled=False,
        sessions_dir=tmp_path / "sessions",
        session_id="parent",
        cwd=str(tmp_path),
        repo_root=str(tmp_path),
    )


def _build_cancellation_session(
    tmp_path: Path,
    *,
    child_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    AgentSession,
    _ScriptedClient,
    SessionStore,
    UsageSummary,
    _RecordingSurface,
    list[_BlockingChildSession],
    threading.Event,
    threading.Event,
]:
    store = _session_store(tmp_path)
    usage_summary = UsageSummary()
    surface = _RecordingSurface()
    children: list[_BlockingChildSession] = []
    children_lock = threading.Lock()
    started_count = [0]
    started_lock = threading.Lock()
    all_started = threading.Event()
    cleanup_release = threading.Event()

    def _create_child(**_kwargs: Any) -> _BlockingChildSession:
        with children_lock:
            index = len(children)
            child = _BlockingChildSession(
                index=index,
                started_count=started_count,
                started_lock=started_lock,
                all_started=all_started,
                expected_children=child_count,
                cleanup_release=cleanup_release,
                mutation_path=tmp_path / f"late-mutation-{index}.txt",
            )
            children.append(child)
            return child

    registry = {
        "explorer": SubagentDefinition(
            name="explorer",
            description="readonly explorer",
            system_prompt="Inspect the repository.",
            mode="readonly",
            allow_tools=("fs_read",),
        )
    }
    monkeypatch.setattr(agent_loop, "create_session", _create_child)
    built_tools = build_tools(
        root=tmp_path,
        console=None,
        surface=surface,
        store=store,
        mode="auto",
        yes=True,
        cfg=AppConfig(model="test-model", routing_mode="code_only"),
        api_key="test-key",
        max_steps=4,
        subagents_enabled=True,
        subagent_registry=registry,
        usage_summary=usage_summary,
        create_session_factory=_create_child,
    )
    subagent_tool = built_tools["subagent_run"]
    tool_calls = [
        ToolCall(
            id=f"call-{index}",
            name="subagent_run",
            arguments={"name": "explorer", "task": f"Inspect area {index}"},
        )
        for index in range(child_count)
    ]
    client = _ScriptedClient(tool_calls)
    session = AgentSession(
        subagent_registry=registry,
        cfg=AppConfig(model="test-model", routing_mode="code_only"),
        root=tmp_path,
        mode="auto",
        yes=True,
        stream=False,
        routing_mode="code_only",
        max_steps=4,
        console=Console(file=io.StringIO(), force_terminal=False),
        surface=surface,
        store=store,
        client=client,  # type: ignore[arg-type]
        model_registry=_Registry(),  # type: ignore[arg-type]
        usage_summary=usage_summary,
        usage_role="main",
        tool_output_offloader=None,
        conversation_compactor=None,
        tool_output_offload_enabled=False,
        conversation_summarization_enabled=False,
        compaction_profile="chat",
        tools={"subagent_run": subagent_tool},
        tool_list=[subagent_tool.as_openai_tool()],
        messages=[{"role": "system", "content": "system prompt"}],
        verification_enabled=False,
        skills_enabled=False,
    )
    return (
        session,
        client,
        store,
        usage_summary,
        surface,
        children,
        all_started,
        cleanup_release,
    )


def _assert_parent_cancellation(
    child_count: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        session,
        client,
        store,
        usage_summary,
        surface,
        children,
        all_started,
        cleanup_release,
    ) = _build_cancellation_session(
        tmp_path,
        child_count=child_count,
        monkeypatch=monkeypatch,
    )
    shutdown_calls: list[tuple[bool, bool]] = []
    if child_count > 1:
        real_executor = turn_core.ThreadPoolExecutor

        class _RecordingExecutor(real_executor):
            def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
                shutdown_calls.append((wait, cancel_futures))
                super().shutdown(wait=wait, cancel_futures=cancel_futures)

        monkeypatch.setattr(turn_core, "ThreadPoolExecutor", _RecordingExecutor)
    token = _CancellationToken()
    outcome: list[BaseException | int] = []

    def _run_parent() -> None:
        try:
            outcome.append(
                session.run_turn(
                    "Use explorer subagents to inspect the repository.",
                    cancellation_token=token,
                )
            )
        except BaseException as exc:  # noqa: BLE001 - cancellation may be KeyboardInterrupt
            outcome.append(exc)

    worker = threading.Thread(target=_run_parent, daemon=True)
    worker.start()
    try:
        assert all_started.wait(timeout=2.0), "subagent child did not start"
        cancelled_at = perf_counter()
        token.cancel()
        worker.join(timeout=1.5)
        elapsed = perf_counter() - cancelled_at
        assert not worker.is_alive(), "parent turn did not return promptly after cancellation"
        assert elapsed < 1.5
    finally:
        cleanup_release.set()
        worker.join(timeout=2.0)
        session.close()

    assert len(children) == child_count
    assert all(child.received_tokens == [token] for child in children)
    assert all(child.close_calls == 1 for child in children)
    assert not list(tmp_path.glob("late-mutation-*.txt"))
    assert len(outcome) == 1
    assert isinstance(outcome[0], KeyboardInterrupt)
    assert "cancelled_by_user" in str(outcome[0])
    assert client.calls == 1

    events = store.events_snapshot()
    start_payloads = [event["payload"] for event in events if event["type"] == "subagent_start"]
    end_payloads = [event["payload"] for event in events if event["type"] == "subagent_end"]
    assert len(start_payloads) == child_count
    assert len(end_payloads) == child_count
    assert all(payload["status"] == "cancelled" for payload in end_payloads)
    assert all(payload["failure_category"] == "cancelled" for payload in end_payloads)
    assert all(payload["error_code"] == "subagent_cancelled" for payload in end_payloads)
    assert not any(payload["status"] == "success" for payload in end_payloads)
    assert len(surface.starts) == child_count
    assert len(surface.ends) == child_count
    assert all(event.status == "cancelled" for event in surface.ends)

    child_usage_events = [
        event
        for event in events
        if event["type"] == "llm_usage" and event["payload"].get("role") == "main:subagent:explorer"
    ]
    assert len(child_usage_events) == child_count
    child_usage_records = [
        record for record in usage_summary.records() if record.role == "main:subagent:explorer"
    ]
    assert len(child_usage_records) == child_count

    tool_result_ids = [
        event["payload"]["tool_call_id"] for event in events if event["type"] == "tool_result"
    ]
    assert tool_result_ids == [f"call-{index}" for index in range(child_count)]
    schema_properties = session.tools["subagent_run"].as_openai_tool()["function"]["parameters"][
        "properties"
    ]
    assert "cancellation_token" not in schema_properties
    assert "_cancellation_token" not in schema_properties
    if child_count > 1:
        assert shutdown_calls == [(False, True)]


def test_serial_subagent_receives_parent_cancellation_and_cleans_up_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_parent_cancellation(1, tmp_path, monkeypatch)


def test_parallel_subagents_receive_parent_cancellation_and_clean_up_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_parent_cancellation(2, tmp_path, monkeypatch)
