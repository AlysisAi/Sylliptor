from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest

from sylliptor_agent_cli import agent_loop
from sylliptor_agent_cli.agent_loop import SYSTEM_PROMPT, ToolDef, build_tools, create_session
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.runtime_kind import RuntimeKind
from sylliptor_agent_cli.step_budget import StepBudgetRuntime
from sylliptor_agent_cli.subagents import (
    SubagentDefinition,
    built_in_subagents,
    load_subagent_registry,
)
from sylliptor_agent_cli.surface import ApprovalDecision, ApprovalRequest, NoopSurface
from sylliptor_agent_cli.surface.types import (
    SubagentEndEvent,
    SubagentStartEvent,
    ToolEndEvent,
    ToolOutputEvent,
    ToolStartEvent,
)
from sylliptor_agent_cli.usage_tracker import UsageRecord, UsageSummary


class _RecordingStore:
    def __init__(self) -> None:
        self.session_id = "main-session"
        self.events: list[tuple[str, dict[str, Any]]] = []

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((event_type, payload))


def _store_event_payloads(store: _RecordingStore, event_type: str) -> list[dict[str, Any]]:
    return [payload for kind, payload in store.events if kind == event_type]


def _last_store_event_payload(store: _RecordingStore, event_type: str) -> dict[str, Any]:
    payloads = _store_event_payloads(store, event_type)
    assert payloads
    return payloads[-1]


class _FakeUsageSummary:
    def __init__(
        self,
        *,
        prompt_tokens: int = 7,
        completion_tokens: int = 3,
        total_tokens: int = 10,
        api_usage_calls: int = 1,
        estimate_usage_calls: int = 0,
    ) -> None:
        self._records: list[UsageRecord] = []
        self._totals = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "api_usage_calls": api_usage_calls,
            "estimate_usage_calls": estimate_usage_calls,
        }

    def totals(self) -> dict[str, Any]:
        return dict(self._totals)

    def records(self) -> list[UsageRecord]:
        return list(self._records)


class _FakeSubSession:
    def __init__(
        self,
        *,
        tools: dict[str, ToolDef],
        messages: list[dict[str, Any]] | None = None,
        exit_code: int = 0,
        usage_summary: Any | None = None,
    ) -> None:
        self.store = types.SimpleNamespace(session_id="sub-001")
        self.tools = tools
        self.tool_list = [tool.as_openai_tool() for tool in tools.values()]
        self.messages = messages or [{"role": "assistant", "content": "subagent final"}]
        self.usage_summary = usage_summary or _FakeUsageSummary()
        self.exit_code = exit_code
        self.closed = False
        self.run_calls: list[str] = []

    def run_turn(self, task: str) -> int:
        self.run_calls.append(task)
        return self.exit_code

    def close(self) -> None:
        self.closed = True


class _RecordingApprovalSurface(NoopSurface):
    def __init__(self, *, allow: bool) -> None:
        self.allow = allow
        self.approval_requests: list[ApprovalRequest] = []
        self.noise_events: list[tuple[str, str]] = []

    def on_progress_update(self, message: str) -> None:
        self.noise_events.append(("progress", message))

    def on_assistant_token(self, delta: str) -> None:
        self.noise_events.append(("token", delta))

    def on_assistant_message_done(self, text: str) -> None:
        self.noise_events.append(("assistant_done", text))

    def on_error(self, err: str) -> None:
        self.noise_events.append(("error", err))

    def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        self.approval_requests.append(request)
        return ApprovalDecision(allow=self.allow)


class _RecordingNestedSurface(NoopSurface):
    def __init__(self) -> None:
        self.subagent_starts: list[SubagentStartEvent] = []
        self.subagent_ends: list[SubagentEndEvent] = []
        self.tool_starts: list[ToolStartEvent] = []
        self.tool_outputs: list[ToolOutputEvent] = []
        self.tool_ends: list[ToolEndEvent] = []

    def on_subagent_start(self, event: SubagentStartEvent) -> None:
        self.subagent_starts.append(event)

    def on_subagent_end(self, event: SubagentEndEvent) -> None:
        self.subagent_ends.append(event)

    def on_tool_start(self, event: ToolStartEvent) -> None:
        self.tool_starts.append(event)

    def on_tool_output(self, event: ToolOutputEvent) -> None:
        self.tool_outputs.append(event)

    def on_tool_end(self, event: ToolEndEvent) -> None:
        self.tool_ends.append(event)


class _ChildToolSession:
    def __init__(
        self,
        *,
        tools: dict[str, ToolDef],
        surface: Any,
        usage_summary: UsageSummary | None = None,
    ) -> None:
        self.store = types.SimpleNamespace(session_id="sub-child")
        self.tools = tools
        self.tool_list = [tool.as_openai_tool() for tool in tools.values()]
        self.surface = surface
        self.messages: list[dict[str, Any]] = []
        self.usage_summary = usage_summary or UsageSummary()
        self.closed = False
        self.run_calls: list[str] = []

    def run_turn(self, task: str) -> int:
        self.run_calls.append(task)
        self.surface.on_progress_update("child progress noise")
        self.surface.on_assistant_token("child token noise")
        self.tools["fs_write"].run({"path": "approved.txt", "content": "ok\n"})
        self.messages.append({"role": "assistant", "content": "approved write complete"})
        return 0

    def close(self) -> None:
        self.closed = True


class _ChildToolTraceSession(_FakeSubSession):
    def __init__(self, *, surface: Any, tools: dict[str, ToolDef]) -> None:
        super().__init__(
            tools=tools,
            messages=[{"role": "assistant", "content": "nested trace complete"}],
        )
        self.surface = surface

    def run_turn(self, task: str) -> int:
        self.run_calls.append(task)
        self.surface.on_tool_start(
            ToolStartEvent(
                tool_call_id="child_call_1",
                name="fs_read",
                args={"path": "README.md"},
                step=1,
            )
        )
        self.surface.on_tool_output(
            ToolOutputEvent(
                tool_call_id="child_call_1",
                name="fs_read",
                chunk=json.dumps(
                    {"path": "README.md", "content": "abc", "truncated": False},
                    ensure_ascii=True,
                ),
            )
        )
        self.surface.on_tool_end(
            ToolEndEvent(
                tool_call_id="child_call_1",
                name="fs_read",
                status="done",
                elapsed_ms=9,
                meta={},
            )
        )
        return 0


def _cell(value: Any) -> Any:
    return (lambda: value).__closure__[0]


def _rewrite_closure(func: Any, **replacements: Any) -> Any:
    closure = func.__closure__
    if closure is None:
        raise AssertionError("Expected closure-backed function.")
    new_cells = []
    for freevar_name, freevar_cell in zip(func.__code__.co_freevars, closure, strict=True):
        if freevar_name in replacements:
            new_cells.append(_cell(replacements[freevar_name]))
        else:
            new_cells.append(freevar_cell)
    return types.FunctionType(
        func.__code__,
        func.__globals__,
        name=func.__name__,
        argdefs=func.__defaults__,
        closure=tuple(new_cells),
    )


def _build_main_tools(
    *,
    tmp_path: Path,
    subagents_enabled: bool,
    mode: str = "auto",
    subagent_depth: int = 0,
    subagent_registry: dict[str, SubagentDefinition] | None = None,
    store: _RecordingStore | None = None,
    usage_summary: UsageSummary | None = None,
    surface: Any | None = None,
    non_interactive: bool = False,
    cfg: AppConfig | None = None,
    max_steps: int = 8,
    step_budget_runtime: StepBudgetRuntime | None = None,
) -> dict[str, ToolDef]:
    recording_store = store or _RecordingStore()
    effective_cfg = cfg or AppConfig(model="test-model")
    return build_tools(
        root=tmp_path,
        console=None,
        surface=surface,
        store=recording_store,  # type: ignore[arg-type]
        mode=mode,
        yes=True,
        cfg=effective_cfg,
        api_key="test-key",
        max_steps=max_steps,
        subagents_enabled=subagents_enabled,
        subagent_depth=subagent_depth,
        subagent_registry=subagent_registry,
        usage_summary=usage_summary,
        non_interactive=non_interactive,
        step_budget_runtime=step_budget_runtime,
    )


def _usage_record(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    usage_source: str,
    cost_usd: float | None = None,
) -> UsageRecord:
    return UsageRecord(
        timestamp="2026-03-09T00:00:00+00:00",
        role="main:subagent:sandboxed",
        requested_model=model,
        response_model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        input_cost_per_token=0.1 if cost_usd is not None else None,
        output_cost_per_token=0.2 if cost_usd is not None else None,
        cost_usd=cost_usd,
        usage_source=usage_source,
    )


def _readonly_subagent_tools() -> dict[str, ToolDef]:
    return {
        "fs_read": ToolDef(
            name="fs_read",
            description="read",
            parameters={"type": "object", "properties": {}, "required": []},
            run=lambda _args: {"ok": True},
        )
    }


def test_subagent_tool_toggle_presence(tmp_path: Path) -> None:
    tools_disabled = _build_main_tools(tmp_path=tmp_path, subagents_enabled=False)
    tools_enabled = _build_main_tools(tmp_path=tmp_path, subagents_enabled=True)

    assert "subagent_run" not in tools_disabled
    assert "subagent_run" in tools_enabled


def test_build_tools_non_one_shot_does_not_require_full_session_store_shape(
    tmp_path: Path,
) -> None:
    store = _RecordingStore()

    assert not hasattr(store, "path")
    assert not hasattr(store, "session_artifact_root")

    tools = _build_main_tools(
        tmp_path=tmp_path,
        subagents_enabled=True,
        store=store,
    )

    assert "fs_read" in tools
    assert "subagent_run" in tools


def test_subagent_runtime_guard_when_disabled_reports_clear_error(tmp_path: Path) -> None:
    tools = _build_main_tools(tmp_path=tmp_path, subagents_enabled=True)
    subagent_run = tools["subagent_run"].run
    guarded_run = _rewrite_closure(subagent_run, subagents_enabled=False)

    result = guarded_run({"name": "explorer", "task": "Summarize src layout"})
    assert result == {"error": "Subagents are disabled for this session."}


def test_subagent_recursion_is_blocked_and_unregistered_for_nested_depth(tmp_path: Path) -> None:
    nested_tools = _build_main_tools(
        tmp_path=tmp_path,
        subagents_enabled=True,
        subagent_depth=1,
    )
    assert "subagent_run" not in nested_tools

    tools = _build_main_tools(tmp_path=tmp_path, subagents_enabled=True)
    subagent_run = tools["subagent_run"].run
    nested_guard_run = _rewrite_closure(subagent_run, subagent_depth=1)
    result = nested_guard_run({"name": "explorer", "task": "Inspect files"})
    assert result == {"error": "Subagents cannot invoke subagents (nesting is blocked)."}


def test_subagent_allowlist_denylist_and_default_readonly_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, Any] = {}
    fake_sub_session = _FakeSubSession(
        tools={
            "fs_read": ToolDef(
                name="fs_read",
                description="read",
                parameters={"type": "object", "properties": {}, "required": []},
                run=lambda _args: {"ok": True},
            ),
            "fs_write": ToolDef(
                name="fs_write",
                description="write",
                parameters={"type": "object", "properties": {}, "required": []},
                run=lambda _args: {"ok": True},
            ),
            "subagent_run": ToolDef(
                name="subagent_run",
                description="recursive",
                parameters={"type": "object", "properties": {}, "required": []},
                run=lambda _args: {"ok": True},
            ),
        },
        messages=[
            {"role": "tool", "content": "INTERMEDIATE-TOOL-OUTPUT"},
            {"role": "assistant", "content": "Final summarized answer"},
        ],
    )

    def _fake_create_session(**kwargs: Any) -> _FakeSubSession:
        captured_kwargs.update(kwargs)
        return fake_sub_session

    monkeypatch.setattr(agent_loop, "create_session", _fake_create_session)

    registry = {
        "sandboxed": SubagentDefinition(
            name="sandboxed",
            description="sandboxed test agent",
            system_prompt="You are sandboxed.",
            prompt_trust="untrusted",
            mode="readonly",
            allow_tools=("fs_read", "fs_write", "subagent_run"),
            deny_tools=("fs_write",),
        )
    }
    recording_store = _RecordingStore()
    tools = _build_main_tools(
        tmp_path=tmp_path,
        subagents_enabled=True,
        subagent_registry=registry,
        store=recording_store,
    )

    result = tools["subagent_run"].run({"name": "sandboxed", "task": "Inspect repository"})

    assert captured_kwargs["mode"] == "readonly"
    assert captured_kwargs["runtime_kind"] == RuntimeKind.SUBAGENT
    assert captured_kwargs["subagents_enabled"] is False
    assert captured_kwargs["subagent_depth"] == 1
    assert captured_kwargs["one_shot_execution"] is False
    assert captured_kwargs.get("trusted_system_prompt_override") is None
    assert captured_kwargs.get("trusted_system_prompt_append") is None
    assert captured_kwargs["untrusted_prompt_prelude"] == "You are sandboxed."
    assert fake_sub_session.run_calls == ["Inspect repository"]
    assert result["result"] == "Final summarized answer"
    assert result["sandbox"]["mode"] == "readonly"
    assert result["sandbox"]["tools"] == ["fs_read"]
    assert "INTERMEDIATE-TOOL-OUTPUT" not in json.dumps(result)
    assert {event_type for event_type, _ in recording_store.events} == {
        "subagent_start",
        "subagent_end",
    }


def test_subagent_mode_override_applies_for_single_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, Any] = {}

    def _fake_create_session(**kwargs: Any) -> _FakeSubSession:
        captured_kwargs.update(kwargs)
        return _FakeSubSession(
            tools={
                "fs_read": ToolDef(
                    name="fs_read",
                    description="read",
                    parameters={"type": "object", "properties": {}, "required": []},
                    run=lambda _args: {"ok": True},
                )
            }
        )

    monkeypatch.setattr(agent_loop, "create_session", _fake_create_session)

    registry = {
        "sandboxed": SubagentDefinition(
            name="sandboxed",
            description="sandboxed test agent",
            system_prompt="You are sandboxed.",
            mode="readonly",
        )
    }
    tools = _build_main_tools(
        tmp_path=tmp_path,
        subagents_enabled=True,
        subagent_registry=registry,
    )

    _ = tools["subagent_run"].run(
        {"name": "sandboxed", "task": "Inspect repository", "mode": "auto"}
    )
    assert captured_kwargs["mode"] == "auto"


@pytest.mark.parametrize(
    ("parent_mode", "requested_mode", "expected_mode"),
    [
        ("readonly", "auto", None),
        ("review", "auto", "review"),
        ("auto", "fullaccess", "auto"),
        ("fullaccess", "fullaccess", "fullaccess"),
    ],
)
def test_subagent_mode_request_is_capped_by_parent_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_mode: str,
    requested_mode: str,
    expected_mode: str | None,
) -> None:
    captured_kwargs: dict[str, Any] = {}

    def _fake_create_session(**kwargs: Any) -> _FakeSubSession:
        captured_kwargs.update(kwargs)
        return _FakeSubSession(
            tools={
                "fs_read": ToolDef(
                    name="fs_read",
                    description="read",
                    parameters={"type": "object", "properties": {}, "required": []},
                    run=lambda _args: {"ok": True},
                )
            }
        )

    monkeypatch.setattr(agent_loop, "create_session", _fake_create_session)

    registry = {
        "sandboxed": SubagentDefinition(
            name="sandboxed",
            description="sandboxed test agent",
            system_prompt="You are sandboxed.",
            mode="readonly",
        )
    }
    tools = _build_main_tools(
        tmp_path=tmp_path,
        subagents_enabled=True,
        mode=parent_mode,
        subagent_registry=registry,
    )

    if expected_mode is None:
        assert "subagent_run" not in tools
        return

    _ = tools["subagent_run"].run(
        {"name": "sandboxed", "task": "Inspect repository", "mode": requested_mode}
    )

    assert captured_kwargs["mode"] == expected_mode


def test_subagent_definition_mode_is_capped_by_parent_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, Any] = {}

    def _fake_create_session(**kwargs: Any) -> _FakeSubSession:
        captured_kwargs.update(kwargs)
        return _FakeSubSession(
            tools={
                "fs_read": ToolDef(
                    name="fs_read",
                    description="read",
                    parameters={"type": "object", "properties": {}, "required": []},
                    run=lambda _args: {"ok": True},
                )
            }
        )

    monkeypatch.setattr(agent_loop, "create_session", _fake_create_session)

    registry = {
        "sandboxed": SubagentDefinition(
            name="sandboxed",
            description="sandboxed test agent",
            system_prompt="You are sandboxed.",
            mode="fullaccess",
        )
    }
    tools = _build_main_tools(
        tmp_path=tmp_path,
        subagents_enabled=True,
        mode="review",
        subagent_registry=registry,
    )

    _ = tools["subagent_run"].run({"name": "sandboxed", "task": "Inspect repository"})
    assert captured_kwargs["mode"] == "review"


@pytest.mark.parametrize(
    ("subagent_name", "expected_max_steps"),
    [
        ("explorer", 12),
        ("reviewer", 10),
        ("test-strategist", 11),
    ],
)
def test_subagent_profile_defaults_use_adaptive_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    subagent_name: str,
    expected_max_steps: int,
) -> None:
    captured_kwargs: dict[str, Any] = {}

    def _fake_create_session(**kwargs: Any) -> _FakeSubSession:
        captured_kwargs.update(kwargs)
        return _FakeSubSession(tools=_readonly_subagent_tools())

    monkeypatch.setattr(agent_loop, "create_session", _fake_create_session)

    recording_store = _RecordingStore()
    tools = _build_main_tools(
        tmp_path=tmp_path,
        subagents_enabled=True,
        subagent_registry=built_in_subagents(),
        store=recording_store,
        max_steps=40,
        step_budget_runtime=StepBudgetRuntime(active_turn_budget=20),
    )

    result = tools["subagent_run"].run({"name": subagent_name, "task": "Inspect repository"})
    start_payload = _last_store_event_payload(recording_store, "subagent_start")

    assert result["result"] == "subagent final"
    assert captured_kwargs["max_steps"] == expected_max_steps
    assert captured_kwargs["enable_chat_turn_step_budget"] is False
    assert start_payload["max_steps"] == expected_max_steps
    assert start_payload["parent_turn_budget"] == 20
    assert start_payload["step_budget"]["reason"] == "adaptive_subagent"
    assert start_payload["step_budget"]["profile"] == subagent_name


def test_subagent_explicit_max_steps_uses_fixed_override_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, Any] = {}

    def _fake_create_session(**kwargs: Any) -> _FakeSubSession:
        captured_kwargs.update(kwargs)
        return _FakeSubSession(tools=_readonly_subagent_tools())

    monkeypatch.setattr(agent_loop, "create_session", _fake_create_session)

    recording_store = _RecordingStore()
    tools = _build_main_tools(
        tmp_path=tmp_path,
        subagents_enabled=True,
        subagent_registry=built_in_subagents(),
        store=recording_store,
        max_steps=40,
        step_budget_runtime=StepBudgetRuntime(active_turn_budget=20),
    )

    _ = tools["subagent_run"].run(
        {"name": "reviewer", "task": "Inspect repository", "max_steps": 7}
    )
    start_payload = _last_store_event_payload(recording_store, "subagent_start")

    assert captured_kwargs["max_steps"] == 7
    assert start_payload["max_steps"] == 7
    assert start_payload["step_budget"]["resolved_max_steps"] == 7
    assert start_payload["step_budget"]["reason"] == "fixed_override"
    assert start_payload["step_budget"]["override_applied"] is True


def test_subagent_budget_clamps_to_cfg_subagent_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, Any] = {}

    def _fake_create_session(**kwargs: Any) -> _FakeSubSession:
        captured_kwargs.update(kwargs)
        return _FakeSubSession(tools=_readonly_subagent_tools())

    monkeypatch.setattr(agent_loop, "create_session", _fake_create_session)

    recording_store = _RecordingStore()
    tools = _build_main_tools(
        tmp_path=tmp_path,
        subagents_enabled=True,
        subagent_registry=built_in_subagents(),
        store=recording_store,
        cfg=AppConfig(model="test-model", subagent_max_steps=9),
        max_steps=40,
        step_budget_runtime=StepBudgetRuntime(active_turn_budget=20),
    )

    _ = tools["subagent_run"].run({"name": "explorer", "task": "Inspect repository"})
    start_payload = _last_store_event_payload(recording_store, "subagent_start")

    assert captured_kwargs["max_steps"] == 9
    assert start_payload["max_steps"] == 9
    assert start_payload["step_budget"]["hard_cap"] == 9
    assert start_payload["step_budget"]["resolved_max_steps"] == 9


def test_subagent_budget_clamps_to_active_parent_turn_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, Any] = {}

    def _fake_create_session(**kwargs: Any) -> _FakeSubSession:
        captured_kwargs.update(kwargs)
        return _FakeSubSession(tools=_readonly_subagent_tools())

    monkeypatch.setattr(agent_loop, "create_session", _fake_create_session)

    recording_store = _RecordingStore()
    tools = _build_main_tools(
        tmp_path=tmp_path,
        subagents_enabled=True,
        subagent_registry=built_in_subagents(),
        store=recording_store,
        max_steps=40,
        step_budget_runtime=StepBudgetRuntime(active_turn_budget=10),
    )

    _ = tools["subagent_run"].run({"name": "explorer", "task": "Inspect repository"})
    start_payload = _last_store_event_payload(recording_store, "subagent_start")

    assert captured_kwargs["max_steps"] == 10
    assert start_payload["parent_turn_budget"] == 10
    assert start_payload["step_budget"]["hard_cap"] == 10
    assert start_payload["step_budget"]["resolved_max_steps"] == 10


def test_subagent_budget_falls_back_to_parent_session_max_steps_without_active_turn_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, Any] = {}

    def _fake_create_session(**kwargs: Any) -> _FakeSubSession:
        captured_kwargs.update(kwargs)
        return _FakeSubSession(tools=_readonly_subagent_tools())

    monkeypatch.setattr(agent_loop, "create_session", _fake_create_session)

    recording_store = _RecordingStore()
    tools = _build_main_tools(
        tmp_path=tmp_path,
        subagents_enabled=True,
        subagent_registry=built_in_subagents(),
        store=recording_store,
        max_steps=10,
        step_budget_runtime=StepBudgetRuntime(),
    )

    _ = tools["subagent_run"].run({"name": "explorer", "task": "Inspect repository"})
    start_payload = _last_store_event_payload(recording_store, "subagent_start")

    assert captured_kwargs["max_steps"] == 10
    assert start_payload["parent_turn_budget"] == 10
    assert start_payload["step_budget"]["hard_cap"] == 10
    assert start_payload["step_budget"]["resolved_max_steps"] == 10


def test_subagent_start_payload_includes_step_budget_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_create_session(**_kwargs: Any) -> _FakeSubSession:
        return _FakeSubSession(tools=_readonly_subagent_tools())

    monkeypatch.setattr(agent_loop, "create_session", _fake_create_session)

    recording_store = _RecordingStore()
    tools = _build_main_tools(
        tmp_path=tmp_path,
        subagents_enabled=True,
        subagent_registry=built_in_subagents(),
        store=recording_store,
        max_steps=40,
        step_budget_runtime=StepBudgetRuntime(active_turn_budget=20),
    )

    _ = tools["subagent_run"].run(
        {
            "name": "test-strategist",
            "task": "Inspect src/sylliptor_agent_cli/agent_loop.py and tests/test_subagents.py",
        }
    )
    start_payload = _last_store_event_payload(recording_store, "subagent_start")

    assert start_payload["max_steps"] == 13
    assert start_payload["parent_turn_budget"] == 20
    assert start_payload["step_budget"]["kind"] == "subagent"
    assert start_payload["step_budget"]["profile"] == "test-strategist"
    assert start_payload["step_budget"]["signals_used"]["explicit_path_count"] == 2


def test_subagent_trusted_prompt_uses_system_append_not_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, Any] = {}

    def _fake_create_session(**kwargs: Any) -> _FakeSubSession:
        captured_kwargs.update(kwargs)
        return _FakeSubSession(
            tools={
                "fs_read": ToolDef(
                    name="fs_read",
                    description="read",
                    parameters={"type": "object", "properties": {}, "required": []},
                    run=lambda _args: {"ok": True},
                )
            }
        )

    monkeypatch.setattr(agent_loop, "create_session", _fake_create_session)

    registry = {
        "sandboxed": SubagentDefinition(
            name="sandboxed",
            description="trusted built-in style subagent",
            system_prompt="You are sandboxed.",
            prompt_trust="trusted",
            mode="readonly",
        )
    }
    tools = _build_main_tools(
        tmp_path=tmp_path,
        subagents_enabled=True,
        subagent_registry=registry,
    )

    _ = tools["subagent_run"].run({"name": "sandboxed", "task": "Inspect repository"})

    assert captured_kwargs.get("trusted_system_prompt_override") is None
    assert captured_kwargs["trusted_system_prompt_append"] == "You are sandboxed."
    assert captured_kwargs.get("untrusted_prompt_prelude") is None


def test_subagent_review_mode_forwards_approvals_without_nested_surface_noise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, Any] = {}
    parent_surface = _RecordingApprovalSurface(allow=True)

    def _fake_create_session(**kwargs: Any) -> _ChildToolSession:
        captured_kwargs.update(kwargs)
        child_tools = build_tools(
            root=tmp_path,
            console=None,
            surface=kwargs["surface"],
            store=_RecordingStore(),  # type: ignore[arg-type]
            mode=kwargs["mode"],
            yes=kwargs["yes"],
            cfg=AppConfig(model="test-model"),
            api_key="test-key",
            max_steps=4,
            subagents_enabled=False,
            non_interactive=kwargs["non_interactive"],
        )
        return _ChildToolSession(tools=child_tools, surface=kwargs["surface"])

    monkeypatch.setattr(agent_loop, "create_session", _fake_create_session)

    registry = {
        "sandboxed": SubagentDefinition(
            name="sandboxed",
            description="sandboxed test agent",
            system_prompt="You are sandboxed.",
            mode="auto",
        )
    }

    tools = _build_main_tools(
        tmp_path=tmp_path,
        subagents_enabled=True,
        mode="review",
        subagent_registry=registry,
        surface=parent_surface,
    )

    result = tools["subagent_run"].run({"name": "sandboxed", "task": "Write approved file"})

    assert captured_kwargs["mode"] == "review"
    assert result["result"] == "approved write complete"
    assert (tmp_path / "approved.txt").read_text(encoding="utf-8") == "ok\n"
    assert [request.kind for request in parent_surface.approval_requests] == ["fs_write"]
    assert parent_surface.noise_events == []


def test_subagent_live_tool_events_are_forwarded_to_parent_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_surface = _RecordingNestedSurface()

    def _fake_create_session(**kwargs: Any) -> _ChildToolTraceSession:
        child_tools = {
            "fs_read": ToolDef(
                name="fs_read",
                description="read",
                parameters={"type": "object", "properties": {}, "required": []},
                run=lambda _args: {"ok": True},
            )
        }
        return _ChildToolTraceSession(surface=kwargs["surface"], tools=child_tools)

    monkeypatch.setattr(agent_loop, "create_session", _fake_create_session)

    registry = {
        "explorer": SubagentDefinition(
            name="explorer",
            description="sandboxed explorer",
            system_prompt="You are explorer.",
            mode="readonly",
            allow_tools=("fs_read",),
        )
    }

    tools = _build_main_tools(
        tmp_path=tmp_path,
        subagents_enabled=True,
        mode="auto",
        subagent_registry=registry,
        surface=parent_surface,
    )

    result = tools["subagent_run"].run({"name": "explorer", "task": "Inspect README"})

    assert result["result"] == "nested trace complete"
    assert [event.name for event in parent_surface.subagent_starts] == ["explorer"]
    assert parent_surface.subagent_starts[0].mode == "readonly"
    assert len(parent_surface.tool_starts) == 1
    assert parent_surface.tool_starts[0].subagent_name == "explorer"
    assert parent_surface.tool_starts[0].subagent_mode == "readonly"
    assert parent_surface.tool_starts[0].nesting_depth == 1
    assert parent_surface.tool_starts[0].tool_call_id.startswith("subagent:explorer:")
    assert len(parent_surface.tool_outputs) == 1
    assert parent_surface.tool_outputs[0].subagent_name == "explorer"
    assert len(parent_surface.tool_ends) == 1
    assert parent_surface.tool_ends[0].subagent_name == "explorer"
    assert len(parent_surface.subagent_ends) == 1
    assert parent_surface.subagent_ends[0].status == "success"
    assert parent_surface.subagent_ends[0].steps_completed == 1


def test_subagent_review_mode_non_interactive_fails_fast_without_approval_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_surface = _RecordingApprovalSurface(allow=True)

    def _fake_create_session(**kwargs: Any) -> _ChildToolSession:
        child_tools = build_tools(
            root=tmp_path,
            console=None,
            surface=kwargs["surface"],
            store=_RecordingStore(),  # type: ignore[arg-type]
            mode=kwargs["mode"],
            yes=kwargs["yes"],
            cfg=AppConfig(model="test-model"),
            api_key="test-key",
            max_steps=4,
            subagents_enabled=False,
            non_interactive=kwargs["non_interactive"],
        )
        return _ChildToolSession(tools=child_tools, surface=kwargs["surface"])

    monkeypatch.setattr(agent_loop, "create_session", _fake_create_session)

    registry = {
        "sandboxed": SubagentDefinition(
            name="sandboxed",
            description="sandboxed test agent",
            system_prompt="You are sandboxed.",
            mode="review",
        )
    }

    tools = _build_main_tools(
        tmp_path=tmp_path,
        subagents_enabled=True,
        mode="review",
        subagent_registry=registry,
        surface=parent_surface,
        non_interactive=True,
    )

    result = tools["subagent_run"].run({"name": "sandboxed", "task": "Write approved file"})

    assert "Confirmation required for sensitive command" in str(result.get("error") or "")
    assert parent_surface.approval_requests == []
    assert not (tmp_path / "approved.txt").exists()


def test_subagent_usage_replays_each_child_record_into_parent_summary_and_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_usage = UsageSummary()
    recording_store = _RecordingStore()
    child_usage = UsageSummary()
    child_usage.add_record(
        _usage_record(
            model="model-a",
            prompt_tokens=11,
            completion_tokens=5,
            usage_source="api",
            cost_usd=2.1,
        )
    )
    child_usage.add_record(
        _usage_record(
            model="model-a",
            prompt_tokens=7,
            completion_tokens=3,
            usage_source="estimate",
            cost_usd=None,
        )
    )

    fake_sub_session = _FakeSubSession(
        tools={
            "fs_read": ToolDef(
                name="fs_read",
                description="read",
                parameters={"type": "object", "properties": {}, "required": []},
                run=lambda _args: {"ok": True},
            )
        },
        usage_summary=child_usage,
        messages=[{"role": "assistant", "content": "Subagent final answer"}],
    )

    def _fake_create_session(**_kwargs: Any) -> _FakeSubSession:
        return fake_sub_session

    monkeypatch.setattr(agent_loop, "create_session", _fake_create_session)

    registry = {
        "sandboxed": SubagentDefinition(
            name="sandboxed",
            description="sandboxed test agent",
            system_prompt="You are sandboxed.",
            mode="readonly",
        )
    }

    tools = _build_main_tools(
        tmp_path=tmp_path,
        subagents_enabled=True,
        mode="auto",
        subagent_registry=registry,
        store=recording_store,
        usage_summary=parent_usage,
    )

    result = tools["subagent_run"].run({"name": "sandboxed", "task": "Inspect repository"})

    assert result["result"] == "Subagent final answer"
    totals = parent_usage.totals()
    assert totals["prompt_tokens"] == 18
    assert totals["completion_tokens"] == 8
    assert totals["total_tokens"] == 26
    assert totals["calls"] == 2
    assert totals["api_usage_calls"] == 1
    assert totals["estimate_usage_calls"] == 1
    llm_usage_events = [
        payload for event_type, payload in recording_store.events if event_type == "llm_usage"
    ]
    assert len(llm_usage_events) == 2
    assert [payload["usage_source"] for payload in llm_usage_events] == ["api", "estimate"]


def test_failed_subagent_run_still_replays_child_usage_into_parent_summary_and_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_usage = UsageSummary()
    recording_store = _RecordingStore()
    child_usage = UsageSummary()
    child_usage.add_record(
        _usage_record(
            model="model-b",
            prompt_tokens=13,
            completion_tokens=4,
            usage_source="api",
            cost_usd=1.7,
        )
    )
    child_usage.add_record(
        _usage_record(
            model="model-b",
            prompt_tokens=6,
            completion_tokens=2,
            usage_source="estimate",
            cost_usd=None,
        )
    )

    class _FailingSubSession(_FakeSubSession):
        def run_turn(self, task: str) -> int:
            self.run_calls.append(task)
            raise RuntimeError("child exploded")

    fake_sub_session = _FailingSubSession(
        tools={
            "fs_read": ToolDef(
                name="fs_read",
                description="read",
                parameters={"type": "object", "properties": {}, "required": []},
                run=lambda _args: {"ok": True},
            )
        },
        usage_summary=child_usage,
        messages=[{"role": "assistant", "content": "Subagent partial answer"}],
    )

    def _fake_create_session(**_kwargs: Any) -> _FakeSubSession:
        return fake_sub_session

    monkeypatch.setattr(agent_loop, "create_session", _fake_create_session)

    registry = {
        "sandboxed": SubagentDefinition(
            name="sandboxed",
            description="sandboxed test agent",
            system_prompt="You are sandboxed.",
            mode="readonly",
        )
    }

    tools = _build_main_tools(
        tmp_path=tmp_path,
        subagents_enabled=True,
        mode="auto",
        subagent_registry=registry,
        store=recording_store,
        usage_summary=parent_usage,
    )

    result = tools["subagent_run"].run({"name": "sandboxed", "task": "Inspect repository"})

    assert "execution failed: child exploded" in str(result.get("error") or "")
    totals = parent_usage.totals()
    assert totals["prompt_tokens"] == 19
    assert totals["completion_tokens"] == 6
    assert totals["total_tokens"] == 25
    assert totals["calls"] == 2
    assert totals["api_usage_calls"] == 1
    assert totals["estimate_usage_calls"] == 1
    llm_usage_events = [
        payload for event_type, payload in recording_store.events if event_type == "llm_usage"
    ]
    assert len(llm_usage_events) == 2
    assert any(
        event_type == "subagent_end" and payload.get("status") == "failed"
        for event_type, payload in recording_store.events
    )


def test_subagent_loader_discovers_project_and_user_agent_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_agents = tmp_path / ".sylliptor_agents"
    project_agents.mkdir(parents=True, exist_ok=True)
    (project_agents / "project_agent.md").write_text(
        "---\n"
        "name: project-agent\n"
        "description: Project custom agent\n"
        "allow_tools:\n"
        "  - fs_read\n"
        "---\n"
        "You are the project agent.\n",
        encoding="utf-8",
    )
    (project_agents / "disabled_agent.md").write_text(
        "---\nname: disabled-agent\nenabled: false\n---\nThis should not load.\n",
        encoding="utf-8",
    )

    fake_user_config_root = tmp_path / "user-config"
    user_agents = fake_user_config_root / "agents"
    user_agents.mkdir(parents=True, exist_ok=True)
    (user_agents / "user_agent.md").write_text(
        "---\nname: user-agent\ndescription: User custom agent\n---\nYou are the user agent.\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sylliptor_agent_cli.subagents.user_config_dir",
        lambda appname, appauthor: str(fake_user_config_root),
    )

    registry = load_subagent_registry(root=tmp_path)

    assert "explorer" in registry
    assert "reviewer" in registry
    assert "test-strategist" in registry
    assert "project-agent" in registry
    assert "user-agent" in registry
    assert "disabled-agent" not in registry
    assert registry["user-agent"].mode == "readonly"
    assert registry["project-agent"].prompt_trust == "untrusted"
    assert registry["user-agent"].prompt_trust == "untrusted"


def test_built_in_subagents_allow_navigation_tools() -> None:
    registry = built_in_subagents()

    assert "fs_read_lines" in registry["explorer"].allow_tools
    assert "fs_read_lines" in registry["reviewer"].allow_tools
    assert "fs_read_lines" in registry["test-strategist"].allow_tools
    assert "history_search" in registry["explorer"].allow_tools
    assert "history_search" in registry["reviewer"].allow_tools
    assert "history_search" in registry["test-strategist"].allow_tools
    assert "symbol_search" in registry["explorer"].allow_tools
    assert "symbol_search" in registry["reviewer"].allow_tools
    assert "symbol_search" in registry["test-strategist"].allow_tools
    assert "git_history" in registry["explorer"].allow_tools
    assert "git_history" in registry["reviewer"].allow_tools
    assert "git_history" in registry["test-strategist"].allow_tools
    assert "web_search" not in registry["explorer"].allow_tools
    assert "web_search" not in registry["reviewer"].allow_tools
    assert "web_search" not in registry["test-strategist"].allow_tools


def test_subagent_tool_schema_includes_name_enum_when_registry_is_available(
    tmp_path: Path,
) -> None:
    registry = {
        "alpha": SubagentDefinition(
            name="alpha",
            description="alpha agent",
            system_prompt="You are alpha.",
        ),
        "beta": SubagentDefinition(
            name="beta",
            description="beta agent",
            system_prompt="You are beta.",
        ),
    }
    tools = _build_main_tools(
        tmp_path=tmp_path,
        subagents_enabled=True,
        subagent_registry=registry,
    )

    schema = tools["subagent_run"].as_openai_tool()["function"]["parameters"]["properties"]["name"]
    assert schema["type"] == "string"
    assert schema["enum"] == ["alpha", "beta"]


def test_subagent_loader_supports_claude_style_tool_aliases(tmp_path: Path) -> None:
    project_agents = tmp_path / ".sylliptor_agents"
    project_agents.mkdir(parents=True, exist_ok=True)
    (project_agents / "claude_alias.md").write_text(
        "---\n"
        "name: claude-alias\n"
        "tools:\n"
        "  - fs_read\n"
        "  - search_rg\n"
        "disallowedTools:\n"
        "  - search_rg\n"
        "---\n"
        "You are a claude-style custom agent.\n",
        encoding="utf-8",
    )

    registry = load_subagent_registry(root=tmp_path)
    alias_agent = registry["claude-alias"]

    assert alias_agent.allow_tools == ("fs_read", "search_rg")
    assert alias_agent.deny_tools == ("search_rg",)


def test_create_session_injects_subagent_context_when_enabled(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
        subagents_enabled=True,
        subagent_depth=0,
    )
    try:
        user_messages = [
            str(message.get("content") or "")
            for message in session.messages
            if str(message.get("role") or "") == "user"
        ]
        subagent_context = next(
            (text for text in user_messages if "<subagent_context>" in text),
            "",
        )
        assert subagent_context
        assert "subagents_enabled: true" in subagent_context
        assert "explorer" in subagent_context
        assert "reviewer" in subagent_context
        assert "test-strategist" in subagent_context

        subagent_idx = next(
            i for i, text in enumerate(user_messages) if "<subagent_context>" in text
        )
        env_idx = next(i for i, text in enumerate(user_messages) if "<environment_context>" in text)
        assert subagent_idx < env_idx
    finally:
        session.close()


def test_create_session_does_not_inject_subagent_context_when_disabled(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
        subagents_enabled=False,
    )
    try:
        assert not any(
            "<subagent_context>" in str(message.get("content") or "")
            for message in session.messages
            if str(message.get("role") or "") == "user"
        )
    finally:
        session.close()


def test_create_session_does_not_inject_subagent_context_for_nested_subagent(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
        subagents_enabled=True,
        subagent_depth=1,
    )
    try:
        assert not any(
            "<subagent_context>" in str(message.get("content") or "")
            for message in session.messages
            if str(message.get("role") or "") == "user"
        )
    finally:
        session.close()


def test_create_session_appends_subagent_system_guidance_when_enabled(tmp_path: Path) -> None:
    cfg = AppConfig(model="test-model")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
        subagents_enabled=True,
        subagent_depth=0,
    )
    try:
        system_prompt = next(
            (
                str(message.get("content") or "")
                for message in session.messages
                if str(message.get("role") or "") == "system"
            ),
            "",
        )
        assert "Subagent delegation" in system_prompt
        assert (
            "Run unrelated investigations in parallel in one tool batch instead of serializing them."
            in system_prompt
        )
    finally:
        session.close()


def test_create_session_omits_subagent_system_guidance_when_subagents_unavailable(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model")

    disabled_session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
        subagents_enabled=False,
    )
    try:
        disabled_prompt = next(
            (
                str(message.get("content") or "")
                for message in disabled_session.messages
                if str(message.get("role") or "") == "system"
            ),
            "",
        )
        assert "Subagent delegation" not in disabled_prompt
    finally:
        disabled_session.close()

    nested_session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
        subagents_enabled=True,
        subagent_depth=1,
    )
    try:
        nested_prompt = next(
            (
                str(message.get("content") or "")
                for message in nested_session.messages
                if str(message.get("role") or "") == "system"
            ),
            "",
        )
        assert "Subagent delegation" not in nested_prompt
    finally:
        nested_session.close()


def test_create_session_untrusted_prompt_prelude_preserves_base_system_prompt(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="test-model")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
        untrusted_prompt_prelude="Custom subagent markdown guidance.",
    )
    try:
        system_prompt = str(session.messages[0]["content"] or "")
        assert SYSTEM_PROMPT.splitlines()[0] in system_prompt
        assert "Never exfiltrate, disclose, simulate, or infer secrets" in system_prompt
        assert "Custom subagent markdown guidance." not in system_prompt

        prelude_message = next(
            (
                str(message.get("content") or "")
                for message in session.messages
                if "<scoped_prompt_prelude>" in str(message.get("content") or "")
            ),
            "",
        )
        assert "Custom subagent markdown guidance." in prelude_message
        assert "higher-priority system, developer, and direct user instructions" in prelude_message
    finally:
        session.close()
