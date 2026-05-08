from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sylliptor_agent_cli import agent_loop as agent_loop_mod
from sylliptor_agent_cli.agent_loop import create_session
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.llm.openai_compat import LLMResponse, ToolCall
from sylliptor_agent_cli.session_store import read_session_events


class _LoopingToolClient:
    model = "test-model"
    temperature = 0.2

    def __init__(self) -> None:
        self.calls = 0
        self.tool_enabled_calls = 0
        self.call_tools: list[list[dict[str, Any]] | None] = []

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        on_text_delta=None,  # type: ignore[no-untyped-def]
        temperature: float | None = None,
    ) -> LLMResponse:
        _ = messages, stream, on_text_delta, temperature
        self.calls += 1
        self.call_tools.append(tools)
        if tools is None:
            return LLMResponse(
                content=(
                    "Completed work: partial repo inspection.\n"
                    "Remaining work: implementation is unfinished.\n"
                    "Known issues or risks: step budget exhausted."
                ),
                tool_calls=[],
                raw={},
            )
        self.tool_enabled_calls += 1
        return LLMResponse(
            content="Still working.",
            tool_calls=[
                ToolCall(
                    id=f"call-{self.tool_enabled_calls}",
                    name="fs_list",
                    arguments={"path": "."},
                )
            ],
            raw={},
        )


class _FinalReplyClient:
    model = "test-model"
    temperature = 0.2

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

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
        self.calls += 1
        return LLMResponse(content=self.reply, tool_calls=[], raw={})


def _route_decision(route: str) -> SimpleNamespace:
    return SimpleNamespace(
        route=route,
        confidence=0.99,
        language="",
        script="",
        explicit_language_override=False,
    )


def _event_payloads(path: Path, event_type: str) -> list[dict[str, Any]]:
    return [
        dict(event.get("payload") or {})
        for event in read_session_events(path)
        if event.get("type") == event_type
    ]


def test_repo_turn_emits_resolution_event_and_uses_resolved_ceiling(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(agent_loop_mod, "_route_turn", lambda **_kwargs: _route_decision("repo"))
    cfg = AppConfig(model="test-model", routing_mode="auto", step_budget_policy="adaptive")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="review",
        yes=True,
        max_steps=32,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
        enable_chat_turn_step_budget=True,
        verification_enabled=False,
    )
    client = _LoopingToolClient()
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Keep working on the repo change.")
        active_turn_budget = session.step_budget_runtime.active_turn_budget
        last_resolution = session.step_budget_runtime.last_resolution
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 0
    assert last_resolution is not None
    assert last_resolution.resolved_max_steps < session.max_steps
    assert client.tool_enabled_calls == last_resolution.resolved_max_steps
    assert client.calls == last_resolution.resolved_max_steps + 1
    assert client.call_tools[-1] is None
    assert active_turn_budget == last_resolution.resolved_max_steps
    assert last_resolution is not None
    assert last_resolution.reason == "adaptive_chat_turn"
    resolution_events = _event_payloads(log_path, "turn_step_budget_resolved")
    assert len(resolution_events) == 1
    assert resolution_events[0]["resolved_max_steps"] == last_resolution.resolved_max_steps
    assert _event_payloads(log_path, "error") == []
    handoff_events = _event_payloads(log_path, "interactive_step_budget_handoff")
    assert handoff_events[-1]["max_steps"] == last_resolution.resolved_max_steps
    assert _event_payloads(log_path, "forced_final_summary_requested")


def test_simple_agent_fixed_override_uses_fixed_turn_ceiling(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(agent_loop_mod, "_route_turn", lambda **_kwargs: _route_decision("repo"))
    cfg = AppConfig(model="test-model", routing_mode="auto", step_budget_policy="adaptive")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="review",
        yes=True,
        max_steps=12,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
        enable_chat_turn_step_budget=True,
        chat_turn_fixed_override=4,
    )
    client = _LoopingToolClient()
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Keep working on the repo change.")
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 0
    assert client.tool_enabled_calls == 4
    assert client.calls == 5
    assert client.call_tools[-1] is None
    resolution_events = _event_payloads(log_path, "turn_step_budget_resolved")
    assert len(resolution_events) == 1
    assert resolution_events[0]["resolved_max_steps"] == 4
    assert resolution_events[0]["reason"] == "fixed_override"
    assert resolution_events[0]["override_applied"] is True
    assert _event_payloads(log_path, "error") == []
    assert _event_payloads(log_path, "interactive_step_budget_handoff")


def test_disabled_session_keeps_fixed_runtime_behavior(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(agent_loop_mod, "_route_turn", lambda **_kwargs: _route_decision("repo"))
    cfg = AppConfig(model="test-model", routing_mode="auto", step_budget_policy="adaptive")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="review",
        yes=True,
        max_steps=12,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
    )
    client = _LoopingToolClient()
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Keep working on the repo change.")
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 1
    assert client.tool_enabled_calls == 12
    assert client.calls == 13
    assert client.call_tools[-1] is None
    assert session.step_budget_runtime.active_turn_budget is None
    assert _event_payloads(log_path, "turn_step_budget_resolved") == []
    error_events = _event_payloads(log_path, "error")
    assert error_events[-1]["max_steps"] == 12


def test_non_repo_fast_path_clears_active_turn_budget_without_new_resolution_event(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_route_turn(*, instruction: str, **_kwargs: Any) -> SimpleNamespace:
        if "repo" in instruction:
            return _route_decision("repo")
        return _route_decision("chat")

    monkeypatch.setattr(agent_loop_mod, "_route_turn", fake_route_turn)
    monkeypatch.setattr(
        agent_loop_mod,
        "_respond_non_repo_turn",
        lambda **_kwargs: "I am doing well.",
    )
    cfg = AppConfig(model="test-model", routing_mode="auto", step_budget_policy="adaptive")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="review",
        yes=True,
        max_steps=32,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
        enable_chat_turn_step_budget=True,
        verification_enabled=False,
    )
    client = _FinalReplyClient("Blocked by missing requirements.")
    session.client = client  # type: ignore[assignment]

    try:
        first_exit_code = session.run_turn("Handle this repo task.")
        assert session.step_budget_runtime.last_resolution is not None
        assert (
            session.step_budget_runtime.active_turn_budget
            == session.step_budget_runtime.last_resolution.resolved_max_steps
        )
        assert session.step_budget_runtime.active_turn_budget < session.max_steps
        second_exit_code = session.run_turn("How are you?")
        active_turn_budget = session.step_budget_runtime.active_turn_budget
        log_path = session.store.path
    finally:
        session.close()

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert client.calls == 1
    assert active_turn_budget is None
    assert len(_event_payloads(log_path, "turn_step_budget_resolved")) == 1


def test_run_agent_defaults_chat_turn_budget_off(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _DummySession:
        def run_turn(self, instruction: str, *, image_paths: list[str] | None = None) -> int:
            captured["instruction"] = instruction
            captured["image_paths"] = image_paths
            return 0

        def close(self) -> None:
            captured["closed"] = True

    def fake_create_session(**kwargs: Any) -> _DummySession:
        captured.update(kwargs)
        return _DummySession()

    monkeypatch.setattr(agent_loop_mod, "create_session", fake_create_session)

    exit_code = agent_loop_mod.run_agent(
        cfg=AppConfig(model="test-model"),
        root=tmp_path,
        instruction="hi",
        mode="review",
        yes=True,
        max_steps=7,
        no_log=True,
        api_key_override="override-key",
    )

    assert exit_code == 0
    assert captured["enable_chat_turn_step_budget"] is False
    assert captured["chat_turn_fixed_override"] is None
    assert captured["closed"] is True
