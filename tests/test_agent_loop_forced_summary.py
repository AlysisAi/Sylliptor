from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from sylliptor_agent_cli.agent_loop import create_session
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.llm.openai_compat import LLMResponse, ToolCall
from sylliptor_agent_cli.session_store import read_session_events
from sylliptor_agent_cli.surface.noop_surface import NoopSurface

SMOKE_MODEL = "gpt-4o-mini"


class _ForcedSummarySurface(NoopSurface):
    def __init__(self) -> None:
        super().__init__()
        self.final_messages: list[str] = []
        self.errors: list[str] = []

    def on_assistant_message_done(self, text: str) -> None:
        self.final_messages.append(text)

    def on_error(self, err: str) -> None:
        self.errors.append(err)


class _BudgetExhaustionClient:
    model = SMOKE_MODEL
    temperature = 0.2

    def __init__(self, *, finalization_mode: str = "ok") -> None:
        self.finalization_mode = finalization_mode
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
        _ = on_text_delta, temperature
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "stream": stream,
            }
        )
        tool_enabled_calls = sum(1 for item in self.calls if item["tools"] is not None)
        if tools is None:
            if self.finalization_mode == "error":
                raise RuntimeError("forced finalization failed")
            if self.finalization_mode == "blank":
                return LLMResponse(content="   ", tool_calls=[], raw={})
            return LLMResponse(
                content=(
                    "Completed work: inspected the repo and made partial progress.\n"
                    "Remaining work: the requested change is unfinished.\n"
                    "Known issues or risks: the turn hit the step budget before completion."
                ),
                tool_calls=[],
                raw={},
            )
        return LLMResponse(
            content="Still working.",
            tool_calls=[
                ToolCall(
                    id=f"call-{tool_enabled_calls}",
                    name="fs_list",
                    arguments={"path": "."},
                )
            ],
            raw={},
        )


class _ScriptedClient:
    model = SMOKE_MODEL
    temperature = 0.2

    def __init__(
        self,
        responses: list[LLMResponse],
        *,
        finalization_response: LLMResponse | None = None,
    ) -> None:
        self._responses = responses
        self._finalization_response = finalization_response
        self.calls: list[dict[str, Any]] = []
        self._tool_enabled_calls = 0

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
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "stream": stream,
            }
        )
        if tools is None and self._finalization_response is not None:
            return self._finalization_response
        response = self._responses[self._tool_enabled_calls]
        self._tool_enabled_calls += 1
        return response


def _event_payloads(path: Path, event_type: str) -> list[dict[str, Any]]:
    return [
        dict(event.get("payload") or {})
        for event in read_session_events(path)
        if event.get("type") == event_type
    ]


def _assert_forced_summary_artifacts(
    *,
    log_path: Path,
    surface: _ForcedSummarySurface,
) -> str:
    assert _event_payloads(log_path, "forced_final_summary_requested")
    assert _event_payloads(log_path, "forced_final_summary_completed")
    assistant_events = _event_payloads(log_path, "assistant_message")
    final_events = _event_payloads(log_path, "final")
    assert assistant_events
    assert final_events
    summary = surface.final_messages[-1]
    assert assistant_events[-1]["content"] == summary
    assert final_events[-1]["content"] == summary
    return summary


def _assert_last_forced_summary_request(
    client: _ScriptedClient,
    *,
    latest_assistant_text: str,
    termination_cause: str,
) -> None:
    assert client.calls
    assert client.calls[-1]["tools"] is None
    request_messages = client.calls[-1]["messages"]
    assert request_messages[-2] == {
        "role": "assistant",
        "content": latest_assistant_text,
    }
    assert str(request_messages[-1].get("role")) == "system"
    assert f"Stop reason: {termination_cause}" in str(request_messages[-1].get("content"))


def _install_stub_subagent_run(session: Any, *, result_text: str) -> None:
    tool = session.tools["subagent_run"]

    def fake_run(args: dict[str, Any]) -> dict[str, Any]:
        return {
            "subagent": str(args.get("name") or "explorer"),
            "subagent_session_id": "stub-subagent",
            "result": result_text,
            "usage": {},
            "sandbox": {"mode": "readonly", "tools": ["fs_read"]},
        }

    session.tools["subagent_run"] = replace(tool, run=fake_run)


def test_matching_tool_bridge_text_and_final_emit_once(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    surface = _ForcedSummarySurface()
    session = create_session(
        cfg=AppConfig(model=SMOKE_MODEL, routing_mode="code_only", stream=False),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        session_id_override="assistant-dedupe-same-final",
        surface=surface,
    )
    client = _ScriptedClient(
        [
            LLMResponse(
                content="Done.",
                tool_calls=[ToolCall(id="tc1", name="fs_list", arguments={"path": "."})],
                raw={},
            ),
            LLMResponse(content="Done.", tool_calls=[], raw={}),
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Inspect the repo and finish the task.")
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 0
    assert surface.final_messages == ["Done."]
    assistant_events = _event_payloads(log_path, "assistant_message")
    final_events = _event_payloads(log_path, "final")
    assert [event["content"] for event in assistant_events] == ["Done."]
    assert [event["content"] for event in final_events] == ["Done."]
    assert assistant_events[0]["tool_calls"] == ["fs_list"]


def test_distinct_tool_bridge_text_and_final_both_emit(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    surface = _ForcedSummarySurface()
    session = create_session(
        cfg=AppConfig(model=SMOKE_MODEL, routing_mode="code_only", stream=False),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        session_id_override="assistant-dedupe-distinct-final",
        surface=surface,
    )
    client = _ScriptedClient(
        [
            LLMResponse(
                content="Inspecting files.",
                tool_calls=[ToolCall(id="tc1", name="fs_list", arguments={"path": "."})],
                raw={},
            ),
            LLMResponse(content="Done.", tool_calls=[], raw={}),
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Inspect the repo and finish the task.")
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 0
    assert surface.final_messages == ["Inspecting files.", "Done."]
    assistant_events = _event_payloads(log_path, "assistant_message")
    final_events = _event_payloads(log_path, "final")
    assert [event["content"] for event in assistant_events] == [
        "Inspecting files.",
        "Done.",
    ]
    assert [event["content"] for event in final_events] == ["Done."]


def test_generic_max_steps_exhaustion_emits_forced_final_summary(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    surface = _ForcedSummarySurface()
    session = create_session(
        cfg=AppConfig(model=SMOKE_MODEL, routing_mode="code_only"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=3,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        session_id_override="forced-summary-generic-max-steps",
        surface=surface,
    )
    client = _BudgetExhaustionClient()
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Keep working on the repo task until it is done.")
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 1
    assert len([call for call in client.calls if call["tools"] is not None]) == 3
    assert len(client.calls) == 4
    assert client.calls[-1]["tools"] is None
    assert client.calls[-1]["stream"] is False
    final_request_messages = client.calls[-1]["messages"]
    assert str(final_request_messages[-1].get("role")) == "system"
    assert "Stop reason: the overall step budget is exhausted" in str(
        final_request_messages[-1].get("content")
    )
    assert "No more tool calls are allowed." in str(final_request_messages[-1].get("content"))
    assert surface.final_messages
    assert "Completed work:" in surface.final_messages[-1]
    assert "Remaining work:" in surface.final_messages[-1]
    assert "Known issues or risks:" in surface.final_messages[-1]

    assistant_events = _event_payloads(log_path, "assistant_message")
    final_events = _event_payloads(log_path, "final")
    assert assistant_events
    assert final_events
    assert assistant_events[-1]["content"] == surface.final_messages[-1]
    assert final_events[-1]["content"] == surface.final_messages[-1]
    assert _event_payloads(log_path, "forced_final_summary_requested")
    assert _event_payloads(log_path, "forced_final_summary_completed")


@pytest.mark.parametrize("finalization_mode", ["error", "blank"])
def test_forced_final_summary_uses_fallback_when_needed(
    tmp_path: Path,
    finalization_mode: str,
) -> None:
    sessions_dir = tmp_path / "sessions"
    surface = _ForcedSummarySurface()
    session = create_session(
        cfg=AppConfig(model=SMOKE_MODEL, routing_mode="code_only"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=2,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        session_id_override=f"forced-summary-fallback-{finalization_mode}",
        surface=surface,
    )
    client = _BudgetExhaustionClient(finalization_mode=finalization_mode)
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Keep working on the repo task until it is done.")
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 1
    assert len([call for call in client.calls if call["tools"] is not None]) == 2
    assert len(client.calls) == 3
    assert client.calls[-1]["tools"] is None
    assert surface.final_messages
    assert "The turn stopped before it could finish" in surface.final_messages[-1]
    assert "Listed directories: ." in surface.final_messages[-1]
    assert (
        "A reliable final completion summary could not be produced"
        not in surface.final_messages[-1]
    )
    assert "Remaining work:" in surface.final_messages[-1]
    assert "Known issues or risks:" in surface.final_messages[-1]

    fallback_events = _event_payloads(log_path, "forced_final_summary_fallback")
    assert fallback_events
    assert fallback_events[-1]["fallback_reason"] in {"finalization_error", "blank_response"}
    assert _event_payloads(log_path, "forced_final_summary_completed") == []
    assistant_events = _event_payloads(log_path, "assistant_message")
    final_events = _event_payloads(log_path, "final")
    assert assistant_events[-1]["content"] == surface.final_messages[-1]
    assert final_events[-1]["content"] == surface.final_messages[-1]


def test_exploration_retry_exhausted_fallback_summary_uses_truthful_termination_wording(
    tmp_path: Path,
) -> None:
    (tmp_path / "repeat.txt").write_text("x\n", encoding="utf-8")
    sessions_dir = tmp_path / "sessions"
    surface = _ForcedSummarySurface()
    session = create_session(
        cfg=AppConfig(model=SMOKE_MODEL, routing_mode="code_only"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=10,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="forced-summary-exploration-retry-exhausted-fallback",
        surface=surface,
    )
    client = _ScriptedClient(
        [
            *[
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id=f"tc{idx}", name="fs_read", arguments={"path": "repeat.txt"})
                    ],
                    raw={},
                )
                for idx in range(1, 6)
            ],
        ],
        finalization_response=LLMResponse(content="   ", tool_calls=[], raw={}),
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 1
    fallback_events = _event_payloads(log_path, "forced_final_summary_fallback")
    assert fallback_events
    assert fallback_events[-1]["reason"] == "exploration_retry_exhausted"
    assert fallback_events[-1]["termination_cause"] == "exploration retries are exhausted"
    assert surface.final_messages
    assert "exploration retries are exhausted" in surface.final_messages[-1]
    assert "before the turn terminated" in surface.final_messages[-1]
    assert "budget ran out" not in surface.final_messages[-1]
    assistant_events = _event_payloads(log_path, "assistant_message")
    final_events = _event_payloads(log_path, "final")
    assert assistant_events[-1]["content"] == surface.final_messages[-1]
    assert final_events[-1]["content"] == surface.final_messages[-1]


def test_post_explore_retry_exhausted_emits_forced_final_summary(tmp_path: Path) -> None:
    (tmp_path / "src" / "mini_notes").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "mini_notes" / "cli.py").write_text("print('x')\n", encoding="utf-8")
    sessions_dir = tmp_path / "sessions"
    surface = _ForcedSummarySurface()
    session = create_session(
        cfg=AppConfig(model=SMOKE_MODEL, routing_mode="code_only"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=12,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        subagents_enabled=True,
        session_log_dir_override=sessions_dir,
        session_id_override="forced-summary-post-explore-retry-exhausted",
        surface=surface,
    )
    _install_stub_subagent_run(
        session,
        result_text="Edit targets: src/mini_notes/cli.py, src/mini_notes/logic.py, tests/test_cli.py",
    )
    client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="subagent_run",
                        arguments={"name": "explorer", "task": "Map repo"},
                    )
                ],
                raw={},
            ),
            *[
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=f"tc{idx}",
                            name="fs_read",
                            arguments={"path": "src/mini_notes/cli.py"},
                        )
                    ],
                    raw={},
                )
                for idx in range(2, 9)
            ],
        ],
        finalization_response=LLMResponse(
            content=(
                "Completed work: inspected `src/mini_notes/cli.py` and mapped likely targets.\n"
                "Remaining work: implementation has not started.\n"
                "Known issues or risks: post-explore retries were exhausted before edits began."
            ),
            tool_calls=[],
            raw={},
        ),
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 1
    assert surface.errors
    assert len([call for call in client.calls if call["tools"] is not None]) < session.max_steps
    assert client.calls[-1]["tools"] is None
    assert _event_payloads(log_path, "one_shot_post_explore_incomplete_after_retries")
    requested = _event_payloads(log_path, "forced_final_summary_requested")
    assert requested[-1]["reason"] == "post_explore_retry_exhausted"
    assert requested[-1]["termination_cause"] == "post-explore bootstrap retries are exhausted"
    summary = _assert_forced_summary_artifacts(log_path=log_path, surface=surface)
    assert "Known issues or risks: post-explore retries were exhausted" in summary


def test_exploration_retry_exhausted_emits_forced_final_summary(tmp_path: Path) -> None:
    (tmp_path / "repeat.txt").write_text("x\n", encoding="utf-8")
    sessions_dir = tmp_path / "sessions"
    surface = _ForcedSummarySurface()
    session = create_session(
        cfg=AppConfig(model=SMOKE_MODEL, routing_mode="code_only"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=10,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="forced-summary-exploration-retry-exhausted",
        surface=surface,
    )
    client = _ScriptedClient(
        [
            *[
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id=f"tc{idx}", name="fs_read", arguments={"path": "repeat.txt"})
                    ],
                    raw={},
                )
                for idx in range(1, 6)
            ],
        ],
        finalization_response=LLMResponse(
            content=(
                "Completed work: repeated exploration confirmed the same file state.\n"
                "Remaining work: implementation and verification are still pending.\n"
                "Known issues or risks: exploration retries were exhausted before progress."
            ),
            tool_calls=[],
            raw={},
        ),
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 1
    assert surface.errors
    assert len([call for call in client.calls if call["tools"] is not None]) == 5
    assert client.calls[-1]["tools"] is None
    assert _event_payloads(log_path, "one_shot_exploration_incomplete_after_retries")
    requested = _event_payloads(log_path, "forced_final_summary_requested")
    assert requested[-1]["reason"] == "exploration_retry_exhausted"
    assert requested[-1]["termination_cause"] == "exploration retries are exhausted"
    summary = _assert_forced_summary_artifacts(log_path=log_path, surface=surface)
    assert "Known issues or risks: exploration retries were exhausted" in summary


def test_edit_retry_exhausted_emits_forced_final_summary(tmp_path: Path) -> None:
    (tmp_path / "target.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    sessions_dir = tmp_path / "sessions"
    surface = _ForcedSummarySurface()
    session = create_session(
        cfg=AppConfig(model=SMOKE_MODEL, routing_mode="code_only"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=12,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="forced-summary-edit-retry-exhausted",
        surface=surface,
    )
    client = _ScriptedClient(
        [
            *[
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=f"tc{idx}",
                            name="fs_edit",
                            arguments={
                                "path": "target.txt",
                                "edits": [
                                    {
                                        "op": "replace_wrong",
                                        "target": "alpha",
                                        "replacement": "ALPHA",
                                    }
                                ],
                            },
                        )
                    ],
                    raw={},
                )
                for idx in range(1, 13)
            ],
        ],
        finalization_response=LLMResponse(
            content=(
                "Completed work: attempted localized edits on `target.txt`.\n"
                "Remaining work: the requested change is still unfinished.\n"
                "Known issues or risks: failed edit retries were exhausted before a successful write."
            ),
            tool_calls=[],
            raw={},
        ),
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 1
    assert surface.errors
    assert len([call for call in client.calls if call["tools"] is not None]) < session.max_steps
    assert client.calls[-1]["tools"] is None
    assert _event_payloads(log_path, "one_shot_edit_incomplete_after_retries")
    requested = _event_payloads(log_path, "forced_final_summary_requested")
    assert requested[-1]["reason"] == "edit_retry_exhausted"
    assert requested[-1]["termination_cause"] == "failed edit retries are exhausted"
    summary = _assert_forced_summary_artifacts(log_path=log_path, surface=surface)
    assert "Known issues or risks: failed edit retries were exhausted" in summary


def test_repeated_non_final_progress_termination_emits_forced_final_summary(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    surface = _ForcedSummarySurface()
    session = create_session(
        cfg=AppConfig(model=SMOKE_MODEL, routing_mode="code_only"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="forced-summary-repeated-non-final-progress",
        surface=surface,
    )
    repeated_progress_text = "I will implement search next."
    client = _ScriptedClient(
        [
            LLMResponse(content=repeated_progress_text, tool_calls=[], raw={}),
            LLMResponse(content=repeated_progress_text, tool_calls=[], raw={}),
        ],
        finalization_response=LLMResponse(
            content=(
                "Completed work: inspected the task and identified the next implementation step.\n"
                "Remaining work: implementation is still incomplete.\n"
                "Known issues or risks: repeated non-final progress stopped the run."
            ),
            tool_calls=[],
            raw={},
        ),
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 1
    assert len(client.calls) == 3
    incomplete_events = _event_payloads(log_path, "one_shot_incomplete_after_retries")
    assert incomplete_events
    assert incomplete_events[-1]["reason"] == "repeated_progress"
    requested = _event_payloads(log_path, "forced_final_summary_requested")
    assert requested[-1]["reason"] == "non_final_progress_retry_exhausted"
    assert requested[-1]["termination_cause"] == "repeated non-final progress is detected"
    _assert_last_forced_summary_request(
        client,
        latest_assistant_text=repeated_progress_text,
        termination_cause="repeated non-final progress is detected",
    )
    summary = _assert_forced_summary_artifacts(log_path=log_path, surface=surface)
    assert "Known issues or risks: repeated non-final progress stopped the run." in summary


def test_non_final_progress_continuation_cap_emits_forced_final_summary(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    surface = _ForcedSummarySurface()
    session = create_session(
        cfg=AppConfig(model=SMOKE_MODEL, routing_mode="code_only"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=8,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="forced-summary-non-final-progress-cap",
        surface=surface,
    )
    final_progress_text = "I will add tests next."
    client = _ScriptedClient(
        [
            LLMResponse(content="I will inspect the parser next.", tool_calls=[], raw={}),
            LLMResponse(content="I will update the parser next.", tool_calls=[], raw={}),
            LLMResponse(content=final_progress_text, tool_calls=[], raw={}),
        ],
        finalization_response=LLMResponse(
            content=(
                "Completed work: gathered partial implementation progress signals.\n"
                "Remaining work: the requested change is still unfinished.\n"
                "Known issues or risks: the non-final progress continuation cap was reached."
            ),
            tool_calls=[],
            raw={},
        ),
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement search command and update tests.")
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 1
    assert len(client.calls) == 4
    incomplete_events = _event_payloads(log_path, "one_shot_incomplete_after_retries")
    assert incomplete_events
    assert incomplete_events[-1]["reason"] == "continuation_cap"
    requested = _event_payloads(log_path, "forced_final_summary_requested")
    assert requested[-1]["reason"] == "non_final_progress_continuation_cap_reached"
    assert (
        requested[-1]["termination_cause"] == "the non-final progress continuation limit is reached"
    )
    _assert_last_forced_summary_request(
        client,
        latest_assistant_text=final_progress_text,
        termination_cause="the non-final progress continuation limit is reached",
    )
    summary = _assert_forced_summary_artifacts(log_path=log_path, surface=surface)
    assert "Known issues or risks: the non-final progress continuation cap was reached." in summary


def test_completion_gate_terminal_failure_emits_forced_final_summary(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    surface = _ForcedSummarySurface()
    session = create_session(
        cfg=AppConfig(
            model=SMOKE_MODEL,
            routing_mode="code_only",
            verify_commands=["pytest -q"],
        ),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=8,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        session_log_dir_override=sessions_dir,
        session_id_override="forced-summary-completion-gate-terminal",
        surface=surface,
    )
    latest_final_text = "Implemented the requested code change."
    client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "main.py", "content": "print('done')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(content=latest_final_text, tool_calls=[], raw={}),
            LLMResponse(content=latest_final_text, tool_calls=[], raw={}),
        ],
        finalization_response=LLMResponse(
            content=(
                "Completed work: updated `main.py`.\n"
                "Remaining work: verification still needs to run.\n"
                "Known issues or risks: completion-gate repair attempts were exhausted."
            ),
            tool_calls=[],
            raw={},
        ),
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 1
    assert len(client.calls) == 4
    incomplete_events = _event_payloads(
        log_path, "one_shot_completion_gate_incomplete_after_retries"
    )
    assert incomplete_events
    assert incomplete_events[-1]["stage"] == "verification_not_attempted"
    requested = _event_payloads(log_path, "forced_final_summary_requested")
    assert requested[-1]["reason"] == "completion_gate_terminal_failure"
    assert requested[-1]["termination_cause"] == "completion-gate repair attempts are exhausted"
    _assert_last_forced_summary_request(
        client,
        latest_assistant_text=latest_final_text,
        termination_cause="completion-gate repair attempts are exhausted",
    )
    summary = _assert_forced_summary_artifacts(log_path=log_path, surface=surface)
    assert "Known issues or risks: completion-gate repair attempts were exhausted." in summary
