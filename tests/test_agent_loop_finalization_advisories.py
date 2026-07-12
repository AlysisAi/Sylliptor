from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import sylliptor_agent_cli.agent_loop as agent_loop_mod
from sylliptor_agent_cli.agent.turn.core import _spec_faithfulness_advisory_message
from sylliptor_agent_cli.agent_loop import create_session
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.llm.openai_compat import LLMResponse, ToolCall
from sylliptor_agent_cli.session_store import read_session_events

_LIVE_BG_LINE_1 = (
    "- You have 1 background process(es) started with shell_background; they are terminated "
    "when this run ends. If the task requires a server/daemon to still be running after you "
    "finish, start it with shell_service_start (durable) instead, and re-verify."
)


class _RecordingClient:
    model = "test-model"
    temperature = 0.2

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
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
        _ = tools, tool_choice, stream, on_text_delta, temperature
        self.call_records.append({"messages": list(messages)})
        response = self._responses[self.calls]
        self.calls += 1
        return response


class _FakeTerminalManager:
    def __init__(self, statuses: tuple[str, ...]) -> None:
        self._statuses = statuses

    def list(self) -> tuple[SimpleNamespace, ...]:
        return tuple(SimpleNamespace(status=status) for status in self._statuses)

    def shutdown_all(self) -> None:
        pass


def _event_payloads(path: Path, event_type: str) -> list[dict[str, Any]]:
    return [
        dict(event.get("payload") or {})
        for event in read_session_events(path)
        if event.get("type") == event_type
    ]


def _controller_details(path: Path) -> list[str]:
    return [
        str(payload.get("detail") or "")
        for payload in _event_payloads(path, "controller_intervention")
    ]


def _route_decision(route: str, execution_posture: str) -> SimpleNamespace:
    return SimpleNamespace(
        route=route,
        execution_posture=execution_posture,
        confidence=0.99,
        language="",
        script="",
        explicit_language_override=False,
        language_source="default",
        decision_source="test",
        execution_posture_source="test",
        tool_family="none",
        tool_candidates=(),
    )


def _create_one_shot_session(tmp_path: Path, *, session_id: str):
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=AppConfig(model="test-model", routing_mode="code_only"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=8,
        no_log=False,
        api_key_override="override-key",
        one_shot_execution=True,
        verification_enabled=False,
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
    )
    return sessions_dir, session


def test_spec_faithfulness_advisory_live_background_warning_is_one_shot_only() -> None:
    one_shot_message = _spec_faithfulness_advisory_message(
        one_shot_execution=True,
        live_background_processes=1,
    )
    interactive_message = _spec_faithfulness_advisory_message(
        one_shot_execution=False,
        live_background_processes=1,
    )
    no_live_process_message = _spec_faithfulness_advisory_message(
        one_shot_execution=True,
        live_background_processes=0,
    )

    assert one_shot_message.splitlines().count(_LIVE_BG_LINE_1) == 1
    assert _LIVE_BG_LINE_1 not in interactive_message.splitlines()
    assert _LIVE_BG_LINE_1 not in no_live_process_message.splitlines()


def test_clean_one_shot_final_gets_one_spec_advisory_then_accepts(tmp_path: Path) -> None:
    sessions_dir, session = _create_one_shot_session(
        tmp_path,
        session_id="clean-one-shot-spec-advisory",
    )
    client = _RecordingClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "answer.txt", "content": "42\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Completed work: wrote answer.txt.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Completed work: answer.txt matches the requested output.",
                tool_calls=[],
                raw={},
            ),
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Create answer.txt containing 42.")
    finally:
        session.close()

    log_path = sessions_dir / "clean-one-shot-spec-advisory.jsonl"
    assert exit_code == 0
    assert client.calls == 3
    assert _controller_details(log_path).count("spec_faithfulness_advisory") == 1
    nudge_events = _event_payloads(log_path, "completion_gate_nudge")
    assert [event["stage"] for event in nudge_events] == ["spec_faithfulness_advisory"]
    assert "Final check before you finish:" in nudge_events[0]["message"]
    assert _event_payloads(log_path, "completion_gate_accepted_with_open_problems") == []


def test_one_shot_finalization_advisory_mentions_live_background_process(
    tmp_path: Path,
) -> None:
    sessions_dir, session = _create_one_shot_session(
        tmp_path,
        session_id="live-background-spec-advisory",
    )
    session.terminal_manager = _FakeTerminalManager(  # type: ignore[assignment]
        ("running", "exited")
    )
    client = _RecordingClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "answer.txt", "content": "42\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Completed work: wrote answer.txt.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Completed work: answer.txt matches the requested output.",
                tool_calls=[],
                raw={},
            ),
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Create answer.txt containing 42.")
    finally:
        session.close()

    log_path = sessions_dir / "live-background-spec-advisory.jsonl"
    nudge_events = _event_payloads(log_path, "completion_gate_nudge")
    final_call_messages = client.call_records[-1]["messages"]

    assert exit_code == 0
    assert [event["stage"] for event in nudge_events] == ["spec_faithfulness_advisory"]
    assert nudge_events[0]["live_background_processes"] == 1
    assert nudge_events[0]["message"].splitlines().count(_LIVE_BG_LINE_1) == 1
    assert any(
        message.get("role") == "system" and _LIVE_BG_LINE_1 in str(message.get("content") or "")
        for message in final_call_messages
    )


def test_problem_final_after_clarification_advisory_still_gets_checklist(
    tmp_path: Path,
) -> None:
    sessions_dir, session = _create_one_shot_session(
        tmp_path,
        session_id="clarification-then-problem-checklist",
    )
    client = _RecordingClient(
        [
            LLMResponse(
                content="Which output path should I use?",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(content="Completed work.", tool_calls=[], raw={}),
            LLMResponse(content="Completed work.", tool_calls=[], raw={}),
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Create the requested output file.")
    finally:
        session.close()

    log_path = sessions_dir / "clarification-then-problem-checklist.jsonl"
    details = _controller_details(log_path)
    assert exit_code == 0
    assert details.count("one_shot_clarification_advisory") == 1
    assert details.count("completion_gate_checklist") == 1
    assert _event_payloads(log_path, "completion_gate_accepted_with_open_problems")


@pytest.mark.parametrize(
    ("case_id", "final_text", "clarification_expected", "suppression_expected"),
    [
        (
            "confident-summary-with-what",
            "Summary: what it does is produce answer.txt with the computed value. "
            "The output file exists at the requested path.",
            False,
            True,
        ),
        (
            "question-ending",
            "I need one detail: which directory should I use?",
            True,
            False,
        ),
        (
            "long-rhetorical-question",
            (
                "Report: what about edge cases? The implementation handles the documented "
                "inputs, writes the expected artifact, preserves existing files, and records "
                "the verification notes. The remaining details describe behavior rather than "
                "ask for user input. "
                "This paragraph is intentionally long enough to exceed the short-question "
                "guard so a rhetorical question in the middle of a final report does not turn "
                "the whole response into a clarification request. "
                "The final answer is declarative and does not ask the user for anything."
            ),
            False,
            True,
        ),
        ("short-question", "Which file?", True, False),
    ],
)
def test_clarification_classifier_question_shape_guard(
    tmp_path: Path,
    case_id: str,
    final_text: str,
    clarification_expected: bool,
    suppression_expected: bool,
) -> None:
    sessions_dir, session = _create_one_shot_session(
        tmp_path,
        session_id=f"clarification-guard-{case_id}",
    )
    client = _RecordingClient(
        [
            LLMResponse(content=final_text, tool_calls=[], raw={}),
            LLMResponse(content="Finished.", tool_calls=[], raw={}),
            LLMResponse(content="Finished.", tool_calls=[], raw={}),
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Create the requested output file.")
    finally:
        session.close()

    log_path = sessions_dir / f"clarification-guard-{case_id}.jsonl"
    details = _controller_details(log_path)
    suppressed = [
        event
        for event in _event_payloads(log_path, "controller_intervention")
        if event.get("detail") == "clarification_suppressed_by_guard"
    ]

    assert exit_code == 0
    assert bool(details.count("one_shot_clarification_advisory")) is clarification_expected
    assert bool(suppressed) is suppression_expected
    for event in suppressed:
        metadata = event["metadata"]
        assert metadata["text_len"] == len(final_text.strip())
        assert metadata["ends_with_question"] is False
        assert event["headline_counted"] is False


def test_continuation_nudge_does_not_starve_clean_final_spec_advisory(
    tmp_path: Path,
) -> None:
    sessions_dir, session = _create_one_shot_session(
        tmp_path,
        session_id="continuation-then-spec-advisory",
    )
    client = _RecordingClient(
        [
            LLMResponse(
                content="I will inspect the repo, make the edit, and then verify it.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "answer.txt", "content": "ready\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Completed work: wrote answer.txt.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="Completed work: answer.txt matches the requested output.",
                tool_calls=[],
                raw={},
            ),
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Create answer.txt.")
    finally:
        session.close()

    log_path = sessions_dir / "continuation-then-spec-advisory.jsonl"
    details = _controller_details(log_path)
    assert exit_code == 0
    assert details.count("non_final_progress_continuation_nudge") == 1
    assert details.count("spec_faithfulness_advisory") == 1


def test_read_only_chat_final_does_not_get_spec_advisory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    monkeypatch.setattr(
        agent_loop_mod,
        "_route_turn",
        lambda **_kwargs: _route_decision("repo", "execute"),
    )
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=AppConfig(model="test-model", routing_mode="auto"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        enable_chat_turn_step_budget=True,
        verification_enabled=False,
        session_log_dir_override=sessions_dir,
        session_id_override="read-only-chat-no-spec-advisory",
    )
    client = _RecordingClient(
        [
            LLMResponse(
                content="The repo contains README.md at the workspace root.",
                tool_calls=[],
                raw={},
            )
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("can you inspect the repo we are working?")
    finally:
        session.close()

    log_path = sessions_dir / "read-only-chat-no-spec-advisory.jsonl"
    assert exit_code == 0
    assert "spec_faithfulness_advisory" not in _controller_details(log_path)
    assert _event_payloads(log_path, "completion_gate_nudge") == []
