from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sylliptor_agent_cli import agent_loop as agent_loop_mod
from sylliptor_agent_cli.agent.turn import core as turn_core_mod
from sylliptor_agent_cli.agent_loop import create_session
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.llm.openai_compat import LLMResponse, ToolCall
from sylliptor_agent_cli.session_store import read_session_events
from sylliptor_agent_cli.verify_gate import VerifyCommandResult, VerifyRunResult


class _ScriptedClient:
    model = "test-model"
    temperature = 0.2

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
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
        response = self._responses[self.calls]
        self.calls += 1
        return response


class _RouterSequenceClient:
    model = "test-model"
    temperature = 0.0

    def __init__(self, responses: list[dict[str, Any] | str | Exception]) -> None:
        self._responses = list(responses)
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
        if first_system != agent_loop_mod._ROUTER_SYSTEM_PROMPT:
            self.response_calls += 1
            raise AssertionError("Router sequence client should only handle route calls.")

        self.route_calls += 1
        response = self._responses[self.route_calls - 1]
        if isinstance(response, Exception):
            raise response
        if isinstance(response, str):
            return LLMResponse(content=response, tool_calls=[], raw={})

        route = str(response.get("route") or "repo")
        payload = {
            "route": route,
            "confidence": float(response.get("confidence", 0.99)),
            "reply": str(response.get("reply") or ""),
            "language": str(response.get("language") or ""),
            "script": str(response.get("script") or ""),
            "explicit_language_override": bool(response.get("explicit_language_override", False)),
        }
        if not bool(response.get("omit_execution_posture", False)):
            payload["execution_posture"] = str(
                response.get("execution_posture")
                or ("execute" if route == "repo" else "advisory_non_execution")
            )
        return LLMResponse(content=json.dumps(payload), tool_calls=[], raw={})


def _event_payloads(path: Path, event_type: str) -> list[dict[str, Any]]:
    return [
        dict(event.get("payload") or {})
        for event in read_session_events(path)
        if event.get("type") == event_type
    ]


def _assert_no_interactive_guard_fallbacks(path: Path) -> None:
    assert _event_payloads(path, "interactive_completion_gate_failed") == []
    assert _event_payloads(path, "interactive_completion_gate_incomplete_after_retries") == []
    assert _event_payloads(path, "interactive_step_budget_handoff") == []
    assert _event_payloads(path, "forced_final_summary_requested") == []


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


def _fake_verify_run(
    *,
    root: Path,
    commands: list[str],
    artifact_path: Path,
    cfg: AppConfig,
) -> VerifyRunResult:
    _ = root, cfg
    command_results = [
        VerifyCommandResult(
            command=command,
            effective_command=command,
            exit_code=0,
            output="ok\n",
            real_execution=True,
        )
        for command in commands
    ]
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("verification ok\n", encoding="utf-8")
    return VerifyRunResult(
        commands=list(commands),
        command_results=command_results,
        artifact_path=artifact_path,
    )


def _fake_infra_unavailable_verify_run(
    *,
    root: Path,
    commands: list[str],
    artifact_path: Path,
    cfg: AppConfig,
) -> VerifyRunResult:
    _ = root, cfg
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        "$ ./gradlew test\n/bin/bash: ./gradlew: No such file or directory\n",
        encoding="utf-8",
    )
    return VerifyRunResult(
        commands=list(commands),
        command_results=[
            VerifyCommandResult(
                command="./gradlew test",
                effective_command="./gradlew test",
                exit_code=127,
                output="/bin/bash: ./gradlew: No such file or directory\n",
                stderr="/bin/bash: ./gradlew: No such file or directory\n",
                real_execution=False,
                non_execution_reason="execution_layer_failure",
            )
        ],
        artifact_path=artifact_path,
        failure_category="infra_unavailable",
    )


def _write_skill(root: Path, name: str, body: str) -> None:
    bundle = root / ".sylliptor_skills" / name
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test skill\n---\n\n{body}\n",
        encoding="utf-8",
    )


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
        ["git", "add", "README.md"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def _create_interactive_session(
    tmp_path: Path,
    *,
    session_id: str,
) -> tuple[Path, Any]:
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=AppConfig(model="test-model", routing_mode="auto", verify_commands=["pytest -q"]),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
        enable_chat_turn_step_budget=True,
    )
    return sessions_dir, session


def test_interactive_read_only_inspection_ignores_router_execute_posture(
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
        cfg=AppConfig(model="test-model", routing_mode="auto", verify_commands=["pytest -q"]),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        session_id_override="interactive-read-only-inspect",
        enable_chat_turn_step_budget=True,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="fs_list", arguments={"path": "."})],
                raw={},
            ),
            LLMResponse(
                content="The repo contains README.md at the workspace root.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("can you inspect the repo we are working?")
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 0
    route_events = _event_payloads(log_path, "route_decision")
    assert route_events[-1]["execution_posture"] == "execute"
    assert route_events[-1]["classified_turn_intent_kind"] == "read_only"
    resolved_events = _event_payloads(log_path, "turn_intent_resolved")
    assert resolved_events[-1]["turn_intent"] == "read_only"
    finalized_events = _event_payloads(log_path, "turn_intent_finalized")
    assert finalized_events[-1]["observed_tool_intent"] == "read_only"
    assert _event_payloads(log_path, "interactive_no_material_edits_detected") == []
    assert _event_payloads(log_path, "no_material_edits_bootstrap_nudge") == []
    assert _event_payloads(log_path, "completion_gate_nudge") == []
    assert _event_payloads(log_path, "forced_final_summary_requested") == []


def test_interactive_read_only_tools_override_execute_classifier_for_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    monkeypatch.setattr(
        agent_loop_mod,
        "_route_turn",
        lambda **_kwargs: _route_decision("repo", "execute"),
    )
    monkeypatch.setattr(
        turn_core_mod,
        "_classify_one_shot_repo_turn_intent",
        lambda _instruction: "execute",
    )

    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=AppConfig(model="test-model", routing_mode="auto", verify_commands=["pytest -q"]),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        session_id_override="interactive-read-only-tools-override",
        enable_chat_turn_step_budget=True,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="fs_list", arguments={"path": "."})],
                raw={},
            ),
            LLMResponse(
                content="README.md is present at the workspace root.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("handle the repository request")
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 0
    resolved_events = _event_payloads(log_path, "turn_intent_resolved")
    assert resolved_events[-1]["turn_intent"] == "mutating_execution"
    finalized_events = _event_payloads(log_path, "turn_intent_finalized")
    assert finalized_events[-1]["observed_tool_intent"] == "read_only"
    assert finalized_events[-1]["completion_gate_turn_intent"] == "read_only"
    assert _event_payloads(log_path, "interactive_no_material_edits_detected") == []
    assert _event_payloads(log_path, "completion_gate_nudge") == []
    assert _event_payloads(log_path, "forced_final_summary_requested") == []


def test_interactive_read_only_tools_do_not_bypass_completion_claim_gate(
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
        cfg=AppConfig(model="test-model", routing_mode="auto", verify_commands=["pytest -q"]),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=5,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        session_id_override="interactive-read-only-tools-completion-claim",
        enable_chat_turn_step_budget=True,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="fs_list", arguments={"path": "."})],
                raw={},
            ),
            LLMResponse(content="I updated README.md.", tool_calls=[], raw={}),
            LLMResponse(
                content=(
                    "BLOCKED: category: missing_information The requested change needs "
                    "a concrete target beyond the files I inspected."
                ),
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Fix the parser bug.")
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 0
    no_material_events = _event_payloads(log_path, "interactive_no_material_edits_detected")
    assert no_material_events
    assert no_material_events[-1]["observed_tool_intent"] == "read_only"
    assert no_material_events[-1]["completion_gate_turn_intent"] == "execute"
    assert _event_payloads(log_path, "no_material_edits_bootstrap_nudge")
    assert _event_payloads(log_path, "completion_gate_nudge")
    assert _event_payloads(log_path, "forced_final_summary_requested") == []


def test_interactive_read_only_inspection_without_tools_does_not_retry_grounding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_git_repo(tmp_path)
    monkeypatch.setattr(
        agent_loop_mod,
        "_route_turn",
        lambda **_kwargs: _route_decision("repo", "execute"),
    )

    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=AppConfig(model="test-model", routing_mode="auto", verify_commands=["pytest -q"]),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        session_id_override="interactive-read-only-inspect-no-tools",
        enable_chat_turn_step_budget=True,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="This repository is available; I can inspect files on request.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("can you inspect the repo we are working?")
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 0
    resolved_events = _event_payloads(log_path, "turn_intent_resolved")
    assert resolved_events[-1]["turn_intent"] == "read_only"
    assert _event_payloads(log_path, "normal_chat_first_turn_repo_execute_retry") == []
    assert _event_payloads(log_path, "interactive_no_material_edits_detected") == []
    assert _event_payloads(log_path, "completion_gate_nudge") == []
    assert _event_payloads(log_path, "forced_final_summary_requested") == []


def test_interactive_structured_infra_blocker_finalizes_without_forced_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "_route_turn",
        lambda **_kwargs: _route_decision("repo", "execute"),
    )
    monkeypatch.setattr(agent_loop_mod, "run_task_verification", _fake_infra_unavailable_verify_run)

    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=AppConfig(
            model="test-model",
            routing_mode="auto",
            verify_commands=["./gradlew test"],
        ),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        session_id_override="interactive-structured-blocker",
        enable_chat_turn_step_budget=True,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('changed')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc2", name="verify_run", arguments={})],
                raw={},
            ),
            LLMResponse(
                content=(
                    "BLOCKED: configured `./gradlew test` cannot run because "
                    "`./gradlew` is missing from this workspace."
                ),
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Implement the requested code change.")
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 0
    accepted = _event_payloads(log_path, "completion_gate_blocker_accepted")
    assert accepted
    assert accepted[-1]["blocked_response_allows_completion"] is True
    assert _event_payloads(log_path, "interactive_completion_gate_failed") == []
    assert _event_payloads(log_path, "failed_verification_repair_attempt") == []
    assert _event_payloads(log_path, "forced_final_summary_requested") == []


def test_interactive_completion_certificate_allows_required_output_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "_route_turn",
        lambda **_kwargs: _route_decision("repo", "execute"),
    )
    monkeypatch.setattr(agent_loop_mod, "run_task_verification", _fake_verify_run)

    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=AppConfig(model="test-model", routing_mode="auto", verify_commands=["pytest -q"]),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=5,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        session_id_override="interactive-certificate-required-output",
        enable_chat_turn_step_budget=True,
    )
    assert session.one_shot_execution is False
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "result.txt", "content": "ok\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc2", name="verify_run", arguments={})],
                raw={},
            ),
            LLMResponse(content="Created result.txt.", tool_calls=[], raw={}),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Create result.txt.")
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 0
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "ok\n"
    _assert_no_interactive_guard_fallbacks(log_path)
    finalized = _event_payloads(log_path, "turn_intent_finalized")
    assert finalized
    assert finalized[-1]["acceptance_problems"] == []


def test_interactive_local_materialization_overrides_non_repo_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "_route_turn",
        lambda **_kwargs: _route_decision("general", "advisory_non_execution"),
    )
    monkeypatch.setattr(agent_loop_mod, "run_task_verification", _fake_verify_run)

    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=AppConfig(model="test-model", routing_mode="auto", verify_commands=["pytest -q"]),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=5,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        session_id_override="interactive-local-materialization-route",
        enable_chat_turn_step_budget=True,
    )
    assert session.one_shot_execution is False
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "answer.txt", "content": "2\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": ["pytest -q"]},
                    )
                ],
                raw={},
            ),
            LLMResponse(content="Created answer.txt and verified it.", tool_calls=[], raw={}),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Tell me the count and save it to answer.txt.")
        log_path = session.store.path
    finally:
        session.close()

    route_events = _event_payloads(log_path, "route_decision")
    finalized = _event_payloads(log_path, "turn_intent_finalized")

    assert exit_code == 0
    assert (tmp_path / "answer.txt").read_text(encoding="utf-8") == "2\n"
    assert route_events[-1]["route"] == "repo"
    assert route_events[-1]["original_route"] == "general"
    assert route_events[-1]["execution_posture"] == "execute"
    assert route_events[-1]["route_override_reason"] == (
        "local_materialization_requires_repo_execution"
    )
    assert route_events[-1]["local_materialization_required"] is True
    assert finalized[-1]["runtime_kind"] == "interactive_chat"
    _assert_no_interactive_guard_fallbacks(log_path)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("BLOCKED:", False),
        ("BLOCKED: cannot continue", False),
        ("BLOCKED: category: docker Docker is unavailable in this sandbox.", True),
        (
            "BLOCKED: configured `./gradlew test` cannot run because `./gradlew` is missing.",
            True,
        ),
    ],
)
def test_structured_blocker_requires_concrete_detail(text: str, expected: bool) -> None:
    assert agent_loop_mod._assistant_text_has_well_formed_blocker(text) is expected


@pytest.mark.parametrize(
    "follow_up_instruction",
    [
        "what did you change",
        "summarize what changed",
        "give me a short summary of the changes",
    ],
)
def test_interactive_follow_up_summary_uses_router_non_execution_posture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    follow_up_instruction: str,
) -> None:
    route_decisions = iter(
        [
            _route_decision("repo", "execute"),
            _route_decision("repo", "advisory_non_execution"),
        ]
    )
    monkeypatch.setattr(agent_loop_mod, "_route_turn", lambda **_kwargs: next(route_decisions))
    monkeypatch.setattr(agent_loop_mod, "run_task_verification", _fake_verify_run)

    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=AppConfig(model="test-model", routing_mode="auto", verify_commands=["pytest -q"]),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        session_id_override="interactive-summary-follow-up",
        enable_chat_turn_step_budget=True,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('ok')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": ["python -m pytest tests/test_cli.py -v"]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the parser fix and ran targeted verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="The last turn updated src/app.py and verified the change with targeted pytest.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        first_exit_code = session.run_turn("Implement the parser fix now.")
        second_exit_code = session.run_turn(follow_up_instruction)
        log_path = session.store.path
    finally:
        session.close()

    assert first_exit_code == 0
    assert second_exit_code == 0
    route_events = _event_payloads(log_path, "route_decision")
    assert route_events[-1]["execution_posture"] == "advisory_non_execution"
    _assert_no_interactive_guard_fallbacks(log_path)


@pytest.mark.parametrize(
    ("follow_up_instruction", "degraded_router_response", "session_suffix"),
    [
        (
            "what did you change",
            {"route": "repo", "omit_execution_posture": True},
            "omit-posture",
        ),
        (
            "summarize what changed",
            {"route": "repo", "execution_posture": "definitely-not-valid"},
            "malformed-posture",
        ),
        (
            "give me a short summary of the changes",
            {"route": "repo", "omit_execution_posture": True},
            "omit-posture-long-summary",
        ),
    ],
)
def test_interactive_summary_follow_up_stays_non_execution_when_router_posture_fallback_is_needed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    follow_up_instruction: str,
    degraded_router_response: dict[str, Any],
    session_suffix: str,
) -> None:
    monkeypatch.setattr(agent_loop_mod, "run_task_verification", _fake_verify_run)

    _sessions_dir, session = _create_interactive_session(
        tmp_path,
        session_id=f"interactive-summary-fallback-{session_suffix}",
    )
    session.router_client = _RouterSequenceClient(
        [
            {"route": "repo", "execution_posture": "execute"},
            degraded_router_response,
        ]
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('ok')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": ["python -m pytest tests/test_cli.py -v"]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the parser fix and ran targeted verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="The last turn updated src/app.py and verified the change with targeted pytest.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        first_exit_code = session.run_turn("Implement the parser fix now.")
        second_exit_code = session.run_turn(follow_up_instruction)
        log_path = session.store.path
    finally:
        session.close()

    assert first_exit_code == 0
    assert second_exit_code == 0
    route_events = _event_payloads(log_path, "route_decision")
    assert route_events[-1]["route"] == "repo"
    assert route_events[-1]["execution_posture"] == "advisory_non_execution"
    assert route_events[-1]["execution_posture_source"] == "fallback"
    assert route_events[-1]["router_execution_posture"] == "advisory_non_execution"
    assert route_events[-1]["router_execution_posture_source"] == "fallback"
    _assert_no_interactive_guard_fallbacks(log_path)


@pytest.mark.parametrize(
    ("follow_up_instruction", "degraded_router_response", "session_suffix"),
    [
        ("what did you change", RuntimeError("router unavailable"), "router-error"),
        ("summarize what changed", "not json", "malformed-router-output"),
        ("explain the fix", RuntimeError("router unavailable"), "router-error-explain"),
    ],
)
def test_interactive_router_failure_summary_follow_up_stays_repo_advisory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    follow_up_instruction: str,
    degraded_router_response: str | Exception,
    session_suffix: str,
) -> None:
    _init_git_repo(tmp_path)
    monkeypatch.setattr(agent_loop_mod, "run_task_verification", _fake_verify_run)

    _sessions_dir, session = _create_interactive_session(
        tmp_path,
        session_id=f"interactive-router-failure-follow-up-{session_suffix}",
    )
    session.router_client = _RouterSequenceClient(
        [
            {"route": "repo", "execution_posture": "execute"},
            degraded_router_response,
        ]
    )
    client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/app.py", "content": "print('ok')\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": ["python -m pytest tests/test_cli.py -v"]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the parser fix and ran targeted verification.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="I updated src/app.py and reran targeted pytest for the parser fix.",
                tool_calls=[],
                raw={},
            ),
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        first_exit_code = session.run_turn("Implement the parser fix now.")
        second_exit_code = session.run_turn(follow_up_instruction)
        log_path = session.store.path
    finally:
        session.close()

    assert first_exit_code == 0
    assert second_exit_code == 0
    route_events = _event_payloads(log_path, "route_decision")
    assistant_messages = _event_payloads(log_path, "assistant_message")
    assert route_events[-1]["route"] == "repo"
    assert route_events[-1]["original_route"] == "repo"
    assert route_events[-1]["execution_posture"] == "advisory_non_execution"
    assert route_events[-1]["execution_posture_source"] == "fallback"
    assert route_events[-1]["router_execution_posture"] == "advisory_non_execution"
    assert route_events[-1]["router_execution_posture_source"] == "fallback"
    assert route_events[-1]["router_decision_source"] == "fallback_contextual"
    assert route_events[-1]["route_selection_source"] == "fallback_contextual"
    assert route_events[-1]["route_context"]["active_workspace_task"] is True
    assert client.calls == 4
    assert assistant_messages[-1]["content"] == (
        "I updated src/app.py and reran targeted pytest for the parser fix."
    )
    assert (
        "Could you clarify what you want me to help with?" not in assistant_messages[-1]["content"]
    )
    _assert_no_interactive_guard_fallbacks(log_path)


def test_interactive_execute_follow_up_keeps_execution_gate_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_decisions = iter(
        [
            _route_decision("repo", "advisory_non_execution"),
            _route_decision("repo", "execute"),
        ]
    )
    monkeypatch.setattr(agent_loop_mod, "_route_turn", lambda **_kwargs: next(route_decisions))
    monkeypatch.setattr(agent_loop_mod, "run_task_verification", _fake_verify_run)

    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=AppConfig(model="test-model", routing_mode="auto", verify_commands=["pytest -q"]),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        session_id_override="interactive-execute-follow-up",
        enable_chat_turn_step_budget=True,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="The parser is failing because retries are missing.", tool_calls=[], raw={}
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/retries.py", "content": "RETRIES = 3\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": ["python -m pytest tests/test_cli.py -v"]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the retry logic and verified it.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        first_exit_code = session.run_turn("Explain the current parser issue.")
        second_exit_code = session.run_turn("Go ahead and implement it now.")
        log_path = session.store.path
    finally:
        session.close()

    assert first_exit_code == 0
    assert second_exit_code == 0
    _assert_no_interactive_guard_fallbacks(log_path)


@pytest.mark.parametrize(
    ("degraded_router_response", "session_suffix"),
    [
        ({"route": "repo", "omit_execution_posture": True}, "omit-posture"),
        ({"route": "repo", "execution_posture": "broken-posture"}, "malformed-posture"),
    ],
)
def test_interactive_explicit_execute_follow_up_remains_execute_when_router_posture_fallback_is_needed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    degraded_router_response: dict[str, Any],
    session_suffix: str,
) -> None:
    monkeypatch.setattr(agent_loop_mod, "run_task_verification", _fake_verify_run)

    _sessions_dir, session = _create_interactive_session(
        tmp_path,
        session_id=f"interactive-explicit-execute-fallback-{session_suffix}",
    )
    session.router_client = _RouterSequenceClient(
        [
            {"route": "repo", "execution_posture": "advisory_non_execution"},
            degraded_router_response,
        ]
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="The parser is failing because retries are missing.",
                tool_calls=[],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/retries.py", "content": "RETRIES = 3\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="verify_run",
                        arguments={"commands": ["python -m pytest tests/test_cli.py -v"]},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Implemented the retry logic and verified it.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        first_exit_code = session.run_turn("Explain the current parser issue.")
        second_exit_code = session.run_turn("Go ahead and implement it now.")
        log_path = session.store.path
    finally:
        session.close()

    assert first_exit_code == 0
    assert second_exit_code == 0
    route_events = _event_payloads(log_path, "route_decision")
    assert route_events[-1]["route"] == "repo"
    assert route_events[-1]["execution_posture"] == "execute"
    assert route_events[-1]["execution_posture_source"] == "fallback"
    assert route_events[-1]["router_execution_posture"] == "execute"
    assert route_events[-1]["router_execution_posture_source"] == "fallback"
    _assert_no_interactive_guard_fallbacks(log_path)


def test_interactive_shell_run_verification_with_workspace_cd_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "_route_turn",
        lambda **_kwargs: _route_decision("repo", "execute"),
    )

    def fake_shell_run(
        *,
        root: Path,
        cmd: str,
        cwd: str | None = None,
        runner=None,
    ) -> dict[str, Any]:
        _ = root, cwd, runner
        return {
            "cmd": cmd,
            "effective_cmd": cmd,
            "exit_code": 0,
            "stdout": "ok\n",
            "stderr": "",
        }

    monkeypatch.setattr(agent_loop_mod, "shell_run", fake_shell_run)

    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=AppConfig(model="test-model", routing_mode="auto", verify_commands=["pytest -q"]),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        session_id_override="interactive-shell-verify-cd-prefix",
        enable_chat_turn_step_budget=True,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={"path": "src/batching.py", "content": "THRESHOLD = 4\n"},
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="shell_run",
                        arguments={
                            "cmd": "cd /tmp/x && python -m pytest tests/test_batching.py -v"
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Updated batching logic and ran targeted pytest from a workspace wrapper command.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Fix the batching threshold and verify it.")
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 0
    tool_results = _event_payloads(log_path, "tool_result")
    shell_payloads = [payload for payload in tool_results if payload.get("name") == "shell_run"]
    assert len(shell_payloads) == 1
    assert shell_payloads[0]["result"]["effective_cmd"] == (
        "cd /tmp/x && python -m pytest tests/test_batching.py -v"
    )
    _assert_no_interactive_guard_fallbacks(log_path)


def test_interactive_explicit_skill_turn_reaches_progress_under_constrained_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_skill(
        tmp_path, "pytest", "Read the task, inspect the test target, then make the smallest fix."
    )
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
        max_steps=8,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        session_id_override="interactive-explicit-skill-progress",
        enable_chat_turn_step_budget=True,
        chat_turn_fixed_override=4,
    )
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="tc1", name="skill_read", arguments={"name": "pytest"})],
                raw={},
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc2",
                        name="fs_write",
                        arguments={
                            "path": "docs/skill-note.md",
                            "content": "explicit skill applied\n",
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Updated docs/skill-note.md using the explicit skill guidance.",
                tool_calls=[],
                raw={},
            ),
        ]
    )  # type: ignore[assignment]

    explicit_skill_context = (
        "<explicit_skill_context>\n"
        "name: pytest\n"
        "source: /skill pytest\n"
        "description: test skill\n"
        "</explicit_skill_context>"
    )

    try:
        exit_code = session.run_turn(
            "Apply the pytest skill instructions to this task.",
            ephemeral_user_messages=[explicit_skill_context],
        )
        log_path = session.store.path
    finally:
        session.close()

    assert exit_code == 0
    _assert_no_interactive_guard_fallbacks(log_path)
