"""Capability arbitration over router verdicts.

Routing decides which capabilities are provisioned, not what the agent must
do. These tests pin the two arbitration rules:

- "one_shot_provisioning": non-interactive turns bound to a writable
  repo-backed workspace always keep the repo toolset provisioned; the router
  verdict never selects the answer-without-tools path.
- "classifier_disagreement": interactive turns escalate a non-execute router
  verdict (route general/tool, or repo with advisory/plan posture) to
  repo/execute when the turn-intent classifier says the turn is execute-class.
  Agreement on advisory/plan is respected unchanged, which preserves normal
  Q&A, and route "chat" is never escalated.

The arbitration decision consumes only classifier outputs and runtime facts —
no raw-text matching — and the router-vs-intent disagreement metric
(`router_intent_execution_disagreement` on every `route_decision` event) is
logged even when the kill-switch disables overrides.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from sylliptor_agent_cli import agent_loop as agent_loop_mod
from sylliptor_agent_cli.agent.routing import (
    _arbitrate_route_capability,
    _route_arbitration_enabled,
    _router_intent_execution_disagreement,
)
from sylliptor_agent_cli.agent_loop import create_session
from sylliptor_agent_cli.config import AppConfig, ConfigError, set_config_value
from sylliptor_agent_cli.llm.openai_compat import LLMResponse, ToolCall
from sylliptor_agent_cli.session_store import read_session_events

_ROUTES = ("chat", "general", "repo", "tool")
_POSTURES = ("execute", "advisory_non_execution", "plan_or_analysis_only")
_INTENTS = ("execute", "advisory_non_execution", "plan_or_analysis_only")

# A declarative defect report with no identity overlap with the test
# workspace and no file paths: the pre-existing one-shot defect net
# (identity-overlap based) and the local-materialization override both miss
# it, so any repo provisioning observed in these tests comes from arbitration
# alone. The turn-intent classifier classifies it as execute-class.
_NON_OVERLAPPING_DEFECT_REPORT = (
    "The pagination is broken: page 2 shows duplicate rows from page 1. "
    "Expected each page to contain distinct rows."
)

# Advisory-shaped general question: both the router stub and the turn-intent
# classifier agree the turn is advisory, so arbitration must not touch it.
_UNRELATED_ADVISORY_QUESTION = "How does asyncio work in Python?"


class _FailClient:
    model = "test-model"
    temperature = 0.2

    def chat(self, **_kwargs: Any) -> LLMResponse:
        raise AssertionError("Repo agent client should not be called for this turn.")


class _ScriptedClient:
    model = "test-model"
    temperature = 0.2

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self.calls = 0

    def chat(self, **_kwargs: Any) -> LLMResponse:
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return response


class _RouterStubClient:
    model = "test-model"
    temperature = 0.0

    def __init__(
        self,
        *,
        route: str,
        execution_posture: str,
        confidence: float = 0.9,
        route_reply: str = "",
        response_reply: str = "",
    ) -> None:
        self.route = route
        self.execution_posture = execution_posture
        self.confidence = confidence
        self.route_reply = route_reply
        self.response_reply = response_reply
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
        on_reasoning_delta=None,  # type: ignore[no-untyped-def]
        temperature: float | None = None,
    ) -> LLMResponse:
        _ = tools, stream, on_text_delta, on_reasoning_delta, temperature
        first_system = ""
        if messages and isinstance(messages[0], dict):
            first_system = str(messages[0].get("content") or "")
        if first_system == agent_loop_mod._ROUTER_SYSTEM_PROMPT:
            self.route_calls += 1
            self.last_route_messages = list(messages)
            payload = {
                "route": self.route,
                "execution_posture": self.execution_posture,
                "confidence": self.confidence,
                "reply": self.route_reply,
                "language": "",
                "script": "",
                "explicit_language_override": False,
            }
            return LLMResponse(content=json.dumps(payload), tool_calls=[], raw={})
        self.response_calls += 1
        return LLMResponse(content=self.response_reply, tool_calls=[], raw={})


def _init_repo_with_package(root: Path) -> None:
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
    (root / "README.md").write_text("# Acme Toolkit\n\nA demo toolkit.\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "acme-toolkit"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    package_dir = root / "src" / "acme_toolkit"
    package_dir.mkdir(parents=True)
    (package_dir / "listing.py").write_text(
        "def paginate(rows, size):\n    return rows[:size]\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def _session_event_payload(path: Path, event_type: str) -> dict[str, Any]:
    for event in read_session_events(path):
        if str(event.get("type") or "") == event_type:
            payload = event.get("payload")
            if isinstance(payload, dict):
                return payload
    raise AssertionError(f"event not found: {event_type}")


def _session_event_payload_or_none(path: Path, event_type: str) -> dict[str, Any] | None:
    for event in read_session_events(path):
        if str(event.get("type") or "") == event_type:
            payload = event.get("payload")
            if isinstance(payload, dict):
                return payload
    return None


def _make_session(
    root: Path,
    *,
    one_shot: bool,
    cfg: AppConfig | None = None,
) -> Any:
    return create_session(
        cfg=cfg or AppConfig(model="test-model", routing_mode="auto"),
        root=root,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=root / "sessions",
        one_shot_execution=one_shot,
        verification_enabled=False,
    )


# ---------------------------------------------------------------------------
# Truth-table unit tests for the pure arbitration function
# ---------------------------------------------------------------------------


def _spec_expected_verdict(
    *,
    interactive: bool,
    route: str,
    posture: str,
    intent: str,
    decision_source: str = "router",
) -> tuple[str, str, str] | None:
    """Independent restatement of the two arbitration rules from the spec.

    Returns (rule, route, posture) for an override cell, None for a
    no-override cell. Applies only to writable repo-backed workspaces.
    """

    if not interactive:
        if route != "repo":
            return ("one_shot_provisioning", "repo", posture)
        return None
    if decision_source.startswith("fallback"):
        return None
    if intent != "execute":
        return None
    if route in {"general", "tool"}:
        return ("classifier_disagreement", "repo", "execute")
    if route == "repo" and posture in {"advisory_non_execution", "plan_or_analysis_only"}:
        return ("classifier_disagreement", "repo", "execute")
    return None


def test_arbitration_truth_table_repo_backed_writable() -> None:
    """Exhaustive sweep over interactive x route x posture x intent x source."""

    for interactive in (True, False):
        for route in _ROUTES:
            for posture in _POSTURES:
                for intent in _INTENTS:
                    for decision_source in ("router", "fallback", "fallback_contextual"):
                        verdict = _arbitrate_route_capability(
                            route=route,
                            execution_posture=posture,
                            classified_turn_intent=intent,
                            workspace_is_repo_backed=True,
                            workspace_writable=True,
                            interactive=interactive,
                            decision_source=decision_source,
                        )
                        expected = _spec_expected_verdict(
                            interactive=interactive,
                            route=route,
                            posture=posture,
                            intent=intent,
                            decision_source=decision_source,
                        )
                        cell = (
                            f"interactive={interactive} {route}/{posture} "
                            f"intent={intent} source={decision_source}"
                        )
                        if expected is None:
                            assert verdict.rule is None, cell
                            assert verdict.route == route, cell
                            assert verdict.execution_posture == posture, cell
                        else:
                            assert (
                                verdict.rule,
                                verdict.route,
                                verdict.execution_posture,
                            ) == expected, cell


@pytest.mark.parametrize(
    ("interactive", "route", "posture", "intent", "expected_rule"),
    [
        # Rule A: one-shot provisioning fires for every non-repo route, and
        # preserves the router posture for every posture value.
        (False, "general", "advisory_non_execution", "execute", "one_shot_provisioning"),
        (False, "tool", "advisory_non_execution", "execute", "one_shot_provisioning"),
        (
            False,
            "tool",
            "plan_or_analysis_only",
            "advisory_non_execution",
            "one_shot_provisioning",
        ),
        (
            False,
            "chat",
            "advisory_non_execution",
            "advisory_non_execution",
            "one_shot_provisioning",
        ),
        (False, "general", "execute", "plan_or_analysis_only", "one_shot_provisioning"),
        # Rule A: route=repo is already provisioned; posture is not touched
        # (non-interactive posture resolution follows the intent classifier).
        (False, "repo", "advisory_non_execution", "execute", None),
        (False, "repo", "execute", "execute", None),
        # Rule B: non-execute router verdict + execute-class intent escalates.
        (True, "general", "advisory_non_execution", "execute", "classifier_disagreement"),
        (True, "general", "execute", "execute", "classifier_disagreement"),
        (True, "tool", "plan_or_analysis_only", "execute", "classifier_disagreement"),
        (True, "tool", "execute", "execute", "classifier_disagreement"),
        (True, "tool", "advisory_non_execution", "execute", "classifier_disagreement"),
        # Rule B milder variant: repo route with advisory/plan posture.
        (True, "repo", "advisory_non_execution", "execute", "classifier_disagreement"),
        (True, "repo", "plan_or_analysis_only", "execute", "classifier_disagreement"),
        # Rule B: agreement on advisory/plan preserves normal Q&A.
        (True, "general", "advisory_non_execution", "advisory_non_execution", None),
        (True, "general", "advisory_non_execution", "plan_or_analysis_only", None),
        (True, "repo", "advisory_non_execution", "advisory_non_execution", None),
        # Rule B: an execute-class router verdict is never touched.
        (True, "repo", "execute", "execute", None),
        (True, "repo", "execute", "advisory_non_execution", None),
        # Rule B: chat is never escalated (the intent classifier defaults to
        # "execute" on social turns, which must not hijack greetings).
        (True, "chat", "advisory_non_execution", "execute", None),
        (True, "chat", "execute", "execute", None),
    ],
)
def test_arbitration_named_cells(
    interactive: bool,
    route: str,
    posture: str,
    intent: str,
    expected_rule: str | None,
) -> None:
    verdict = _arbitrate_route_capability(
        route=route,
        execution_posture=posture,
        classified_turn_intent=intent,
        workspace_is_repo_backed=True,
        workspace_writable=True,
        interactive=interactive,
        decision_source="router",
    )
    assert verdict.rule == expected_rule
    if expected_rule is None:
        assert verdict.route == route
        assert verdict.execution_posture == posture
    else:
        assert verdict.route == "repo"
        if expected_rule == "classifier_disagreement":
            assert verdict.execution_posture == "execute"
        else:
            assert verdict.execution_posture == posture


@pytest.mark.parametrize(
    ("interactive", "route", "posture", "intent", "expected_rule"),
    [
        # Fallback-sourced verdicts: Rule B never fires (no independent router
        # verdict to arbitrate against) ...
        (True, "general", "advisory_non_execution", "execute", None),
        (True, "general", "execute", "execute", None),
        (True, "repo", "advisory_non_execution", "execute", None),
        # ... but Rule A still fires: one-shot provisioning is independent of
        # router errors by construction, including outright router failure.
        (False, "general", "advisory_non_execution", "execute", "one_shot_provisioning"),
        (False, "general", "execute", "advisory_non_execution", "one_shot_provisioning"),
        (False, "repo", "advisory_non_execution", "execute", None),
    ],
)
@pytest.mark.parametrize("decision_source", ["fallback", "fallback_contextual"])
def test_arbitration_fallback_source_cells(
    interactive: bool,
    route: str,
    posture: str,
    intent: str,
    expected_rule: str | None,
    decision_source: str,
) -> None:
    verdict = _arbitrate_route_capability(
        route=route,
        execution_posture=posture,
        classified_turn_intent=intent,
        workspace_is_repo_backed=True,
        workspace_writable=True,
        interactive=interactive,
        decision_source=decision_source,
    )
    assert verdict.rule == expected_rule


def test_arbitration_never_fires_without_repo_backed_writable_workspace() -> None:
    for workspace_is_repo_backed, workspace_writable in (
        (False, True),
        (True, False),
        (False, False),
    ):
        for interactive in (True, False):
            for route in _ROUTES:
                for posture in _POSTURES:
                    for intent in _INTENTS:
                        verdict = _arbitrate_route_capability(
                            route=route,
                            execution_posture=posture,
                            classified_turn_intent=intent,
                            workspace_is_repo_backed=workspace_is_repo_backed,
                            workspace_writable=workspace_writable,
                            interactive=interactive,
                        )
                        assert verdict.rule is None
                        assert verdict.route == route
                        assert verdict.execution_posture == posture


def test_router_intent_execution_disagreement_metric() -> None:
    # Router execute-class verdict is exactly repo/execute.
    assert (
        _router_intent_execution_disagreement(
            route="repo", execution_posture="execute", classified_turn_intent="execute"
        )
        is False
    )
    assert (
        _router_intent_execution_disagreement(
            route="repo",
            execution_posture="execute",
            classified_turn_intent="advisory_non_execution",
        )
        is True
    )
    assert (
        _router_intent_execution_disagreement(
            route="general",
            execution_posture="advisory_non_execution",
            classified_turn_intent="execute",
        )
        is True
    )
    assert (
        _router_intent_execution_disagreement(
            route="general",
            execution_posture="advisory_non_execution",
            classified_turn_intent="advisory_non_execution",
        )
        is False
    )
    assert (
        _router_intent_execution_disagreement(
            route="repo",
            execution_posture="advisory_non_execution",
            classified_turn_intent="execute",
        )
        is True
    )


# ---------------------------------------------------------------------------
# Kill-switch resolution
# ---------------------------------------------------------------------------


def test_route_arbitration_enabled_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = AppConfig(model="test-model")
    monkeypatch.delenv("SYLLIPTOR_ROUTE_ARBITRATION", raising=False)
    assert _route_arbitration_enabled(cfg) is True
    assert _route_arbitration_enabled(None) is True

    cfg.route_arbitration_enabled = False
    assert _route_arbitration_enabled(cfg) is False

    # Env wins over config, in both directions.
    monkeypatch.setenv("SYLLIPTOR_ROUTE_ARBITRATION", "on")
    assert _route_arbitration_enabled(cfg) is True
    cfg.route_arbitration_enabled = True
    monkeypatch.setenv("SYLLIPTOR_ROUTE_ARBITRATION", "off")
    assert _route_arbitration_enabled(cfg) is False
    monkeypatch.setenv("SYLLIPTOR_ROUTE_ARBITRATION", "0")
    assert _route_arbitration_enabled(cfg) is False

    # Unrecognized env values fall back to the config value.
    monkeypatch.setenv("SYLLIPTOR_ROUTE_ARBITRATION", "sometimes")
    assert _route_arbitration_enabled(cfg) is True


def test_route_arbitration_enabled_config_key() -> None:
    cfg = AppConfig(model="test-model")
    set_config_value(cfg, "route_arbitration_enabled", "false")
    assert cfg.route_arbitration_enabled is False
    set_config_value(cfg, "route_arbitration_enabled", "on")
    assert cfg.route_arbitration_enabled is True
    with pytest.raises(ConfigError):
        set_config_value(cfg, "route_arbitration_enabled", "sometimes")


# ---------------------------------------------------------------------------
# Integration: one-shot provisioning (Rule A)
# ---------------------------------------------------------------------------


def test_one_shot_provisioning_overrides_non_execute_router_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SYLLIPTOR_ROUTE_ARBITRATION", raising=False)
    _init_repo_with_package(tmp_path)
    session = _make_session(tmp_path, one_shot=True)
    event_path = session.store.path
    repo_client = _ScriptedClient(
        [
            LLMResponse(
                content="Fixing the pagination offset.",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="fs_write",
                        arguments={
                            "path": "src/acme_toolkit/listing.py",
                            "content": (
                                "def paginate(rows, size, page=0):\n"
                                "    start = page * size\n"
                                "    return rows[start : start + size]\n"
                            ),
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(content="Fixed the duplicate-page pagination bug.", tool_calls=[], raw={}),
        ]
    )
    session.client = repo_client  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        execution_posture="advisory_non_execution",
        confidence=0.9,
        route_reply="You should check your pagination offset math.",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn(_NON_OVERLAPPING_DEFECT_REPORT)
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")
    override_payload = _session_event_payload(event_path, "route_arbitration_override")

    assert exit_code == 0
    # The identity-overlap defect net missed this report (no workspace-identity
    # overlap), so provisioning came from arbitration alone.
    assert route_payload["route_override_reason"] is None
    assert route_payload["route"] == "repo"
    assert route_payload["original_route"] == "general"
    assert route_payload["arbitrated"] is True
    assert route_payload["route_arbitration_rule"] == "one_shot_provisioning"
    assert route_payload["route_selection_source"] == "route_arbitration"
    assert route_payload["router_intent_execution_disagreement"] is True
    # Rule A escalates the route but preserves the router posture (one-shot
    # posture resolution follows the intent classifier, not the router).
    assert route_payload["execution_posture"] == "advisory_non_execution"
    assert override_payload["rule"] == "one_shot_provisioning"
    assert override_payload["pre_arbitration_route"] == "general"
    assert override_payload["pre_arbitration_execution_posture"] == "advisory_non_execution"
    assert override_payload["arbitrated_execution_posture"] == "advisory_non_execution"
    assert override_payload["signals"]["interactive"] is False
    assert override_payload["signals"]["workspace_kind"] == "git_repo"
    assert override_payload["signals"]["classified_turn_intent"] == "execute"
    assert override_payload["signals"]["router_confidence"] == pytest.approx(0.9)
    # The repo toolset was actually provisioned and used: no advisory
    # tech-support answer, real steps attempted.
    assert router.response_calls == 0
    assert repo_client.calls >= 2
    assert "page * size" in (tmp_path / "src" / "acme_toolkit" / "listing.py").read_text(
        encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Integration: interactive agreement case is untouched (Rule B negative)
# ---------------------------------------------------------------------------


def test_interactive_advisory_agreement_is_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SYLLIPTOR_ROUTE_ARBITRATION", raising=False)
    _init_repo_with_package(tmp_path)
    session = _make_session(tmp_path, one_shot=False)
    event_path = session.store.path
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        execution_posture="advisory_non_execution",
        route_reply="asyncio schedules coroutines on an event loop.",
        response_reply="asyncio schedules coroutines on an event loop.",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn(_UNRELATED_ADVISORY_QUESTION)
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")

    assert exit_code == 0
    assert route_payload["route"] == "general"
    assert route_payload["arbitrated"] is False
    assert route_payload["route_arbitration_rule"] is None
    assert route_payload["router_intent_execution_disagreement"] is False
    assert _session_event_payload_or_none(event_path, "route_arbitration_override") is None
    final_payload = _session_event_payload(event_path, "final")
    assert "event loop" in str(final_payload.get("content") or "")


# ---------------------------------------------------------------------------
# Integration: kill-switch restores legacy behavior, disagreement still logged
# ---------------------------------------------------------------------------


def test_kill_switch_env_off_restores_legacy_and_still_logs_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYLLIPTOR_ROUTE_ARBITRATION", "off")
    _init_repo_with_package(tmp_path)
    session = _make_session(tmp_path, one_shot=True)
    event_path = session.store.path
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        execution_posture="advisory_non_execution",
        confidence=0.9,
        route_reply="You should check your pagination offset math.",
        response_reply="You should check your pagination offset math.",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn(_NON_OVERLAPPING_DEFECT_REPORT)
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")

    assert exit_code == 0
    # Legacy behavior: the router verdict stands, no override event.
    assert route_payload["route"] == "general"
    assert route_payload["arbitrated"] is False
    assert route_payload["route_arbitration_rule"] is None
    assert route_payload["route_arbitration_enabled"] is False
    assert _session_event_payload_or_none(event_path, "route_arbitration_override") is None
    # The router-vs-intent disagreement is still measured.
    assert route_payload["router_intent_execution_disagreement"] is True
    # The legacy advisory answer was actually delivered.
    final_payload = _session_event_payload(event_path, "final")
    assert "pagination offset math" in str(final_payload.get("content") or "")


def test_kill_switch_config_off_restores_legacy_interactive_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SYLLIPTOR_ROUTE_ARBITRATION", raising=False)
    _init_repo_with_package(tmp_path)
    cfg = AppConfig(model="test-model", routing_mode="auto", route_arbitration_enabled=False)
    session = _make_session(tmp_path, one_shot=False, cfg=cfg)
    event_path = session.store.path
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        execution_posture="advisory_non_execution",
        route_reply="You should check your pagination offset math.",
        response_reply="You should check your pagination offset math.",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn(_NON_OVERLAPPING_DEFECT_REPORT)
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")

    assert exit_code == 0
    assert route_payload["route"] == "general"
    assert route_payload["arbitrated"] is False
    assert route_payload["route_arbitration_enabled"] is False
    assert route_payload["router_intent_execution_disagreement"] is True
    assert _session_event_payload_or_none(event_path, "route_arbitration_override") is None
    final_payload = _session_event_payload(event_path, "final")
    assert "pagination offset math" in str(final_payload.get("content") or "")


# ---------------------------------------------------------------------------
# Readonly sessions: the mode-to-writable wiring must gate both rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("one_shot", [True, False])
def test_readonly_session_mode_disables_arbitration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, one_shot: bool
) -> None:
    # Interactive plan mode runs with mode="readonly"; a wiring regression that
    # hardcodes workspace_writable would silently escalate those turns.
    monkeypatch.delenv("SYLLIPTOR_ROUTE_ARBITRATION", raising=False)
    _init_repo_with_package(tmp_path)
    session = create_session(
        cfg=AppConfig(model="test-model", routing_mode="auto"),
        root=tmp_path,
        mode="readonly",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "sessions",
        one_shot_execution=one_shot,
        verification_enabled=False,
    )
    event_path = session.store.path
    session.client = _FailClient()  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        execution_posture="advisory_non_execution",
        route_reply="You should check your pagination offset math.",
        response_reply="You should check your pagination offset math.",
    )
    session.router_client = router

    try:
        exit_code = session.run_turn(_NON_OVERLAPPING_DEFECT_REPORT)
    finally:
        session.close()

    route_payload = _session_event_payload(event_path, "route_decision")

    assert exit_code == 0
    assert route_payload["route"] == "general"
    assert route_payload["arbitrated"] is False
    assert route_payload["route_arbitration_rule"] is None
    assert _session_event_payload_or_none(event_path, "route_arbitration_override") is None


# ---------------------------------------------------------------------------
# Prompt immutability: arbitration must not change any prompt bytes
# ---------------------------------------------------------------------------


def _run_one_shot_turn_and_capture_router_messages(
    root: Path,
    log_dir: Path,
) -> list[dict[str, Any]]:
    session = create_session(
        cfg=AppConfig(model="test-model", routing_mode="auto"),
        root=root,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=log_dir,
        one_shot_execution=True,
        verification_enabled=False,
    )
    repo_client = _ScriptedClient([LLMResponse(content="Answered.", tool_calls=[], raw={})])
    session.client = repo_client  # type: ignore[assignment]
    router = _RouterStubClient(
        route="general",
        execution_posture="advisory_non_execution",
        confidence=0.9,
        route_reply="Advisory reply.",
        response_reply="Advisory reply.",
    )
    session.router_client = router
    try:
        session.run_turn(_NON_OVERLAPPING_DEFECT_REPORT)
    finally:
        session.close()
    return router.last_route_messages


def test_router_prompt_bytes_identical_with_arbitration_on_and_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The session log dirs live OUTSIDE the repo root so the first run does
    # not perturb the second run's repo scan (anchor paths).
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_repo_with_package(repo_root)

    monkeypatch.delenv("SYLLIPTOR_ROUTE_ARBITRATION", raising=False)
    messages_on = _run_one_shot_turn_and_capture_router_messages(
        repo_root, tmp_path / "sessions-on"
    )
    monkeypatch.setenv("SYLLIPTOR_ROUTE_ARBITRATION", "off")
    messages_off = _run_one_shot_turn_and_capture_router_messages(
        repo_root, tmp_path / "sessions-off"
    )

    assert messages_on, "router was not called with arbitration on"
    assert messages_off, "router was not called with arbitration off"
    assert messages_on[0]["content"] == agent_loop_mod._ROUTER_SYSTEM_PROMPT
    serialized_on = json.dumps(messages_on, ensure_ascii=True, sort_keys=True)
    serialized_off = json.dumps(messages_off, ensure_ascii=True, sort_keys=True)
    assert serialized_on.encode("utf-8") == serialized_off.encode("utf-8")
