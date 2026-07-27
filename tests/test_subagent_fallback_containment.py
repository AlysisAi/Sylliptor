from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from sylliptor_agent_cli import agent_loop
from sylliptor_agent_cli.agent_loop import ToolDef, build_tools
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.internal_artifacts import (
    INTERNAL_ARTIFACT_MESSAGE_KEY,
    INTERNAL_FALLBACK_SOURCE,
    SUBAGENT_INCOMPLETE_ERROR_CODE,
    ArtifactVisibility,
    SubagentIncompleteStatus,
    mark_message_internal,
    message_is_internal,
    resolve_incomplete_reason,
    subagent_report_is_internal,
    summary_input_messages,
)
from sylliptor_agent_cli.llm.metadata import strip_provider_metadata_from_message
from sylliptor_agent_cli.subagents import SubagentDefinition

# The shape the runtime writes when a turn runs out of deadline or steps. The
# tests never assert on this prose -- containment must not depend on wording --
# but a realistic dump makes the "did it leak?" assertions meaningful.
FALLBACK_DUMP = (
    "The turn stopped before it could finish (the run deadline is exhausted).\n\n"
    "Completed work:\n"
    "- Read files: lib/matplotlib/axes/_axes.py.\n\n"
    "Remaining work:\n"
    "- Continue from the recorded tool results instead of restarting from scratch.\n"
    "- Finish the requested implementation or report a concrete blocker.\n\n"
    "Known issues or risks:\n"
    "- The run deadline was exhausted before the turn could finish.\n"
    "- This fallback was generated from runtime state before the turn terminated."
)


class _RecordingStore:
    def __init__(self, *, artifact_root: Path | None = None) -> None:
        self.session_id = "main-session"
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.artifact_persistence_enabled = artifact_root is not None
        self._artifact_root = artifact_root
        self._lock = threading.Lock()

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self.events.append((event_type, payload))

    @property
    def session_artifact_layout(self):
        from sylliptor_agent_cli.session_artifacts import SessionArtifactLayout

        assert self._artifact_root is not None
        return SessionArtifactLayout(filesystem_root=self._artifact_root)


class _FakeUsageSummary:
    def totals(self) -> dict[str, Any]:
        return {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}

    def records(self) -> list[Any]:
        return []


class _FakeSubSessionStore:
    def __init__(self, *, session_id: str, events: list[dict[str, Any]]) -> None:
        self.session_id = session_id
        self._events = list(events)

    def events_snapshot(self) -> list[dict[str, Any]]:
        return list(self._events)


class _FakeSubSession:
    def __init__(
        self,
        *,
        store_events: list[dict[str, Any]],
        exit_code: int = 0,
        session_id: str = "sub-001",
    ) -> None:
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
        self.store = _FakeSubSessionStore(session_id=session_id, events=store_events)
        self.usage_summary = _FakeUsageSummary()
        self.exit_code = exit_code
        self.closed = False

    def run_turn(self, task: str) -> int:
        return self.exit_code

    def close(self) -> None:
        self.closed = True


def _fallback_events(*, termination_kind: str = "deadline_exhausted") -> list[dict[str, Any]]:
    return [
        {
            "type": "forced_final_summary_fallback",
            "payload": {
                "termination_kind": termination_kind,
                "fallback_reason": "local_summary_due_to_deadline",
            },
        },
        {
            "type": "final",
            "payload": {
                "content": FALLBACK_DUMP,
                "internal_fallback": True,
                "artifact_visibility": "internal",
                "internal_fallback_kind": termination_kind,
            },
        },
    ]


def _build_parent_tools(*, tmp_path: Path, store: _RecordingStore) -> dict[str, ToolDef]:
    registry = {
        "explorer": SubagentDefinition(
            name="explorer",
            description="explores",
            system_prompt="Inspect the repository.",
            mode="readonly",
        )
    }
    return build_tools(
        root=tmp_path,
        console=None,
        surface=None,
        store=store,  # type: ignore[arg-type]
        mode="auto",
        yes=True,
        cfg=AppConfig(model="test-model"),
        api_key="test-key",
        max_steps=8,
        subagents_enabled=True,
        subagent_depth=0,
        subagent_registry=registry,
    )


def _payloads(store: _RecordingStore, event_type: str) -> list[dict[str, Any]]:
    return [payload for kind, payload in store.events if kind == event_type]


# ---------------------------------------------------------------------------
# Test 1: a deadline-exhausted subagent returns a status, not the dump
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exit_code", [0, 1])
def test_exhausted_subagent_returns_structured_status_not_prose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    # exit_code 0 is the step-budget path, 1 the deadline path; both used to
    # hand the dump up, as `result` and as `final_text` respectively.
    monkeypatch.setattr(
        agent_loop,
        "create_session",
        lambda **_kwargs: _FakeSubSession(store_events=_fallback_events(), exit_code=exit_code),
    )
    store = _RecordingStore(artifact_root=tmp_path / "artifacts")
    tools = _build_parent_tools(tmp_path=tmp_path, store=store)

    result = tools["subagent_run"].run({"name": "explorer", "task": "Find the bug"})

    assert result["status"] == "incomplete"
    assert result["error_code"] == SUBAGENT_INCOMPLETE_ERROR_CODE
    assert result["incomplete_reason"] == "deadline_exhausted"
    assert isinstance(result["steps_used"], int)
    assert "deadline_s" in result
    # The dump must not appear anywhere in what the parent receives.
    for value in result.values():
        assert FALLBACK_DUMP not in str(value)
    assert "Remaining work:" not in str(result)
    assert "result" not in result, "an incomplete run must not report a deliverable"
    assert "final_text" not in result


def test_exhausted_subagent_emits_telemetry_and_stores_an_internal_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        agent_loop,
        "create_session",
        lambda **_kwargs: _FakeSubSession(store_events=_fallback_events()),
    )
    artifact_root = tmp_path / "artifacts"
    store = _RecordingStore(artifact_root=artifact_root)
    tools = _build_parent_tools(tmp_path=tmp_path, store=store)

    result = tools["subagent_run"].run({"name": "explorer", "task": "Find the bug"})

    incomplete = _payloads(store, "subagent_incomplete")
    assert len(incomplete) == 1
    payload = incomplete[0]
    assert payload["subagent"] == "explorer"
    assert payload["reason"] == "deadline_exhausted"
    assert payload["artifact_visibility"] == ArtifactVisibility.INTERNAL.value
    assert "steps_used" in payload
    assert "deadline_s" in payload

    locator = payload["report_artifact"]
    assert locator, "the internal report should have been persisted"
    assert result["report_artifact"] == locator
    stored = list(artifact_root.rglob("*.md"))
    assert len(stored) == 1
    assert stored[0].read_text(encoding="utf-8") == FALLBACK_DUMP

    end_payload = _payloads(store, "subagent_end")[-1]
    assert end_payload["status"] == "incomplete"
    assert FALLBACK_DUMP not in str(end_payload)


def test_step_budget_exhaustion_is_reported_as_its_own_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        agent_loop,
        "create_session",
        lambda **_kwargs: _FakeSubSession(
            store_events=_fallback_events(termination_kind="step_budget_exhausted")
        ),
    )
    store = _RecordingStore(artifact_root=tmp_path / "artifacts")
    tools = _build_parent_tools(tmp_path=tmp_path, store=store)

    result = tools["subagent_run"].run({"name": "explorer", "task": "Find the bug"})

    assert result["incomplete_reason"] == "step_budget_exhausted"


def test_containment_survives_without_artifact_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Under --no-log there is nowhere to write the artifact; the dump must still
    # not be handed to the parent.
    monkeypatch.setattr(
        agent_loop,
        "create_session",
        lambda **_kwargs: _FakeSubSession(store_events=_fallback_events()),
    )
    store = _RecordingStore(artifact_root=None)
    tools = _build_parent_tools(tmp_path=tmp_path, store=store)

    result = tools["subagent_run"].run({"name": "explorer", "task": "Find the bug"})

    assert result["status"] == "incomplete"
    assert result.get("report_artifact", "") == ""
    assert FALLBACK_DUMP not in str(result)


# ---------------------------------------------------------------------------
# Test 2: a successful subagent is unaffected
# ---------------------------------------------------------------------------


def test_successful_subagent_still_returns_its_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = "Found the bug in lib/matplotlib/axes/_axes.py: the limits are set before scaling."
    monkeypatch.setattr(
        agent_loop,
        "create_session",
        lambda **_kwargs: _FakeSubSession(
            store_events=[{"type": "final", "payload": {"content": report}}]
        ),
    )
    store = _RecordingStore(artifact_root=tmp_path / "artifacts")
    tools = _build_parent_tools(tmp_path=tmp_path, store=store)

    result = tools["subagent_run"].run({"name": "explorer", "task": "Find the bug"})

    assert result["result"] == report
    assert result["result_source"] == "store_final"
    assert "status" not in result or result.get("status") != "incomplete"
    assert _payloads(store, "subagent_incomplete") == []


def test_llm_written_stop_summary_is_still_a_deliverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # When the model itself wrote the closing summary there is no fallback marker,
    # so it stays a normal report even though the turn stopped early.
    summary = "I ran out of time after locating the bug in _axes.py; the fix is a one-liner."
    monkeypatch.setattr(
        agent_loop,
        "create_session",
        lambda **_kwargs: _FakeSubSession(
            store_events=[
                {"type": "forced_final_summary_completed", "payload": {"content_length": 80}},
                {"type": "final", "payload": {"content": summary}},
            ]
        ),
    )
    store = _RecordingStore(artifact_root=tmp_path / "artifacts")
    tools = _build_parent_tools(tmp_path=tmp_path, store=store)

    result = tools["subagent_run"].run({"name": "explorer", "task": "Find the bug"})

    assert result["result"] == summary
    assert _payloads(store, "subagent_incomplete") == []


# ---------------------------------------------------------------------------
# Test 3: the marker mechanism itself (by construction, not by pattern)
# ---------------------------------------------------------------------------


def test_summary_input_excludes_marked_messages() -> None:
    visible = {"role": "assistant", "content": "the real answer"}
    internal = mark_message_internal(
        {"role": "assistant", "content": FALLBACK_DUMP}, kind="deadline_exhausted"
    )
    messages = [visible, internal, {"role": "user", "content": "do the thing"}]

    projected = summary_input_messages(messages)

    assert internal not in projected
    assert visible in projected
    assert len(projected) == 2
    assert FALLBACK_DUMP not in str(projected)


def test_exclusion_does_not_depend_on_the_report_wording() -> None:
    # Same marker, completely different text: exclusion must key on the marker.
    internal = mark_message_internal(
        {"role": "assistant", "content": "Υπόλοιπη εργασία: τίποτα."}, kind="other"
    )
    assert summary_input_messages([internal]) == []
    # And an unmarked message that merely looks like a dump stays visible.
    lookalike = {"role": "assistant", "content": FALLBACK_DUMP}
    assert summary_input_messages([lookalike]) == [lookalike]


def test_marked_message_is_stripped_before_the_provider_call() -> None:
    internal = mark_message_internal(
        {"role": "assistant", "content": "internal"}, kind="deadline_exhausted"
    )
    assert message_is_internal(internal) is True
    wire = strip_provider_metadata_from_message(internal)
    assert INTERNAL_ARTIFACT_MESSAGE_KEY not in wire
    assert wire["content"] == "internal"


def test_message_is_internal_is_false_for_ordinary_messages() -> None:
    assert message_is_internal({"role": "assistant", "content": "hi"}) is False
    assert message_is_internal(None) is False
    assert message_is_internal("not a message") is False


def test_subagent_report_is_internal_keys_on_the_recorded_source() -> None:
    assert subagent_report_is_internal(INTERNAL_FALLBACK_SOURCE) is True
    assert subagent_report_is_internal("store_final") is False
    assert subagent_report_is_internal("") is False


def test_incomplete_reason_prefers_the_recorded_termination_kind() -> None:
    assert (
        resolve_incomplete_reason(termination_kind="step_budget_exhausted", deadline_exhausted=True)
        == "step_budget_exhausted"
    )
    assert (
        resolve_incomplete_reason(termination_kind="", deadline_exhausted=True)
        == "deadline_exhausted"
    )
    assert (
        resolve_incomplete_reason(termination_kind="", deadline_exhausted=False)
        == "step_budget_exhausted"
    )


# ---------------------------------------------------------------------------
# Test 4: the marker is set where the fallback is produced
# ---------------------------------------------------------------------------


def _session_events(sessions_dir: Path, session_id: str, event_type: str) -> list[dict[str, Any]]:
    from sylliptor_agent_cli.session_store import read_session_events

    return [
        event.get("payload") or {}
        for event in read_session_events(sessions_dir / f"{session_id}.jsonl")
        if str(event.get("type") or "") == event_type
    ]


class _BrokenClient:
    model = "test-model"
    temperature = 0.2

    def chat(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("provider unavailable")


def _forced_summary_session(tmp_path: Path, session_id: str):
    from sylliptor_agent_cli.agent_loop import create_session

    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=AppConfig(model="test-model", routing_mode="code_only"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        verification_enabled=False,
        session_log_dir_override=sessions_dir,
        session_id_override=session_id,
    )
    return session, sessions_dir


def test_locally_generated_stop_report_marks_its_final_event(tmp_path: Path) -> None:
    session, sessions_dir = _forced_summary_session(tmp_path, "fallback-marked")
    session.client = _BrokenClient()  # type: ignore[assignment]
    try:
        emitted = session._emit_forced_final_summary_before_termination(
            reason="deadline_exhausted",
            termination_cause="the run deadline is exhausted",
            termination_kind="deadline_exhausted",
            max_steps=None,
        )
    finally:
        session.store.close()

    assert "Remaining work:" in emitted, "the local fallback should still be shown here"
    finals = _session_events(sessions_dir, "fallback-marked", "final")
    assert len(finals) == 1
    assert finals[0]["internal_fallback"] is True
    assert finals[0]["artifact_visibility"] == ArtifactVisibility.INTERNAL.value
    assert finals[0]["internal_fallback_kind"] == "deadline_exhausted"


def test_nested_stop_report_is_not_pushed_to_the_parent_surface(tmp_path: Path) -> None:
    # The nested surface forwards a child's assistant messages up to the parent's
    # panel. Containing the dump in the tool result is not enough if the user
    # watches it stream past on the way there.
    from sylliptor_agent_cli.surface import NestedSubagentSurface

    class _RecordingParentSurface:
        def __init__(self) -> None:
            self.rendered: list[str] = []

        def emit_message_delta(self, text: str, **_kwargs: Any) -> None:
            self.rendered.append(text)

        def on_assistant_message_done(self, text: str) -> None:
            self.rendered.append(text)

    parent_surface = _RecordingParentSurface()
    nested = NestedSubagentSurface(
        parent_surface, subagent_name="explorer", subagent_mode="readonly"
    )
    session, sessions_dir = _forced_summary_session(tmp_path, "fallback-nested-surface")
    session.client = _BrokenClient()  # type: ignore[assignment]
    session.surface = nested  # type: ignore[assignment]
    session.subagent_depth = 1
    try:
        session._emit_forced_final_summary_before_termination(
            reason="deadline_exhausted",
            termination_cause="the run deadline is exhausted",
            termination_kind="deadline_exhausted",
            max_steps=None,
        )
    finally:
        session.store.close()

    assert parent_surface.rendered == [], "the child's stop report reached the parent's panel"
    assert nested.last_assistant_message_done == ""
    # It is still recorded, just not shown.
    finals = _session_events(sessions_dir, "fallback-nested-surface", "final")
    assert finals[0]["internal_fallback"] is True


def test_top_level_stop_report_is_still_shown_to_the_user(tmp_path: Path) -> None:
    # Containment applies to nested runs. For a top-level run the local stop
    # report *is* the honest answer and must keep reaching the user.
    class _RecordingSurface:
        def __init__(self) -> None:
            self.done: list[str] = []

        def on_assistant_message_done(self, text: str) -> None:
            self.done.append(text)

    surface = _RecordingSurface()
    session, _sessions_dir = _forced_summary_session(tmp_path, "fallback-top-level")
    session.client = _BrokenClient()  # type: ignore[assignment]
    session.surface = surface  # type: ignore[assignment]
    assert session.subagent_depth == 0
    try:
        emitted = session._emit_forced_final_summary_before_termination(
            reason="deadline_exhausted",
            termination_cause="the run deadline is exhausted",
            termination_kind="deadline_exhausted",
            max_steps=None,
        )
    finally:
        session.store.close()

    assert surface.done == [emitted]
    assert "Remaining work:" in emitted


def test_model_written_stop_summary_is_not_marked(tmp_path: Path) -> None:
    class _WorkingClient:
        model = "test-model"
        temperature = 0.2

        def chat(self, *_args: Any, **_kwargs: Any) -> Any:
            from sylliptor_agent_cli.llm.openai_compat import LLMResponse

            return LLMResponse(
                content="I located the bug but ran out of time before fixing it.",
                tool_calls=[],
                raw={},
            )

    session, sessions_dir = _forced_summary_session(tmp_path, "fallback-unmarked")
    session.client = _WorkingClient()  # type: ignore[assignment]
    try:
        session._emit_forced_final_summary_before_termination(
            reason="max_steps_exhausted",
            termination_cause="the overall step budget is exhausted",
            termination_kind="step_budget_exhausted",
            max_steps=4,
        )
    finally:
        session.store.close()

    finals = _session_events(sessions_dir, "fallback-unmarked", "final")
    assert len(finals) == 1
    assert "internal_fallback" not in finals[0]
    assert "artifact_visibility" not in finals[0]


def test_summary_builder_consumes_the_filtered_transcript(tmp_path: Path) -> None:
    session, sessions_dir = _forced_summary_session(tmp_path, "fallback-filtered")
    seen: list[list[dict[str, Any]]] = []

    class _CapturingClient:
        model = "test-model"
        temperature = 0.2

        def chat(self, *_args: Any, **kwargs: Any) -> Any:
            seen.append(list(kwargs.get("messages") or (_args[0] if _args else [])))
            raise RuntimeError("provider unavailable")

    session.client = _CapturingClient()  # type: ignore[assignment]
    session.messages.append({"role": "user", "content": "fix the axes bug"})
    session.messages.append(
        mark_message_internal(
            {"role": "assistant", "content": FALLBACK_DUMP}, kind="deadline_exhausted"
        )
    )
    try:
        session._emit_forced_final_summary_before_termination(
            reason="deadline_exhausted",
            termination_cause="the run deadline is exhausted",
            termination_kind="deadline_exhausted",
            max_steps=None,
        )
    finally:
        session.store.close()

    assert seen, "the summary builder should have been called"
    for request in seen:
        assert FALLBACK_DUMP not in str(request)
    assert sessions_dir.exists()


def test_incomplete_status_message_tells_the_parent_what_to_do() -> None:
    status = SubagentIncompleteStatus(
        subagent="explorer", reason="deadline_exhausted", steps_used=12, deadline_s=0.0
    )
    assert "stopped before finishing" in status.message
    assert "internal and was not returned" in status.message
    assert status.tool_result()["error"] == status.message
    assert status.tool_result()["status"] == "incomplete"
