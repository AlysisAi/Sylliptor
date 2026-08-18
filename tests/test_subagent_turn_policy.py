from __future__ import annotations

from typing import Any

from sylliptor_agent_cli import agent_loop
from sylliptor_agent_cli.agent_loop import ToolDef, create_session
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.llm.openai_compat import LLMResponse, ToolCall
from sylliptor_agent_cli.session_store import read_session_events
from sylliptor_agent_cli.subagents import built_in_subagents


class _ScriptedClient:
    model = "test-model"
    temperature = 0.2

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
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
        if not self._responses:
            raise AssertionError("unexpected extra model call")
        return self._responses.pop(0)


class _UnexpectedClient:
    model = "test-model"
    temperature = 0.2

    def __init__(self) -> None:
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
        raise AssertionError("model should not be called")


def _event_payloads(path, event_type: str) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
    return [
        dict(event.get("payload") or {})
        for event in read_session_events(path)
        if event.get("type") == event_type
    ]


def _replace_subagent_run_with_fake(session: Any) -> list[dict[str, Any]]:
    original = session.tools["subagent_run"]
    calls: list[dict[str, Any]] = []

    def _fake_subagent_run(args: dict[str, Any]) -> dict[str, Any]:
        calls.append(dict(args))
        return {
            "subagent": str(args.get("name") or "explorer"),
            "subagent_session_id": "fake-subagent",
            "result": "subagent report",
            "usage": {},
            "sandbox": {"mode": "readonly", "tools": ["fs_read"]},
        }

    session.tools["subagent_run"] = ToolDef(
        name="subagent_run",
        description=original.description,
        parameters=original.parameters,
        run=_fake_subagent_run,
        metadata=original.metadata,
    )
    return calls


def test_subagent_turn_policy_never_manufactures_delegation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Router-free path: no semantic contract exists, so a turn never derives a
    # required_by_user delegation gate (nor a user_opt_out block) from the
    # instruction text — in any language. Posture alone selects the advisory
    # level: execute turns get "recommended", advisory turns get "available".
    registry = built_in_subagents()
    tools = {"subagent_run": object()}  # type: ignore[dict-item]

    for instruction in (
        "Please use a subagent to inspect the parser before answering.",
        "Run the explorer to map the parser.",
        "استخدم وكيلاً فرعياً لفحص المستودع.",
        "Fix the parser, but do not use subagents for this one.",
        "サブエージェントを使わずに、このリポジトリを確認してください。",
    ):
        advisory = agent_loop._resolve_subagent_turn_policy(
            instruction=instruction,
            subagents_enabled=True,
            subagent_depth=0,
            subagent_registry=registry,
            turn_tools=tools,  # type: ignore[arg-type]
            repo_turn_execution_intent="advisory_non_execution",
        )
        assert advisory.level == "available", instruction
        assert advisory.reason == "repo_non_execution_turn", instruction

        execute = agent_loop._resolve_subagent_turn_policy(
            instruction=instruction,
            subagents_enabled=True,
            subagent_depth=0,
            subagent_registry=registry,
            turn_tools=tools,  # type: ignore[arg-type]
            repo_turn_execution_intent="execute",
        )
        assert execute.level == "recommended", instruction
        assert execute.reason == "repo_execution_turn", instruction

    context = agent_loop._subagent_turn_context_message(
        agent_loop._resolve_subagent_turn_policy(
            instruction="Inspect this repo and fix any issue you find.",
            subagents_enabled=True,
            subagent_depth=0,
            subagent_registry=registry,
            turn_tools=tools,  # type: ignore[arg-type]
            repo_turn_execution_intent="execute",
        )
    )
    assert context is not None
    assert "policy: recommended" in context
    assert "Make an explicit delegation decision" in context
    assert "Call subagent_run before finalizing" not in context


def test_subagent_turn_policy_reports_disabled_and_missing_tool_as_off(tmp_path) -> None:  # type: ignore[no-untyped-def]
    registry = built_in_subagents()
    tools = {"subagent_run": object()}  # type: ignore[dict-item]

    disabled = agent_loop._resolve_subagent_turn_policy(
        instruction="Please use a subagent to inspect the repo.",
        subagents_enabled=False,
        subagent_depth=0,
        subagent_registry=registry,
        turn_tools=tools,  # type: ignore[arg-type]
        repo_turn_execution_intent="advisory_non_execution",
    )
    assert disabled.level == "off"
    assert disabled.reason == "subagents_disabled"
    assert agent_loop._subagent_turn_context_message(disabled) is None

    tool_missing = agent_loop._resolve_subagent_turn_policy(
        instruction="Please use a subagent to inspect the repo.",
        subagents_enabled=True,
        subagent_depth=0,
        subagent_registry=registry,
        turn_tools={},
        repo_turn_execution_intent="advisory_non_execution",
    )
    assert tool_missing.level == "off"
    assert tool_missing.reason == "subagent_tool_not_exposed"


def test_repo_turn_injects_delegation_decision_context_and_delegation_executes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Router-free path: no semantic contract forces or forbids delegation. The
    # turn gets the advisory <subagent_turn_context> (recommended on an
    # execute-capable turn), and a model-initiated subagent_run simply runs.
    (tmp_path / "README.md").write_text("repo notes\n", encoding="utf-8")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=AppConfig(model="test-model", routing_mode="code_only"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        enable_chat_turn_step_budget=True,
    )
    subagent_calls = _replace_subagent_run_with_fake(session)
    client = _ScriptedClient(
        [
            LLMResponse(
                content="Delegating now.",
                tool_calls=[
                    ToolCall(
                        id="call-subagent",
                        name="subagent_run",
                        arguments={
                            "name": "explorer",
                            "task": "Inspect README.md and report the relevant notes.",
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(content="Done after using the subagent.", tool_calls=[], raw={}),
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn(
            "Please use a subagent to read README.md and tell me what it says. Do not modify files."
        )
        session_path = session.store.path
    finally:
        session.close()

    assert exit_code == 0
    assert len(subagent_calls) == 1
    assert subagent_calls[0]["name"] == "explorer"
    assert len(client.calls) == 2
    first_call_messages = "\n".join(
        str(message.get("content") or "") for message in client.calls[0]["messages"]
    )
    assert "<subagent_turn_context>" in first_call_messages
    assert "policy: recommended" in first_call_messages
    assert "Make an explicit delegation decision" in first_call_messages
    # No manufactured delegation gate exists on the router-free path.
    assert "Call subagent_run before finalizing" not in first_call_messages
    assert _event_payloads(session_path, "subagent_required_nudge") == []


def test_turn_proceeds_without_subagent_gate_when_subagents_disabled(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Router-free path: subagents disabled means the policy is silently off —
    # no unavailable-notice message, no gate events, and the turn completes on
    # the first model reply.
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=AppConfig(model="test-model", routing_mode="code_only", subagents_enabled=False),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=3,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        enable_chat_turn_step_budget=True,
        subagents_enabled=False,
    )
    client = _ScriptedClient(
        [
            LLMResponse(
                content="I inspected directly because subagent delegation is unavailable.",
                tool_calls=[],
                raw={},
            )
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Please use a subagent to inspect the repo.")
        session_path = session.store.path
    finally:
        session.close()

    assert exit_code == 0
    assert len(client.calls) == 1
    first_call_messages = "\n".join(
        str(message.get("content") or "") for message in client.calls[0]["messages"]
    )
    assert "<subagent_turn_context>" not in first_call_messages
    assert _event_payloads(session_path, "subagent_request_unavailable") == []
    assert _event_payloads(session_path, "subagent_required_nudge") == []


def test_interactive_repo_exploration_gets_subagent_nudge(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "README.md").write_text("known issue\n", encoding="utf-8")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=AppConfig(model="test-model", routing_mode="code_only"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        enable_chat_turn_step_budget=True,
    )
    client = _ScriptedClient(
        [
            LLMResponse(
                content="Listing files.",
                tool_calls=[ToolCall(id="call-list", name="fs_list", arguments={"path": "."})],
                raw={},
            ),
            LLMResponse(
                content="Reading notes.",
                tool_calls=[
                    ToolCall(
                        id="call-read",
                        name="fs_read",
                        arguments={"path": "README.md"},
                    )
                ],
                raw={},
            ),
            LLMResponse(content="Blocked by no concrete issue found.", tool_calls=[], raw={}),
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Inspect this repo and fix any issue you find.")
        session_path = session.store.path
    finally:
        session.close()

    assert exit_code == 0
    assert len(client.calls) == 3
    third_call_messages = "\n".join(
        str(message.get("content") or "") for message in client.calls[2]["messages"]
    )
    assert "Subagent delegation check" in third_call_messages
    events = _event_payloads(session_path, "subagent_exploration_nudge")
    assert events
    assert events[-1]["consecutive_exploration_only_steps"] == 2


def test_model_initiated_subagent_call_runs_despite_opt_out_phrasing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Router-free path: no semantic contract exists, so opt-out phrasing in
    # the instruction cannot manufacture a tool block — honoring "do not use
    # subagents" is the model's job, and a subagent_run it does issue simply
    # executes.
    (tmp_path / "README.md").write_text("repo notes\n", encoding="utf-8")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=AppConfig(model="test-model", routing_mode="code_only"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=3,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        enable_chat_turn_step_budget=True,
    )
    subagent_calls = _replace_subagent_run_with_fake(session)
    client = _ScriptedClient(
        [
            LLMResponse(
                content="Trying a subagent.",
                tool_calls=[
                    ToolCall(
                        id="call-subagent",
                        name="subagent_run",
                        arguments={
                            "name": "explorer",
                            "task": "Inspect README.md and report the notes.",
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(content="README.md contains repo notes.", tool_calls=[], raw={}),
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Tell me what README.md says, but do not use subagents.")
        session_path = session.store.path
    finally:
        session.close()

    assert exit_code == 0
    assert len(subagent_calls) == 1
    tool_results = _event_payloads(session_path, "tool_result")
    assert tool_results
    assert tool_results[0]["name"] == "subagent_run"
    assert "error" not in (tool_results[0].get("result") or {})
