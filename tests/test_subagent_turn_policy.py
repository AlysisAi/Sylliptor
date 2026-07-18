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


def test_subagent_turn_policy_requires_explicit_request_and_respects_opt_out(tmp_path) -> None:  # type: ignore[no-untyped-def]
    registry = built_in_subagents()
    tools = {"subagent_run": object()}  # type: ignore[dict-item]

    required = agent_loop._resolve_subagent_turn_policy(
        instruction="Please use a subagent to inspect the parser before answering.",
        subagents_enabled=True,
        subagent_depth=0,
        subagent_registry=registry,
        turn_tools=tools,  # type: ignore[arg-type]
        repo_turn_execution_intent="advisory_non_execution",
    )
    assert required.level == "required_by_user"
    context = agent_loop._subagent_turn_context_message(required)
    assert context is not None
    assert "policy: required_by_user" in context
    assert "Call subagent_run before finalizing" in context

    for instruction in (
        "Run the explorer to map the parser.",
        "Use the implementer for this scoped change.",
        "Ask the debugger to isolate the failure.",
        "Run the code-reviewer on the current diff.",
        "Use the test strategist to identify regression cases.",
        "Use the frontend engineer to build the responsive settings page.",
        "Run the visual designer to create the empty-state illustration.",
    ):
        role_request = agent_loop._resolve_subagent_turn_policy(
            instruction=instruction,
            subagents_enabled=True,
            subagent_depth=0,
            subagent_registry=registry,
            turn_tools=tools,  # type: ignore[arg-type]
            repo_turn_execution_intent="advisory_non_execution",
        )
        assert role_request.level == "required_by_user", instruction

    opted_out = agent_loop._resolve_subagent_turn_policy(
        instruction="Fix the parser, but do not use subagents for this one.",
        subagents_enabled=True,
        subagent_depth=0,
        subagent_registry=registry,
        turn_tools=tools,  # type: ignore[arg-type]
        repo_turn_execution_intent="execute",
    )
    assert opted_out.level == "off"
    assert opted_out.reason == "user_opt_out"


def test_explicit_subagent_request_gets_repair_nudge_before_finalizing(tmp_path) -> None:  # type: ignore[no-untyped-def]
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
            LLMResponse(content="I can answer directly.", tool_calls=[], raw={}),
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
    assert len(client.calls) == 3
    second_call_messages = "\n".join(
        str(message.get("content") or "") for message in client.calls[1]["messages"]
    )
    assert "The current user request explicitly asked for subagent" in second_call_messages
    assert _event_payloads(session_path, "subagent_required_nudge")


def test_explicit_subagent_request_accepts_final_after_required_nudge_cap(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "README.md").write_text("repo notes\n", encoding="utf-8")
    sessions_dir = tmp_path / "sessions"
    session = create_session(
        cfg=AppConfig(model="test-model", routing_mode="code_only"),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=5,
        no_log=False,
        api_key_override="override-key",
        session_log_dir_override=sessions_dir,
        enable_chat_turn_step_budget=True,
    )
    subagent_calls = _replace_subagent_run_with_fake(session)
    final_text = "I inspected directly and will proceed without delegating."
    client = _ScriptedClient(
        [
            LLMResponse(content=final_text, tool_calls=[], raw={}),
            LLMResponse(content=final_text, tool_calls=[], raw={}),
            LLMResponse(content=final_text, tool_calls=[], raw={}),
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
    assert subagent_calls == []
    assert len(client.calls) == 3
    nudge_events = _event_payloads(session_path, "subagent_required_nudge")
    assert len(nudge_events) == 2
    not_honored_events = _event_payloads(session_path, "subagent_required_not_honored")
    assert not_honored_events
    assert not_honored_events[-1]["content"] == final_text
    assert _event_payloads(session_path, "subagent_required_incomplete_after_retries") == []


def test_explicit_subagent_request_proceeds_when_subagents_disabled(tmp_path) -> None:  # type: ignore[no-untyped-def]
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
    assert "subagent_run is unavailable in this session (subagents_disabled)" in first_call_messages
    events = _event_payloads(session_path, "subagent_request_unavailable")
    assert events
    assert events[-1]["reason"] == "subagents_disabled"
    proceeding_events = _event_payloads(session_path, "subagent_request_unavailable_proceeding")
    assert proceeding_events
    assert proceeding_events[-1]["reason"] == "subagents_disabled"


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


def test_user_opt_out_blocks_subagent_tool_even_if_model_calls_it(tmp_path) -> None:  # type: ignore[no-untyped-def]
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
                            "task": "Inspect README.md despite the user opt-out.",
                        },
                    )
                ],
                raw={},
            ),
            LLMResponse(
                content="Blocked by the user request to avoid subagents.",
                tool_calls=[],
                raw={},
            ),
        ]
    )
    session.client = client  # type: ignore[assignment]

    try:
        exit_code = session.run_turn("Tell me what README.md says, but do not use subagents.")
        session_path = session.store.path
    finally:
        session.close()

    assert exit_code == 0
    assert subagent_calls == []
    tool_results = _event_payloads(session_path, "tool_result")
    assert tool_results
    assert tool_results[0]["name"] == "subagent_run"
    assert "explicitly requested no subagents" in tool_results[0]["result"]["error"]
