from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest

from sylliptor_agent_cli.agent_loop import create_session
from sylliptor_agent_cli.agentbox_integration import (
    AgentBoxTelemetry,
    sanitize_task_hint,
    tool_category,
)
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.llm.types import LLMResponse, LLMUsage, ToolCall


class _ScriptedClient:
    def __init__(self, responses: list[LLMResponse], *, model: str = "test-model") -> None:
        self.model = model
        self.temperature = 0.2
        self._responses = responses
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


class _FakeAgentBox:
    instances: list[_FakeAgentBox] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.events: list[tuple[Any, ...]] = []
        self.__class__.instances.append(self)

    def session(self, workspace: str | None = None) -> _FakeSessionContext:
        return _FakeSessionContext(self, workspace)

    def close(self) -> None:
        self.events.append(("client.close",))


class _FakeSessionContext:
    def __init__(self, client: _FakeAgentBox, workspace: str | None) -> None:
        self.client = client
        self.workspace = workspace

    def __enter__(self) -> _FakeSession:
        self.client.events.append(("session.start", self.workspace))
        return _FakeSession(self.client)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        _ = exc, tb
        self.client.events.append(("session.end", exc_type is not None))


class _FakeSession:
    def __init__(self, client: _FakeAgentBox) -> None:
        self.client = client

    def task(self, hint: str) -> None:
        self.client.events.append(("task.update", hint))

    def turn(self) -> _FakeTurnContext:
        return _FakeTurnContext(self.client)

    def tokens(self, in_: int = 0, out: int = 0, usd: float | None = None) -> None:
        self.client.events.append(("tokens", in_, out, usd))

    def tool(self, name: str, category: str = "other", count: int = 1) -> None:
        self.client.events.append(("tool.activity", name, category, count))


class _FakeTurnContext:
    def __init__(self, client: _FakeAgentBox) -> None:
        self.client = client

    def __enter__(self) -> None:
        self.client.events.append(("turn.start",))
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        _ = exc, tb
        self.client.events.append(("turn.end", exc_type is not None))


def _install_fake_agentbox(monkeypatch: pytest.MonkeyPatch) -> type[_FakeAgentBox]:
    _FakeAgentBox.instances.clear()
    module = types.ModuleType("agentbox_sdk")
    module.AgentBox = _FakeAgentBox  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agentbox_sdk", module)
    return _FakeAgentBox


def _enable_agentbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENTBOX_ENABLED", "1")
    monkeypatch.setenv("AGENTBOX_PLANE_URL", "http://agentbox.local")
    monkeypatch.setenv("AGENTBOX_TOKEN", "test-token")
    monkeypatch.setenv("AGENTBOX_QUEUE_DIR", str(tmp_path / "queue"))


def _session_for(
    root: Path,
    *,
    max_steps: int = 4,
    no_log: bool = True,
) -> Any:
    return create_session(
        cfg=AppConfig(model="test-model", routing_mode="code_only"),
        root=root,
        mode="auto",
        yes=True,
        max_steps=max_steps,
        no_log=no_log,
        api_key_override="override-key",
    )


def test_agentbox_disabled_mode_does_not_import_or_emit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTBOX_ENABLED", "0")
    monkeypatch.setenv("AGENTBOX_PLANE_URL", "http://agentbox.local")
    monkeypatch.setenv("AGENTBOX_TOKEN", "test-token")
    monkeypatch.delitem(sys.modules, "agentbox_sdk", raising=False)

    assert AgentBoxTelemetry.from_env(root=tmp_path, runtime_version="test") is None
    assert "agentbox_sdk" not in sys.modules

    session = _session_for(tmp_path, max_steps=2)
    session.client = _ScriptedClient([LLMResponse(content="done", tool_calls=[], raw={})])

    try:
        assert session.run_turn("Do a tiny task.") == 0
    finally:
        session.close()

    assert session.agentbox_telemetry is None
    assert "agentbox_sdk" not in sys.modules


def test_agentbox_run_turn_emits_metadata_only_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_agentbox = _install_fake_agentbox(monkeypatch)
    _enable_agentbox(monkeypatch, tmp_path)
    root = tmp_path / "repo-name"
    root.mkdir()
    (root / "README.md").write_text("hello\n", encoding="utf-8")

    session = _session_for(root)
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="Reading.",
                tool_calls=[ToolCall(id="tc1", name="fs_read", arguments={"path": "README.md"})],
                raw={},
                usage=LLMUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            ),
            LLMResponse(
                content="Done.",
                tool_calls=[],
                raw={},
                usage=LLMUsage(prompt_tokens=8, completion_tokens=3, total_tokens=11),
            ),
            LLMResponse(
                content="Verified.",
                tool_calls=[],
                raw={},
                usage=LLMUsage(prompt_tokens=4, completion_tokens=1, total_tokens=5),
            ),
        ]
    )

    try:
        assert (
            session.run_turn("Inspect /workspace/sylliptor/src/auth.py and summarize `def x()`.")
            == 0
        )
        assert session.run_turn("Run the verification step.") == 0
    finally:
        session.close()

    agentbox = fake_agentbox.instances[0]
    assert agentbox.events.count(("session.start", "repo-name")) == 1
    assert agentbox.events.count(("session.end", False)) == 1
    assert agentbox.events.count(("turn.start",)) == 2
    assert agentbox.events.count(("turn.end", False)) == 2
    assert ("tool.activity", "fs_read", "read", 1) in agentbox.events
    assert ("tokens", 5, 2, None) in agentbox.events
    assert ("tokens", 13, 5, None) in agentbox.events
    assert ("tokens", 4, 1, None) in agentbox.events

    task_events = [event[1] for event in agentbox.events if event[0] == "task.update"]
    assert task_events == [
        "Inspect file and summarize.",
        "Run the verification step.",
    ]
    assert not any("/Users" in event or "def x" in event for event in task_events)


def test_agentbox_task_sanitizer_removes_content_and_paths() -> None:
    hint = sanitize_task_hint(
        "Refactor /workspace/project/src/auth.py then ```secret diff``` and `def login()` "
        + ("x" * 200)
    )

    assert 0 < len(hint) <= 140
    assert "/Users" not in hint
    assert "secret diff" not in hint
    assert "def login" not in hint


def test_agentbox_tool_category_mapping_matches_event_contract() -> None:
    assert tool_category("fs_read") == "read"
    assert tool_category("fs_write") == "edit"
    assert tool_category("shell_run") == "exec"
    assert tool_category("web_fetch") == "net"
    assert tool_category("custom_agentbox_probe") == "other"


def test_unreachable_agentbox_plane_does_not_fail_sylliptor_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_sdk_path = os.environ.get("AGENTBOX_SDK_PATH")
    if not configured_sdk_path:
        pytest.skip("AGENTBOX_SDK_PATH is not configured")
    sdk_path = Path(configured_sdk_path).expanduser()
    if not sdk_path.exists():
        pytest.skip("Configured AgentBox SDK checkout is not available")
    monkeypatch.syspath_prepend(str(sdk_path))
    monkeypatch.delitem(sys.modules, "agentbox_sdk", raising=False)
    monkeypatch.setenv("AGENTBOX_ENABLED", "1")
    monkeypatch.setenv("AGENTBOX_PLANE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("AGENTBOX_TOKEN", "test-token")
    monkeypatch.setenv("AGENTBOX_HOME", str(tmp_path / "agentbox-home"))
    monkeypatch.setenv("AGENTBOX_QUEUE_DIR", str(tmp_path / "agentbox-queue"))

    session = _session_for(tmp_path, max_steps=2)
    session.client = _ScriptedClient(
        [
            LLMResponse(
                content="done",
                tool_calls=[],
                raw={},
                usage=LLMUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            )
        ]
    )

    started = time.monotonic()
    try:
        assert session.run_turn("Complete a short no-op task.") == 0
    finally:
        session.close()

    assert time.monotonic() - started < 5
