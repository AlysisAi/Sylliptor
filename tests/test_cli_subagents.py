from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from sylliptor_agent_cli import cli as cli_mod
from sylliptor_agent_cli.config import AppConfig


def test_chat_command_subagents_toggles_current_session(monkeypatch) -> None:
    rebuild_calls: list[dict[str, Any]] = []

    def _fake_rebuild_session_tools_for_mode(*, session: Any, mode: str) -> None:
        rebuild_calls.append({"session": session, "mode": mode})

    monkeypatch.setattr(
        cli_mod,
        "_rebuild_session_tools_for_mode",
        _fake_rebuild_session_tools_for_mode,
    )

    session = type("Session", (), {})()
    session.subagents_enabled = False
    session.cfg = AppConfig(model="test-model")
    session.mode = "review"

    console = Console(file=io.StringIO(), force_terminal=False)
    forge_state = cli_mod._ForgeChatState()
    result = cli_mod._handle_chat_command(
        input_text="/subagent on",
        root=Path("."),
        session=session,
        pending_images=[],
        console=console,
        forge_state=forge_state,
        plan_mode_state=cli_mod._ChatPlanModeState(),
    )

    assert result == "handled"
    assert session.subagents_enabled is True
    assert session.cfg.subagents_enabled is True
    assert len(rebuild_calls) == 1
    assert rebuild_calls[0]["mode"] == "review"


class _RecordingSubagentTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(args))
        return {
            "subagent": str(args.get("name", "")),
            "result": "subagent done",
            "sandbox": {
                "mode": "readonly",
                "tools": ["fs_read", "search_rg"],
            },
        }


def test_subagent_command_auto_enables_subagents_and_runs_requested_subagent(
    monkeypatch,
) -> None:
    tool = _RecordingSubagentTool()
    rebuild_calls: list[dict[str, Any]] = []

    def _fake_rebuild_session_tools_for_mode(*, session: Any, mode: str) -> None:
        rebuild_calls.append({"session": session, "mode": mode})
        session.tools = {"subagent_run": tool}

    monkeypatch.setattr(
        cli_mod,
        "_rebuild_session_tools_for_mode",
        _fake_rebuild_session_tools_for_mode,
    )

    session = type("Session", (), {})()
    session.subagents_enabled = False
    session.cfg = AppConfig(model="test-model")
    session.mode = "review"
    session.tools = {}
    session.subagent_registry = {
        "explorer": type(
            "Def",
            (),
            {"name": "explorer", "mode": "readonly", "description": "Explore repo"},
        )()
    }

    stream = io.StringIO()
    console = Console(file=stream, force_terminal=False)
    forge_state = cli_mod._ForgeChatState()
    result = cli_mod._handle_chat_command(
        input_text="/subagent explorer inspect auth boundaries",
        root=Path("."),
        session=session,
        pending_images=[],
        console=console,
        forge_state=forge_state,
        plan_mode_state=cli_mod._ChatPlanModeState(),
    )

    assert result == "handled"
    assert session.subagents_enabled is True
    assert session.cfg.subagents_enabled is True
    assert len(rebuild_calls) == 1
    assert rebuild_calls[0]["mode"] == "review"
    assert tool.calls == [{"name": "explorer", "task": "inspect auth boundaries"}]
    output = stream.getvalue()
    assert "Subagents enabled for this session." in output
    assert "Subagent Result" in output
    assert "mode: readonly" in output
    assert "fs_read, search_rg" in output
    assert "subagent done" in output


def test_subagent_without_args_shows_usage_panel_when_picker_unavailable(
    monkeypatch,
) -> None:
    def _fake_select_chat_subagent_interactive(
        *,
        registry: dict[str, Any],
        console: Console,
    ) -> tuple[str | None, bool]:
        _ = (registry, console)
        return None, False

    monkeypatch.setattr(
        cli_mod,
        "_select_chat_subagent_interactive",
        _fake_select_chat_subagent_interactive,
        raising=False,
    )

    session = type("Session", (), {})()
    session.subagents_enabled = False
    session.cfg = AppConfig(model="test-model")
    session.mode = "review"
    session.tools = {}
    session.subagent_registry = {
        "explorer": type(
            "Def", (), {"name": "explorer", "description": "Explore repo", "mode": "readonly"}
        )(),
        "reviewer": type(
            "Def",
            (),
            {"name": "reviewer", "description": "Review diff", "mode": "readonly"},
        )(),
    }

    stream = io.StringIO()
    console = Console(file=stream, force_terminal=False)
    forge_state = cli_mod._ForgeChatState()
    result = cli_mod._handle_chat_command(
        input_text="/subagent",
        root=Path("."),
        session=session,
        pending_images=[],
        console=console,
        forge_state=forge_state,
        plan_mode_state=cli_mod._ChatPlanModeState(),
    )

    assert result == "handled"
    output = stream.getvalue()
    assert "Subagent Usage" in output
    assert "Usage: /subagent <name> <task>" in output
    assert "/subagent on|off|status" in output
    assert "/explorer" not in output
    assert "explorer: Explore repo" in output


def test_subagent_status_reports_current_state() -> None:
    session = type("Session", (), {})()
    session.subagents_enabled = True
    session.cfg = AppConfig(model="test-model")
    session.mode = "review"
    session.tools = {}
    session.subagent_registry = {}

    stream = io.StringIO()
    console = Console(file=stream, force_terminal=False)
    forge_state = cli_mod._ForgeChatState()
    result = cli_mod._handle_chat_command(
        input_text="/subagent status",
        root=Path("."),
        session=session,
        pending_images=[],
        console=console,
        forge_state=forge_state,
        plan_mode_state=cli_mod._ChatPlanModeState(),
    )

    assert result == "handled"
    output = stream.getvalue()
    assert output.startswith("Subagents: on\n")
    assert "Available now: none" in output
    assert "Unavailable: visual-designer" in output
    assert "image_generation.enabled true" in output.replace("\n", " ")


def test_nested_subagent_command_is_rejected_before_any_agent_runs() -> None:
    tool = _RecordingSubagentTool()
    session = type("Session", (), {})()
    session.subagents_enabled = True
    session.cfg = AppConfig(model="test-model")
    session.mode = "review"
    session.tools = {"subagent_run": tool}
    session.subagent_registry = {}

    stream = io.StringIO()
    result = cli_mod._handle_chat_command(
        input_text=(
            "/subagent frontend-engineer /subagent visual-designer Create a square forest image."
        ),
        root=Path("."),
        session=session,
        pending_images=[],
        console=Console(file=stream, force_terminal=False),
        forge_state=cli_mod._ForgeChatState(),
        plan_mode_state=cli_mod._ChatPlanModeState(),
    )

    assert result == "handled"
    assert tool.calls == []
    output = " ".join(stream.getvalue().split())
    assert "cannot start another /subagent command" in output
    assert "/subagent visual-designer Create a square forest image." in output


def test_nested_subagent_rejection_escapes_user_controlled_rich_markup() -> None:
    tool = _RecordingSubagentTool()
    session = type("Session", (), {})()
    session.subagents_enabled = True
    session.cfg = AppConfig(model="test-model")
    session.mode = "review"
    session.tools = {"subagent_run": tool}
    session.subagent_registry = {}

    stream = io.StringIO()
    result = cli_mod._handle_chat_command(
        input_text="/subagent explorer /subagent [bold]visual[/bold] Create an image.",
        root=Path("."),
        session=session,
        pending_images=[],
        console=Console(file=stream, force_terminal=False),
        forge_state=cli_mod._ForgeChatState(),
        plan_mode_state=cli_mod._ChatPlanModeState(),
    )

    assert result == "handled"
    assert tool.calls == []
    assert "[bold]visual[/bold]" in stream.getvalue()


@pytest.mark.parametrize(
    ("command", "reported_command"),
    [
        ("/subagents", "/subagents"),
        ("/agents", "/agents"),
        ("/explorer", "/explorer"),
        ("/explore", "/explore"),
        ("/review", "/review"),
        ("/reviewer", "/reviewer"),
        ("/tests", "/tests"),
        ("/test-strategist", "/test-strategist"),
        ("/engineer", "/engineer"),
        ("/agent explorer inspect auth boundaries", "/agent"),
    ],
)
def test_removed_subagent_commands_fall_through_to_unknown_command(
    command: str,
    reported_command: str,
) -> None:
    session = type("Session", (), {})()
    session.subagents_enabled = False
    session.cfg = AppConfig(model="test-model")
    session.mode = "review"
    session.tools = {}
    session.subagent_registry = {}

    stream = io.StringIO()
    console = Console(file=stream, force_terminal=False)
    forge_state = cli_mod._ForgeChatState()
    result = cli_mod._handle_chat_command(
        input_text=command,
        root=Path("."),
        session=session,
        pending_images=[],
        console=console,
        forge_state=forge_state,
        plan_mode_state=cli_mod._ChatPlanModeState(),
    )

    assert result == "handled"
    output = stream.getvalue()
    assert f"Unknown command: {reported_command}." in output
    assert "Try /help." in output
