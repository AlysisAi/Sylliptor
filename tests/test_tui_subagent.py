"""TUI-native ``/subagent`` behaviour.

Covers the three things that made the command read as a pasted CLI dump inside
the alt-screen: it ran a nested agent session on the prompt_toolkit UI thread
(freezing the pane and killing Ctrl+C), it rendered its result as a box-drawn
Rich panel, and its picker hint promised a spawn that Enter never performed.
"""

from __future__ import annotations

import inspect
import io
from pathlib import Path
from typing import Any

from rich.console import Console

from sylliptor_agent_cli import cli as cli_mod
from sylliptor_agent_cli.cli_impl.chat.loop import _is_deferrable_subagent_command
from sylliptor_agent_cli.cli_impl.commands.chat_tui_panels import _chat_subagent_result_body
from sylliptor_agent_cli.config import AppConfig

# --------------------------------------------------------- deferral predicate


def test_explicit_subagent_with_task_defers_to_worker() -> None:
    # These spawn a nested agent session — they must never run on the UI thread.
    assert _is_deferrable_subagent_command("/subagent explorer map the auth flow")
    assert _is_deferrable_subagent_command("  /SubAgent explorer map it  ")
    # A name that merely starts like an action word is still a spawn.
    assert _is_deferrable_subagent_command("/subagent onboarding-helper do a thing")


def test_instant_and_bare_subagent_forms_stay_on_the_fast_path() -> None:
    # on|off|status are local toggles; the bare form is intercepted by the picker;
    # "<name>" with no task is answered inline with a one-line hint. None block.
    for text in (
        "/subagent",
        "/subagent on",
        "/subagent off",
        "/subagent status",
        "/subagent explorer",
        "/skill explorer do a thing",
        "",
    ):
        assert not _is_deferrable_subagent_command(text), text


# ------------------------------------------------------------- result body


def _result(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "subagent": "explorer",
        "result": "Found the auth flow in `auth/session.py`.",
        "sandbox": {"mode": "readonly", "tools": ["fs_read", "search_rg"]},
    }
    base.update(over)
    return base


def test_result_body_is_markdown_not_a_box_drawn_panel() -> None:
    body = _chat_subagent_result_body(subagent_name="explorer", result=_result())
    # Attribution reuses the ↩ mark the subagent trace already uses, as a dim
    # markdown line — no Rich markup, no box-drawing characters.
    assert body.startswith("*↩ explorer · readonly · fs_read, search_rg*")
    assert "Found the auth flow in `auth/session.py`." in body
    assert "[bold]" not in body
    assert not any(ch in body for ch in "─│╭╮╰╯┌┐└┘")


def test_result_body_tolerates_missing_metadata_and_empty_text() -> None:
    body = _chat_subagent_result_body(subagent_name="explorer", result={})
    assert body.startswith("*↩ explorer*")
    assert "_(no text result)_" in body
    # A non-dict sandbox must not raise; a custom name attributes the same way.
    assert _chat_subagent_result_body(
        subagent_name="x", result={"sandbox": "nope", "result": "ok"}
    ).startswith("*↩ x*")


def test_subagent_error_notice_escapes_all_dynamic_rich_markup() -> None:
    from sylliptor_agent_cli.cli_impl.chat.loop import _subagent_error_notice

    plain, markup = _subagent_error_notice(
        {
            "error": "Unknown [bold]agent[/bold]",
            "unavailable_reason": "Missing [link=https://example.test]capability[/link]",
            "resolution": "Set [green]config[/green]",
            "available_subagents": ["safe", "[red]spoof[/red]"],
        }
    )

    assert "[bold]agent[/bold]" in plain
    assert r"\[bold]agent\[/bold]" in markup
    assert r"\[link=https://example.test]capability\[/link]" in markup
    assert r"\[green]config\[/green]" in markup
    assert r"\[red]spoof\[/red]" in markup


def test_result_body_truncates_long_tool_lists_and_reports_steps() -> None:
    body = _chat_subagent_result_body(
        subagent_name="explorer",
        result=_result(
            sandbox={"mode": "readonly", "tools": ["a", "b", "c", "d", "e", "f"]},
            steps_completed=7,
        ),
    )
    assert "a, b, c, d +2" in body
    assert "7 steps" in body


def test_result_body_falls_back_to_final_text() -> None:
    body = _chat_subagent_result_body(
        subagent_name="explorer",
        result={"final_text": "from final_text"},
    )
    assert "from final_text" in body


# ------------------------------------------------------------------- sinks


def test_subagent_sinks_threaded_through_every_hop() -> None:
    # Regression: the TUI passes these by keyword through a chain of forwarding
    # shims, and every hop must accept them. The prompt_helpers shim in particular
    # pins them by NAME (it does not take **kwargs), so a hop that omits them
    # silently drops the command back onto the box-drawn console path rather than
    # failing loudly. Mirrors test_tui_forge's planner-sink guard.
    from sylliptor_agent_cli.cli_impl.chat import commands as cmd

    for fn in (cmd._handle_chat_command, cli_mod._handle_chat_command):
        params = inspect.signature(fn).parameters
        assert "subagent_result_sink" in params, fn.__module__
        assert "subagent_notice_sink" in params, fn.__module__

    # _run_explicit_subagent is where the sinks are actually consumed.
    from sylliptor_agent_cli.cli_impl.chat.loop import _run_explicit_subagent

    run_params = inspect.signature(_run_explicit_subagent).parameters
    assert "result_sink" in run_params
    assert "notice_sink" in run_params


class _RecordingSubagentTool:
    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "subagent": str(args.get("name", "")),
            "result": "subagent done",
            "sandbox": {"mode": "readonly", "tools": ["fs_read"]},
        }


def _session_with_tool(monkeypatch) -> Any:
    tool = _RecordingSubagentTool()

    def _fake_rebuild(*, session: Any, mode: str) -> None:
        session.tools = {"subagent_run": tool}

    monkeypatch.setattr(cli_mod, "_rebuild_session_tools_for_mode", _fake_rebuild)
    session = type("Session", (), {})()
    session.subagents_enabled = False
    session.cfg = AppConfig(model="test-model")
    session.mode = "review"
    session.tools = {}
    session.subagent_registry = {}
    return session


def _run(session: Any, **kwargs: Any) -> str:
    stream = io.StringIO()
    result = cli_mod._handle_chat_command(
        input_text="/subagent explorer inspect auth",
        root=Path("."),
        session=session,
        pending_images=[],
        console=Console(file=stream, force_terminal=False),
        forge_state=cli_mod._ForgeChatState(),
        plan_mode_state=cli_mod._ChatPlanModeState(),
        **kwargs,
    )
    assert result == "handled"
    return stream.getvalue()


def test_result_sink_receives_the_result_and_console_stays_clean(monkeypatch) -> None:
    session = _session_with_tool(monkeypatch)
    seen: list[tuple[str, dict[str, Any]]] = []
    notices: list[tuple[str, str]] = []

    output = _run(
        session,
        subagent_result_sink=lambda name, result: seen.append((name, result)),
        subagent_notice_sink=lambda role, text: notices.append((role, text)),
    )

    assert [name for name, _ in seen] == ["explorer"]
    assert seen[0][1]["result"] == "subagent done"
    # With both sinks attached nothing is left for the flat captured dump.
    assert "Subagent Result" not in output
    assert output.strip() == ""
    # The auto-enable side effect is announced through the sink, where the TUI
    # can render it — not buried in captured console text.
    assert notices == [("trace", "Subagents enabled for this session · auto-delegation on")]


def test_without_sinks_the_classic_rich_panel_is_preserved(monkeypatch) -> None:
    # The non-TUI CLI must keep its existing rendering verbatim.
    session = _session_with_tool(monkeypatch)
    output = _run(session)
    assert "Subagent Result" in output
    assert "subagent done" in output


def test_a_broken_result_sink_falls_back_to_the_console_panel(monkeypatch) -> None:
    session = _session_with_tool(monkeypatch)

    def _boom(name: str, result: dict[str, Any]) -> None:
        raise RuntimeError("sink exploded")

    output = _run(session, subagent_result_sink=_boom)
    # The run already happened — its result must still reach the user somehow.
    assert "Subagent Result" in output


def test_real_tui_command_runner_defers_only_the_spawning_form(tmp_path, monkeypatch) -> None:
    """The runner the TUI is actually handed must route the spawn to the worker.

    The unit tests above pin the predicate and app.py's own tests pin the
    "_deferred_execute → worker thread" machinery, but nothing otherwise proves
    _tui_command_runner consults the predicate at all — and it is nested inside
    chat(), so the only way to reach the real object is to capture it off the
    run_tui call (same interception test_cli_ux uses for on_config_saved).
    """
    import os
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from sylliptor_agent_cli.cli import app as sylliptor_app
    from sylliptor_agent_cli.cli_impl import tui as tui_pkg

    captured: dict[str, Any] = {}

    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path / "cfg"))
    monkeypatch.setattr(cli_mod, "_is_non_interactive_terminal", lambda: False)
    monkeypatch.setattr(tui_pkg, "is_tui_enabled", lambda: True)
    monkeypatch.setattr(
        cli_mod,
        "create_session",
        lambda **kw: SimpleNamespace(
            cfg=kw["cfg"], root=kw["root"], mode=kw["mode"], close=lambda: None
        ),
    )

    def _fake_run_tui(_state: object, **kwargs: object):
        captured["command_runner"] = kwargs["command_runner"]
        return (("exit", ""), [])

    monkeypatch.setattr(tui_pkg, "run_tui", _fake_run_tui)

    result = CliRunner().invoke(
        sylliptor_app,
        [
            "chat",
            "--path",
            os.fspath(tmp_path),
            "--model",
            "m",
            "--base-url",
            "https://x.example/v1",
            "--api-key",
            "k",
            "--no-log",
        ],
        env={
            "SYLLIPTOR_CONFIG_DIR": os.fspath(tmp_path / "cfg"),
            "SYLLIPTOR_DATA_DIR": os.fspath(tmp_path / "data"),
            "SYLLIPTOR_API_KEY": "",
            "OPENAI_API_KEY": "",
        },
    )
    assert result.exit_code == 0, result.output

    runner = captured.get("command_runner")
    assert callable(runner), "the TUI was never handed a command_runner"

    # The spawning form is handed to the worker as a deferred callable.
    action, _out, instruction, run_kwargs = runner(None, "/subagent explorer map auth", 100)
    assert action == "run", "explicit /subagent must not run on the UI thread"
    assert instruction == "/subagent explorer map auth"
    assert callable((run_kwargs or {}).get("_deferred_execute"))

    # The instant toggles must NOT be deferred — they stay on the fast path.
    for text in ("/subagent on", "/subagent status"):
        action, _o, _i, kw = runner(None, text, 100)
        assert not (kw or {}).get("_deferred_execute"), text

    # "/subagent <name>" with no task is answered inline with ONE line — never
    # the classic full usage panel (the dense every-description wall).
    action, out, _i, kw = runner(None, "/subagent explorer", 100)
    assert action == "handled"
    assert not (kw or {}).get("_deferred_execute")
    assert "\n" not in out
    assert "Available subagents:" not in out
    assert "Usage: /subagent" not in out


# --------------------------------------------------- minimal in-run identity


def _tui_surface(events: list[str | None]):
    from sylliptor_agent_cli.cli_impl.tui.surface import TuiSurface
    from sylliptor_agent_cli.cli_impl.tui.transcript import TuiTranscript

    t = TuiTranscript()
    s = TuiSurface(t, auto_approve=lambda: True, on_subagent_activity=events.append)
    return t, s


def _start_event(**over: Any) -> Any:
    from sylliptor_agent_cli.surface.types import SubagentStartEvent

    base: dict[str, Any] = {
        "name": "explorer",
        "mode": "readonly",
        "description": (
            "Use this agent when you need to search the codebase broadly. "
            "It reads excerpts rather than whole files, so it locates code."
        ),
    }
    base.update(over)
    return SubagentStartEvent(**base)


def _end_event(**over: Any) -> Any:
    from sylliptor_agent_cli.surface.types import SubagentEndEvent

    base: dict[str, Any] = {
        "name": "explorer",
        "mode": "readonly",
        "status": "success",
        "elapsed_ms": 1200,
        "steps_completed": 3,
    }
    base.update(over)
    return SubagentEndEvent(**base)


def test_subagent_start_renders_one_minimal_identity_line() -> None:
    # Entering a built-in subagent shows exactly one line naming that agent and
    # its activity tagline — not a dump of the nested
    # run or its full multi-sentence description. The REAL built-in description
    # matters: a different one means a custom definition shadows the name (see
    # the shadowing test below).
    from sylliptor_agent_cli.subagents import built_in_subagents

    events: list[str | None] = []
    t, s = _tui_surface(events)
    s.on_subagent_start(_start_event(description=built_in_subagents()["explorer"].description))
    lines = [text for role, text in t.entries if role == "subagent"]
    assert len(lines) == 1
    assert lines[0] == "↪ explorer · readonly — charting the codebase"
    assert "Use this" not in lines[0]
    # The badge is pinned and the live status names the agent + its tagline.
    assert events == ["explorer"]
    assert t.status is not None and "explorer" in t.status
    assert "charting the codebase" in t.status


def test_custom_subagent_start_falls_back_to_condensed_description() -> None:
    # A custom agent has no built-in tagline — the line falls back to the
    # condensed description (boilerplate stripped).
    events: list[str | None] = []
    t, s = _tui_surface(events)
    s.on_subagent_start(_start_event(name="my-custom"))
    lines = [text for role, text in t.entries if role == "subagent"]
    assert len(lines) == 1
    assert lines[0].startswith("↪ my-custom · readonly — ")
    assert "Search the codebase broadly" in lines[0]
    assert "Use this agent" not in lines[0]


def test_subagent_badge_shows_even_at_trace_off_but_line_does_not() -> None:
    events: list[str | None] = []
    t, s = _tui_surface(events)
    s.set_trace_level("off")
    s.on_subagent_start(_start_event())
    assert events == ["explorer"]  # identity is not trace — the badge still pins
    assert not any(role == "subagent" for role, _ in t.entries)


def test_subagent_start_without_description_keeps_the_line_short() -> None:
    # No tagline (custom agent) and no description → just the identity line.
    events: list[str | None] = []
    t, s = _tui_surface(events)
    s.on_subagent_start(_start_event(name="my-custom", description=""))
    lines = [text for role, text in t.entries if role == "subagent"]
    assert lines == ["↪ my-custom · readonly"]


def test_nested_tool_steps_stay_out_of_the_transcript() -> None:
    # The nested run's step-by-step ✓ chatter is what made entering a subagent
    # read as a flood; at the default trace level it lives only in the live
    # status line (named, so the user knows who is working).
    from sylliptor_agent_cli.surface.types import ToolEndEvent, ToolStartEvent

    events: list[str | None] = []
    t, s = _tui_surface(events)
    s.on_subagent_start(_start_event())
    before = list(t.entries)
    s.on_tool_start(
        ToolStartEvent(
            tool_call_id="sub:1",
            name="web_search",
            args={"query": "auth flow"},
            step=1,
            subagent_name="explorer",
            subagent_mode="readonly",
            nesting_depth=1,
        )
    )
    assert t.status is not None
    assert "explorer" in t.status and "auth flow" in t.status
    s.on_tool_end(
        ToolEndEvent(
            tool_call_id="sub:1",
            name="web_search",
            status="done",
            elapsed_ms=10,
            subagent_name="explorer",
            subagent_mode="readonly",
            nesting_depth=1,
        )
    )
    assert t.entries == before  # no ✓ line committed for the nested success
    assert t.status is not None and "explorer" in t.status  # still visibly working


def test_nested_tool_failures_keep_their_error_line_with_attribution() -> None:
    from sylliptor_agent_cli.surface.types import ToolEndEvent, ToolStartEvent

    events: list[str | None] = []
    t, s = _tui_surface(events)
    s.on_subagent_start(_start_event())
    s.on_tool_start(
        ToolStartEvent(
            tool_call_id="sub:2",
            name="web_search",
            args={"query": "auth flow"},
            step=1,
            subagent_name="explorer",
            subagent_mode="readonly",
            nesting_depth=1,
        )
    )
    s.on_tool_end(
        ToolEndEvent(
            tool_call_id="sub:2",
            name="web_search",
            status="error",
            elapsed_ms=10,
            meta={"error": "network unreachable"},
            subagent_name="explorer",
            subagent_mode="readonly",
            nesting_depth=1,
        )
    )
    errors = [text for role, text in t.entries if role == "error"]
    assert len(errors) == 1
    assert errors[0].startswith("✗ explorer ▸ ")
    assert "network unreachable" in errors[0]
    # The argument preview survives so the user can tell WHICH invocation
    # failed when the agent retries the same tool with different arguments.
    assert "auth flow" in errors[0]


def test_trace_off_keeps_nested_activity_quiet() -> None:
    # /trace off means a quiet surface: nested tool events show the generic
    # "Working…" (start) and clear (end) — never a named "↪ …" status line —
    # and the subagent's end never strands a leftover status.
    from sylliptor_agent_cli.surface.types import ToolEndEvent, ToolStartEvent

    events: list[str | None] = []
    t, s = _tui_surface(events)
    s.set_trace_level("off")
    s.on_subagent_start(_start_event())
    s.on_tool_start(
        ToolStartEvent(
            tool_call_id="sub:off",
            name="web_search",
            args={"query": "auth flow"},
            step=1,
            subagent_name="explorer",
            subagent_mode="readonly",
            nesting_depth=1,
        )
    )
    assert t.status == "Working…"  # generic, not "↪ explorer · …"
    s.on_tool_end(
        ToolEndEvent(
            tool_call_id="sub:off",
            name="web_search",
            status="done",
            elapsed_ms=10,
            subagent_name="explorer",
            subagent_mode="readonly",
            nesting_depth=1,
        )
    )
    assert t.status is None
    s.on_subagent_end(_end_event())
    assert t.status is None  # nothing stranded after the run
    assert not any(role in ("subagent", "trace") for role, _ in t.entries)


def test_full_trace_level_opts_back_into_nested_detail() -> None:
    from sylliptor_agent_cli.surface.types import ToolEndEvent, ToolStartEvent

    events: list[str | None] = []
    t, s = _tui_surface(events)
    s.set_trace_level("full")
    s.on_subagent_start(_start_event())
    s.on_tool_start(
        ToolStartEvent(
            tool_call_id="sub:3",
            name="web_search",
            args={"query": "auth flow"},
            step=1,
            subagent_name="explorer",
            subagent_mode="readonly",
            nesting_depth=1,
        )
    )
    s.on_tool_end(
        ToolEndEvent(
            tool_call_id="sub:3",
            name="web_search",
            status="done",
            elapsed_ms=10,
            subagent_name="explorer",
            subagent_mode="readonly",
            nesting_depth=1,
        )
    )
    assert any(role == "trace" and text.startswith("✓") for role, text in t.entries)


def test_tui_subagent_identity_precedes_first_nested_tool_trace() -> None:
    from sylliptor_agent_cli.surface.types import ToolStartEvent

    events: list[str | None] = []
    t, s = _tui_surface(events)
    s.set_trace_level("full")
    s.on_subagent_start(_start_event())
    s.on_tool_start(
        ToolStartEvent(
            tool_call_id="sub:ordered",
            name="fs_read",
            args={"path": "README.md"},
            step=1,
            subagent_name="explorer",
            subagent_mode="readonly",
            nesting_depth=1,
        )
    )

    visible_roles = [role for role, _text in t.entries if role in {"subagent", "trace"}]
    assert visible_roles[:2] == ["subagent", "trace"]


def test_subagent_end_clears_badge_and_appends_finish_line() -> None:
    events: list[str | None] = []
    t, s = _tui_surface(events)
    s.on_subagent_start(_start_event())
    s.on_subagent_end(_end_event())
    assert events == ["explorer", None]
    assert any(
        role == "trace" and text.startswith("↩ explorer · finished · 3 steps")
        for role, text in t.entries
    )
    assert t.status is None


def test_new_turn_clears_a_stranded_subagent_badge() -> None:
    # An interrupted run may never deliver its end event; the next submission
    # must not inherit its badge.
    events: list[str | None] = []
    _t, s = _tui_surface(events)
    s.on_subagent_start(_start_event())
    s.on_user_message("next question")
    assert events == ["explorer", None]


def test_footer_pins_the_active_subagent() -> None:
    from sylliptor_agent_cli.cli_impl.tui.footer import footer_fragments
    from sylliptor_agent_cli.cli_impl.tui.state import TuiState

    def _plain(fragments) -> str:
        return "".join(text for _style, text in fragments)

    assert TuiState().active_subagent == ""
    idle = _plain(footer_fragments(TuiState(model_name="m", username="u"), width=120))
    assert "↪" not in idle
    busy_state = TuiState(model_name="m", username="u", active_subagent="explorer")
    busy = _plain(footer_fragments(busy_state, width=120))
    assert "↪ explorer" in busy
    long_state = TuiState(model_name="m", active_subagent="a-very-long-subagent-name")
    clipped = _plain(footer_fragments(long_state, width=120))
    assert "↪ a-very-long-sub…" in clipped


def test_footer_badge_wears_the_subagents_own_accent() -> None:
    # The badge is tinted per agent — WHICH subagent is glanceable, not just
    # that one is running.
    from sylliptor_agent_cli.cli_impl.tui.footer import footer_fragments
    from sylliptor_agent_cli.cli_impl.tui.state import TuiState
    from sylliptor_agent_cli.cli_impl.tui.subagent_identity import subagent_identity

    def _badge_style(name: str) -> str:
        fragments = footer_fragments(TuiState(model_name="m", active_subagent=name), width=120)
        return next(style for style, text in fragments if "↪" in text)

    assert subagent_identity("explorer").color in _badge_style("explorer")
    assert subagent_identity("debugger").color in _badge_style("debugger")
    assert _badge_style("explorer") != _badge_style("debugger")


def test_concurrent_subagents_keep_the_badge_and_status_honest() -> None:
    # A parallel readonly batch really does run several subagents against the
    # SAME surface (turn/core.py prelaunch path), so the badge must survive one
    # of them finishing: pop is by name, the badge falls back to whoever is
    # still running, and the status only clears when the last one ends.
    events: list[str | None] = []
    t, s = _tui_surface(events)
    s.on_subagent_start(_start_event(name="explorer"))
    s.on_subagent_start(_start_event(name="debugger"))
    assert events == ["explorer", "debugger"]
    # The FIRST agent finishes while the second is still working: the badge
    # re-pins to the survivor (not None, not the finished name) and the live
    # status is not wiped mid-run.
    s.on_subagent_end(_end_event(name="explorer"))
    assert events[-1] == "debugger"
    # The activity line rolls to the survivor — never left attributed to the
    # agent whose ↩ line just printed.
    assert t.status is not None and "debugger" in t.status
    # The last agent finishes: badge clears, status clears.
    s.on_subagent_end(_end_event(name="debugger"))
    assert events[-1] is None
    assert t.status is None


def test_end_event_with_no_matching_start_is_dropped_whole() -> None:
    # A stray end (a failure path that already reported, or an abandoned turn's
    # agent finishing late) must not pop someone else's badge, print a phantom
    # ↩ line, or touch the status.
    events: list[str | None] = []
    t, s = _tui_surface(events)
    s.on_subagent_start(_start_event(name="explorer"))
    s.on_subagent_end(_end_event(name="debugger"))
    assert events == ["explorer"]  # no badge re-sync for the stray end
    assert not any("↩ debugger" in text for _r, text in t.entries)


def test_stale_end_from_another_thread_cannot_pop_a_live_same_named_agent() -> None:
    # After an interrupt the abandoned turn's pool threads may deliver end
    # events for the SAME name the next turn is running. A run's start and end
    # share a thread, so a foreign thread's end must find no entry and be
    # dropped — the live agent keeps its badge, status word, and ↩-less
    # transcript.
    import threading

    events: list[str | None] = []
    t, s = _tui_surface(events)
    s.on_subagent_start(_start_event(name="explorer"))
    stale = threading.Thread(target=lambda: s.on_subagent_end(_end_event(name="explorer")))
    stale.start()
    stale.join()
    assert events == ["explorer"]  # badge untouched (no None, no re-pin)
    assert not any(text.startswith("↩ explorer") for _r, text in t.entries)
    # The live run can still end normally on its own thread.
    s.on_subagent_end(_end_event(name="explorer"))
    assert events == ["explorer", None]


def test_clear_subagent_activity_makes_later_ends_silent() -> None:
    # The soft-interrupt path: the app clears the surface's live-subagent state;
    # the abandoned run's late end must then be a no-op instead of re-pinning a
    # sibling into an idle footer.
    events: list[str | None] = []
    t, s = _tui_surface(events)
    s.on_subagent_start(_start_event(name="explorer"))
    s.on_subagent_start(_start_event(name="debugger"))
    s.clear_subagent_activity()
    assert events[-1] is None
    before = list(t.entries)
    s.on_subagent_end(_end_event(name="debugger"))
    assert events[-1] is None  # no sibling re-pin after the clear
    assert t.entries == before  # no phantom ↩ line either


def test_nested_approval_declined_is_reported_as_the_users_decision() -> None:
    # Declining an approval inside a subagent is the USER's call — it must read
    # "approval declined", never be misreported as a tool failure.
    from sylliptor_agent_cli.surface.types import ToolEndEvent

    events: list[str | None] = []
    t, s = _tui_surface(events)
    s.on_subagent_start(_start_event())
    s.on_tool_end(
        ToolEndEvent(
            tool_call_id="sub:4",
            name="shell_run",
            status="failed",
            elapsed_ms=10,
            meta={"approval_declined": True, "error": "approval declined by user"},
            subagent_name="explorer",
            subagent_mode="readonly",
            nesting_depth=1,
        )
    )
    errors = [text for role, text in t.entries if role == "error"]
    assert len(errors) == 1
    assert errors[0].startswith("✗ explorer ▸ ")
    assert "approval declined" in errors[0]
    assert "failed" not in errors[0]


def test_headless_run_tui_pins_the_badge_and_the_turn_end_backstop_clears_it() -> None:
    # The unit tests above hand TuiSurface a fake callback; this drives the REAL
    # run_tui wiring: on_subagent_activity → state.active_subagent while the
    # turn runs, and the worker's turn-end backstop clearing it even when the
    # end event never arrives (interrupted/crashed nested run).
    from sylliptor_agent_cli.cli_impl.tui import run_tui
    from sylliptor_agent_cli.cli_impl.tui.state import TuiState

    state = TuiState(model_name="test-model", username="t")
    seen_mid_turn: list[str] = []

    class _SpawningSession:
        def __init__(self, surface: Any) -> None:
            self.surface = surface

        def run_turn(self, text: str, *, cancellation_token: Any = None) -> int:
            self.surface.on_user_message(text)
            self.surface.on_subagent_start(_start_event())
            # The badge must be live in the shared state WHILE the subagent
            # works — this is the real wiring, no hand-made callback.
            seen_mid_turn.append(state.active_subagent)
            # No on_subagent_end: simulate a nested run that never reports back.
            self.surface.on_assistant_message_done("done")
            return 0

        def close(self) -> None:
            return None

    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as pipe:
        pipe.send_text("spawn something\r/exit\r")
        run_tui(
            state,
            owl_color=False,
            input=pipe,
            output=DummyOutput(),
            session_builder=lambda surface: _SpawningSession(surface),
            background_turns=False,
        )

    assert seen_mid_turn == ["explorer"]
    assert state.active_subagent == ""  # turn-end backstop cleared the orphan


def test_notice_sink_reports_tool_unavailable_as_an_error(monkeypatch) -> None:
    def _fake_rebuild(*, session: Any, mode: str) -> None:
        session.tools = {}  # rebuild yields no subagent_run

    monkeypatch.setattr(cli_mod, "_rebuild_session_tools_for_mode", _fake_rebuild)
    session = type("Session", (), {})()
    session.subagents_enabled = False
    session.cfg = AppConfig(model="test-model")
    session.mode = "review"
    session.tools = {}
    session.subagent_registry = {}

    notices: list[tuple[str, str]] = []
    _run(session, subagent_notice_sink=lambda role, text: notices.append((role, text)))

    assert ("error", "subagent_run tool is unavailable in this session.") in notices


# ------------------------------------------------------- per-agent identity


def test_builtin_subagent_identities_are_distinct() -> None:
    from sylliptor_agent_cli.cli_impl.tui.subagent_identity import subagent_identity
    from sylliptor_agent_cli.subagents import built_in_subagents

    names = sorted(built_in_subagents())
    identities = [subagent_identity(name) for name in names]
    # Every built-in wears its own colour and tagline — no two agents look the
    # same anywhere the identity is rendered.
    assert len({ident.color for ident in identities}) == len(names)
    assert all(ident.tagline for ident in identities)
    assert len({ident.tagline for ident in identities}) == len(names)


def test_subagent_identity_resolves_aliases_and_is_deterministic_for_customs() -> None:
    from sylliptor_agent_cli.cli_impl.tui.subagent_identity import subagent_identity

    # "explore" is an alias of "explorer" — same identity, not a custom one.
    assert subagent_identity("explore") == subagent_identity("explorer")
    assert subagent_identity(" Explorer ") == subagent_identity("explorer")
    # A custom agent gets no tagline and the SAME colour every time
    # (name-derived, not random).
    custom = subagent_identity("my-custom")
    assert custom.tagline == ""
    assert custom == subagent_identity("my-custom")
    assert custom.color.startswith("#")
    # Nonsense input must never raise.
    assert subagent_identity("").color.startswith("#")


def test_no_task_hint_is_one_identity_line_not_the_usage_wall() -> None:
    from sylliptor_agent_cli.cli_impl.chat.loop import _subagent_no_task_hint
    from sylliptor_agent_cli.subagents import built_in_subagents

    registry = built_in_subagents()
    hint = _subagent_no_task_hint(registry=registry, raw_name="debugger")
    assert hint == (
        "debugger · hunting the root cause — not started; run one task: /subagent debugger <task>"
    )
    # The alias resolves to the canonical agent.
    assert "explorer" in _subagent_no_task_hint(registry=registry, raw_name="explore")
    # One line — never the panel listing every subagent's full description.
    for name in registry:
        line = _subagent_no_task_hint(registry=registry, raw_name=name)
        assert "\n" not in line
        assert "Available subagents:" not in line


def test_no_task_hint_names_the_unknown_agent_compactly() -> None:
    from sylliptor_agent_cli.cli_impl.chat.loop import _subagent_no_task_hint
    from sylliptor_agent_cli.subagents import built_in_subagents

    hint = _subagent_no_task_hint(registry=built_in_subagents(), raw_name="nope")
    assert hint.startswith("Unknown subagent 'nope'")
    assert "picker" in hint
    assert "\n" not in hint
    # Empty registry still answers in one line.
    empty = _subagent_no_task_hint(registry={}, raw_name="nope")
    assert "available: none" in empty


def test_no_task_hint_explains_capability_gated_visual_role() -> None:
    from sylliptor_agent_cli.cli_impl.chat.loop import _subagent_no_task_hint
    from sylliptor_agent_cli.subagents import built_in_subagents

    hint = _subagent_no_task_hint(
        registry=built_in_subagents(include_visual_designer=False),
        raw_name="visual-designer",
        cfg=AppConfig(model="test-model"),
        available_tool_names=set(),
    )

    assert hint.startswith("visual-designer is unavailable:")
    assert "Image generation is disabled" in hint
    assert "image_generation.enabled true" in hint


def test_identity_accents_never_impersonate_fixed_marks() -> None:
    # Violet is Forge, green the chat accent, and cyan the brand. No subagent
    # may wear any of them, or a running agent would read as a mode indicator.
    from sylliptor_agent_cli.cli_impl.tui.subagent_identity import (
        _BUILTIN_IDENTITIES,
        _FALLBACK_COLORS,
    )

    reserved_colors = {"#bc8cff", "#3fb950", "#56b6c2"}
    for name, ident in _BUILTIN_IDENTITIES.items():
        assert ident.color.lower() not in reserved_colors, name
    for color in _FALLBACK_COLORS:
        assert color.lower() not in reserved_colors


def test_footer_badge_styles_are_valid_prompt_toolkit_styles() -> None:
    # A malformed accent would crash only the live TUI (tests read plain text);
    # parse every builtin's badge style plus a custom one through the real
    # prompt_toolkit style machinery.
    from prompt_toolkit.styles import Style

    from sylliptor_agent_cli.cli_impl.tui.footer import footer_fragments
    from sylliptor_agent_cli.cli_impl.tui.state import TuiState
    from sylliptor_agent_cli.cli_impl.tui.subagent_identity import _BUILTIN_IDENTITIES

    style = Style([])
    for name in [*_BUILTIN_IDENTITIES, "my-custom"]:
        fragments = footer_fragments(TuiState(model_name="m", active_subagent=name), width=120)
        badge_style = next(s for s, text in fragments if "↪" in text)
        style.get_attrs_for_style_str(badge_style)  # must not raise


def test_tagline_is_suppressed_when_a_custom_definition_shadows_a_builtin() -> None:
    from sylliptor_agent_cli.cli_impl.tui.subagent_identity import subagent_tagline
    from sylliptor_agent_cli.subagents import built_in_subagents

    builtin_desc = built_in_subagents()["explorer"].description
    assert subagent_tagline("explorer", builtin_desc) == "charting the codebase"
    # No description at hand → trust the name.
    assert subagent_tagline("explorer", "") == "charting the codebase"
    # A custom definition overriding the built-in name keeps its own story.
    assert subagent_tagline("explorer", "Audits license headers only.") == ""
    assert subagent_tagline("some-custom", "whatever") == ""


def test_shadowed_builtin_spawn_line_tells_the_custom_story() -> None:
    from sylliptor_agent_cli.surface.types import ToolEndEvent

    events: list[str | None] = []
    t, s = _tui_surface(events)
    s.on_subagent_start(_start_event(name="explorer", description="Audits license headers only."))
    lines = [text for role, text in t.entries if role == "subagent"]
    assert len(lines) == 1
    assert "charting the codebase" not in lines[0]
    assert "Audits license headers only" in lines[0]
    assert t.status is not None and "charting the codebase" not in t.status
    # The between-step status rollback keeps the honest word too.
    s.on_tool_end(
        ToolEndEvent(
            tool_call_id="sub:9",
            name="fs_read",
            status="done",
            elapsed_ms=5,
            subagent_name="explorer",
            subagent_mode="readonly",
            nesting_depth=1,
        )
    )
    assert t.status is not None and "charting the codebase" not in t.status


def test_picker_rows_are_plain_names_and_submit_raw_names() -> None:
    from sylliptor_agent_cli.cli_impl.chat.loop import _subagent_picker_row_specs
    from sylliptor_agent_cli.subagents import built_in_subagents

    registry = built_in_subagents()
    rows = _subagent_picker_row_specs(registry=registry)
    assert [row["value"] for row in rows] == sorted(registry)
    for row in rows:
        # The label is the undecorated name — the same string the runner and
        # prefill expect as the submitted value.
        assert row["label"] == row["value"]
        assert row["description"]


def test_no_subagent_surface_wears_a_per_agent_symbol() -> None:
    # A subagent is identified by its NAME, never by a mark of its own. The
    # shared ↪/↩ marks (a nested run started / ended) are the only symbols any
    # subagent surface may carry; a per-agent glyph creeping back into ANY of
    # the four render sites is the regression this pins.
    from sylliptor_agent_cli.cli_impl.chat.loop import (
        _subagent_no_task_hint,
        _subagent_picker_row_specs,
    )
    from sylliptor_agent_cli.subagents import built_in_subagents

    banned = set("▣✧◎✦◉❖△◆◇")
    registry = built_in_subagents()

    rendered: list[str] = [
        str(row["label"]) for row in _subagent_picker_row_specs(registry=registry)
    ]
    for name in [*registry, "my-custom"]:
        events: list[str | None] = []
        t, s = _tui_surface(events)
        s.on_subagent_start(_start_event(name=name))
        rendered.extend(text for role, text in t.entries if role == "subagent")
        rendered.append(_subagent_no_task_hint(registry=registry, raw_name=name))
        rendered.append(_chat_subagent_result_body(subagent_name=name, result=_result()))

    assert rendered
    for text in rendered:
        assert not (set(text) & banned), text
