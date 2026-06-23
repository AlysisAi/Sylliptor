"""Phase 2 TUI tests: the conversation model, the agent surface, and a headless
end-to-end run that streams a fake agent turn into the transcript.

Turns run inline (``background_turns=False``) so ordering is deterministic.
"""

from __future__ import annotations

from sylliptor_agent_cli.cli_impl.tui import run_tui
from sylliptor_agent_cli.cli_impl.tui.state import TuiState
from sylliptor_agent_cli.cli_impl.tui.surface import TuiSurface
from sylliptor_agent_cli.cli_impl.tui.transcript import TuiTranscript
from sylliptor_agent_cli.surface.types import (
    ApprovalRequest,
    ToolEndEvent,
    ToolStartEvent,
)

# --------------------------- transcript model ---------------------------


def test_transcript_streams_assistant_into_one_block():
    t = TuiTranscript()
    t.append_user("hi")
    t.begin_turn()
    t.stream_assistant("Hel")
    t.stream_assistant("lo")
    t.finish_assistant("Hello")
    assert t.entries == [("user", "hi"), ("assistant", "Hello")]


def test_transcript_finish_uses_final_when_no_stream():
    t = TuiTranscript()
    t.begin_turn()
    t.finish_assistant("done")
    assert ("assistant", "done") in t.entries


def test_transcript_load_history_keeps_user_assistant_drops_tools():
    # /resume reload: prior conversation replaces the pane; only user/assistant
    # text turns survive (tool calls/results and blank turns are dropped).
    t = TuiTranscript()
    t.append_user("stale")  # pre-existing content must be cleared first
    t.load_history(
        [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "tool output"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer\n"},
        ]
    )
    assert t.entries == [
        ("user", "first question"),
        ("assistant", "first answer"),
        ("user", "second question"),
        ("assistant", "second answer"),
    ]
    # No assistant block is left "open", so every reply renders as completed.
    assert t.snapshot()[2] is None


def test_transcript_load_history_empty_clears():
    t = TuiTranscript()
    t.append_user("stale")
    t.load_history([])
    assert t.entries == []


def test_surface_replace_history_reloads_transcript():
    t = TuiTranscript()
    surface = TuiSurface(t, auto_approve=lambda: False)
    surface.replace_history(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    )
    assert t.entries == [("user", "q"), ("assistant", "a")]


def test_surface_append_note_uses_given_role():
    # The resume outcome line picks its role so it can flip the welcome→chat pane
    # (assistant) or stay a dim status (system) as needed.
    t = TuiTranscript()
    surface = TuiSurface(t, auto_approve=lambda: False)
    surface.append_note("Resumed session: x (0 turns loaded).", role="assistant")
    surface.append_note("plain status")  # defaults to system
    surface.append_note("   ")  # blank is dropped
    assert t.entries == [
        ("assistant", "Resumed session: x (0 turns loaded)."),
        ("system", "plain status"),
    ]


def test_transcript_status_is_transient():
    t = TuiTranscript()
    t.set_status("Thinking…")
    assert t.status == "Thinking…"
    t.stream_assistant("x")
    assert t.status is None


def test_transcript_invalidate_fires_on_mutation():
    hits = {"n": 0}
    t = TuiTranscript(invalidate=lambda: hits.__setitem__("n", hits["n"] + 1))
    t.append_user("hi")
    assert hits["n"] >= 1


# --------------------------- surface ---------------------------


def test_surface_streams_tokens_and_done():
    t = TuiTranscript()
    s = TuiSurface(t, auto_approve=lambda: True)
    s.on_user_message("hi")
    s.on_assistant_token("Hello ")
    s.on_assistant_token("world")
    s.on_assistant_message_done("Hello world")
    assert ("assistant", "Hello world") in t.entries


def test_surface_renders_tool_trace():
    t = TuiTranscript()
    s = TuiSurface(t, auto_approve=lambda: True)
    s.on_tool_start(ToolStartEvent(tool_call_id="1", name="read_file", args={}, step=1))
    # While running, the tool shows via the live status (which drives the single
    # under-question activity indicator), not a committed "⚙ start" line.
    assert t.status
    assert not any(role == "trace" for role, _ in t.entries)
    s.on_tool_end(ToolEndEvent(tool_call_id="1", name="read_file", status="done", elapsed_ms=1200))
    roles = [role for role, _ in t.entries]
    assert roles.count("trace") == 1  # only the completion line is recorded
    assert any(text.startswith("✓") for _r, text in t.entries)
    assert t.status is None


def test_surface_refreshes_hud_mid_turn():
    # The footer HUD (context/tokens/cost) must advance DURING a long multi-step
    # turn, not only when it ends: the surface calls on_hud_refresh at safe points
    # (message-done, tool-end) on the worker thread, throttled to avoid re-running
    # on every step.
    t = TuiTranscript()
    calls = {"n": 0}
    s = TuiSurface(
        t,
        auto_approve=lambda: True,
        on_hud_refresh=lambda: calls.__setitem__("n", calls["n"] + 1),
    )
    s.on_user_message("go")
    s.on_assistant_message_done("calling a tool")
    assert calls["n"] >= 1  # refreshed mid-turn, before the turn completed
    after_msg = calls["n"]
    s._hud_last_refresh = 0.0  # step past the throttle window
    s.on_tool_end(ToolEndEvent(tool_call_id="1", name="shell_run", status="done", elapsed_ms=10))
    assert calls["n"] == after_msg + 1  # tool-end is another safe refresh point
    # Throttle: a second immediate tool-end must NOT re-fire.
    s.on_tool_end(ToolEndEvent(tool_call_id="2", name="shell_run", status="done", elapsed_ms=10))
    assert calls["n"] == after_msg + 1


def test_surface_hud_refresh_optional():
    # Without an on_hud_refresh callback the surface must behave exactly as before
    # (no crash, no extra work) — the hook is purely additive.
    t = TuiTranscript()
    s = TuiSurface(t, auto_approve=lambda: True)
    s.on_assistant_message_done("done")
    s.on_tool_end(ToolEndEvent(tool_call_id="1", name="read_file", status="done", elapsed_ms=5))
    assert any(text.startswith("✓") for _r, text in t.entries)


def test_surface_renders_failed_tool_as_error():
    t = TuiTranscript()
    s = TuiSurface(t, auto_approve=lambda: True)
    s.on_tool_end(
        ToolEndEvent(
            tool_call_id="1",
            name="shell_run",
            status="error",
            elapsed_ms=50,
            meta={"error": "boom"},
        )
    )
    assert any(role == "error" and "boom" in text for role, text in t.entries)


def test_surface_auto_approve_allows():
    t = TuiTranscript()
    s = TuiSurface(t, auto_approve=lambda: True)
    decision = s.request_approval(
        ApprovalRequest(kind="fs_write", reason="r", preview="p", files=["a.py"])
    )
    assert decision.allow is True


def test_surface_denies_when_auto_off_and_no_ui():
    t = TuiTranscript()
    s = TuiSurface(t, auto_approve=lambda: False, request_approval_ui=None)
    decision = s.request_approval(
        ApprovalRequest(kind="fs_write", reason="r", preview="p", files=["a.py"])
    )
    assert decision.allow is False
    assert any(role == "warn" for role, _ in t.entries)


def test_surface_emit_error_warning_delegate_to_render():
    t = TuiTranscript()
    s = TuiSurface(t, auto_approve=lambda: True)
    s.emit_error("terminal_error", "boom", False)
    s.emit_warning("careful")
    assert any(role == "error" and "boom" in text for role, text in t.entries)
    assert any(role == "warn" and "careful" in text for role, text in t.entries)


def test_surface_emit_probe_not_mistaken_for_noop():
    # Regression for the high-severity bug: a synthesized no-op emit_error (e.g.
    # via __getattr__) makes the runtime's capability probe believe the surface
    # handles errors and skip the on_error render path. Real class-level methods
    # must differ from NoopSurface, and absent additive emit_* must stay absent.
    from sylliptor_agent_cli.surface.noop_surface import NoopSurface

    assert getattr(TuiSurface, "emit_error", None) is not getattr(NoopSurface, "emit_error", None)
    assert getattr(TuiSurface, "emit_warning", None) is not getattr(
        NoopSurface, "emit_warning", None
    )
    s = TuiSurface(TuiTranscript(), auto_approve=lambda: True)
    assert getattr(s, "emit_message_delta", None) is None
    assert getattr(s, "emit", None) is None


def test_runtime_emit_surface_error_reaches_transcript():
    # End-to-end: drive the actual runtime helper that chooses emit_* vs on_*.
    from sylliptor_agent_cli.agent.turn.core import _emit_surface_error

    t = TuiTranscript()
    s = TuiSurface(t, auto_approve=lambda: True)
    _emit_surface_error(s, "terminal_error", "TOOL BLEW UP", False)
    assert any(role == "error" and "TOOL BLEW UP" in text for role, text in t.entries)


# --------------------------- headless end-to-end ---------------------------


class _FakeSession:
    """Minimal stand-in for AgentSession that drives the surface."""

    def __init__(self, surface: TuiSurface) -> None:
        self.surface = surface
        self.closed = False

    def run_turn(self, text: str, *, cancellation_token=None) -> int:
        self.surface.on_user_message(text)
        self.surface.on_assistant_token("Echo: ")
        self.surface.on_assistant_token(text)
        self.surface.on_assistant_message_done(f"Echo: {text}")
        return 0

    def close(self) -> None:
        self.closed = True


def _run_headless(state: TuiState, keys: str, **kwargs):
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        return run_tui(state, owl_color=False, input=pipe, output=DummyOutput(), **kwargs)


def test_headless_runs_agent_turn():
    state = TuiState(model_name="deepseek-chat", username="t")
    sessions: list[_FakeSession] = []

    def _builder(surface):
        sess = _FakeSession(surface)
        sessions.append(sess)
        return sess

    completed = {"n": 0}
    _result, transcript = _run_headless(
        state,
        "hi there\r/exit\r",
        session_builder=_builder,
        on_turn_complete=lambda: completed.__setitem__("n", completed["n"] + 1),
        background_turns=False,
    )
    assert ("user", "hi there") in transcript
    assert ("assistant", "Echo: hi there") in transcript
    assert completed["n"] == 1
    assert sessions and sessions[0].surface is not None


def test_user_band_rows_full_width_with_prompt():
    from sylliptor_agent_cli.cli_impl.tui.app import _user_band_rows

    width = 40
    rows = _user_band_rows("hi", width)
    assert len(rows) == 3  # blank pad + text + blank pad
    for row in rows:
        assert sum(len(t) for _s, t in row) == width  # every row spans full width
    text_row = "".join(t for _s, t in rows[1])
    assert text_row.startswith("› hi")


def test_user_band_rows_wraps_long_message():
    from sylliptor_agent_cli.cli_impl.tui.app import _user_band_rows

    width = 24
    rows = _user_band_rows("a fairly long message that wraps", width)
    for row in rows:
        assert sum(len(t) for _s, t in row) == width
    assert len(rows) >= 4  # pad + >=2 text rows + pad


def test_assistant_rows_have_marker():
    from sylliptor_agent_cli.cli_impl.tui.app import _assistant_rows

    rows = _assistant_rows("Hello\nworld")
    first = "".join(t for _s, t in rows[0])
    assert first.startswith("✦ Hello")
    assert "".join(t for _s, t in rows[1]) == "  world"


def _row_width(row) -> int:
    return sum(len(t) for _s, t in row)


def test_assistant_rows_plain_wrap_to_width_keeps_follow_accurate():
    # Regression: the transcript window wraps lines on screen (wrap_lines=True), so
    # an over-wide emitted row becomes extra UNcounted screen rows and the follow
    # math undershoots, hiding the live "thinking" line behind the footer. Every
    # emitted row must be <= width so logical rows == screen rows.
    from sylliptor_agent_cli.cli_impl.tui.app import _assistant_rows

    width = 30
    rows = _assistant_rows("word " * 40, width, markdown=False)
    assert len(rows) > 1
    assert all(_row_width(row) <= width for row in rows)
    assert "".join(t for _s, t in rows[0]).startswith("✦ ")


def test_assistant_rows_hard_break_long_url():
    from sylliptor_agent_cli.cli_impl.tui.app import _assistant_rows

    width = 24
    url = "https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/canadamexicousa2026"
    rows = _assistant_rows(url, width, markdown=False)
    assert all(_row_width(row) <= width for row in rows)


def test_plain_role_rows_wrap_long_line_to_width():
    from sylliptor_agent_cli.cli_impl.tui.app import _plain_role_rows

    width = 28
    text = "X Search Web failed (42.3s): OpenRouter web_search timed out during response read"
    rows = _plain_role_rows("class:tui.transcript.error", text, width)
    assert len(rows) > 1
    assert all(_row_width(row) <= width for row in rows)
    joined = " ".join("".join(t for _s, t in row) for row in rows)
    assert "OpenRouter" in joined and "timed" in joined


def test_wrap_line_preserves_blank_and_breaks_long_token():
    from sylliptor_agent_cli.cli_impl.tui.app import _wrap_line

    assert _wrap_line("", 10) == [""]
    chunks = _wrap_line("a" * 25, 10)
    assert chunks and all(len(c) <= 10 for c in chunks)
    assert "".join(chunks) == "a" * 25


def test_followup_placeholder_is_short_and_distinct():
    from sylliptor_agent_cli.cli_impl.tui.content import (
        INPUT_PLACEHOLDER,
        INPUT_PLACEHOLDER_FOLLOWUP,
    )

    assert INPUT_PLACEHOLDER_FOLLOWUP != INPUT_PLACEHOLDER
    assert len(INPUT_PLACEHOLDER_FOLLOWUP) < len(INPUT_PLACEHOLDER)
    assert INPUT_PLACEHOLDER_FOLLOWUP.lower().strip(" .…") != "ask anything"


def test_scroll_target_clamps_and_reports_follow():
    from sylliptor_agent_cli.cli_impl.tui.app import _scroll_target

    assert _scroll_target(20, 20, -10) == (10, False)  # scroll up off the tail
    assert _scroll_target(10, 20, 10) == (20, True)  # back to the tail → follow
    assert _scroll_target(3, 20, -10) == (0, False)  # cannot pass the top
    assert _scroll_target(0, 0, -10) == (0, True)  # content fits → always tail


def test_scrollable_control_routes_wheel_events():
    from prompt_toolkit.mouse_events import MouseEventType

    from sylliptor_agent_cli.cli_impl.tui.app import _ScrollableControl

    seen: list = []
    ctrl = _ScrollableControl(lambda: [], on_scroll=lambda d: seen.append(d))

    class _Evt:
        def __init__(self, et):
            self.event_type = et

    assert ctrl.mouse_handler(_Evt(MouseEventType.SCROLL_UP)) is None
    assert ctrl.mouse_handler(_Evt(MouseEventType.SCROLL_DOWN)) is None
    assert ctrl.mouse_handler(_Evt(MouseEventType.MOUSE_UP)) is NotImplemented
    assert seen == [-1, 1]


def test_headless_pageup_pagedown_do_not_crash():
    state = TuiState(model_name="m", username="t")
    result, transcript = _run_headless(
        state,
        "hello\r\x1b[5~\x1b[6~/exit\r",  # message, PageUp, PageDown, exit
        session_builder=_FakeSession,
        command_runner=_fake_command_runner([]),
        background_turns=False,
    )
    assert result == "/exit"
    assert ("user", "hello") in transcript


def _fake_command_runner(calls):
    def runner(session, text, width):
        calls.append((text, width))
        low = text.strip().lower()
        if low in ("/exit", "exit"):
            return ("exit", "", None, None)
        if low == "/status":
            return ("handled", "Status: ok", None, None)
        return ("run", "", text, {})

    return runner


def test_headless_slash_command_handled_renders_output():
    state = TuiState(model_name="m", username="t")
    calls: list = []
    _result, transcript = _run_headless(
        state,
        "/status\r/exit\r",
        session_builder=_FakeSession,
        command_runner=_fake_command_runner(calls),
        background_turns=False,
    )
    assert ("user", "/status") in transcript
    assert any(role == "system" and "Status:" in text for role, text in transcript)
    assert not any(role == "assistant" for role, _ in transcript)
    assert calls and calls[0][0] == "/status"


def test_headless_slash_help_opens_popup_not_routed_to_runner():
    # /help is intercepted natively: it opens the centered popup instead of being
    # echoed as a user line or routed to the command runner. Pressing q closes it,
    # then /exit leaves.
    state = TuiState(model_name="m", username="t")
    calls: list = []
    _result, transcript = _run_headless(
        state,
        "/help\rq/exit\r",
        session_builder=_FakeSession,
        command_runner=_fake_command_runner(calls),
        background_turns=False,
    )
    assert ("user", "/help") not in transcript
    assert not any("/help" in text for _role, text in transcript)
    assert all(text.strip().lower() != "/help" for text, _w in calls)


def test_help_popup_rows_render_green_commands_and_descriptions():
    from sylliptor_agent_cli.cli_impl.tui.app import _help_inner_width, _help_rows_for_sections

    sections = [
        ("Getting Started", [("/help", "commands & config"), ("/status", "session details")]),
        ("Execution", [("/mode", "change execution mode")]),
    ]
    width = _help_inner_width(100)
    rows = _help_rows_for_sections(sections, width)
    # Every row is padded to the panel width (solid background block).
    assert all(sum(len(t) for _s, t in row) == width for row in rows)
    # Commands render in the green command style, left-aligned in a shared column.
    cmd_rows = [row for row in rows if any(s == "class:tui.help.cmd" for s, _t in row)]
    assert len(cmd_rows) == 3  # /help, /status, /mode
    cmd_texts = ["".join(t for s, t in row if s == "class:tui.help.cmd") for row in cmd_rows]
    assert any(c.startswith("/help") for c in cmd_texts)
    # Shared left column width → every command cell is padded to the same length.
    assert len({len(c) for c in cmd_texts}) == 1
    # Section headers and a closing hint are present.
    assert any(any(s == "class:tui.help.section" for s, _t in row) for row in rows)
    assert any(any(s == "class:tui.help.hint" for s, _t in row) for row in rows)


def test_kv_panel_rows_render_toned_values_and_full_width():
    from sylliptor_agent_cli.cli_impl.tui.app import _render_kv_panel_rows

    sections = [
        ("Session", [("mode", "fast (auto)", "accent"), ("dirty", "no", "accent")]),
        ("Web search", [("status", "unavailable", "err"), ("note", "x" * 80, "plain")]),
    ]
    rows = _render_kv_panel_rows(sections, 50)
    # Every row padded to the panel width (solid background block).
    assert all(sum(len(t) for _s, t in row) == 50 for row in rows)
    # Keys render in the dim key column; healthy values in green; errors in red.
    assert any(any(s == "class:tui.help.key" for s, _t in row) for row in rows)
    assert any(any(s == "class:tui.help.accent" for s, _t in row) for row in rows)
    assert any(any(s == "class:tui.help.err" for s, _t in row) for row in rows)
    # Long values wrap with a hanging indent (more body rows than logical values).
    assert any(any(s == "class:tui.help.section" for s, _t in row) for row in rows)
    assert any(any(s == "class:tui.help.hint" for s, _t in row) for row in rows)


def test_headless_status_panel_opens_via_provider_not_routed_to_runner():
    # /status is intercepted by its panel provider: it opens the centered popup
    # instead of being echoed as a user line or routed to the command runner.
    # Pressing q closes it, then /exit leaves.
    state = TuiState(model_name="m", username="t")
    calls: list = []
    opened = {"n": 0}

    def _status_provider(arg=""):
        opened["n"] += 1
        return {
            "title": "Session Status",
            "sections": [("Session", [("mode", "auto", "accent")])],
        }

    _result, transcript = _run_headless(
        state,
        "/status\rq/exit\r",
        session_builder=_FakeSession,
        command_runner=_fake_command_runner(calls),
        panel_providers={"/status": _status_provider},
        background_turns=False,
    )
    assert opened["n"] == 1  # provider was invoked → panel opened
    assert ("user", "/status") not in transcript
    assert not any(text.strip().lower() == "/status" for text, _w in calls)


def test_slash_completer_lists_commands_and_excludes_removed_stream():
    # The dropdown content: typing "/" lists commands; "/stream" was deleted so it
    # must not appear; prefix filtering narrows the list.
    from prompt_toolkit.document import Document

    from sylliptor_agent_cli.cli_impl.chat_slash_completer import ChatSlashCompleter

    completer = ChatSlashCompleter(mode_provider=lambda: "chat")

    def comps(text: str) -> list[str]:
        return [c.text for c in completer.get_completions(Document(text, len(text)), None)]

    top = comps("/")
    assert "/status" in top
    assert "/help" in top
    assert "/stream" not in top  # deleted command must not surface in the dropdown
    narrowed = comps("/st")
    assert "/status" in narrowed
    assert all(c.startswith("/st") for c in narrowed)


def test_cancellation_token_contract_raises_keyboardinterrupt():
    # run_turn's _throw_if_cancelled calls token.throw_if_cancelled(...) and relies
    # on it raising to abort mid-stream. Lock that contract so interrupt can't
    # silently regress.
    import pytest

    from sylliptor_agent_cli.cli_impl.tui.app import _Cancellation

    tok = _Cancellation()
    assert tok.is_cancelled is False
    tok.throw_if_cancelled("noop")  # no-op before cancel
    tok.cancel()
    assert tok.is_cancelled is True
    with pytest.raises(KeyboardInterrupt):
        tok.throw_if_cancelled("cancelled_by_user")


def test_surface_drops_output_for_cancelled_worker():
    # After a soft-interrupt the worker's token is cancelled; the surface must drop
    # its (late) streamed output and auto-deny approvals so an abandoned turn can't
    # paint into the transcript or pop a modal.
    from sylliptor_agent_cli.cli_impl.tui.surface import set_active_cancellation

    class _CancelledTok:
        is_cancelled = True

    t = TuiTranscript()
    s = TuiSurface(t, auto_approve=lambda: False)
    set_active_cancellation(_CancelledTok())
    try:
        s.on_reasoning_token("thinking")
        s.on_assistant_token("hello")
        s.on_assistant_message_done("hello")
        assert all("hello" not in text for _role, text in t.entries)
        assert all("thinking" not in text for _role, text in t.entries)
        decision = s.request_approval(
            ApprovalRequest(kind="fs_write", reason="r", preview="p", files=["a.py"])
        )
        assert decision.allow is False
    finally:
        set_active_cancellation(None)  # reset thread-local; don't leak to other tests


def test_approval_modal_rows_render_colored_keys_and_full_width():
    from types import SimpleNamespace

    from sylliptor_agent_cli.cli_impl.tui.app import _render_approval_rows

    req = SimpleNamespace(
        kind="fs_write", command="", files=["approval_demo.txt"], reason="review mode"
    )
    rows = _render_approval_rows(req, 60)
    # Solid background block + colour-coded y/a/n keys + bright target.
    assert all(sum(len(t) for _s, t in row) == 60 for row in rows)
    styles = {s for row in rows for s, _t in row}
    assert "class:tui.approve.head" in styles  # amber headline (non-destructive)
    assert "class:tui.approve.target" in styles
    assert {"class:tui.approve.key.yes", "class:tui.approve.key.no"} <= styles
    # A destructive command turns the headline red.
    danger = SimpleNamespace(kind="shell_run", command="rm -rf build", files=[], reason="x")
    danger_styles = {s for row in _render_approval_rows(danger, 60) for s, _t in row}
    assert "class:tui.approve.head.danger" in danger_styles


def _mode_picker_spec(on_select):
    return {
        "title": "Mode",
        "rows": [
            {"label": "safe (review)", "description": "d", "value": "review", "current": True},
            {"label": "fast (auto)", "description": "d", "value": "auto", "current": False},
            {"label": "read (readonly)", "description": "d", "value": "readonly", "current": False},
        ],
        "on_select": on_select,
    }


def test_headless_mode_picker_digit_selects_and_applies():
    # Bare /mode opens the picker (not routed to the runner, not echoed); pressing
    # the number applies that option via on_select and echoes its messages.
    state = TuiState(model_name="m", username="t")
    calls: list = []
    picked = {"value": None}

    def on_select(value):
        picked["value"] = value
        return [("system", f"Mode -> {value}")]

    _result, transcript = _run_headless(
        state,
        "/mode\r2/exit\r",  # open picker, press "2", then exit
        session_builder=_FakeSession,
        command_runner=_fake_command_runner(calls),
        picker_providers={"/mode": lambda: _mode_picker_spec(on_select)},
        background_turns=False,
    )
    assert picked["value"] == "auto"  # digit 2 chose the second option
    assert ("system", "Mode -> auto") in transcript
    assert ("user", "/mode") not in transcript
    assert not any(text.strip().lower() == "/mode" for text, _w in calls)


def test_headless_mode_picker_arrow_then_enter_selects():
    # Down arrow moves the highlight off the current row; Enter applies it.
    state = TuiState(model_name="m", username="t")
    picked = {"value": None}

    def on_select(value):
        picked["value"] = value
        return [("system", f"Mode -> {value}")]

    _result, _transcript = _run_headless(
        state,
        "/mode\r\x1b[B\r/exit\r",  # open, Down (review->auto), Enter, exit
        session_builder=_FakeSession,
        command_runner=_fake_command_runner([]),
        picker_providers={"/mode": lambda: _mode_picker_spec(on_select)},
        background_turns=False,
    )
    assert picked["value"] == "auto"


def test_headless_mode_with_arg_falls_through_to_runner():
    # "/mode fast" (with an arg) must NOT open the picker — it routes to the runner.
    state = TuiState(model_name="m", username="t")
    calls: list = []
    opened = {"n": 0}

    def provider():
        opened["n"] += 1
        return _mode_picker_spec(lambda v: None)

    _run_headless(
        state,
        "/mode fast\r/exit\r",
        session_builder=_FakeSession,
        command_runner=_fake_command_runner(calls),
        picker_providers={"/mode": provider},
        background_turns=False,
    )
    assert opened["n"] == 0  # picker never opened
    assert any(text.strip().lower() == "/mode fast" for text, _w in calls)


def test_headless_with_completer_does_not_crash():
    # Attaching the slash completer (fires on every keystroke via
    # complete_while_typing) must not break normal input/command routing.
    from sylliptor_agent_cli.cli_impl.chat_slash_completer import ChatSlashCompleter

    state = TuiState(model_name="m", username="t")
    calls: list = []
    result, transcript = _run_headless(
        state,
        "/status\r/exit\r",
        session_builder=_FakeSession,
        command_runner=_fake_command_runner(calls),
        completer=ChatSlashCompleter(mode_provider=lambda: "chat"),
        background_turns=False,
    )
    assert result == "/exit"
    assert any(role == "system" and "Status:" in text for role, text in transcript)


def test_headless_plain_message_runs_turn_via_runner():
    state = TuiState(model_name="m", username="t")
    _result, transcript = _run_headless(
        state,
        "hello\r/exit\r",
        session_builder=_FakeSession,
        command_runner=_fake_command_runner([]),
        background_turns=False,
    )
    assert ("user", "hello") in transcript
    assert ("assistant", "Echo: hello") in transcript


def test_headless_slash_clear_empties_transcript():
    state = TuiState(model_name="m", username="t")
    _result, transcript = _run_headless(
        state,
        "hello\r/clear\r/exit\r",
        session_builder=_FakeSession,
        command_runner=_fake_command_runner([]),
        background_turns=False,
    )
    assert ("user", "hello") not in transcript
    assert ("assistant", "Echo: hello") not in transcript


def test_headless_without_session_uses_stub():
    state = TuiState(model_name="deepseek-chat", username="t")
    _result, transcript = _run_headless(state, "hello\r/exit\r")
    assert ("user", "hello") in transcript
    assert any(role == "system" for role, _ in transcript)
