"""Phase 1 TUI tests: pure content builders + a headless Application smoke test.

The Application is driven with a prompt_toolkit pipe input and a dummy output so
no real terminal is required (works in CI / on Windows).
"""

from __future__ import annotations

import pytest

from sylliptor_agent_cli.cli_impl.tui import run_tui
from sylliptor_agent_cli.cli_impl.tui.config import is_tui_enabled
from sylliptor_agent_cli.cli_impl.tui.content import (
    HEADING_TEXT,
    HINT_TEXT,
    pretty_model_label,
)
from sylliptor_agent_cli.cli_impl.tui.footer import footer_fragments
from sylliptor_agent_cli.cli_impl.tui.owl import load_owl_animation
from sylliptor_agent_cli.cli_impl.tui.state import TuiState


def _plain(fragments) -> str:
    return "".join(text for _style, text in fragments)


# --------------------------- flag ---------------------------


def test_tui_enabled_by_default(monkeypatch):
    monkeypatch.delenv("SYLLIPTOR_TUI", raising=False)
    assert is_tui_enabled() is True


@pytest.mark.parametrize(
    "value,expected", [("1", True), ("true", True), ("0", False), ("off", False), ("", True)]
)
def test_tui_flag_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("SYLLIPTOR_TUI", value)
    assert is_tui_enabled() is expected


# --------------------------- content ---------------------------


def test_static_text_matches_target():
    from sylliptor_agent_cli.cli_impl.tui.content import CREDIT_TEXT

    # The heading is the Sylliptor wordmark (the prompt question lives in the box).
    assert HEADING_TEXT == "Sylliptor"
    assert CREDIT_TEXT == "crafted by AlysisAI"
    # Hint leads with Sylliptor's signature command; @ file-mentions aren't wired.
    assert "/forge" in HINT_TEXT
    assert "@" not in HINT_TEXT


def test_input_placeholder_is_sylliptor_greeting():
    from sylliptor_agent_cli.cli_impl.tui.content import INPUT_PLACEHOLDER

    assert "Sylliptor" in INPUT_PLACEHOLDER
    assert "coding buddy" in INPUT_PLACEHOLDER


@pytest.mark.parametrize(
    "model,expected",
    [
        ("deepseek-chat", "DeepSeek Chat"),
        ("gpt-4o", "GPT 4o"),
        ("xiaomi/mimo-v2.5-pro", "MiMo V2.5 Pro"),
        ("", "model"),
        ("openai/gpt-4o-mini", "GPT 4o Mini"),
    ],
)
def test_pretty_model_label(model, expected):
    assert pretty_model_label(model) == expected


# --------------------------- footer ---------------------------


def test_footer_auto_on():
    state = TuiState(model_name="deepseek-chat", username="perdikis")
    text = _plain(footer_fragments(state, width=90))
    assert "sylliptor" in text
    assert "DeepSeek Chat" in text
    assert "context 100%" in text
    assert "0 tokens" in text and "$0.0000" in text
    assert "perdikis" in text
    assert "auto-approve on" in text and "shift+tab" in text
    # Distinct from Cline: no "(0)", no "▶▶", no Plan/Act toggle.
    assert "(0)" not in text
    assert "▶▶" not in text
    assert "Plan" not in text and "Act" not in text and "(Tab)" not in text


def test_footer_auto_off():
    state = TuiState(model_name="deepseek-chat", username="perdikis", auto_approve=False)
    text = _plain(footer_fragments(state, width=90))
    assert "auto-approve off" in text


def test_footer_shows_workspace_and_branch():
    state = TuiState(
        model_name="m",
        username="perdikis",
        workspace="~/coder-plugin-install",
        branch="feat/tui-rebuild",
    )
    text = _plain(footer_fragments(state, width=120))
    assert "perdikis" in text
    assert "~/coder-plugin-install" in text
    assert "feat/tui-rebuild" in text


def test_footer_context_indicator_value():
    text = _plain(
        footer_fragments(TuiState(model_name="m", username="u", context_pct=42.0), width=90)
    )
    assert "context 42%" in text


def test_footer_forge_badge_hidden_by_default():
    text = _plain(footer_fragments(TuiState(model_name="m", username="u"), width=120))
    assert "FORGE" not in text


def test_footer_forge_badge_shown_when_active():
    state = TuiState(
        model_name="m",
        username="perdikis",
        exec_mode="review",
        forge_mode=True,
        forge_run_id="run-1a2b",
    )
    text = _plain(footer_fragments(state, width=120))
    assert "FORGE" in text
    assert "⚒" not in text  # no wide emoji (it threw off the width math)
    assert "run-1a2b" in text
    # The execution-mode badge still renders alongside it (with a separator).
    assert "safe" in text
    # Order: FORGE chip precedes the exec-mode badge on line 2.
    line2 = text.split("\n")[1]
    assert line2.index("FORGE") < line2.index("safe")


def test_forge_placeholder_constant():
    from sylliptor_agent_cli.cli_impl.tui.content import INPUT_PLACEHOLDER_FORGE

    assert "Forge" in INPUT_PLACEHOLDER_FORGE
    assert "/goal" in INPUT_PLACEHOLDER_FORGE


def test_footer_is_two_lines_and_right_aligned():
    state = TuiState(model_name="m", username="u", tokens=1234)
    lines = _plain(footer_fragments(state, width=100)).split("\n")
    assert len(lines) == 2
    assert lines[0].rstrip().endswith("$0.0000")  # cost right-aligned, line 1
    assert lines[0].startswith("◇ sylliptor")  # brand mark + wordmark, line 1 left
    assert "1,234 tokens" in lines[0]  # spelled + comma-grouped
    assert lines[1].startswith("u")  # username left, line 2
    assert lines[1].rstrip().endswith("shift+tab")  # hint right-aligned, line 2


def test_footer_never_overflows_width():
    state = TuiState(
        model_name="some-very-long-model-name",
        username="averylongusername",
        workspace="~/a/very/long/workspace/path/that/keeps/going/and/going",
        branch="feature/a-really-quite-long-branch-name-here",
    )
    for width in (40, 60, 80, 120):
        lines = _plain(footer_fragments(state, width=width)).split("\n")
        assert len(lines) == 2
        for line in lines:
            assert len(line) <= width


# --------------------------- owl ---------------------------


def test_owl_frames_load():
    owl = load_owl_animation(color_enabled=False)
    # The repo ships 21 frames; loading must succeed and advancing must cycle.
    assert owl.available is True
    assert owl.frame_count >= 1
    first = owl.current_ansi()
    owl.advance()
    assert owl.current_ansi() is not None
    assert first is not None and first.value  # non-empty ASCII art


# --------------------------- headless app ---------------------------


def _run_headless(state: TuiState, keys: str):
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        return run_tui(state, owl_color=False, input=pipe, output=DummyOutput())


def test_app_exits_on_exit_word():
    state = TuiState(model_name="deepseek-chat", username="t")
    result, transcript = _run_headless(state, "/exit\r")
    assert result == "/exit"
    assert transcript == []


def test_app_records_submission_then_exits():
    state = TuiState(model_name="deepseek-chat", username="t")
    result, transcript = _run_headless(state, "hello there\r/exit\r")
    assert ("user", "hello there") in transcript
    assert result == "/exit"


def test_app_shift_tab_toggles_auto_approve_then_exits():
    state = TuiState(model_name="deepseek-chat", username="t", auto_approve=True)
    # Shift+Tab toggles auto-approve off, then exit.
    _run_headless(state, "\x1b[Z/exit\r")
    assert state.auto_approve is False


# --------------------------- mouse capture / copy-paste ---------------------------


def test_toggle_mouse_capture():
    state = TuiState(model_name="m")
    assert state.mouse_capture is False  # selection/copy works by default
    assert state.toggle_mouse_capture() is True  # opt into wheel-scroll
    assert state.mouse_capture is True
    assert state.toggle_mouse_capture() is False


def test_footer_omits_mouse_mode_chip():
    # The "copy (F2: scroll)" / "scroll (F2: copy)" chip was removed from the footer
    # as bottom-right noise; the F2 key still toggles the mode. Neither phrasing of
    # the chip should appear, in either mouse mode, and "shift+tab" stays the tail.
    default = _plain(footer_fragments(TuiState(model_name="m", username="u"), width=120))
    line2 = default.split("\n")[1]
    assert "F2" not in line2
    assert "copy (F2: scroll)" not in line2
    assert line2.rstrip().endswith("shift+tab")
    scroll = _plain(
        footer_fragments(TuiState(model_name="m", username="u", mouse_capture=True), width=120)
    )
    assert "F2" not in scroll.split("\n")[1]
    assert "scroll (F2: copy)" not in scroll


def test_footer_cost_unknown_shows_na():
    # Unmetered/free model with real usage: cost is None → honest "n/a", never $0.0000.
    state = TuiState(model_name="m", username="u", tokens=5000, cost_usd=None, cost_unknown_calls=3)
    line1 = _plain(footer_fragments(state, width=120)).split("\n")[0]
    assert "n/a" in line1
    assert "$0.0000" not in line1
    assert "+3" in line1  # unmetered-calls flag


def test_footer_cost_known_shows_dollars():
    state = TuiState(model_name="m", username="u", tokens=5000, cost_usd=0.1234)
    line1 = _plain(footer_fragments(state, width=120)).split("\n")[0]
    assert "$0.1234" in line1
    assert "n/a" not in line1


def test_app_f2_toggles_mouse_capture_then_exits():
    state = TuiState(model_name="deepseek-chat", username="t")
    assert state.mouse_capture is False
    # F2 (\x1bOQ) opts into wheel-scroll (mouse captured), then exit.
    _run_headless(state, "\x1bOQ/exit\r")
    assert state.mouse_capture is True


# --------------------------- welcome landing vs startup notices ---------------------------


def test_has_conversation_ignores_startup_notices():
    # Startup notices (the streaming-disabled warning, system/trace lines) must
    # NOT count as a conversation, otherwise they dismiss the owl landing the
    # moment the app opens.
    from sylliptor_agent_cli.cli_impl.tui.app import _has_conversation

    assert _has_conversation([]) is False
    assert _has_conversation([("warn", "streaming is disabled")]) is False
    assert _has_conversation([("system", "x"), ("trace", "y")]) is False
    # A real turn dismisses the landing.
    assert _has_conversation([("warn", "w"), ("user", "hi")]) is True
    assert _has_conversation([("assistant", "yo")]) is True


def test_startup_warning_keeps_welcome_then_exits():
    # Regression: a streaming-disabled warning emitted while the session is built
    # used to flip the transcript to "has messages" and hide the owl landing.
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from sylliptor_agent_cli.cli_impl.tui.app import _has_conversation

    class _WarningSession:
        def __init__(self, surface) -> None:
            surface.emit_warning("streaming is disabled for this run")

        def run_turn(self, text, *, cancellation_token=None, **_kwargs):  # pragma: no cover
            return 0

        def close(self) -> None:  # pragma: no cover - parity with real session
            pass

    state = TuiState(model_name="gpt-5.5", username="t")
    with create_pipe_input() as pipe:
        pipe.send_text("/exit\r")
        result, transcript = run_tui(
            state,
            owl_color=False,
            input=pipe,
            output=DummyOutput(),
            session_builder=_WarningSession,
        )
    assert result == "/exit"
    # The warning is retained (it surfaces once chatting) …
    assert ("warn", "streaming is disabled for this run") in transcript
    # … but it is not a turn, so the welcome landing stayed up.
    assert _has_conversation(transcript) is False
