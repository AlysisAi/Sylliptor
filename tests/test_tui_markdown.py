"""Phase 3 TUI tests: markdown rendering of completed assistant replies.

A finished reply with block-level markdown (headings, lists, fenced code) is
rendered through Rich into styled rows; plain prose and still-streaming text are
left untouched so a half-open code fence never flickers mid-stream.
"""

from __future__ import annotations

from sylliptor_agent_cli.cli_impl.tui.app import _assistant_rows
from sylliptor_agent_cli.cli_impl.tui.markdown import (
    looks_like_markdown,
    render_markdown_rows,
)
from sylliptor_agent_cli.cli_impl.tui.transcript import TuiTranscript

_CODE_REPLY = "Here you go:\n\n```python\ndef f(x):\n    return x + 1\n```"
_LIST_REPLY = "Steps:\n\n- first\n- second\n\n## Heading\n\nmore text"


def _row_text(row: list[tuple[str, str]]) -> str:
    return "".join(text for _style, text in row)


# ------------------------------ heuristic ------------------------------


def test_looks_like_markdown_detects_blocks():
    assert looks_like_markdown("# Title")
    assert looks_like_markdown("- a\n- b")
    assert looks_like_markdown("1. one\n2. two")
    assert looks_like_markdown("```\ncode\n```")
    assert looks_like_markdown("| a | b |\n| - | - |")


def test_looks_like_markdown_skips_plain_prose():
    assert not looks_like_markdown("")
    assert not looks_like_markdown("   ")
    assert not looks_like_markdown("just a single sentence with no markup")
    # Single newline-joined prose must stay plain (else markdown reflows it).
    assert not looks_like_markdown("Hello\nworld")


# --------------------------- render_markdown_rows ---------------------------


def test_render_returns_none_for_plain_text():
    assert render_markdown_rows("Hello\nworld", 80) is None
    assert render_markdown_rows("", 80) is None


def test_render_code_block_keeps_code_and_styles_it():
    rows = render_markdown_rows(_CODE_REPLY, 60)
    assert rows is not None
    joined = "\n".join(_row_text(r) for r in rows)
    assert "def f(x):" in joined
    assert "return x + 1" in joined
    # The fenced block is syntax-highlighted on a background (monokai), so at
    # least one fragment on the code row carries a non-empty style.
    code_row = next(r for r in rows if "def f(x):" in _row_text(r))
    assert any(style for style, _text in code_row), "code row should be styled"


def test_render_list_and_heading():
    rows = render_markdown_rows(_LIST_REPLY, 60)
    assert rows is not None
    joined = "\n".join(_row_text(r) for r in rows)
    assert "first" in joined and "second" in joined
    assert "Heading" in joined
    # Rich renders bullets as "•".
    assert "•" in joined


def test_render_never_raises_and_rows_fit_width():
    width = 40
    rows = render_markdown_rows(_LIST_REPLY, width)
    assert rows is not None
    for row in rows:
        assert len(_row_text(row)) <= width


# ----------------------------- _assistant_rows -----------------------------


def test_assistant_rows_markdown_puts_marker_on_first_visible_row():
    rows = _assistant_rows(_CODE_REPLY, width=60, markdown=True)
    # Exactly one row carries the accent marker prefix.
    marker_rows = [r for r in rows if _row_text(r).startswith("✦ ")]
    assert len(marker_rows) == 1
    assert _row_text(marker_rows[0]).startswith("✦ Here you go")
    # The code survives the marker/indent wrapping.
    joined = "\n".join(_row_text(r) for r in rows)
    assert "def f(x):" in joined


def test_assistant_rows_streaming_stays_plain():
    # A half-open fence is markdown-shaped but must NOT be rendered while
    # streaming — it should pass through as plain lines.
    partial = "Here you go:\n\n```python\ndef f(x):"
    rows = _assistant_rows(partial, width=60, markdown=False)
    joined = "\n".join(_row_text(r) for r in rows)
    assert "```python" in joined  # fence kept verbatim, not consumed by Rich
    assert rows[0][0][1] == "✦ "


def test_assistant_rows_plain_text_unchanged():
    # Regression: non-markdown text renders exactly as before (marker + indent).
    rows = _assistant_rows("Hello\nworld")
    assert _row_text(rows[0]).startswith("✦ Hello")
    assert _row_text(rows[1]) == "  world"


# ------------------------------- snapshot -------------------------------


def test_snapshot_exposes_streaming_index():
    t = TuiTranscript()
    t.append_user("hi")
    t.begin_turn()
    t.stream_assistant("partial")
    entries, _status, streaming_index = t.snapshot()
    assert entries[streaming_index] == ("assistant", "partial")
    t.finish_assistant("partial done")
    _entries, _status2, idx_after = t.snapshot()
    assert idx_after is None  # block closed → renders as markdown


# --------------------------- headless integration ---------------------------


class _MarkdownSession:
    """Fake agent session that streams a markdown reply (with a fenced code
    block) into the surface, the way a real turn would."""

    def __init__(self, surface) -> None:
        self.surface = surface

    def run_turn(self, text: str, *, cancellation_token=None, **_kwargs) -> int:
        self.surface.on_user_message(text)
        # Deltas concatenate to exactly _CODE_REPLY (finish_assistant keeps the
        # streamed content when present).
        for delta in (
            "Here you go:\n\n",
            "```python\n",
            "def f(x):\n",
            "    return x + 1\n",
            "```",
        ):
            self.surface.on_assistant_token(delta)
        self.surface.on_assistant_message_done(_CODE_REPLY)
        return 0

    def close(self) -> None:  # pragma: no cover - parity with real session
        pass


def test_headless_markdown_reply_renders_without_crashing():
    # Drives a full ``run_tui`` render loop (a markdown reply must survive the
    # transcript render path, not just ``_assistant_rows`` in isolation).
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from sylliptor_agent_cli.cli_impl.tui import run_tui
    from sylliptor_agent_cli.cli_impl.tui.state import TuiState

    state = TuiState(model_name="deepseek-chat", username="t")
    # No command_runner: the first line runs a turn, "/exit" is an exit word that
    # tears the app down (a command_runner returning "run" would loop forever).
    with create_pipe_input() as pipe:
        pipe.send_text("hi\r/exit\r")
        _result, transcript = run_tui(
            state,
            owl_color=False,
            input=pipe,
            output=DummyOutput(),
            session_builder=_MarkdownSession,
            background_turns=False,
        )
    assert ("assistant", _CODE_REPLY) in transcript
