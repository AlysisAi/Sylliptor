"""Phase 3 TUI tests: live model reasoning ("thinking").

Covers the full path — the openai_compat reasoning stream callback, the
transcript reasoning block (stream → collapse with elapsed seconds), the surface
opt-in handler, the dim collapsible rendering, and a headless run that streams
reasoning then an answer.
"""

from __future__ import annotations

import httpx

from sylliptor_agent_cli.cli_impl.tui.app import _activity_rows, _reasoning_rows
from sylliptor_agent_cli.cli_impl.tui.surface import TuiSurface
from sylliptor_agent_cli.cli_impl.tui.transcript import TuiTranscript
from sylliptor_agent_cli.llm.openai_compat import OpenAICompatClient


def _row_text(row: list[tuple[str, str]]) -> str:
    return "".join(text for _style, text in row)


# ----------------------------- LLM stream layer -----------------------------


def test_openai_compat_streams_reasoning_to_callback():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"delta":{"reasoning_content":"Let me "}}]}\n\n'
                b'data: {"choices":[{"delta":{"reasoning_content":"think."}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"Answer"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    client = OpenAICompatClient(
        base_url="https://api.deepseek.com/v1",
        api_key="test",
        model="deepseek-reasoner",
        temperature=0.0,
        transport=httpx.MockTransport(handler),
    )
    reasoning: list[str] = []
    content: list[str] = []
    resp = client.chat(
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        on_reasoning_delta=reasoning.append,
        on_text_delta=content.append,
    )
    assert "".join(reasoning) == "Let me think."
    assert "".join(content) == "Answer"
    assert resp.content == "Answer"


def test_non_repo_turn_streams_reasoning_to_callback():
    # Conversational ("who are you") turns go through the non-repo responder; it
    # must thread reasoning to the surface too, not just the main repo turn.
    from sylliptor_agent_cli.agent.routing import _respond_non_repo_turn
    from sylliptor_agent_cli.llm.types import LLMResponse

    class _Client:
        base_url = "x"
        model = "m"

        def chat(
            self,
            *,
            messages,
            tools=None,
            stream=False,
            on_text_delta=None,
            on_reasoning_delta=None,
            temperature=None,
            **_kw,
        ):
            if on_reasoning_delta:
                on_reasoning_delta("Let me ")
                on_reasoning_delta("consider this.")
            if on_text_delta:
                on_text_delta("Sure!")
            return LLMResponse(content="Sure!", tool_calls=[], raw={})

    reasoning: list[str] = []
    text: list[str] = []
    result = _respond_non_repo_turn(
        client=_Client(),
        instruction="who are you",
        route="chat",
        language="en",
        script="latin",
        explicit_language_override=False,
        temperature=0.0,
        stream=True,
        on_text_delta=text.append,
        on_reasoning_delta=reasoning.append,
    )
    assert "".join(reasoning) == "Let me consider this."
    assert str(result) == "Sure!"


def test_non_repo_chat_falls_back_when_client_lacks_reasoning_param():
    # A narrow client without on_reasoning_delta still streams content (the
    # responder must not crash on older/test-double signatures).
    from sylliptor_agent_cli.agent.routing import _non_repo_chat
    from sylliptor_agent_cli.llm.types import LLMResponse

    class _NarrowClient:
        base_url = "x"
        model = "m"

        def chat(self, *, messages, tools=None, stream=False, on_text_delta=None, temperature=None):
            if on_text_delta:
                on_text_delta("ok")
            return LLMResponse(content="ok", tool_calls=[], raw={})

    text: list[str] = []
    resp = _non_repo_chat(
        client=_NarrowClient(),
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        stream=True,
        on_text_delta=text.append,
        on_reasoning_delta=lambda _d: None,
    )
    assert resp.content == "ok"
    assert "".join(text) == "ok"


def test_openai_compat_reasoning_callback_is_optional():
    # No on_reasoning_delta passed → still streams content fine (back-compat).
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"delta":{"reasoning_content":"hidden"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    client = OpenAICompatClient(
        base_url="https://api.deepseek.com/v1",
        api_key="test",
        model="deepseek-reasoner",
        temperature=0.0,
        transport=httpx.MockTransport(handler),
    )
    resp = client.chat(messages=[{"role": "user", "content": "hi"}], stream=True)
    assert resp.content == "ok"


# ------------------------------ transcript model ------------------------------


def test_transcript_streams_reasoning_block():
    t = TuiTranscript()
    t.begin_turn()
    t.stream_reasoning("I should ")
    t.stream_reasoning("check X.")
    idx, secs = t.reasoning_snapshot()
    assert idx is not None
    assert t.entries[idx] == ("reasoning", "I should check X.")
    assert idx not in secs  # still open → no recorded duration yet


def test_reasoning_collapses_when_answer_starts():
    t = TuiTranscript()
    t.begin_turn()
    t.stream_reasoning("thinking...")
    r_idx, _secs = t.reasoning_snapshot()
    t.stream_assistant("Hello")
    idx_after, secs = t.reasoning_snapshot()
    assert idx_after is None  # the first answer token closes the block
    assert r_idx in secs and secs[r_idx] >= 0  # elapsed seconds recorded
    assert t.entries[0][0] == "reasoning"
    assert ("assistant", "Hello") in t.entries


def test_reasoning_closes_on_tool_trace():
    t = TuiTranscript()
    t.begin_turn()
    t.stream_reasoning("plan the work")
    t.append("trace", "⚙ running a tool")
    idx_after, secs = t.reasoning_snapshot()
    assert idx_after is None  # a tool following thinking also collapses it
    assert 0 in secs


def test_reasoning_supports_multiple_blocks_per_turn():
    t = TuiTranscript()
    t.begin_turn()
    t.stream_reasoning("step one thinking")
    t.append("trace", "⚙ tool")
    t.stream_reasoning("step two thinking")
    t.stream_assistant("done")
    reasoning_entries = [i for i, (role, _t) in enumerate(t.entries) if role == "reasoning"]
    assert len(reasoning_entries) == 2


def test_clear_resets_reasoning():
    t = TuiTranscript()
    t.stream_reasoning("x")
    t.clear()
    idx, secs = t.reasoning_snapshot()
    assert idx is None and secs == {}
    assert t.entries == []


# --------------------------------- surface ---------------------------------


def test_surface_routes_reasoning_token():
    t = TuiTranscript()
    surface = TuiSurface(t, auto_approve=lambda: True)
    surface.on_reasoning_token("hmm, let me see")
    idx, _secs = t.reasoning_snapshot()
    assert idx is not None
    assert t.entries[idx] == ("reasoning", "hmm, let me see")


# ------------------------------ _reasoning_rows ------------------------------


def test_reasoning_rows_collapsed_summary():
    rows = _reasoning_rows("a long inner monologue", 60, live=False, secs=12, expanded=False)
    assert len(rows) == 1  # just the chevron header, no body
    text = _row_text(rows[0])
    assert text.startswith("▸ ")  # collapsed chevron
    assert "thought for 12s" in text


def test_reasoning_rows_live_shows_full_text():
    rows = _reasoning_rows("line one\nline two", 60, live=True, secs=0, expanded=False)
    joined = "\n".join(_row_text(r) for r in rows)
    assert "line one" in joined and "line two" in joined
    assert _row_text(rows[0]).startswith("▾ ")  # open chevron header ("thinking…")
    # Body lines hang off the dim left rail.
    assert any(_row_text(r).startswith("│ ") for r in rows[1:])


def test_reasoning_rows_live_header_animates_with_spinner_and_elapsed():
    rows = _reasoning_rows(
        "partial thought", 60, live=True, secs=0, expanded=False, spinner="⠋", elapsed=3
    )
    header = _row_text(rows[0])
    assert header.startswith("⠋ ")  # the spinner leads the live header
    assert "thinking… 3s" in header


def test_activity_indicator_under_question():
    # The single live activity line shown under the question while busy — either
    # "thinking…" or a running tool name.
    rows = _activity_rows("⠙", "thinking…", 5)
    assert len(rows) == 1
    text = _row_text(rows[0])
    assert text.startswith("⠙ ")
    assert "thinking… 5s" in text
    # Works for a running tool label too, and shows no seconds before any elapse.
    assert _row_text(_activity_rows("⠋", "Search Web…", 0)[0]) == "⠋ Search Web…"


def test_reasoning_rows_expanded_shows_full_when_closed():
    collapsed = _reasoning_rows("detail here", 60, live=False, secs=5, expanded=False)
    expanded = _reasoning_rows("detail here", 60, live=False, secs=5, expanded=True)
    assert "detail here" not in "\n".join(_row_text(r) for r in collapsed)
    assert "detail here" in "\n".join(_row_text(r) for r in expanded)


# --------------------------- headless integration ---------------------------


class _ReasoningSession:
    """Fake session that streams reasoning, then a short answer."""

    def __init__(self, surface) -> None:
        self.surface = surface

    def run_turn(self, text: str, *, cancellation_token=None, **_kwargs) -> int:
        self.surface.on_user_message(text)
        self.surface.on_reasoning_token("I will greet them politely.")
        self.surface.on_assistant_token("Hi!")
        self.surface.on_assistant_message_done("Hi!")
        return 0

    def close(self) -> None:  # pragma: no cover - parity with real session
        pass


def test_headless_reasoning_then_answer_renders():
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from sylliptor_agent_cli.cli_impl.tui import run_tui
    from sylliptor_agent_cli.cli_impl.tui.state import TuiState

    state = TuiState(model_name="deepseek-reasoner", username="t")
    with create_pipe_input() as pipe:
        pipe.send_text("hi\r/exit\r")
        _result, transcript = run_tui(
            state,
            owl_color=False,
            input=pipe,
            output=DummyOutput(),
            session_builder=_ReasoningSession,
            background_turns=False,
        )
    roles = [role for role, _text in transcript]
    assert "reasoning" in roles  # the thinking block was recorded
    assert ("assistant", "Hi!") in transcript  # and the answer followed it
