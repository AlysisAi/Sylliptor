"""The pinned footer — Sylliptor's own status grammar (not a Cline clone).

Two lines, each split left/right against the terminal width::

    ◇ sylliptor · <Model>              context <P>% · <N> tokens · $<cost>
    <user> · <workspace> · ⎇ <branch>            auto-approve on · shift+tab

The left side is clipped (with an ellipsis) when it would collide with the
right side, so the layout never wraps. Returned as prompt_toolkit
``FormattedText`` with ``class:`` styles resolved by the application's Style.
"""

from __future__ import annotations

from prompt_toolkit.formatted_text import FormattedText

from .content import pretty_model_label
from .state import TuiState

_BRAND_MARK = "◇"
_BRANCH_MARK = "⎇"
_DOT = "  ·  "

# Compact labels for the execution-mode badge (full names live in the /mode popup).
_MODE_SHORT = {
    "review": "safe",
    "auto": "fast",
    "readonly": "read",
    "fullaccess": "full",
}

Fragments = list[tuple[str, str]]


def _visible_len(fragments: Fragments) -> int:
    return sum(len(text) for _style, text in fragments)


def _format_cost(state: TuiState) -> str:
    """Render the session cost honestly.

    A dollar figure when pricing is known, else ``n/a`` — so an unmetered/free
    model with real usage never reads as a fake ``$0.0000``. A trailing ``+N``
    flags calls whose cost could not be metered (partial total)."""
    cost = state.cost_usd
    base = "n/a" if cost is None else f"${cost:.4f}"
    if state.cost_unknown_calls > 0:
        return f"{base} +{state.cost_unknown_calls}"
    return base


def _clip_fragments(fragments: Fragments, max_width: int) -> Fragments:
    """Trim styled fragments from the end so the visible width fits max_width."""
    if max_width <= 0:
        return []
    out: Fragments = []
    used = 0
    for style, text in fragments:
        if used + len(text) <= max_width:
            out.append((style, text))
            used += len(text)
            continue
        remaining = max_width - used
        if remaining >= 2:
            out.append((style, text[: remaining - 1] + "…"))
        elif remaining == 1:
            out.append((style, "…"))
        break
    return out


def _line1(state: TuiState) -> tuple[Fragments, Fragments]:
    left: Fragments = [
        ("class:tui.footer.mark", f"{_BRAND_MARK} "),
        ("class:tui.footer.brand", "sylliptor"),
        ("class:tui.footer.dim", _DOT),
        ("class:tui.footer.model", pretty_model_label(state.model_name)),
    ]
    right: Fragments = [
        ("class:tui.footer.context", f"context {state.context_pct:.0f}%"),
        ("class:tui.footer.dim", _DOT),
        ("class:tui.footer.value", f"{state.tokens:,} tokens"),
        ("class:tui.footer.dim", _DOT),
        ("class:tui.footer.value", _format_cost(state)),
    ]
    return left, right


def _line2(state: TuiState) -> tuple[Fragments, Fragments]:
    left: Fragments = []
    if state.forge_mode:
        # A distinct violet chip so the forge session is unmistakable at a glance
        # (green stays the chat accent). No glyph — a wide emoji would throw off the
        # width math that keeps the footer from wrapping.
        label = "FORGE"
        if state.forge_run_id:
            label = f"{label} {state.forge_run_id}"
        left.append(("class:tui.footer.forge", label))
    if state.exec_mode:
        # Glanceable execution-mode badge; amber when in the unguarded full mode.
        short = _MODE_SHORT.get(state.exec_mode, state.exec_mode)
        mode_style = (
            "class:tui.footer.mode.warn"
            if state.exec_mode == "fullaccess"
            else "class:tui.footer.mode"
        )
        if left:
            left.append(("class:tui.footer.dim", _DOT))
        left.append((mode_style, short))
    if state.username:
        if left:
            left.append(("class:tui.footer.dim", _DOT))
        left.append(("class:tui.footer.user", state.username))
    if state.workspace:
        if left:
            left.append(("class:tui.footer.dim", _DOT))
        left.append(("class:tui.footer.workspace", state.workspace))
    if state.branch:
        if left:
            left.append(("class:tui.footer.dim", _DOT))
        left.append(("class:tui.footer.branch", f"{_BRANCH_MARK} {state.branch}"))

    # Right tail: auto-approve state + the global shift+tab hint. (The mouse-mode
    # "copy (F2: scroll)" chip used to lead here but was removed as footer noise;
    # F2 still toggles wheel-scroll/copy, with a transient reminder when scroll is on.)
    right: Fragments = []
    if state.auto_approve:
        right.append(("class:tui.footer.autoapprove.on", "auto-approve on"))
    else:
        right.append(("class:tui.footer.autoapprove.off", "auto-approve off"))
    right.append(("class:tui.footer.dim", f"{_DOT}shift+tab"))
    return left, right


def _compose(left: Fragments, right: Fragments, width: int) -> Fragments:
    if width <= 0:
        return []
    # Right side has priority and is right-aligned; clip it to the full width if it
    # alone would overflow (narrow terminals), then fit the left into what's left.
    right = _clip_fragments(right, width)
    right_len = _visible_len(right)
    left = _clip_fragments(left, max(0, width - right_len - 1))
    gap = max(0, width - _visible_len(left) - right_len)
    return [*left, ("", " " * gap), *right]


def footer_fragments(state: TuiState, *, width: int = 80) -> FormattedText:
    """Build the 2-line footer as FormattedText for the given terminal width."""
    l1_left, l1_right = _line1(state)
    l2_left, l2_right = _line2(state)
    fragments: Fragments = []
    fragments.extend(_compose(l1_left, l1_right, width))
    fragments.append(("", "\n"))
    fragments.extend(_compose(l2_left, l2_right, width))
    return FormattedText(fragments)


__all__ = ["footer_fragments"]
