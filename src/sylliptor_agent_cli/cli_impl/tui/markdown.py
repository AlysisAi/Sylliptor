"""Markdown → prompt_toolkit fragment rows for the TUI transcript.

A *completed* assistant reply is rendered through Rich's ``Markdown`` (the very
renderer the classic CLI uses, so the two agree on how a reply looks) into ANSI,
then converted to prompt_toolkit ``(style, text)`` rows. Streaming/partial text
and plain prose skip this entirely and render as plain lines, so a half-open
code fence never flashes mid-stream and a one-line answer is not reflowed.

Kept free of any agent imports (Rich + prompt_toolkit only) so it unit-tests in
isolation, and every public entry point is fail-safe: on any error it returns
``None`` and the caller falls back to plain rendering.
"""

from __future__ import annotations

import re
from functools import lru_cache
from io import StringIO

# One visual line: a list of ``(style, text)`` fragments.
Row = list[tuple[str, str]]

# Matches the classic CLI's heuristic (rich_surface._looks_like_markdown) so the
# TUI and the plain console agree on what counts as markdown worth rendering.
_NUMBERED = re.compile(r"\d+\.\s")
_CODE_THEME = "monokai"


def looks_like_markdown(text: str) -> bool:
    """True when ``text`` carries block-level markdown worth rendering.

    Deliberately conservative: inline-only emphasis in a single paragraph is
    left as plain text (rendering it would reflow newlines for no real gain).
    """
    clean = str(text or "")
    if not clean.strip():
        return False
    if "```" in clean:
        return True
    has_multiple_lines = "\n" in clean
    for line in clean.splitlines():
        stripped = line.lstrip()
        if not stripped:
            continue
        if stripped.startswith(("#", ">", "-", "*")):
            return True
        if _NUMBERED.match(stripped):
            return True
        if has_multiple_lines and stripped.count("|") >= 2:
            return True
    return False


@lru_cache(maxsize=256)
def _render_ansi(text: str, width: int) -> str:
    """Render markdown to an ANSI string at a fixed width (cached per text+width).

    Completed replies never change, so a redraw reuses the cached ANSI; only a
    terminal resize (new width) forces a re-render.
    """
    from rich.markdown import Markdown

    from ...surface.console import make_console

    buf = StringIO()
    console = make_console(
        file=buf,
        width=max(8, int(width)),
        force_terminal=True,
        color_system="truecolor",
        highlight=False,  # no repr-highlighting of numbers/strings in prose
        emoji=False,  # ":)" etc. stay literal
        legacy_windows=False,  # emit ANSI escapes, not Win32 console calls
    )
    console.print(Markdown(text, code_theme=_CODE_THEME))
    return buf.getvalue()


def _is_blank(row: Row) -> bool:
    return not "".join(text for _style, text in row).strip()


def _rstrip_row(row: Row) -> Row:
    trimmed = list(row)
    while trimmed:
        style, text = trimmed[-1]
        stripped = str(text).rstrip(" ")
        if stripped:
            if stripped != text:
                trimmed[-1] = (style, stripped)
            break
        trimmed.pop()
    return trimmed


def _fit_row_width(row: Row, width: int) -> list[Row]:
    width = max(1, int(width))
    trimmed = _rstrip_row(row)
    if not trimmed:
        return [[]]
    fitted: list[Row] = []
    current: Row = []
    used = 0
    for style, text in trimmed:
        remaining = str(text)
        if not remaining:
            continue
        while remaining:
            available = width - used
            if available <= 0:
                fitted.append(current)
                current = []
                used = 0
                available = width
            chunk = remaining[:available]
            current.append((style, chunk))
            used += len(chunk)
            remaining = remaining[available:]
    fitted.append(current)
    return fitted


def _fit_rows_width(rows: list[Row], width: int) -> list[Row]:
    fitted: list[Row] = []
    for row in rows:
        fitted.extend(_fit_row_width(row, width))
    return fitted


def render_markdown_rows(text: str, width: int) -> list[Row] | None:
    """Markdown-render ``text`` into rows of fragments, or ``None`` to render plain.

    Returns ``None`` when the text is not markdown (so the caller keeps its plain
    layout) or when rendering fails for any reason. Never raises. ``width`` is the
    target content width in columns.
    """
    if not looks_like_markdown(text):
        return None
    try:
        from prompt_toolkit.formatted_text import ANSI, to_formatted_text
        from prompt_toolkit.formatted_text.utils import split_lines

        ansi = _render_ansi(text, int(width))
        fragments = to_formatted_text(ANSI(ansi.rstrip("\n")))
        rows = [list(line) for line in split_lines(fragments)]
    except Exception:
        return None
    # Trim trailing blank rows Rich pads on (the leading ones, if any, are kept so
    # the caller can drop its accent marker on the first non-blank row).
    while rows and _is_blank(rows[-1]):
        rows.pop()
    return _fit_rows_width(rows, int(width)) or None


__all__ = ["render_markdown_rows", "looks_like_markdown", "Row"]
