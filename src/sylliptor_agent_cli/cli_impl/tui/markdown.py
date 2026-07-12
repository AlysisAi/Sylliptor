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
_FENCED_CODE_RE = re.compile(r"```[^\n]*\n(.*?)(?:\n```|$)", re.S)
_CODE_THEME = "monokai"
_CODE_FALLBACK_STYLE = "class:markdown.code"


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


def _trim_row_right(row: Row) -> Row:
    trimmed = list(row)
    while trimmed:
        style, text = trimmed[-1]
        right_stripped = text.rstrip()
        if right_stripped:
            if right_stripped != text:
                trimmed[-1] = (style, right_stripped)
            break
        trimmed.pop()
    return trimmed


def _split_row_to_width(row: Row, width: int) -> list[Row]:
    safe_width = max(1, int(width))
    rows: list[Row] = []
    current: Row = []
    current_len = 0
    for style, text in row:
        remaining = str(text)
        while remaining:
            available = safe_width - current_len
            if available <= 0:
                rows.append(_trim_row_right(current))
                current = []
                current_len = 0
                available = safe_width
            chunk = remaining[:available]
            current.append((style, chunk))
            current_len += len(chunk)
            remaining = remaining[available:]
    rows.append(_trim_row_right(current))
    return [item for item in rows if item]


def _fenced_code_line_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for match in _FENCED_CODE_RE.finditer(str(text or "")):
        for line in match.group(1).splitlines():
            stripped = line.strip()
            if stripped:
                keys.add(stripped)
    return keys


def _apply_code_fallback_styles(rows: list[Row], text: str) -> list[Row]:
    code_line_keys = _fenced_code_line_keys(text)
    if not code_line_keys:
        return rows
    styled_rows: list[Row] = []
    for row in rows:
        row_text = "".join(fragment_text for _style, fragment_text in row).strip()
        if row_text in code_line_keys and not any(style for style, _text in row):
            styled_rows.append(
                [
                    (_CODE_FALLBACK_STYLE if fragment_text else style, fragment_text)
                    for style, fragment_text in row
                ]
            )
            continue
        styled_rows.append(row)
    return styled_rows


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
        rows = []
        for line in split_lines(fragments):
            trimmed = _trim_row_right(list(line))
            if not trimmed:
                rows.append(trimmed)
                continue
            rows.extend(_split_row_to_width(trimmed, int(width)))
    except Exception:
        return None
    # Trim trailing blank rows Rich pads on (the leading ones, if any, are kept so
    # the caller can drop its accent marker on the first non-blank row).
    while rows and _is_blank(rows[-1]):
        rows.pop()
    rows = _apply_code_fallback_styles(rows, text)
    return rows or None


__all__ = ["render_markdown_rows", "looks_like_markdown", "Row"]
