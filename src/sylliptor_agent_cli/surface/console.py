"""Console construction helpers for terminal-aware Rich output."""

from __future__ import annotations

import os
import sys
from typing import Any

from rich.console import Console as RichTerminal


def _stream_is_tty(stream: Any | None) -> bool:
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except Exception:
        return False


def make_console(
    *,
    force_terminal: bool | None = None,
    no_color: bool | None = None,
    file: Any | None = None,
    **kwargs: Any,
) -> RichTerminal:
    """Create a Rich console that respects host terminal color capability."""
    effective_no_color = bool(no_color) or bool(os.environ.get("NO_COLOR"))
    if effective_no_color:
        return RichTerminal(
            file=file,
            force_terminal=force_terminal,
            no_color=True,
            **kwargs,
        )

    if os.environ.get("WT_SESSION"):
        kwargs.setdefault("color_system", "truecolor")

    stream = file if file is not None else sys.stdout
    if force_terminal is None and not _stream_is_tty(stream):
        return RichTerminal(file=file, force_terminal=False, no_color=True, **kwargs)

    if force_terminal is None:
        return RichTerminal(file=file, **kwargs)
    return RichTerminal(file=file, force_terminal=force_terminal, **kwargs)


__all__ = ["make_console"]
