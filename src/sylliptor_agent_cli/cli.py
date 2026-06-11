# ruff: noqa: F401,F403,F405,E402
from __future__ import annotations


def _harden_stdio() -> None:
    """Make stdout/stderr tolerant of glyphs the console encoding can't map
    (e.g. the rich UI's box-drawing ``│`` on a non-UTF-8 Windows console such as
    Greek cp1253), so the first surfaced warning/error no longer crashes with
    UnicodeEncodeError. ``backslashreplace`` degrades an unmappable glyph to a
    readable escape instead of aborting. We change only the error handler, not
    the encoding, so redirected/piped output keeps its locale codec for
    downstream readers. Runs once at process start, before any output."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="backslashreplace")
        except (ValueError, OSError):  # detached buffer / unsupported stream
            pass


_harden_stdio()

# Source-alignment markers for checks that read this legacy entrypoint after the
# command implementation moved into cli_impl/commands.
# Chat command marker: "/forge"
# Prompt hotkey marker: @kb.add("c-b")
from .cli_impl.commands.cli_surface import *

if __name__ == "__main__":
    app()
