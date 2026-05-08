# ruff: noqa: F401,F403,F405
from __future__ import annotations

# Source-alignment markers for checks that read this legacy entrypoint after the
# command implementation moved into cli_impl/commands.
# Chat command marker: "/forge"
# Prompt hotkey marker: @kb.add("c-b")
from .cli_impl.commands.cli_surface import *

if __name__ == "__main__":
    app()
