"""Mutable view-model for the full-screen TUI.

Holds only what the footer/header need to render. The footer HUD fields
(``context_pct`` / ``tokens`` / ``cost_usd``) are wired live: ``loop.py`` seeds
them when the session is built and refreshes them after each turn (and mid-turn,
throttled) from the session's usage summary + context cache. Other fields are
toggled by the user via Tab / Shift+Tab.
"""

from __future__ import annotations

from dataclasses import dataclass

PLAN_MODE = "plan"
ACT_MODE = "act"


@dataclass
class TuiState:
    model_name: str = ""
    # Non-empty while the shell is available but model calls are not (for
    # example, a selected subscription still needs browser authentication).
    connection_status: str = ""
    tokens: int = 0
    # Session cost. ``None`` means pricing is unknown (an unmetered/free model with
    # real usage) — the footer renders that as "n/a" rather than a misleading
    # "$0.0000". A literal 0.0 means nothing has been spent yet.
    cost_usd: float | None = 0.0
    cost_unknown_calls: int = 0  # calls whose cost couldn't be metered (footer "+N")
    mode: str = ACT_MODE  # "plan" | "act"
    exec_mode: str = ""  # execution mode: review | auto | readonly | fullaccess
    # Forge: True while the user is inside a Forge planning session (ui_mode ==
    # "forge"); drives the footer FORGE badge + the forge-specific placeholder.
    forge_mode: bool = False
    forge_run_id: str = ""  # short run id shown in the FORGE badge, e.g. "run-1a2b"
    auto_approve: bool = True
    username: str = ""
    workspace: str = ""  # short display form, e.g. "~/coder-plugin-install"
    branch: str = ""  # git branch name, e.g. "feat/tui-rebuild"
    usage_hud_enabled: bool = True
    context_pct: float = 100.0  # % of context window remaining

    @property
    def plan_mode(self) -> bool:
        return self.mode == PLAN_MODE

    def toggle_mode(self) -> str:
        self.mode = PLAN_MODE if self.mode == ACT_MODE else ACT_MODE
        return self.mode

    def toggle_auto_approve(self) -> bool:
        self.auto_approve = not self.auto_approve
        return self.auto_approve


__all__ = ["TuiState", "PLAN_MODE", "ACT_MODE"]
