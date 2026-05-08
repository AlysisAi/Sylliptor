from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...config import AppConfig
from ...forge import RunPaths


@dataclass
class _ForgePlannerSessionState:
    transcript: list[dict[str, str]] = field(default_factory=list)
    cfg: AppConfig | None = None
    workspace_context: dict[str, Any] | None = None
    awaiting_clarification: bool = False
    pending_questions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _ForgeEnterCommand:
    entry_mode: str
    usage_error: str | None = None


@dataclass(frozen=True)
class _ForgeChatEntrySelection:
    paths: RunPaths
    plan: dict[str, Any]
    entry_kind: str


@dataclass
class _ForgeChatState:
    ui_mode: str = "chat"
    paths: RunPaths | None = None
    plan: dict[str, Any] | None = None
    assistant_enabled: bool = False
    entry_request_mode: str = "plain"
    planner_session: _ForgePlannerSessionState = field(default_factory=_ForgePlannerSessionState)

    @property
    def planner_transcript(self) -> list[dict[str, str]]:
        return self.planner_session.transcript

    @planner_transcript.setter
    def planner_transcript(self, value: list[dict[str, str]]) -> None:
        self.planner_session.transcript = list(value)

    @property
    def planner_cfg(self) -> AppConfig | None:
        return self.planner_session.cfg

    @planner_cfg.setter
    def planner_cfg(self, value: AppConfig | None) -> None:
        self.planner_session.cfg = value

    @property
    def workspace_context(self) -> dict[str, Any] | None:
        return self.planner_session.workspace_context

    @workspace_context.setter
    def workspace_context(self, value: dict[str, Any] | None) -> None:
        self.planner_session.workspace_context = value

    @property
    def planner_awaiting_clarification(self) -> bool:
        return bool(self.planner_session.awaiting_clarification)

    @planner_awaiting_clarification.setter
    def planner_awaiting_clarification(self, value: bool) -> None:
        self.planner_session.awaiting_clarification = bool(value)

    @property
    def planner_pending_questions(self) -> list[str]:
        return self.planner_session.pending_questions

    @planner_pending_questions.setter
    def planner_pending_questions(self, value: list[str]) -> None:
        self.planner_session.pending_questions = list(value)


@dataclass
class _ChatPlanModeState:
    enabled: bool = False
    restore_mode: str | None = None
    latest_task: str | None = None
    latest_draft: str | None = None


@dataclass(frozen=True)
class _ChatExecutionRequest:
    instruction: str
    routing_mode_override: str | None = None
    ephemeral_system_messages: tuple[str, ...] = ()
    ephemeral_user_messages: tuple[str, ...] = ()
    mode_override: str | None = None
    restore_mode_after: str | None = None
    plan_mode_capture_task: str | None = None


__all__ = [name for name in globals() if not name.startswith("__") or name == "__version__"]
