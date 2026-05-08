from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from prompt_toolkit.completion import Completer, Completion


@dataclass(frozen=True)
class ChatSlashCommandSpec:
    name: str
    usage: str
    description: str


@dataclass(frozen=True)
class _ChatSlashCompletionSpec:
    completion: str
    usage: str
    description: str


SPECS: tuple[ChatSlashCommandSpec, ...] = (
    ChatSlashCommandSpec("help", "/help", "Show available commands"),
    ChatSlashCommandSpec("mode", "/mode", "Change execution mode"),
    ChatSlashCommandSpec("status", "/status", "Show session status"),
    ChatSlashCommandSpec("terminals", "/terminals [list|show|kill]", "Manage background processes"),
    ChatSlashCommandSpec("pwd", "/pwd", "Show active workdir"),
    ChatSlashCommandSpec("usage", "/usage", "Token count & cost; /usage hud on|off toggles HUD"),
    ChatSlashCommandSpec("ctx", "/ctx", "Show context usage"),
    ChatSlashCommandSpec("compact", "/compact [focus]", "Force conversation compaction"),
    ChatSlashCommandSpec("clear", "/clear", "Wipe conversation; keep session id and log"),
    ChatSlashCommandSpec("resume", "/resume [id]", "Resume a previous session"),
    ChatSlashCommandSpec("stream", "/stream on|off", "Toggle streaming"),
    ChatSlashCommandSpec("trace", "/trace", "Set reasoning trace detail"),
    ChatSlashCommandSpec("config", "/config", "Open or show configuration"),
    ChatSlashCommandSpec("toolbar", "/toolbar", "Configure status toolbar"),
    ChatSlashCommandSpec("assets", "/assets", "Open Forge assets"),
    ChatSlashCommandSpec("image", "/image [path]", "Queue an image for the next message"),
    ChatSlashCommandSpec(
        "subagent",
        "/subagent [name] [task]",
        "No args opens picker; /subagent on|off|status toggles delegation",
    ),
    ChatSlashCommandSpec("forge", "/forge [resume]", "Enter or resume Forge"),
    ChatSlashCommandSpec("history", "/history <pattern>", "Search session history"),
    ChatSlashCommandSpec("report", "/report [text]", "Create feedback bundle and issue draft"),
    ChatSlashCommandSpec("feedback", "/feedback [text]", "Alias for /report"),
    ChatSlashCommandSpec("plan", "/plan <task>", "Draft, review, approve, then execute"),
    ChatSlashCommandSpec(
        "skill",
        "/skill [name] [task]",
        "No args lists; <name> shows info; <name> <task> attaches",
    ),
    ChatSlashCommandSpec("exit", "/exit", "Exit chat"),
)

FORGE_SPECS: tuple[ChatSlashCommandSpec, ...] = (
    ChatSlashCommandSpec("assistant", "/assistant on|off|status", "Toggle planner assistant"),
    ChatSlashCommandSpec("execute", "/execute plan", "Run scoped planned tasks"),
    ChatSlashCommandSpec("goal", "/goal <text>", "Set project goal"),
    ChatSlashCommandSpec("task", "/task <title>", "Add a task to the plan"),
    ChatSlashCommandSpec("show", "/show", "Show the current plan summary"),
    ChatSlashCommandSpec("done", "/done", "Save and validate the plan"),
    ChatSlashCommandSpec("back", "/back", "Return to chat without finalizing"),
)

_SHARED_NESTED_SPECS: tuple[_ChatSlashCompletionSpec, ...] = (
    _ChatSlashCompletionSpec("/usage hud", "/usage hud", "Open usage HUD controls"),
    _ChatSlashCompletionSpec("/usage hud on", "/usage hud on", "Enable the persistent usage HUD"),
    _ChatSlashCompletionSpec(
        "/usage hud off", "/usage hud off", "Disable the persistent usage HUD"
    ),
    _ChatSlashCompletionSpec(
        "/usage hud status", "/usage hud status", "Show the current usage HUD state"
    ),
    _ChatSlashCompletionSpec("/terminals list", "/terminals list", "List background processes"),
    _ChatSlashCompletionSpec(
        "/terminals show", "/terminals show <process_id>", "Show background process output"
    ),
    _ChatSlashCompletionSpec(
        "/terminals kill", "/terminals kill <process_id>", "Kill a background process"
    ),
    _ChatSlashCompletionSpec("/terminals help", "/terminals help", "Show terminal command usage"),
    _ChatSlashCompletionSpec("/subagent on", "/subagent on", "Enable subagent delegation"),
    _ChatSlashCompletionSpec("/subagent off", "/subagent off", "Disable subagent delegation"),
    _ChatSlashCompletionSpec(
        "/subagent status", "/subagent status", "Show the current subagent state"
    ),
)

_CHAT_NESTED_SPECS: tuple[_ChatSlashCompletionSpec, ...] = (
    _ChatSlashCompletionSpec("/forge resume", "/forge resume", "Resume the current run pointer"),
    _ChatSlashCompletionSpec("/plan mode", "/plan mode", "Enter persistent readonly planning"),
    _ChatSlashCompletionSpec("/plan approve", "/plan approve", "Execute the stored plan draft"),
)

_FORGE_NESTED_SPECS: tuple[_ChatSlashCompletionSpec, ...] = (
    _ChatSlashCompletionSpec("/execute plan", "/execute plan", "Run scoped planned tasks"),
    _ChatSlashCompletionSpec(
        "/plan markdown", "/plan markdown", "Preview PLAN.md for the current run"
    ),
    _ChatSlashCompletionSpec("/plan md", "/plan md", "Preview PLAN.md for the current run"),
    _ChatSlashCompletionSpec("/plan edit", "/plan edit", "Edit plan.json and reload Forge state"),
)


def get_chat_specs() -> tuple[ChatSlashCommandSpec, ...]:
    return SPECS


def get_forge_specs() -> tuple[ChatSlashCommandSpec, ...]:
    return FORGE_SPECS


def max_completions_for_mode(mode: str) -> int:
    """Return the largest static slash completion list for one menu in this mode."""

    normalized = str(mode or "chat").strip().lower()
    if normalized == "forge":
        top_level_count = len([spec for spec in SPECS if spec.name not in {"clear", "forge"}])
        top_level_count += len(FORGE_SPECS)
        nested_count = len(_SHARED_NESTED_SPECS) + len(_FORGE_NESTED_SPECS)
        return max(top_level_count, nested_count)

    top_level_count = len(SPECS)
    nested_count = len(_SHARED_NESTED_SPECS) + len(_CHAT_NESTED_SPECS)
    return max(top_level_count, nested_count)


class ChatSlashCompleter(Completer):
    def __init__(
        self,
        *,
        mode_provider: Callable[[], str],
        subagent_names_provider: Callable[[], Sequence[str]] | None = None,
        skill_names_provider: Callable[[], Sequence[str]] | None = None,
    ) -> None:
        self._mode_provider = mode_provider
        self._subagent_names_provider = subagent_names_provider
        self._skill_names_provider = skill_names_provider

    def get_completions(self, document: Any, complete_event: Any) -> Iterable[Completion]:
        del complete_event
        text = str(getattr(document, "text_before_cursor", "") or "")
        if not text.startswith("/"):
            return
        completions = self._matching_completions(text)
        for spec in completions:
            yield Completion(
                text=spec.completion,
                start_position=-len(text),
                display=f"{spec.usage:<34} {spec.description}",
            )

    def _matching_completions(self, text: str) -> list[_ChatSlashCompletionSpec]:
        entries = self._completion_entries_for_text(text)
        text_lower = text.lower()
        matches = [entry for entry in entries if entry.completion.lower().startswith(text_lower)]
        exact = [entry for entry in matches if entry.completion.lower() == text_lower]
        inexact = [entry for entry in matches if entry.completion.lower() != text_lower]
        return [*exact, *inexact]

    def _completion_entries_for_text(self, text: str) -> list[_ChatSlashCompletionSpec]:
        if " " not in text[1:]:
            return self._top_level_entries()
        return self._nested_entries(text)

    def _top_level_entries(self) -> list[_ChatSlashCompletionSpec]:
        specs = list(get_chat_specs())
        if self._mode() == "forge":
            specs = [spec for spec in specs if spec.name not in {"clear", "forge"}]
            specs.extend(get_forge_specs())
        return [
            _ChatSlashCompletionSpec(
                completion=f"/{spec.name}",
                usage=spec.usage,
                description=spec.description,
            )
            for spec in specs
        ]

    def _nested_entries(self, text: str) -> list[_ChatSlashCompletionSpec]:
        entries = list(_SHARED_NESTED_SPECS)
        if self._mode() == "forge":
            entries.extend(_FORGE_NESTED_SPECS)
        else:
            entries.extend(_CHAT_NESTED_SPECS)
        lower = text.lower()
        if lower.startswith("/subagent "):
            entries.extend(self._subagent_name_entries())
        if lower.startswith("/skill "):
            entries.extend(self._skill_name_entries())
        return entries

    def _subagent_name_entries(self) -> list[_ChatSlashCompletionSpec]:
        names = self._safe_names(self._subagent_names_provider)
        return [
            _ChatSlashCompletionSpec(
                completion=f"/subagent {name}",
                usage=f"/subagent {name} <task>",
                description="Run this subagent explicitly",
            )
            for name in names
        ]

    def _skill_name_entries(self) -> list[_ChatSlashCompletionSpec]:
        names = self._safe_names(self._skill_names_provider)
        return [
            _ChatSlashCompletionSpec(
                completion=f"/skill {name}",
                usage=f"/skill {name} [task]",
                description="Show skill info or attach it for one turn",
            )
            for name in names
        ]

    @staticmethod
    def _safe_names(provider: Callable[[], Sequence[str]] | None) -> list[str]:
        if provider is None:
            return []
        try:
            values = provider()
        except Exception:
            return []
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            name = str(value or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            ordered.append(name)
        return ordered

    def _mode(self) -> str:
        try:
            mode = str(self._mode_provider() or "chat").strip().lower()
        except Exception:
            return "chat"
        return "forge" if mode == "forge" else "chat"
