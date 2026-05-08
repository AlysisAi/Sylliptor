from __future__ import annotations

from pathlib import Path

from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text

from sylliptor_agent_cli.cli_impl.chat_slash_completer import (
    ChatSlashCompleter,
    get_chat_specs,
    get_forge_specs,
    max_completions_for_mode,
)


def _completion_names(completer: ChatSlashCompleter, text: str) -> list[str]:
    return [
        completion.text.removeprefix("/")
        for completion in completer.get_completions(Document(text=text), None)
    ]


def _display_text(value: object) -> str:
    return fragment_list_to_text(to_formatted_text(value))


def _source_between(source: str, start_marker: str, end_marker: str) -> str:
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def test_get_chat_specs_match_curated_visible_surface() -> None:
    assert [spec.name for spec in get_chat_specs()] == [
        "help",
        "mode",
        "status",
        "terminals",
        "pwd",
        "usage",
        "ctx",
        "compact",
        "clear",
        "resume",
        "stream",
        "trace",
        "config",
        "toolbar",
        "assets",
        "image",
        "subagent",
        "forge",
        "history",
        "report",
        "feedback",
        "plan",
        "skill",
        "exit",
    ]


def test_max_completions_for_mode_covers_static_chat_surface() -> None:
    assert max_completions_for_mode("chat") >= len(get_chat_specs())


def test_max_completions_for_mode_covers_static_forge_surface() -> None:
    assert max_completions_for_mode("forge") >= len(get_forge_specs())


def test_get_chat_completions_filters_by_prefix() -> None:
    completer = ChatSlashCompleter(mode_provider=lambda: "chat")

    assert _completion_names(completer, "/co") == ["compact", "config"]


def test_chat_completions_exclude_hidden_and_removed_commands() -> None:
    completer = ChatSlashCompleter(mode_provider=lambda: "chat")
    names = set(_completion_names(completer, "/"))
    spec_names = {spec.name for spec in get_chat_specs()}

    assert "quit" not in spec_names
    assert "context" not in spec_names
    assert "model" not in spec_names
    assert "model-info" not in spec_names
    assert "paste-image" not in spec_names
    assert "clear-images" not in spec_names
    assert "skills" not in spec_names
    assert "subagents" not in spec_names
    assert not {"back", "done", "show", "goal", "task", "assistant", "execute"} & names


def test_forge_completions_include_general_chat_and_forge_commands() -> None:
    completer = ChatSlashCompleter(mode_provider=lambda: "forge")

    assert _completion_names(completer, "/go") == ["goal"]
    assert _completion_names(completer, "/st") == ["status", "stream"]
    assert _completion_names(completer, "/ta") == ["task"]


def test_nested_chat_completions_cover_usage_subagent_and_skill_names() -> None:
    completer = ChatSlashCompleter(
        mode_provider=lambda: "chat",
        subagent_names_provider=lambda: ["explorer", "reviewer"],
        skill_names_provider=lambda: ["python", "docker"],
    )

    assert _completion_names(completer, "/usage ") == [
        "usage hud",
        "usage hud on",
        "usage hud off",
        "usage hud status",
    ]
    assert _completion_names(completer, "/terminals ") == [
        "terminals list",
        "terminals show",
        "terminals kill",
        "terminals help",
    ]
    assert _completion_names(completer, "/subagent ") == [
        "subagent on",
        "subagent off",
        "subagent status",
        "subagent explorer",
        "subagent reviewer",
    ]
    assert _completion_names(completer, "/skill ") == ["skill python", "skill docker"]


def test_no_completions_for_plain_text() -> None:
    completer = ChatSlashCompleter(mode_provider=lambda: "chat")

    assert _completion_names(completer, "hello world") == []


def test_completion_carries_usage_and_description() -> None:
    completer = ChatSlashCompleter(mode_provider=lambda: "chat")
    completion = list(completer.get_completions(Document(text="/model-i"), None))

    assert completion == []

    usage_completion = list(completer.get_completions(Document(text="/usage "), None))[0]
    assert "/usage hud" in _display_text(usage_completion.display)
    assert "Open usage HUD controls" in _display_text(usage_completion.display)
    assert _display_text(usage_completion.display_meta) == ""


def test_completion_yields_all_matching_top_level_commands() -> None:
    completer = ChatSlashCompleter(mode_provider=lambda: "chat")

    completions = list(completer.get_completions(Document(text="/"), None))

    assert len(completions) == len(get_chat_specs())
    completion_names = [completion.text for completion in completions]
    assert completion_names
    assert "/plan" in completion_names
    assert "/resume" in completion_names
    assert "/config" in completion_names
    assert "/subagent" in completion_names
    assert "/skill" in completion_names
    assert "/forge" in completion_names
    assert "/exit" in completion_names


def test_specs_match_legacy_chat_handler() -> None:
    source = Path("src/sylliptor_agent_cli/cli_impl/chat/commands.py").read_text(encoding="utf-8")
    handler_source = _source_between(
        source,
        "def _handle_chat_command(",
        "def _handle_forge_chat_command(",
    )
    cli_source = Path("src/sylliptor_agent_cli/cli.py").read_text(encoding="utf-8")

    for spec in get_chat_specs():
        token = f'"/{spec.name}"'
        if spec.name == "forge":
            assert "_parse_forge_enter_command" in handler_source
            assert token in cli_source
            continue
        assert token in handler_source


def test_forge_specs_match_legacy_forge_handler() -> None:
    source = Path("src/sylliptor_agent_cli/cli_impl/chat/commands.py").read_text(encoding="utf-8")
    handler_source = source[source.index("def _handle_forge_chat_command(") :]

    for spec in get_forge_specs():
        assert f'"/{spec.name}"' in handler_source
