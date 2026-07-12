from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest
from rich.console import Console

import sylliptor_agent_cli.agent_loop as agent_loop_mod
import sylliptor_agent_cli.cli_impl.chat as chat_impl_mod
from sylliptor_agent_cli import cli as cli_mod
from sylliptor_agent_cli.agent_loop import create_session
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.session_store import read_session_events
from sylliptor_agent_cli.workspace_binding import resolve_workspace_binding


def _fake_git_repo(root: Path) -> None:
    git_dir = root / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True, exist_ok=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "refs" / "heads" / "main").write_text("0" * 40 + "\n", encoding="utf-8")


class _FakeShellRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(
        self, *, root: Path, cwd: Path, cmd: str, timeout_s: int
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(
            {
                "root": root,
                "cwd": cwd,
                "cmd": cmd,
                "timeout_s": timeout_s,
            }
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok\n", stderr="")


def _binding_context_content(session: object) -> str:
    messages = getattr(session, "messages", [])
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if str(message.get("role") or "") != "user":
            continue
        content = str(message.get("content") or "")
        if content.lstrip().startswith("<workspace_binding_context>"):
            return content
    return ""


def _create_chat_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    focus_path: Path | None = None,
    no_log: bool = True,
    active_workdir_relpath_override: str | None = None,
) -> tuple[object, _FakeShellRunner, Path]:
    repo = tmp_path / "repo"
    (repo / "packages" / "app").mkdir(parents=True, exist_ok=True)
    (repo / "packages" / "lib").mkdir(parents=True, exist_ok=True)
    (repo / "package.json").write_text('{"name":"repo-root"}\n', encoding="utf-8")
    (repo / "packages" / "app" / "package.json").write_text(
        '{"name":"packages-app"}\n',
        encoding="utf-8",
    )
    (repo / "packages" / "lib" / "package.json").write_text(
        '{"name":"packages-lib"}\n',
        encoding="utf-8",
    )
    _fake_git_repo(repo)
    target_path = focus_path if focus_path is not None else repo
    workspace_binding = resolve_workspace_binding(target_path, source="cwd")
    runner = _FakeShellRunner()
    monkeypatch.setattr(agent_loop_mod, "build_shell_runner", lambda **_kwargs: runner)
    session = create_session(
        cfg=AppConfig(model="test-model", web_search_mode="off"),
        root=workspace_binding.workspace_context.workspace_root,
        mode="auto",
        runtime_kind="interactive_chat",
        yes=True,
        max_steps=1,
        no_log=no_log,
        api_key_override="override-key",
        console=Console(file=io.StringIO(), force_terminal=False, width=140),
        non_interactive=True,
        session_log_dir_override=tmp_path / "sessions",
        workspace_binding=workspace_binding,
        active_workdir_relpath_override=active_workdir_relpath_override,
    )
    return session, runner, repo


def _run_chat_command(session: object, command: str) -> tuple[object, str]:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=4096)
    result = chat_impl_mod._handle_chat_command_impl(
        cli_mod,
        input_text=command,
        root=Path(getattr(session, "root", Path("."))),
        session=session,
        pending_images=[],
        console=console,
        forge_state=cli_mod._ForgeChatState(),
        plan_mode_state=cli_mod._ChatPlanModeState(),
        plan_mode_escape_supported=False,
    )
    return result, buffer.getvalue()


def test_create_session_initializes_active_workdir_from_git_subdirectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    focus_path = tmp_path / "repo" / "packages" / "app"
    session, _runner, repo = _create_chat_session(
        tmp_path,
        monkeypatch,
        focus_path=focus_path,
    )
    try:
        expected_relpath = "packages/app"

        assert session.focus_relpath == expected_relpath
        assert session.active_workdir_relpath == expected_relpath
        assert agent_loop_mod.resolve_session_active_workdir_path(session) == focus_path.resolve()

        binding_context = _binding_context_content(session)
        assert f"workspace_root: {repo.resolve()}" in binding_context
        assert f"focus_dir: {focus_path.resolve()}" in binding_context
        assert f"active_workdir: {focus_path.resolve()}" in binding_context
        assert f"active_workdir_relpath: {expected_relpath}" in binding_context
    finally:
        session.close()


def test_create_session_active_workdir_override_initializes_session_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _runner, repo = _create_chat_session(
        tmp_path,
        monkeypatch,
        no_log=False,
        active_workdir_relpath_override="packages/app",
    )
    try:
        expected_path = repo / "packages" / "app"

        assert session.active_workdir_relpath == "packages/app"
        assert session.store.cwd == str(expected_path)
        events = list(read_session_events(session.store.path))
        start_event = next(event for event in events if event.get("type") == "session_start")
        assert start_event["payload"]["active_workdir_relpath"] == "packages/app"
        assert start_event["payload"]["active_workdir"] == str(expected_path)
    finally:
        session.close()


def test_create_session_active_workdir_override_fails_before_log_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(agent_loop_mod.SessionWorkdirError):
        _create_chat_session(
            tmp_path,
            monkeypatch,
            no_log=False,
            active_workdir_relpath_override="missing-package",
        )

    sessions_dir = tmp_path / "sessions"
    assert not list(sessions_dir.glob("*.jsonl"))


def test_shell_run_without_cwd_uses_active_workdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    focus_path = tmp_path / "repo" / "packages" / "app"
    session, runner, _repo = _create_chat_session(
        tmp_path,
        monkeypatch,
        focus_path=focus_path,
    )
    try:
        session.tools["shell_run"].run({"cmd": "pwd"})

        assert runner.calls
        assert runner.calls[0]["cwd"] == focus_path.resolve()
    finally:
        session.close()


def test_natural_language_navigation_changes_active_workdir_and_tool_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, str]] = []

    def fake_fs_list(*, root: Path, root_path: str, globs=None, ignore=None) -> dict[str, object]:
        _ = root, globs, ignore
        captured.append(("fs_list", root_path))
        return {"root_path": root_path, "entries": []}

    def fake_search_rg(
        *, root: Path, pattern: str, root_path: str, globs=None, **kwargs: object
    ) -> dict[str, object]:
        _ = root, pattern, globs, kwargs
        captured.append(("search_rg", root_path))
        return {"root_path": root_path, "matches": []}

    def fake_symbol_search(
        *,
        root: Path,
        query: str,
        kind=None,
        root_path: str,
        globs=None,
        max_results=100,
        exact=False,
        **kwargs: object,
    ) -> dict[str, object]:
        _ = root, query, kind, globs, max_results, exact, kwargs
        captured.append(("symbol_search", root_path))
        return {"root_path": root_path, "matches": []}

    monkeypatch.setattr(agent_loop_mod, "fs_list", fake_fs_list)
    monkeypatch.setattr(agent_loop_mod, "search_rg", fake_search_rg)
    monkeypatch.setattr(agent_loop_mod, "symbol_search", fake_symbol_search)

    session, runner, repo = _create_chat_session(tmp_path, monkeypatch)
    try:
        result, output = _run_chat_command(session, "go to packages/app")
        expected_relpath = "packages/app"

        assert result == "handled"
        assert "Active workdir:" in output
        assert session.active_workdir_relpath == expected_relpath

        session.tools["shell_run"].run({"cmd": "pwd"})
        session.tools["fs_list"].run({})
        session.tools["search_rg"].run({"pattern": "TODO"})
        session.tools["symbol_search"].run({"query": "App"})

        assert runner.calls[-1]["cwd"] == (repo / "packages" / "app").resolve()
        assert captured == [
            ("fs_list", expected_relpath),
            ("search_rg", expected_relpath),
            ("symbol_search", expected_relpath),
        ]
    finally:
        session.close()


def test_natural_language_navigation_rejects_paths_that_leave_workspace_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _runner, _repo = _create_chat_session(tmp_path, monkeypatch)
    try:
        result, output = _run_chat_command(session, "go to ..")

        assert result == "handled"
        assert "escapes the bound workspace_root" in output
        assert session.active_workdir_relpath == "."
    finally:
        session.close()


def test_status_and_pwd_show_workspace_focus_and_active_workdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    focus_path = tmp_path / "repo" / "packages" / "app"
    session, _runner, repo = _create_chat_session(
        tmp_path,
        monkeypatch,
        focus_path=focus_path,
    )
    try:
        _status_result, status_output = _run_chat_command(session, "/status")
        _pwd_result, pwd_output = _run_chat_command(session, "/pwd")
        expected_relpath = "packages/app"

        assert "workspace_root" in status_output
        assert str(repo.resolve()) in status_output
        assert "focus_dir" in status_output
        assert "active_workdir" in status_output
        assert expected_relpath in status_output
        assert focus_path.name in status_output

        assert f"active_workdir: {focus_path.resolve()}" in pwd_output
        assert f"active_workdir_relpath: {expected_relpath}" in pwd_output
        assert f"focus_dir: {focus_path.resolve()}" in pwd_output
        assert f"workspace_root: {repo.resolve()}" in pwd_output
    finally:
        session.close()


def test_session_set_workdir_tool_refreshes_binding_context_and_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, runner, repo = _create_chat_session(tmp_path, monkeypatch, no_log=False)
    try:
        result = session.tools["session_set_workdir"].run({"path": "packages/app"})
        session.tools["shell_run"].run({"cmd": "pwd"})
        expected_relpath = "packages/app"

        assert result["active_workdir_relpath"] == expected_relpath
        assert session.active_workdir_relpath == expected_relpath
        assert runner.calls[-1]["cwd"] == (repo / "packages" / "app").resolve()

        binding_context = _binding_context_content(session)
        assert f"active_workdir_relpath: {expected_relpath}" in binding_context
        assert f"active_workdir: {(repo / 'packages' / 'app').resolve()}" in binding_context

        events = list(read_session_events(session.store.path))
        workdir_event = next(
            event for event in events if event.get("type") == "session_workdir_changed"
        )
        cmd_event = next(event for event in events if event.get("type") == "cmd")
        assert workdir_event["payload"]["active_workdir_relpath"] == expected_relpath
        assert cmd_event["cwd"] == str((repo / "packages" / "app").resolve())
        assert cmd_event["active_workdir_relpath"] == expected_relpath
    finally:
        session.close()


def test_mixed_natural_language_navigation_turn_returns_follow_up_instruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _runner, repo = _create_chat_session(tmp_path, monkeypatch)
    try:
        result, output = _run_chat_command(
            session, "switch to packages/app and inspect package.json"
        )

        assert isinstance(result, chat_impl_mod._ChatExecutionRequest)
        assert result.instruction == "inspect package.json"
        assert "Active workdir:" in output
        assert session.active_workdir_relpath == "packages/app"
        assert session.tools["fs_read"].run({"path": "package.json"})["content"].replace(
            "\r\n", "\n"
        ) == (repo / "packages" / "app" / "package.json").read_text(encoding="utf-8")
    finally:
        session.close()


def test_fs_read_defaults_to_active_workdir_after_navigation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _runner, _repo = _create_chat_session(tmp_path, monkeypatch)
    try:
        _run_chat_command(session, "work in packages/app")

        content = session.tools["fs_read"].run({"path": "package.json"})["content"]

        assert content.replace("\r\n", "\n") == '{"name":"packages-app"}\n'
    finally:
        session.close()


def test_fs_write_defaults_to_active_workdir_after_workdir_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _runner, repo = _create_chat_session(tmp_path, monkeypatch)
    try:
        session.tools["session_set_workdir"].run({"path": "packages/app"})
        session.tools["fs_write"].run({"path": "note.txt", "content": "hello\n"})

        assert not (repo / "note.txt").exists()
        assert (repo / "packages" / "app" / "note.txt").read_text(encoding="utf-8").replace(
            "\r\n", "\n"
        ) == "hello\n"
        assert (
            session.tools["fs_read"].run({"path": "note.txt"})["content"].replace("\r\n", "\n")
            == "hello\n"
        )
    finally:
        session.close()


def test_workspace_root_override_still_works_after_workdir_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _runner, repo = _create_chat_session(tmp_path, monkeypatch)
    try:
        session.tools["session_set_workdir"].run({"path": "packages/app"})
        session.tools["fs_write"].run(
            {
                "path": "note.txt",
                "path_base": "workspace_root",
                "content": "root-level\n",
            }
        )

        assert (repo / "note.txt").read_text(encoding="utf-8").replace(
            "\r\n", "\n"
        ) == "root-level\n"
        assert not (repo / "packages" / "app" / "note.txt").exists()
        assert (
            session.tools["fs_read"]
            .run(
                {
                    "path": "package.json",
                    "path_base": "workspace_root",
                }
            )["content"]
            .replace("\r\n", "\n")
            == '{"name":"repo-root"}\n'
        )
    finally:
        session.close()


def test_tool_roots_align_on_active_workdir_with_workspace_root_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, str]] = []

    def fake_fs_list(*, root: Path, root_path: str, globs=None, ignore=None) -> dict[str, object]:
        _ = root, globs, ignore
        captured.append(("fs_list", root_path))
        return {"root_path": root_path, "entries": []}

    def fake_search_rg(
        *, root: Path, pattern: str, root_path: str, globs=None, **kwargs: object
    ) -> dict[str, object]:
        _ = root, pattern, globs, kwargs
        captured.append(("search_rg", root_path))
        return {"root_path": root_path, "matches": []}

    def fake_symbol_search(
        *,
        root: Path,
        query: str,
        kind=None,
        root_path: str,
        globs=None,
        max_results=100,
        exact=False,
        **kwargs: object,
    ) -> dict[str, object]:
        _ = root, query, kind, globs, max_results, exact, kwargs
        captured.append(("symbol_search", root_path))
        return {"root_path": root_path, "matches": []}

    monkeypatch.setattr(agent_loop_mod, "fs_list", fake_fs_list)
    monkeypatch.setattr(agent_loop_mod, "search_rg", fake_search_rg)
    monkeypatch.setattr(agent_loop_mod, "symbol_search", fake_symbol_search)

    session, runner, repo = _create_chat_session(tmp_path, monkeypatch)
    try:
        session.tools["session_set_workdir"].run({"path": "packages/app"})
        expected_relpath = "packages/app"

        session.tools["fs_list"].run({})
        session.tools["search_rg"].run({"pattern": "TODO"})
        session.tools["symbol_search"].run({"query": "App"})
        session.tools["shell_run"].run({"cmd": "pwd"})

        session.tools["fs_list"].run({"path_base": "workspace_root"})
        session.tools["search_rg"].run({"pattern": "TODO", "path_base": "workspace_root"})
        session.tools["symbol_search"].run({"query": "App", "path_base": "workspace_root"})
        session.tools["shell_run"].run({"cmd": "pwd", "cwd_base": "workspace_root"})

        assert runner.calls[0]["cwd"] == (repo / "packages" / "app").resolve()
        assert runner.calls[1]["cwd"] == repo.resolve()
        assert captured == [
            ("fs_list", expected_relpath),
            ("search_rg", expected_relpath),
            ("symbol_search", expected_relpath),
            ("fs_list", "."),
            ("search_rg", "."),
            ("symbol_search", "."),
        ]
    finally:
        session.close()


def test_navigation_false_positive_go_to_definition_is_not_host_handled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _runner, _repo = _create_chat_session(tmp_path, monkeypatch)
    try:
        result, output = _run_chat_command(session, "go to definition")

        assert result == "send"
        assert output == ""
        assert session.active_workdir_relpath == "."
    finally:
        session.close()


def test_navigation_false_positive_directory_name_phrase_is_not_host_handled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _runner, _repo = _create_chat_session(tmp_path, monkeypatch)
    try:
        result, output = _run_chat_command(session, "change the directory name in README")

        assert result == "send"
        assert output == ""
        assert session.active_workdir_relpath == "."
    finally:
        session.close()


def test_natural_language_navigation_logs_active_workdir_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _runner, _repo = _create_chat_session(tmp_path, monkeypatch, no_log=False)
    try:
        result, _output = _run_chat_command(session, "go to packages/app")
        expected_relpath = "packages/app"

        assert result == "handled"
        events = list(read_session_events(session.store.path))
        workdir_event = next(
            event for event in events if event.get("type") == "session_workdir_changed"
        )
        assert workdir_event["payload"]["active_workdir_relpath"] == expected_relpath
        assert (
            cli_mod._load_chat_resume_active_workdir_relpath(session.store.path) == expected_relpath
        )
    finally:
        session.close()


def test_router_client_forces_reasoning_off_while_coding_client_keeps_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Routing/classification calls (the strict-JSON router plus short non-repo
    # replies) must run with model reasoning disabled: on slow reasoning models
    # (e.g. Xiaomi MiMo via the hosted trial proxy) the extra thinking inflates
    # latency/tokens enough to exceed the request timeout, which silently degrades
    # the turn to the generic clarification fallback. Deep reasoning must remain
    # enabled on the coding client.
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    _fake_git_repo(repo)
    workspace_binding = resolve_workspace_binding(repo, source="cwd")
    monkeypatch.setattr(agent_loop_mod, "build_shell_runner", lambda **_kwargs: _FakeShellRunner())
    session = create_session(
        cfg=AppConfig(
            model="test-model",
            web_search_mode="off",
            llm_enable_thinking=True,
            llm_reasoning_effort="high",
        ),
        root=workspace_binding.workspace_context.workspace_root,
        mode="auto",
        runtime_kind="interactive_chat",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="override-key",
        console=Console(file=io.StringIO(), force_terminal=False, width=140),
        non_interactive=True,
        session_log_dir_override=tmp_path / "sessions",
        workspace_binding=workspace_binding,
    )
    try:
        # Coding client honors the configured reasoning settings...
        assert session.client.enable_thinking is True
        assert session.client.reasoning_effort == "high"
        # ...but the router/classification client forces reasoning off so routing
        # stays fast and deterministic.
        assert session.router_client is not None
        assert session.router_client.enable_thinking is False
        assert session.router_client.reasoning_effort in (None, "")
    finally:
        session.close()
