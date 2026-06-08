from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rich.console import Console

from sylliptor_agent_cli import cli as cli_mod
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.runtime_kind import RuntimeKind
from sylliptor_agent_cli.session_store import SessionInfo, SessionStore
from sylliptor_agent_cli.web_research import build_web_research_artifact_from_events


def test_load_chat_resume_messages_reads_user_and_assistant_events(tmp_path: Path) -> None:
    log_path = tmp_path / "resume-target.jsonl"
    events = [
        {"type": "session_start", "payload": {"mode": "auto"}},
        {"type": "user_message", "payload": {"content": "Hello"}},
        {"type": "route_decision", "payload": {"route": "chat"}},
        {"type": "assistant_message", "payload": {"content": "Hi there"}},
        {"type": "final", "payload": {"content": "Hi there"}},
    ]
    log_path.write_text("\n".join(json.dumps(ev) for ev in events) + "\n", encoding="utf-8")

    loaded = cli_mod._load_chat_resume_messages(log_path)
    assert loaded == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]


def test_load_chat_resume_messages_uses_shaped_tool_result_content(tmp_path: Path) -> None:
    log_path = tmp_path / "resume-shaped-tool-result.jsonl"
    shaped_content = json.dumps(
        {
            "transcript_shaped": True,
            "tool": "fs_read",
            "tool_call_id": "tc-read",
            "summary": 'Loaded "large.txt" (5000 chars).',
            "preview": "A" * 80,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )
    events = [
        {"type": "session_start", "payload": {"mode": "auto"}},
        {"type": "user_message", "payload": {"content": "Read the large file."}},
        {
            "type": "assistant_message",
            "payload": {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc-read",
                            "type": "function",
                            "function": {
                                "name": "fs_read",
                                "arguments": json.dumps({"path": "large.txt"}),
                            },
                        }
                    ],
                }
            },
        },
        {
            "type": "tool_result",
            "payload": {
                "name": "fs_read",
                "tool_call_id": "tc-read",
                "result": {"path": "large.txt", "content": "A" * 5000},
                "content": shaped_content,
                "step": 1,
            },
        },
    ]
    log_path.write_text("\n".join(json.dumps(ev) for ev in events) + "\n", encoding="utf-8")

    loaded = cli_mod._load_chat_resume_messages(log_path)

    assert loaded[-1] == {
        "role": "tool",
        "tool_call_id": "tc-read",
        "content": shaped_content,
    }
    assert "A" * 1000 not in loaded[-1]["content"]


def test_load_chat_resume_messages_prefers_user_display_content(tmp_path: Path) -> None:
    log_path = tmp_path / "resume-display.jsonl"
    events = [
        {
            "type": "user_message",
            "payload": {
                "content": (
                    "Fix the bug.\n\nApproved plan:\n1. Inspect code\n\n"
                    "Now execute this task in the repository and follow the approved plan."
                ),
                "display_content": "Fix the bug.",
            },
        },
        {"type": "assistant_message", "payload": {"content": "Done."}},
    ]
    log_path.write_text("\n".join(json.dumps(ev) for ev in events) + "\n", encoding="utf-8")

    loaded = cli_mod._load_chat_resume_messages(log_path)
    assert loaded == [
        {"role": "user", "content": "Fix the bug."},
        {"role": "assistant", "content": "Done."},
    ]


def test_load_chat_resume_messages_accepts_final_or_route_reply_fallback(tmp_path: Path) -> None:
    log_path = tmp_path / "resume-fallback.jsonl"
    events = [
        {"type": "session_start", "payload": {"mode": "auto"}},
        {"type": "user_message", "payload": {"content": "Ping"}},
        {"type": "final", "payload": {"content": "Pong from final"}},
        {"type": "route_decision", "payload": {"route": "chat", "reply": "Natural reply"}},
    ]
    log_path.write_text("\n".join(json.dumps(ev) for ev in events) + "\n", encoding="utf-8")

    loaded = cli_mod._load_chat_resume_messages(log_path)
    assert loaded == [
        {"role": "user", "content": "Ping"},
        {"role": "assistant", "content": "Pong from final"},
        {"role": "assistant", "content": "Natural reply"},
    ]


def test_load_chat_resume_active_workdir_relpath_prefers_latest_change(tmp_path: Path) -> None:
    log_path = tmp_path / "resume-workdir.jsonl"
    events = [
        {
            "type": "session_start",
            "payload": {
                "mode": "auto",
                "active_workdir_relpath": ".",
            },
        },
        {
            "type": "session_workdir_changed",
            "payload": {
                "active_workdir_relpath": "packages/app",
            },
        },
        {
            "type": "session_workdir_changed",
            "payload": {
                "active_workdir_relpath": "packages/lib",
            },
        },
    ]
    log_path.write_text("\n".join(json.dumps(ev) for ev in events) + "\n", encoding="utf-8")

    assert cli_mod._load_chat_resume_active_workdir_relpath(log_path) == "packages/lib"


def test_first_user_message_preview_prefers_user_display_content(tmp_path: Path) -> None:
    log_path = tmp_path / "resume-preview.jsonl"
    events = [
        {
            "type": "user_message",
            "payload": {
                "content": (
                    "Implement the feature.\n\nApproved plan:\n1. Edit src/app.py\n\n"
                    "Now execute this task in the repository and follow the approved plan."
                ),
                "display_content": "Implement the feature.",
            },
        }
    ]
    log_path.write_text("\n".join(json.dumps(ev) for ev in events) + "\n", encoding="utf-8")

    assert cli_mod._first_user_message_preview(log_path) == "Implement the feature."


def test_resolve_chat_resume_target_supports_index_exact_and_unique_prefix(tmp_path: Path) -> None:
    sessions = [
        SessionInfo(session_id="20260101T100000Z_abcd1111", path=tmp_path / "a.jsonl", mtime=3.0),
        SessionInfo(session_id="20260102T100000Z_beef2222", path=tmp_path / "b.jsonl", mtime=2.0),
        SessionInfo(session_id="20260103T100000Z_cafe3333", path=tmp_path / "c.jsonl", mtime=1.0),
    ]

    assert (
        cli_mod._resolve_chat_resume_target(raw_value="2", sessions=sessions)
        == sessions[1].session_id
    )
    assert (
        cli_mod._resolve_chat_resume_target(
            raw_value="20260103T100000Z_cafe3333",
            sessions=sessions,
        )
        == sessions[2].session_id
    )
    assert (
        cli_mod._resolve_chat_resume_target(
            raw_value="20260102T100000Z_beef",
            sessions=sessions,
        )
        == sessions[1].session_id
    )
    assert cli_mod._resolve_chat_resume_target(raw_value="2026", sessions=sessions) is None


def test_collect_chat_resume_candidates_keeps_latest_50(tmp_path: Path, monkeypatch) -> None:
    sessions = [
        SessionInfo(session_id=f"s{i:03d}", path=tmp_path / f"s{i:03d}.jsonl", mtime=1000 - i)
        for i in range(70)
    ]

    monkeypatch.setattr(cli_mod, "list_sessions", lambda _sessions_dir: sessions)
    candidates = cli_mod._collect_chat_resume_candidates(
        sessions_dir=tmp_path,
        current_session_id="s000",
    )

    assert len(candidates) == 50
    assert candidates[0].session_id == "s001"
    assert candidates[-1].session_id == "s050"


def test_chat_resume_panel_highlights_selected_session_case_insensitively(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    first.write_text(
        json.dumps({"type": "user_message", "payload": {"content": "Fix startup latency imports"}})
        + "\n",
        encoding="utf-8",
    )
    second.write_text(
        json.dumps({"type": "user_message", "payload": {"content": "Improve resume picker UI"}})
        + "\n",
        encoding="utf-8",
    )
    sessions = [
        SessionInfo(session_id="20260101T100000Z_ABCD1111", path=first, mtime=2.0),
        SessionInfo(session_id="20260102T100000Z_BEEF2222", path=second, mtime=1.0),
    ]
    panel = cli_mod._chat_resume_panel(
        current_session_id="current",
        sessions=sessions,
        selected_session_id=sessions[0].session_id.lower(),
        interactive=True,
    )
    console = Console(width=140, record=True, color_system=None, force_terminal=False)
    console.print(panel)
    rendered = console.export_text()

    assert "Resume Session" in rendered
    assert "Current Session: * current" in rendered
    assert "> Fix startup latency imports" in rendered
    assert "Improve resume picker UI" in rendered


def test_chat_resume_panel_groups_by_date_and_shows_preview_and_footer(tmp_path: Path) -> None:
    now = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)

    today_path = tmp_path / "today.jsonl"
    yesterday_path = tmp_path / "yesterday.jsonl"
    older_path = tmp_path / "older.jsonl"

    today_path.write_text(
        json.dumps(
            {"type": "user_message", "payload": {"content": "Fix duplicate message bug in chat"}}
        )
        + "\n",
        encoding="utf-8",
    )
    yesterday_path.write_text(
        json.dumps({"type": "user_message", "payload": {"content": "Refactor resume panel layout"}})
        + "\n",
        encoding="utf-8",
    )
    older_path.write_text(
        json.dumps(
            {
                "type": "user_message",
                "payload": {
                    "content": "Investigate startup latency and import graph behavior for sylliptor",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    today_ts = now.timestamp()
    yesterday_ts = (now - timedelta(days=1)).timestamp()
    older_ts = (now - timedelta(days=3, hours=1)).timestamp()
    os.utime(today_path, (today_ts, today_ts))
    os.utime(yesterday_path, (yesterday_ts, yesterday_ts))
    os.utime(older_path, (older_ts, older_ts))

    sessions = [
        SessionInfo(session_id="today", path=today_path, mtime=today_ts),
        SessionInfo(session_id="yesterday", path=yesterday_path, mtime=yesterday_ts),
        SessionInfo(session_id="older", path=older_path, mtime=older_ts),
    ]

    panel = cli_mod._chat_resume_panel(
        current_session_id="current-session",
        sessions=sessions,
        selected_session_id="today",
        interactive=True,
    )

    console = Console(width=150, record=True, color_system=None, force_terminal=False)
    console.print(panel)
    rendered = console.export_text()

    assert "Today -" in rendered
    assert "Yesterday -" in rendered
    assert "Fix duplicate message bug in chat" in rendered
    assert "Refactor resume panel layout" in rendered
    assert "PgUp/PgDn Page" in rendered
    assert "Home/End Jump" in rendered


def test_clamp_resume_scroll_offset_keeps_selection_visible() -> None:
    assert (
        cli_mod._clamp_resume_scroll_offset(
            total_rows=20,
            selected_index=0,
            scroll_offset=7,
            visible_session_rows=5,
        )
        == 0
    )
    assert (
        cli_mod._clamp_resume_scroll_offset(
            total_rows=20,
            selected_index=9,
            scroll_offset=0,
            visible_session_rows=5,
        )
        == 5
    )
    assert (
        cli_mod._clamp_resume_scroll_offset(
            total_rows=20,
            selected_index=18,
            scroll_offset=0,
            visible_session_rows=5,
        )
        == 14
    )


def test_chat_resume_panel_scroll_indicators_and_sticky_group(
    tmp_path: Path,
    monkeypatch,
) -> None:
    now = datetime.now()
    sessions: list[SessionInfo] = []
    for idx in range(8):
        path = tmp_path / f"today-{idx}.jsonl"
        path.write_text(
            json.dumps({"type": "user_message", "payload": {"content": f"Today task {idx}"}})
            + "\n",
            encoding="utf-8",
        )
        ts = (now - timedelta(minutes=idx)).timestamp()
        sessions.append(SessionInfo(session_id=f"today-{idx}", path=path, mtime=ts))
    for idx in range(4):
        path = tmp_path / f"yesterday-{idx}.jsonl"
        path.write_text(
            json.dumps({"type": "user_message", "payload": {"content": f"Yesterday task {idx}"}})
            + "\n",
            encoding="utf-8",
        )
        ts = (now - timedelta(days=1, minutes=idx)).timestamp()
        sessions.append(SessionInfo(session_id=f"yesterday-{idx}", path=path, mtime=ts))

    monkeypatch.setattr(cli_mod, "_terminal_dimensions", lambda: (100, 20))

    panel = cli_mod._chat_resume_panel(
        current_session_id="current",
        sessions=sessions,
        selected_session_id="today-4",
        interactive=True,
        scroll_offset=2,
        visible_session_rows=4,
    )
    console = Console(width=120, record=True, color_system=None, force_terminal=False)
    console.print(panel)
    rendered = console.export_text()

    assert "... 2 more sessions above" in rendered
    assert "more sessions below" in rendered
    assert "Today -" in rendered
    assert "Today task 4" in rendered


def test_chat_resume_panel_shows_resize_warning_when_terminal_too_small(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "small.jsonl"
    path.write_text(
        json.dumps({"type": "user_message", "payload": {"content": "hello world"}}) + "\n",
        encoding="utf-8",
    )
    sessions = [SessionInfo(session_id="small", path=path, mtime=datetime.now().timestamp())]

    monkeypatch.setattr(cli_mod, "_terminal_dimensions", lambda: (30, 8))

    panel = cli_mod._chat_resume_panel(
        current_session_id="current",
        sessions=sessions,
        selected_session_id="small",
        interactive=True,
    )
    console = Console(width=60, record=True, color_system=None, force_terminal=False)
    console.print(panel)
    rendered = console.export_text()

    assert "Terminal too small - please resize" in rendered
    assert "Minimum: 40x10" in rendered


def test_resume_preview_prefers_custom_name_then_summary_then_first_message(tmp_path: Path) -> None:
    session_path = tmp_path / "session.jsonl"
    session_path.write_text(
        json.dumps(
            {"type": "user_message", "payload": {"content": "Fix duplicate chat render bug"}}
        )
        + "\n",
        encoding="utf-8",
    )
    info = SessionInfo(session_id="session", path=session_path, mtime=datetime.now().timestamp())

    metadata_path = cli_mod._session_metadata_path(session_path)
    metadata_path.write_text(
        json.dumps({"summary": "Auto generated summary", "custom_name": "My Custom Session"})
        + "\n",
        encoding="utf-8",
    )
    assert cli_mod._load_resume_preview_text(info) == "My Custom Session"

    metadata_path.write_text(
        json.dumps({"summary": "Auto generated summary"}) + "\n",
        encoding="utf-8",
    )
    assert cli_mod._load_resume_preview_text(info) == "Auto generated summary"

    metadata_path.unlink()
    assert cli_mod._load_resume_preview_text(info) == "Fix duplicate chat render bug"


def test_rename_resume_session_custom_title_persists_metadata(tmp_path: Path) -> None:
    session_path = tmp_path / "rename-target.jsonl"
    session_path.write_text("", encoding="utf-8")
    info = SessionInfo(
        session_id="rename-target", path=session_path, mtime=datetime.now().timestamp()
    )

    renamed, message = cli_mod._rename_resume_session_custom_title(
        info=info,
        new_title="Refactor Resume Session Screen",
    )

    assert renamed is True
    assert "renamed" in message.lower()
    metadata = json.loads(cli_mod._session_metadata_path(session_path).read_text(encoding="utf-8"))
    assert metadata["custom_name"] == "Refactor Resume Session Screen"


def test_ensure_session_summary_metadata_generates_claude_title_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    session_path = tmp_path / "summary-target.jsonl"
    events = [
        {"type": "session_start", "payload": {"mode": "review"}},
        {"type": "user_message", "payload": {"content": "Fix duplicate message rendering"}},
        {"type": "assistant_message", "payload": {"content": "Investigating prompt echo path"}},
        {"type": "user_message", "payload": {"content": "Add inline resume rename flow"}},
        {"type": "assistant_message", "payload": {"content": "Implemented picker key handling"}},
    ]
    session_path.write_text("\n".join(json.dumps(ev) for ev in events) + "\n", encoding="utf-8")

    generated: dict[str, int] = {"calls": 0}

    def _fake_summary(*, session: Any, transcript_messages: list[dict[str, str]]) -> str | None:
        _ = session
        assert len(transcript_messages) >= 4
        generated["calls"] += 1
        return "Fix resume message rendering"

    monkeypatch.setattr(cli_mod, "_generate_session_summary_with_model", _fake_summary)

    fake_session = SimpleNamespace(
        cfg=AppConfig(model="test-model", base_url="https://example.com/v1"),
        client=SimpleNamespace(api_key="k", model="test-model"),
        store=SimpleNamespace(
            enabled=True,
            path=session_path,
            sessions_dir=tmp_path,
            session_id="summary-target",
        ),
    )

    cli_mod._ensure_session_summary_metadata(session=fake_session)
    cli_mod._ensure_session_summary_metadata(session=fake_session)

    metadata = json.loads(cli_mod._session_metadata_path(session_path).read_text(encoding="utf-8"))
    assert metadata["summary"] == "Fix resume message rendering"
    assert metadata["summary_source"] == "generated_model"
    assert generated["calls"] == 1


def test_ensure_session_summary_metadata_can_skip_model_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    session_path = tmp_path / "summary-skip-model.jsonl"
    events = [
        {"type": "session_start", "payload": {"mode": "review"}},
        {"type": "user_message", "payload": {"content": "Fix duplicate message rendering"}},
        {"type": "assistant_message", "payload": {"content": "Investigating prompt echo path"}},
        {"type": "user_message", "payload": {"content": "Add inline resume rename flow"}},
        {"type": "assistant_message", "payload": {"content": "Implemented picker key handling"}},
    ]
    session_path.write_text("\n".join(json.dumps(ev) for ev in events) + "\n", encoding="utf-8")

    generated: dict[str, int] = {"calls": 0}

    def _fake_summary(*, session: Any, transcript_messages: list[dict[str, str]]) -> str | None:
        _ = session, transcript_messages
        generated["calls"] += 1
        return "Should not be used"

    monkeypatch.setattr(cli_mod, "_generate_session_summary_with_model", _fake_summary)

    fake_session = SimpleNamespace(
        cfg=AppConfig(model="test-model", base_url="https://example.com/v1"),
        client=SimpleNamespace(api_key="k", model="test-model"),
        store=SimpleNamespace(
            enabled=True,
            path=session_path,
            sessions_dir=tmp_path,
            session_id="summary-skip-model",
        ),
    )

    cli_mod._ensure_session_summary_metadata(session=fake_session, allow_model_summary=False)

    metadata = json.loads(cli_mod._session_metadata_path(session_path).read_text(encoding="utf-8"))
    assert metadata["summary"] == "Fix duplicate message rendering"
    assert metadata["summary_source"] == "first_user_message"
    assert metadata.get("summary_generated_at") in (None, "")
    assert generated["calls"] == 0


def test_render_chat_resume_history_uses_styled_panels(tmp_path: Path) -> None:
    _ = tmp_path
    console = Console(width=120, record=True, color_system=None, force_terminal=False)
    fake_session = SimpleNamespace(console=console)

    cli_mod._render_chat_resume_history(
        session=fake_session,
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ],
    )

    rendered = console.export_text()
    assert "Loaded 2 historical messages." in rendered
    assert "You" in rendered
    assert "Agent" in rendered
    assert "Hello" in rendered
    assert "Hi there" in rendered


def test_build_chat_resume_context_message_summarizes_tools_and_redacts_secrets(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "resume-context.jsonl"
    events = [
        {
            "type": "session_start",
            "payload": {
                "mode": "auto",
                "model": "test-model",
                "workspace_root": "/workspace/project",
                "focus_relpath": ".",
                "active_workdir_relpath": ".",
            },
        },
        {"type": "user_message", "payload": {"content": "Fix src/app.py"}},
        {
            "type": "tool_call",
            "payload": {"name": "fs_read", "arguments": {"path": "src/app.py"}, "step": 1},
        },
        {
            "type": "tool_result",
            "payload": {
                "name": "fs_read",
                "result": {"path": "src/app.py", "content": "print('hello')"},
                "step": 1,
            },
        },
        {
            "type": "tool_call",
            "payload": {
                "name": "shell_run",
                "arguments": {"cmd": "OPENAI_API_KEY=sk-testsecret123456789 pytest -q"},
                "step": 2,
            },
        },
        {
            "type": "tool_result",
            "payload": {
                "name": "shell_run",
                "result": {
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": "Authorization: Bearer abcdefghijk12345",
                },
                "step": 2,
            },
        },
        {
            "type": "tool_call",
            "payload": {
                "name": "custom_tool",
                "arguments": {
                    "password": "plain-json-secret",
                    "nested": {"openai_api_key": "sk-jsonsecret1234567890"},
                },
                "step": 3,
            },
        },
        {
            "type": "tool_result",
            "payload": {
                "name": "fake_optional_tool",
                "result": {
                    "status": "tool_unavailable",
                    "tool": "fake_optional_tool",
                    "reason": "module not importable: fake_optional_dependency",
                },
                "step": 4,
            },
        },
        {
            "type": "verify_run",
            "payload": {
                "commands": ["pytest -q"],
                "all_passed": False,
                "summary": "1 failed, 3 passed",
            },
        },
        {"type": "assistant_message", "payload": {"content": "I found one failing test."}},
    ]
    log_path.write_text("\n".join(json.dumps(ev) for ev in events) + "\n", encoding="utf-8")

    context = cli_mod._build_chat_resume_context_message(log_path)

    assert context is not None
    assert context.startswith("<resume_context>")
    assert "source_session_id: resume-context" in context
    assert "Fix src/app.py" in context
    assert "tool_call fs_read" in context
    assert "tool_result shell_run status=failed" in context
    assert "tool_result fake_optional_tool status=tool_unavailable" in context
    assert "src/app.py" in context
    assert "pytest -q" in context
    assert "sk-testsecret" not in context
    assert "plain-json-secret" not in context
    assert "sk-jsonsecret" not in context
    assert "abcdefghijk12345" not in context
    assert "[REDACTED]" in context


def test_build_chat_resume_context_message_keeps_recent_tool_activity_bounded(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "resume-large.jsonl"
    events = [
        {
            "type": "session_start",
            "payload": {
                "mode": "auto",
                "model": "test-model",
                "workspace_root": "/workspace/project",
            },
        }
    ]
    for step in range(cli_mod._CHAT_RESUME_CONTEXT_MAX_TOOL_EVENTS + 5):
        events.append(
            {
                "type": "tool_call",
                "payload": {
                    "name": "fs_read",
                    "arguments": {"path": f"src/file_{step}.py"},
                    "step": step,
                },
            }
        )
    log_path.write_text("\n".join(json.dumps(ev) for ev in events) + "\n", encoding="utf-8")

    context = cli_mod._build_chat_resume_context_message(log_path)

    assert context is not None
    assert len(context) <= cli_mod._CHAT_RESUME_CONTEXT_MAX_CHARS
    assert "- ... 5 older tool event(s) omitted" in context
    assert "step=0 tool_call fs_read" not in context
    latest_step = cli_mod._CHAT_RESUME_CONTEXT_MAX_TOOL_EVENTS + 4
    assert f"step={latest_step} tool_call fs_read" in context


def test_build_chat_resume_context_message_bounds_path_candidate_scanning(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "resume-huge-paths.jsonl"
    huge_paths = [
        f"generated/path_{idx}.py"
        for idx in range(cli_mod._CHAT_RESUME_CONTEXT_MAX_PATH_CANDIDATE_SCAN + 100)
    ]
    events = [
        {
            "type": "session_start",
            "payload": {"mode": "auto", "model": "test-model", "workspace_root": "/workspace"},
        },
        {
            "type": "tool_call",
            "payload": {
                "name": "custom_tool",
                "arguments": {"paths": huge_paths},
                "step": 1,
            },
        },
    ]
    log_path.write_text("\n".join(json.dumps(ev) for ev in events) + "\n", encoding="utf-8")

    context = cli_mod._build_chat_resume_context_message(log_path)

    assert context is not None
    assert len(context) <= cli_mod._CHAT_RESUME_CONTEXT_MAX_CHARS
    assert "generated/path_0.py" in context
    assert (
        f"generated/path_{cli_mod._CHAT_RESUME_CONTEXT_MAX_PATH_CANDIDATE_SCAN + 99}.py"
        not in context
    )


def test_resolve_chat_resume_direct_session_id_rejects_path_traversal(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (tmp_path / "outside.jsonl").write_text(
        json.dumps({"type": "user_message", "payload": {"content": "outside"}}) + "\n",
        encoding="utf-8",
    )

    assert (
        cli_mod._resolve_chat_resume_direct_session_id(
            raw_value="../outside",
            sessions_dir=sessions_dir,
        )
        is None
    )
    assert (
        cli_mod._resolve_chat_resume_direct_session_id(
            raw_value="..\\outside",
            sessions_dir=sessions_dir,
        )
        is None
    )


class _FakeStore:
    def __init__(self, *, sessions_dir: Path, session_id: str, enabled: bool = True) -> None:
        self.sessions_dir = sessions_dir
        self.session_id = session_id
        self.enabled = enabled
        self.notes: list[tuple[str, dict[str, object]]] = []

    def append(self, event_type: str, payload: dict[str, object]) -> None:
        self.notes.append((event_type, payload))


class _FakeSession:
    pass


def test_resume_chat_session_inserts_pinned_resume_context_before_history(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    target_id = "resume-with-context"
    target_path = sessions_dir / f"{target_id}.jsonl"
    target_events = [
        {
            "type": "session_start",
            "payload": {
                "mode": "review",
                "model": "resumed-model",
                "active_workdir_relpath": ".",
            },
        },
        {"type": "user_message", "payload": {"content": "Fix src/parser.py"}},
        {
            "type": "tool_call",
            "payload": {"name": "fs_read", "arguments": {"path": "src/parser.py"}, "step": 1},
        },
        {
            "type": "tool_result",
            "payload": {
                "name": "fs_read",
                "result": {"path": "src/parser.py", "content": "def parse(): pass"},
                "step": 1,
            },
        },
        {"type": "assistant_message", "payload": {"content": "I inspected the parser."}},
    ]
    target_path.write_text(
        "\n".join(json.dumps(ev) for ev in target_events) + "\n",
        encoding="utf-8",
    )

    current = _FakeSession()
    current.cfg = AppConfig(model="test-model", max_steps=10)
    current.root = tmp_path
    current.mode = "review"
    current.yes = True
    current.max_steps = 5
    current.console = None
    current.surface = object()
    current.store = _FakeStore(sessions_dir=sessions_dir, session_id="current-session")
    current.client = SimpleNamespace(api_key="override-key")
    current.usage_role = "main"
    current.tool_output_offloader = None
    current.conversation_compactor = None
    current.messages = []
    current.close = lambda: None

    def fake_create_session(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["active_workdir_relpath_override"] == "."
        new_session = _FakeSession()
        new_session.cfg = kwargs["cfg"]
        new_session.root = kwargs["root"]
        new_session.mode = kwargs["mode"]
        new_session.yes = kwargs["yes"]
        new_session.max_steps = kwargs["max_steps"]
        new_session.console = kwargs["console"]
        new_session.surface = kwargs["surface"]
        new_session.store = _FakeStore(
            sessions_dir=sessions_dir,
            session_id=kwargs["session_id_override"],
            enabled=not kwargs["no_log"],
        )
        new_session.client = SimpleNamespace(api_key="override-key")
        new_session.usage_role = kwargs["usage_role"]
        new_session.tool_output_offloader = None
        new_session.conversation_compactor = None
        new_session.messages = [{"role": "system", "content": "startup"}]
        new_session.pinned_prefix_len = 1
        return new_session

    monkeypatch.setattr(cli_mod, "create_session", fake_create_session)

    ok, message, loaded_history = cli_mod._resume_chat_session(
        session=current,
        target_session_id=target_id,
    )

    assert ok is True
    assert "Resumed session:" in message
    assert loaded_history == [
        {"role": "user", "content": "Fix src/parser.py"},
        {"role": "assistant", "content": "I inspected the parser."},
    ]
    assert current.pinned_prefix_len == 2
    assert current.messages[0] == {"role": "system", "content": "startup"}
    assert str(current.messages[1]["content"]).startswith("<resume_context>")
    assert "tool_result fs_read" in str(current.messages[1]["content"])
    assert current.messages[2:] == loaded_history
    system_note_payloads = [
        payload
        for event_type, payload in current.store.notes
        if event_type == "system_note" and payload.get("message") == "chat_resume"
    ]
    assert system_note_payloads
    assert system_note_payloads[-1]["resume_context_loaded"] is True
    assert system_note_payloads[-1]["resume_context_chars"] > 0


def test_resume_chat_session_reports_active_workdir_restore_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    target_id = "resume-bad-workdir"
    target_path = sessions_dir / f"{target_id}.jsonl"
    target_events = [
        {
            "type": "session_start",
            "payload": {
                "mode": "review",
                "model": "resumed-model",
                "active_workdir_relpath": "missing-package",
            },
        },
        {"type": "user_message", "payload": {"content": "Continue the task"}},
    ]
    target_path.write_text(
        "\n".join(json.dumps(ev) for ev in target_events) + "\n",
        encoding="utf-8",
    )

    current = _FakeSession()
    current.cfg = AppConfig(model="test-model", max_steps=10)
    current.root = tmp_path
    current.mode = "review"
    current.yes = True
    current.max_steps = 5
    current.console = None
    current.surface = object()
    current.store = _FakeStore(sessions_dir=sessions_dir, session_id="current-session")
    current.client = SimpleNamespace(api_key="override-key")
    current.usage_role = "main"
    current.tool_output_offloader = None
    current.conversation_compactor = None
    current.messages = []
    current.close = lambda: None
    create_called = {"value": False}

    def fake_create_session(**kwargs):  # type: ignore[no-untyped-def]
        create_called["value"] = True
        raise AssertionError(
            "create_session should not run after active workdir prevalidation fails"
        )

    monkeypatch.setattr(cli_mod, "create_session", fake_create_session)

    ok, message, loaded_history = cli_mod._resume_chat_session(
        session=current,
        target_session_id=target_id,
    )

    assert ok is False
    assert (
        "could not restore active workdir 'missing-package': Directory does not exist:" in message
    )
    assert loaded_history == []
    assert create_called["value"] is False
    assert current.store.session_id == "current-session"
    assert current.messages == []


def test_resume_chat_session_rejects_unsafe_session_id_before_reading(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    outside_path = tmp_path / "outside.jsonl"
    outside_path.write_text(
        json.dumps({"type": "user_message", "payload": {"content": "outside secret"}}) + "\n",
        encoding="utf-8",
    )
    current = _FakeSession()
    current.cfg = AppConfig(model="test-model", max_steps=10)
    current.root = tmp_path
    current.store = _FakeStore(sessions_dir=sessions_dir, session_id="current-session")
    current.messages = []
    create_called = {"value": False}

    def fake_create_session(**kwargs):  # type: ignore[no-untyped-def]
        create_called["value"] = True
        raise AssertionError("unsafe resume target should not create a session")

    monkeypatch.setattr(cli_mod, "create_session", fake_create_session)

    ok, message, loaded_history = cli_mod._resume_chat_session(
        session=current,
        target_session_id="../outside",
    )

    assert ok is False
    assert "Invalid session id" in message
    assert loaded_history == []
    assert create_called["value"] is False
    assert current.messages == []


def test_resume_chat_session_replaces_current_session_state(tmp_path: Path, monkeypatch) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    target_id = "resume-target"
    target_path = sessions_dir / f"{target_id}.jsonl"
    target_events = [
        {
            "type": "session_start",
            "payload": {
                "mode": "readonly",
                "model": "resumed-model",
                "temperature": 0.2,
                "stream": True,
                "routing_mode": "code_only",
                "yes": False,
                "max_steps": 11,
                "step_budget_policy": "fixed",
                "task_max_steps": 41,
                "subagent_max_steps": 9,
                "enable_chat_turn_step_budget": False,
                "chat_turn_fixed_override": 7,
                "compaction_enabled": True,
                "tool_output_offload_enabled": True,
                "conversation_summarization_enabled": False,
            },
        },
        {"type": "user_message", "payload": {"content": "previous question"}},
        {"type": "assistant_message", "payload": {"content": "previous answer"}},
    ]
    target_path.write_text(
        "\n".join(json.dumps(ev) for ev in target_events) + "\n",
        encoding="utf-8",
    )

    old_closed = {"value": False}

    current = _FakeSession()
    current.cfg = AppConfig(
        model="test-model",
        max_steps=37,
        step_budget_policy="adaptive",
        task_max_steps=100,
        subagent_max_steps=16,
    )
    current.root = tmp_path
    current.mode = "auto"
    current.yes = True
    current.max_steps = 5
    current.console = None
    current.surface = object()
    current.store = _FakeStore(sessions_dir=sessions_dir, session_id="current-session")
    current.client = SimpleNamespace(api_key="override-key")
    current.usage_role = "main"
    current.tool_output_offloader = None
    current.conversation_compactor = None
    current.messages = [{"role": "system", "content": "seed-old"}]

    def _close_old() -> None:
        old_closed["value"] = True

    current.close = _close_old

    def fake_create_session(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["session_id_override"] == target_id
        assert kwargs["mode"] == "readonly"
        assert kwargs["runtime_kind"] == RuntimeKind.INTERACTIVE_CHAT
        assert kwargs["session_source"] == "resume"
        assert kwargs["session_source_metadata"] == {
            "from_session_id": "current-session",
            "resumed_session_id": target_id,
            "loaded_message_count": 2,
            "active_workdir_relpath": None,
        }
        assert kwargs["yes"] is False
        assert kwargs["max_steps"] == 11
        assert kwargs["enable_chat_turn_step_budget"] is False
        assert kwargs["chat_turn_fixed_override"] == 7
        cfg = kwargs["cfg"]
        assert cfg.model == "resumed-model"
        assert cfg.max_steps == 11
        assert cfg.step_budget_policy == "fixed"
        assert cfg.task_max_steps == 41
        assert cfg.subagent_max_steps == 9
        assert cfg.temperature == 0.2
        assert cfg.stream is True
        assert cfg.routing_mode == "code_only"
        assert kwargs["enable_compaction"] is True
        assert kwargs["enable_tool_output_offload"] is True
        assert kwargs["enable_conversation_summarization"] is False
        new_session = _FakeSession()
        new_session.cfg = kwargs["cfg"]
        new_session.root = kwargs["root"]
        new_session.mode = kwargs["mode"]
        new_session.yes = kwargs["yes"]
        new_session.max_steps = kwargs["max_steps"]
        new_session.console = kwargs["console"]
        new_session.surface = kwargs["surface"]
        new_session.store = _FakeStore(
            sessions_dir=sessions_dir,
            session_id=kwargs["session_id_override"],
            enabled=not kwargs["no_log"],
        )
        new_session.client = SimpleNamespace(api_key="override-key")
        new_session.usage_role = kwargs["usage_role"]
        new_session.tool_output_offloader = object()
        new_session.conversation_compactor = None
        new_session.messages = [{"role": "system", "content": "seed-new"}]
        return new_session

    monkeypatch.setattr(cli_mod, "create_session", fake_create_session)

    ok, message, loaded_history = cli_mod._resume_chat_session(
        session=current,
        target_session_id=target_id,
    )

    assert ok is True
    assert "Resumed session:" in message
    assert old_closed["value"] is True
    assert current.store.session_id == target_id
    assert loaded_history == [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ]
    assert current.messages[-2:] == [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ]
    assert any(
        event_type == "system_note" and payload.get("message") == "chat_resume"
        for event_type, payload in current.store.notes
    )


def test_resume_chat_session_uses_current_config_defaults_for_older_step_payloads(
    tmp_path: Path, monkeypatch
) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    target_id = "resume-legacy"
    target_path = sessions_dir / f"{target_id}.jsonl"
    target_events = [
        {
            "type": "session_start",
            "payload": {
                "mode": "readonly",
                "model": "resumed-model",
                "yes": False,
            },
        },
        {"type": "user_message", "payload": {"content": "legacy question"}},
    ]
    target_path.write_text(
        "\n".join(json.dumps(ev) for ev in target_events) + "\n",
        encoding="utf-8",
    )

    current = _FakeSession()
    current.cfg = AppConfig(
        model="test-model",
        max_steps=37,
        step_budget_policy="fixed",
        task_max_steps=91,
        subagent_max_steps=13,
    )
    current.root = tmp_path
    current.mode = "auto"
    current.yes = True
    current.max_steps = 5
    current.console = None
    current.surface = object()
    current.store = _FakeStore(sessions_dir=sessions_dir, session_id="current-session")
    current.client = SimpleNamespace(api_key="override-key")
    current.usage_role = "main"
    current.tool_output_offloader = None
    current.conversation_compactor = None
    current.messages = []
    current.close = lambda: None

    def fake_create_session(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["session_source"] == "resume"
        assert kwargs["session_source_metadata"] == {
            "from_session_id": "current-session",
            "resumed_session_id": target_id,
            "loaded_message_count": 1,
            "active_workdir_relpath": None,
        }
        assert kwargs["max_steps"] == 37
        assert kwargs["enable_chat_turn_step_budget"] is True
        assert kwargs["chat_turn_fixed_override"] is None
        cfg = kwargs["cfg"]
        assert cfg.model == "resumed-model"
        assert cfg.max_steps == 37
        assert cfg.step_budget_policy == "fixed"
        assert cfg.task_max_steps == 91
        assert cfg.subagent_max_steps == 13
        new_session = _FakeSession()
        new_session.cfg = kwargs["cfg"]
        new_session.root = kwargs["root"]
        new_session.mode = kwargs["mode"]
        new_session.yes = kwargs["yes"]
        new_session.max_steps = kwargs["max_steps"]
        new_session.console = kwargs["console"]
        new_session.surface = kwargs["surface"]
        new_session.store = _FakeStore(
            sessions_dir=sessions_dir,
            session_id=kwargs["session_id_override"],
            enabled=not kwargs["no_log"],
        )
        new_session.client = SimpleNamespace(api_key="override-key")
        new_session.usage_role = kwargs["usage_role"]
        new_session.tool_output_offloader = None
        new_session.conversation_compactor = None
        new_session.messages = []
        return new_session

    monkeypatch.setattr(cli_mod, "create_session", fake_create_session)

    ok, message, loaded_history = cli_mod._resume_chat_session(
        session=current,
        target_session_id=target_id,
    )

    assert ok is True
    assert "Resumed session:" in message
    assert loaded_history == [{"role": "user", "content": "legacy question"}]


def test_resume_chat_session_preserves_web_provenance_in_reopened_store(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    target_id = "resume-web-provenance"
    target_path = sessions_dir / f"{target_id}.jsonl"
    target_events = [
        {"type": "session_start", "payload": {"mode": "review", "yes": False}},
        {
            "type": "user_message",
            "payload": {"content": "Please inspect https://docs.example.com/spec"},
        },
        {
            "type": "tool_result",
            "payload": {
                "name": "web_search",
                "step": 1,
                "result": {
                    "query": "docs example",
                    "backend": "openai_responses",
                    "sources": [
                        {
                            "title": "Guide",
                            "url": "https://docs.example.com/guide",
                            "snippet": "Official guide",
                        }
                    ],
                },
            },
        },
        {"type": "assistant_message", "payload": {"content": "Loaded docs."}},
    ]
    target_path.write_text(
        "\n".join(json.dumps(ev) for ev in target_events) + "\n",
        encoding="utf-8",
    )

    current = _FakeSession()
    current.cfg = AppConfig(model="test-model", max_steps=10)
    current.root = tmp_path
    current.mode = "review"
    current.yes = True
    current.max_steps = 5
    current.console = None
    current.surface = object()
    current.store = _FakeStore(sessions_dir=sessions_dir, session_id="current-session")
    current.client = SimpleNamespace(api_key="override-key")
    current.usage_role = "main"
    current.tool_output_offloader = None
    current.conversation_compactor = None
    current.messages = []
    current.close = lambda: None

    def fake_create_session(**kwargs):  # type: ignore[no-untyped-def]
        new_session = _FakeSession()
        new_session.cfg = kwargs["cfg"]
        new_session.root = kwargs["root"]
        new_session.mode = kwargs["mode"]
        new_session.yes = kwargs["yes"]
        new_session.max_steps = kwargs["max_steps"]
        new_session.console = kwargs["console"]
        new_session.surface = kwargs["surface"]
        new_session.store = SessionStore(
            enabled=not kwargs["no_log"],
            artifact_persistence_enabled=(not kwargs["no_log"])
            or kwargs["session_log_dir_override"] is not None,
            sessions_dir=kwargs["session_log_dir_override"],
            session_id=kwargs["session_id_override"],
            cwd=str(kwargs["root"]),
            repo_root=str(kwargs["root"]),
        )
        new_session.client = SimpleNamespace(api_key="override-key")
        new_session.usage_role = kwargs["usage_role"]
        new_session.tool_output_offloader = None
        new_session.conversation_compactor = None
        new_session.messages = []
        return new_session

    monkeypatch.setattr(cli_mod, "create_session", fake_create_session)

    ok, message, loaded_history = cli_mod._resume_chat_session(
        session=current,
        target_session_id=target_id,
    )

    assert ok is True
    assert "Resumed session:" in message
    assert loaded_history == [
        {"role": "user", "content": "Please inspect https://docs.example.com/spec"},
        {"role": "assistant", "content": "Loaded docs."},
    ]
    assert current.store.classify_web_fetch_url("https://docs.example.com/spec") == "user_provided"
    assert (
        current.store.classify_web_fetch_url("https://docs.example.com/guide")
        == "returned_by_web_search"
    )


def test_resume_chat_session_merges_newer_web_artifact_ahead_of_log(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    target_id = "resume-web-artifact-ahead"
    target_path = sessions_dir / f"{target_id}.jsonl"
    target_events = [
        {"type": "session_start", "payload": {"mode": "review", "yes": False}},
        {
            "type": "user_message",
            "payload": {"content": "Please inspect https://docs.example.com/spec"},
        },
        {"type": "assistant_message", "payload": {"content": "Loaded docs."}},
    ]
    target_path.write_text(
        "\n".join(json.dumps(ev) for ev in target_events) + "\n",
        encoding="utf-8",
    )
    artifact_root = sessions_dir / target_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_payload = build_web_research_artifact_from_events(
        target_events
        + [
            {
                "type": "tool_result",
                "payload": {
                    "name": "web_search",
                    "step": 1,
                    "result": {
                        "query": "docs example",
                        "backend": "openai_responses",
                        "sources": [
                            {
                                "title": "Guide",
                                "url": "https://docs.example.com/guide",
                                "snippet": "Artifact-only guide",
                            }
                        ],
                    },
                },
            }
        ]
    )
    (artifact_root / "web_research_sources.json").write_text(
        json.dumps(artifact_payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    current = _FakeSession()
    current.cfg = AppConfig(model="test-model", max_steps=10)
    current.root = tmp_path
    current.mode = "review"
    current.yes = True
    current.max_steps = 5
    current.console = None
    current.surface = object()
    current.store = _FakeStore(sessions_dir=sessions_dir, session_id="current-session")
    current.client = SimpleNamespace(api_key="override-key")
    current.usage_role = "main"
    current.tool_output_offloader = None
    current.conversation_compactor = None
    current.messages = []
    current.close = lambda: None

    def fake_create_session(**kwargs):  # type: ignore[no-untyped-def]
        new_session = _FakeSession()
        new_session.cfg = kwargs["cfg"]
        new_session.root = kwargs["root"]
        new_session.mode = kwargs["mode"]
        new_session.yes = kwargs["yes"]
        new_session.max_steps = kwargs["max_steps"]
        new_session.console = kwargs["console"]
        new_session.surface = kwargs["surface"]
        new_session.store = SessionStore(
            enabled=not kwargs["no_log"],
            artifact_persistence_enabled=(not kwargs["no_log"])
            or kwargs["session_log_dir_override"] is not None,
            sessions_dir=kwargs["session_log_dir_override"],
            session_id=kwargs["session_id_override"],
            cwd=str(kwargs["root"]),
            repo_root=str(kwargs["root"]),
        )
        new_session.client = SimpleNamespace(api_key="override-key")
        new_session.usage_role = kwargs["usage_role"]
        new_session.tool_output_offloader = None
        new_session.conversation_compactor = None
        new_session.messages = []
        return new_session

    monkeypatch.setattr(cli_mod, "create_session", fake_create_session)

    ok, message, loaded_history = cli_mod._resume_chat_session(
        session=current,
        target_session_id=target_id,
    )

    assert ok is True
    assert "Resumed session:" in message
    assert loaded_history == [
        {"role": "user", "content": "Please inspect https://docs.example.com/spec"},
        {"role": "assistant", "content": "Loaded docs."},
    ]
    assert current.store.classify_web_fetch_url("https://docs.example.com/spec") == "user_provided"
    assert (
        current.store.classify_web_fetch_url("https://docs.example.com/guide")
        == "returned_by_web_search"
    )
