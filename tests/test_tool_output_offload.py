from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import sylliptor_agent_cli.compaction.tool_output_offload as offload_module
from sylliptor_agent_cli.agent_loop import create_session
from sylliptor_agent_cli.compaction.settings import resolve_compaction_settings
from sylliptor_agent_cli.compaction.tool_output_offload import ToolOutputOffloader
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.session_artifacts import SessionArtifactLayout
from sylliptor_agent_cli.session_store import read_session_events


def test_compaction_settings_default_to_lighter_tool_output_retention() -> None:
    settings = resolve_compaction_settings(AppConfig(model="gpt-5-nano"))

    assert settings.tool_output_offload_threshold_chars == 6000
    assert settings.tool_output_preview_chars == 2000


def test_offloader_does_not_offload_small_output(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    session_artifact_root = tmp_path / "session-store" / "session-1"
    workspace_root.mkdir()
    offloader = ToolOutputOffloader(
        artifact_layout=SessionArtifactLayout(filesystem_root=session_artifact_root),
        workspace_root=workspace_root,
        threshold_chars=200,
        preview_chars=50,
    )
    content_json = json.dumps({"ok": "small"}, ensure_ascii=True)
    result = offloader.maybe_offload(
        tool_name="shell_run",
        tool_call_id="call1",
        step=1,
        result={"ok": "small"},
        content_json=content_json,
    )

    assert result.offloaded is False
    assert result.transcript_shaped is False
    assert result.artifact_locator is None
    assert result.artifact_fs_path is None
    assert result.artifact_readable_via_fs is False
    assert result.message_chars == len(content_json)
    assert result.content_for_message == content_json
    assert not session_artifact_root.exists()


def test_offloader_shows_old_dead_band_output_in_full(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    session_artifact_root = tmp_path / "session-store" / "session-shaped"
    workspace_root.mkdir()
    offloader = ToolOutputOffloader(
        artifact_layout=SessionArtifactLayout(filesystem_root=session_artifact_root),
        workspace_root=workspace_root,
        threshold_chars=6000,
        preview_chars=2000,
    )
    payload = {"path": "README.md", "content": "A" * 1800, "truncated": False}
    content_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

    result = offloader.maybe_offload(
        tool_name="fs_read",
        tool_call_id="call-shaped",
        step=2,
        result=payload,
        content_json=content_json,
    )

    assert result.offloaded is False
    assert result.transcript_shaped is False
    assert result.artifact_locator is None
    assert result.artifact_fs_path is None
    assert result.artifact_readable_via_fs is False
    assert result.message_chars == result.original_chars
    assert result.content_for_message == content_json
    assert not session_artifact_root.exists()


def test_offloader_offloads_large_output_and_writes_artifact_in_session_root(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    session_artifact_root = tmp_path / "session-store" / "session_one"
    workspace_root.mkdir()
    offloader = ToolOutputOffloader(
        artifact_layout=SessionArtifactLayout(filesystem_root=session_artifact_root),
        workspace_root=workspace_root,
        threshold_chars=50,
        preview_chars=40,
    )
    large_text = "x" * 500
    payload = {"stdout": large_text}
    content_json = json.dumps(payload, ensure_ascii=True)

    result = offloader.maybe_offload(
        tool_name="shell/run",
        tool_call_id="call:abc",
        step=3,
        result=payload,
        content_json=content_json,
    )

    assert result.offloaded is True
    assert result.transcript_shaped is True
    assert result.artifact_locator == (
        ".sylliptor/sessions/session_one/tool_outputs/step3_shell_run_call_abc.json"
    )
    assert result.artifact_fs_path is not None
    assert not str(result.artifact_locator).startswith("/")
    assert result.artifact_readable_via_fs is True
    assert result.artifact_location == "workspace_root"
    artifact_path = Path(result.artifact_fs_path)
    assert artifact_path.exists()
    assert artifact_path.is_absolute()
    assert artifact_path.is_relative_to(workspace_root.resolve())

    saved = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert saved["tool_name"] == "shell/run"
    assert saved["tool_call_id"] == "call:abc"
    assert saved["step"] == 3
    assert "result" not in saved
    assert json.loads(saved["content_json"])["stdout"] == large_text
    assert artifact_path.read_text(encoding="utf-8").count(large_text) == 1

    stub = json.loads(result.content_for_message)
    assert stub["offloaded"] is True
    assert stub["artifact_locator"] == result.artifact_locator
    assert "artifact_path" not in stub
    assert stub["artifact_saved"] is True
    assert stub["artifact_readable_via_fs"] is True
    assert stub["artifact_location"] == "workspace_root"
    assert stub["fs_read_path"] == result.artifact_locator
    assert stub["full_output"] == (
        f"Truncated to 40 of {result.original_chars} chars. Full output saved to "
        f"workspace_root at {result.artifact_locator} - use fs_read on that path to view all of it."
    )
    assert stub["raw_saved_in_session_log"] is False
    assert "summary" in stub
    assert "preview" in stub
    assert len(stub["preview"]) <= 40 + len("...(truncated)")
    assert result.error is None


def test_offloader_write_failure_returns_json_stub(tmp_path: Path, monkeypatch) -> None:
    workspace_root = tmp_path / "workspace"
    session_artifact_root = tmp_path / "session-store" / "session-fail"
    workspace_root.mkdir()
    offloader = ToolOutputOffloader(
        artifact_layout=SessionArtifactLayout(filesystem_root=session_artifact_root),
        workspace_root=workspace_root,
        threshold_chars=50,
        preview_chars=40,
    )

    def _raise_write_error(*_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    monkeypatch.setattr(offload_module, "_atomic_private_write_text", _raise_write_error)
    payload = {"stdout": "x" * 500}
    content_json = json.dumps(payload, ensure_ascii=True)

    result = offloader.maybe_offload(
        tool_name="shell_run",
        tool_call_id="call-fail",
        step=2,
        result=payload,
        content_json=content_json,
    )

    assert result.offloaded is False
    assert result.transcript_shaped is True
    assert result.artifact_locator is None
    assert result.artifact_fs_path is None
    assert result.artifact_readable_via_fs is False
    assert result.error is not None
    stub = json.loads(result.content_for_message)
    assert stub["offloaded"] is False
    assert stub["transcript_shaped"] is True
    assert stub["tool"] == "shell_run"
    assert stub["tool_call_id"] == "call-fail"
    assert stub["step"] == 2
    assert "preview" in stub
    assert "summary" in stub
    assert "error" in stub
    assert "the rest is not readable via fs" in stub["full_output"]
    assert "Re-run the command narrowed" in stub["full_output"]


def test_offloader_atomic_publish_preserves_old_artifact_after_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    session_artifact_root = tmp_path / "session-store" / "atomic-session"
    workspace_root.mkdir()
    offloader = ToolOutputOffloader(
        artifact_layout=SessionArtifactLayout(filesystem_root=session_artifact_root),
        workspace_root=workspace_root,
        threshold_chars=50,
        preview_chars=40,
    )
    first_json = json.dumps({"stdout": "first-committed-" + "a" * 500})
    first = offloader.maybe_offload(
        tool_name="shell_run",
        tool_call_id="same-call",
        step=7,
        result={},
        content_json=first_json,
    )
    assert first.offloaded is True
    artifact_path = Path(str(first.artifact_fs_path))
    committed = artifact_path.read_bytes()

    def _crash_before_replace(_source: Path, _target: Path) -> None:
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(offload_module.os, "replace", _crash_before_replace)
    second = offloader.maybe_offload(
        tool_name="shell_run",
        tool_call_id="same-call",
        step=7,
        result={},
        content_json=json.dumps({"stdout": "second-uncommitted-" + "b" * 500}),
    )

    assert second.offloaded is False
    assert artifact_path.read_bytes() == committed
    assert not list(artifact_path.parent.glob(f".{artifact_path.name}.*.tmp"))


@pytest.mark.skipif(os.name == "nt", reason="Windows ACLs are not POSIX mode bits")
def test_offloader_artifact_is_private_on_posix(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    offloader = ToolOutputOffloader(
        artifact_layout=SessionArtifactLayout(
            filesystem_root=tmp_path / "sessions" / "private-session"
        ),
        workspace_root=workspace_root,
        threshold_chars=10,
        preview_chars=5,
    )

    result = offloader.maybe_offload(
        tool_name="shell_run",
        tool_call_id="private",
        step=1,
        result={},
        content_json=json.dumps({"stdout": "private" * 100}),
    )

    artifact_path = Path(str(result.artifact_fs_path))
    assert artifact_path.stat().st_mode & 0o777 == 0o600
    assert artifact_path.parent.stat().st_mode & 0o777 == 0o700


def test_offloader_rejects_workspace_symlink_escape_and_falls_back_to_session_store(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    outside = tmp_path / "workspace-controlled-target"
    session_artifact_root = tmp_path / "trusted-session-store" / "escape-session"
    workspace_root.mkdir()
    outside.mkdir()
    try:
        (workspace_root / ".sylliptor").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are unavailable on this host")
    offloader = ToolOutputOffloader(
        artifact_layout=SessionArtifactLayout(filesystem_root=session_artifact_root),
        workspace_root=workspace_root,
        threshold_chars=20,
        preview_chars=10,
    )

    result = offloader.maybe_offload(
        tool_name="shell_run",
        tool_call_id="escape-call",
        step=1,
        result={},
        content_json=json.dumps({"stdout": "contained" + "x" * 500}),
    )

    assert result.offloaded is True
    assert offloader.artifact_root == session_artifact_root.resolve()
    assert Path(str(result.artifact_fs_path)).is_relative_to(session_artifact_root.resolve())
    assert result.artifact_readable_via_fs is False
    assert list(outside.rglob("*")) == []


def test_offloader_non_fs_readable_stub_guides_to_narrow_rerun(tmp_path: Path) -> None:
    session_artifact_root = tmp_path / "session-store" / "session-external"
    offloader = ToolOutputOffloader(
        artifact_layout=SessionArtifactLayout(filesystem_root=session_artifact_root),
        workspace_root=None,
        threshold_chars=50,
        preview_chars=40,
    )
    payload = {"stdout": "x" * 500}
    content_json = json.dumps(payload, ensure_ascii=True)

    result = offloader.maybe_offload(
        tool_name="shell_run",
        tool_call_id="call-external",
        step=4,
        result=payload,
        content_json=content_json,
    )

    assert result.offloaded is True
    assert result.artifact_readable_via_fs is False
    assert result.artifact_location == "external_session_store"
    stub = json.loads(result.content_for_message)
    assert stub["artifact_readable_via_fs"] is False
    assert stub["artifact_location"] == "external_session_store"
    assert "the rest is not readable via fs" in stub["full_output"]
    assert "Re-run the command narrowed" in stub["full_output"]


def test_create_session_disable_compaction_does_not_create_offloader(tmp_path: Path) -> None:
    cfg = AppConfig(model="gpt-5-nano")
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="readonly",
        yes=False,
        max_steps=1,
        no_log=True,
        api_key_override="test-key",
        enable_compaction=False,
    )
    try:
        assert session.tool_output_offloader is None
        assert session.conversation_compactor is None
    finally:
        session.close()


def test_create_session_can_enable_offload_without_conversation_summarization(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "logs"
    cfg = AppConfig(model="gpt-5-nano")
    cfg.extra_fields = {
        "compaction": {
            "enabled": True,
            "offload_tool_outputs": True,
            "summarize_conversation": True,
        }
    }
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="readonly",
        yes=False,
        max_steps=1,
        no_log=False,
        api_key_override="test-key",
        session_log_dir_override=sessions_dir,
        session_id_override="split-offload",
        enable_compaction=False,
        enable_tool_output_offload=True,
        enable_conversation_summarization=False,
    )
    try:
        assert session.tool_output_offloader is not None
        assert session.conversation_compactor is None
        assert session.tool_output_offload_enabled is True
        assert session.conversation_summarization_enabled is False
    finally:
        session.close()

    events = list(read_session_events(sessions_dir / "split-offload.jsonl"))
    session_start = next(event for event in events if event.get("type") == "session_start")
    payload = dict(session_start.get("payload") or {})
    assert payload["requested_enable_compaction"] is False
    assert payload["requested_tool_output_offload"] is True
    assert payload["requested_conversation_summarization"] is False
    assert payload["logging_enabled"] is True
    assert payload["explicit_session_artifact_root"] is True
    assert payload["tool_output_offload_artifact_persistence_available"] is True
    assert payload["tool_output_offload_enabled"] is True
    assert payload["conversation_summarization_enabled"] is False


def test_create_session_no_log_without_explicit_artifact_root_disables_offload(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(model="gpt-5-nano")
    cfg.extra_fields = {
        "compaction": {
            "enabled": True,
            "offload_tool_outputs": True,
            "summarize_conversation": False,
        }
    }
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="readonly",
        yes=False,
        max_steps=1,
        no_log=True,
        api_key_override="test-key",
        enable_compaction=False,
        enable_tool_output_offload=True,
        enable_conversation_summarization=False,
    )
    try:
        assert session.tool_output_offloader is None
        assert session.tool_output_offload_enabled is False
    finally:
        session.close()


def test_create_session_no_log_with_explicit_artifact_root_keeps_offload_enabled(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "runtime" / "sessions"
    cfg = AppConfig(model="gpt-5-nano")
    cfg.extra_fields = {
        "compaction": {
            "enabled": True,
            "offload_tool_outputs": True,
            "summarize_conversation": False,
        }
    }
    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="readonly",
        yes=False,
        max_steps=1,
        no_log=True,
        api_key_override="test-key",
        session_log_dir_override=sessions_dir,
        session_id_override="offload-runtime",
        enable_compaction=False,
        enable_tool_output_offload=True,
        enable_conversation_summarization=False,
    )
    try:
        assert session.tool_output_offloader is not None
        assert session.tool_output_offload_enabled is True
    finally:
        session.close()
