from __future__ import annotations

from pathlib import Path

import pytest

from sylliptor_agent_cli.tools.history import HistorySearchError, history_search


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_history_search_finds_matches_across_artifacts(tmp_path: Path) -> None:
    base = tmp_path / ".sylliptor" / "sessions" / "sid"
    _write(
        base / "history" / "chunk_0001.jsonl",
        '{"idx":1,"message":{"role":"user","content":"MUST keep API stable"}}\n',
    )
    _write(
        base / "tool_outputs" / "step1_fs_read_call.json",
        '{"tool_name":"fs_read","result":{"stdout":"MUST include tests"}}\n',
    )
    _write(
        base / "memory" / "pins.json",
        '{"pins":[{"kind":"constraint","text":"MUST keep API stable"}]}\n',
    )

    result = history_search(
        root=tmp_path,
        session_id="sid",
        pattern="MUST",
    )

    matches = result["matches"]
    assert len(matches) >= 3
    kinds = {row["kind"] for row in matches}
    assert "history" in kinds
    assert "tool_output" in kinds
    assert "memory" in kinds
    assert all(str(row["path"]).startswith(".sylliptor/sessions/sid/") for row in matches)


def test_history_search_respects_max_results_and_max_file_bytes(tmp_path: Path) -> None:
    base = tmp_path / ".sylliptor" / "sessions" / "sid"
    _write(
        base / "history" / "chunk_0001.jsonl",
        "\n".join(f'{{"idx":{i},"message":"match-{i}"}}' for i in range(10)) + "\n",
    )
    _write(
        base / "tool_outputs" / "large.json",
        "x" * 400 + "\nneedle-at-end\n",
    )

    limited = history_search(
        root=tmp_path,
        session_id="sid",
        pattern="match-",
        max_results=2,
    )
    assert limited["truncated"] is True
    assert len(limited["matches"]) == 2

    byte_limited = history_search(
        root=tmp_path,
        session_id="sid",
        pattern="needle-at-end",
        max_file_bytes=100,
        include_history=False,
        include_memory=False,
    )
    assert byte_limited["matches"] == []


def test_history_search_invalid_regex_raises(tmp_path: Path) -> None:
    with pytest.raises(HistorySearchError):
        history_search(root=tmp_path, session_id="sid", pattern="(")


def test_history_search_missing_session_dir_returns_empty(tmp_path: Path) -> None:
    result = history_search(root=tmp_path, session_id="missing", pattern="anything")
    assert result == {"pattern": "anything", "matches": [], "truncated": False}


def test_history_search_uses_session_artifact_root_for_tool_outputs(tmp_path: Path) -> None:
    legacy_base = tmp_path / ".sylliptor" / "sessions" / "sid"
    external_base = tmp_path.parent / f"{tmp_path.name}-external-sessions" / "sid"
    _write(
        legacy_base / "history" / "chunk_0001.jsonl",
        '{"idx":1,"message":{"role":"user","content":"legacy history"}}\n',
    )
    _write(
        external_base / "history" / "chunk_0002.jsonl",
        '{"idx":2,"message":{"role":"assistant","content":"external history"}}\n',
    )
    _write(
        external_base / "tool_outputs" / "step1_fs_read_call.json",
        '{"tool_name":"fs_read","result":{"stdout":"external tool output"}}\n',
    )
    _write(
        external_base / "memory" / "summary.json",
        '{"goal":"external summary"}\n',
    )

    result = history_search(
        root=tmp_path,
        session_id="sid",
        session_artifact_root=external_base,
        pattern="output|legacy|summary",
    )

    kinds = {row["kind"] for row in result["matches"]}
    assert "history" in kinds
    assert "tool_output" in kinds
    assert "memory" in kinds
    external_paths = [
        str(row["path"]) for row in result["matches"] if "external" in str(row["text"])
    ]
    assert any(path.startswith("session_artifacts/") for path in external_paths)
