from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sylliptor_agent_cli.tools import fs as fs_mod
from sylliptor_agent_cli.tools.fs import FsError, fs_list, fs_read, fs_read_lines


def _init_git_repo(path: Path) -> None:
    subprocess.run(
        ["git", "init", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_fs_read_large_file_respects_max_bytes(tmp_path: Path) -> None:
    path = tmp_path / "big.txt"
    path.write_bytes(b"a" * (512 * 1024))

    result = fs_read(root=tmp_path, path="big.txt", max_bytes=4096)

    assert result["truncated"] is True
    assert len(result["content"].encode("utf-8")) <= 4096


def test_fs_read_default_is_bounded_and_reports_limit_metadata(tmp_path: Path) -> None:
    path = tmp_path / "big-default.txt"
    path.write_bytes(b"a" * 20_000)

    result = fs_read(root=tmp_path, path="big-default.txt")

    assert result["truncated"] is True
    assert result["max_bytes"] == 12_000
    assert result["bytes_read"] == 12_000
    assert len(result["content"].encode("utf-8")) <= 12_000


def test_fs_read_lines_returns_requested_range_with_line_numbers(tmp_path: Path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8", newline="\n")

    result = fs_read_lines(root=tmp_path, path="demo.txt", start_line=2, end_line=3)

    assert result == {
        "path": "demo.txt",
        "start_line": 2,
        "end_line": 3,
        "total_lines": None,
        "content": "2: beta\n3: gamma\n",
        "truncated": False,
    }


def test_fs_read_lines_handles_end_line_past_eof_and_reports_total_lines(tmp_path: Path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8", newline="\n")

    result = fs_read_lines(
        root=tmp_path,
        path="demo.txt",
        start_line=2,
        end_line=10,
        include_line_numbers=False,
    )

    assert result == {
        "path": "demo.txt",
        "start_line": 2,
        "end_line": 3,
        "total_lines": 3,
        "content": "beta\ngamma\n",
        "truncated": False,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start_line": 0}, "Invalid start_line"),
        ({"start_line": 3, "end_line": 2}, "Invalid line range"),
        ({"start_line": 1, "max_lines": 0}, "Invalid max_lines"),
    ],
)
def test_fs_read_lines_rejects_invalid_ranges(
    tmp_path: Path,
    kwargs: dict[str, int],
    message: str,
) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")

    with pytest.raises(FsError, match=message):
        fs_read_lines(root=tmp_path, path="demo.txt", **kwargs)


def test_fs_read_lines_rejects_start_line_beyond_eof(tmp_path: Path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")

    with pytest.raises(FsError, match="beyond end of file"):
        fs_read_lines(root=tmp_path, path="demo.txt", start_line=5)


def test_fs_read_lines_marks_truncated_when_max_lines_caps_output(tmp_path: Path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8", newline="\n")

    result = fs_read_lines(root=tmp_path, path="demo.txt", start_line=2, max_lines=2)

    assert result == {
        "path": "demo.txt",
        "start_line": 2,
        "end_line": 3,
        "total_lines": None,
        "content": "2: two\n3: three\n",
        "truncated": True,
    }


def test_fs_read_lines_reports_missing_file_and_directory_errors(tmp_path: Path) -> None:
    (tmp_path / "subdir").mkdir()

    with pytest.raises(FsError, match="Not found: missing.txt"):
        fs_read_lines(root=tmp_path, path="missing.txt", start_line=1)

    with pytest.raises(FsError, match="Is a directory: subdir"):
        fs_read_lines(root=tmp_path, path="subdir", start_line=1)


def test_fs_read_lines_rejects_root_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("nope\n", encoding="utf-8")

    with pytest.raises(FsError, match="Path escapes root"):
        fs_read_lines(root=tmp_path, path="../outside.txt", start_line=1)


def test_fs_list_ignored_candidates_do_not_consume_visible_result_budget(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("ignored-*.txt\n", encoding="utf-8")
    for idx in range(1, 4):
        (tmp_path / f"ignored-{idx}.txt").write_text("ignored\n", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("visible\n", encoding="utf-8")

    result = fs_list(
        root=tmp_path,
        globs=["ignored-1.txt", "ignored-2.txt", "ignored-3.txt", "visible.txt"],
        max_results=1,
    )

    assert [entry["path"] for entry in result["entries"]] == ["visible.txt"]
    assert result["truncated"] is False


def test_fs_list_keeps_tracked_file_that_matches_gitignore(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("*.txt\n", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", ".gitignore", "-f", "keep.txt"],
        check=True,
        capture_output=True,
        text=True,
    )

    result = fs_list(root=tmp_path, globs=["*.txt"], max_results=10)

    assert [entry["path"] for entry in result["entries"]] == ["keep.txt"]
    assert result["truncated"] is False


def test_fs_list_plain_dir_does_not_probe_git(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "visible.txt").write_text("visible\n", encoding="utf-8")

    def fail_git_probe(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
        raise KeyboardInterrupt("plain directory should not probe git")

    monkeypatch.setattr(fs_mod.subprocess, "run", fail_git_probe)

    result = fs_list(root=tmp_path, globs=["*"], max_results=10)

    assert [entry["path"] for entry in result["entries"]] == ["visible.txt"]
    assert result["truncated"] is False


def test_fs_list_truncated_is_false_when_all_visible_results_fit(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("ignored\n", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("visible\n", encoding="utf-8")

    result = fs_list(
        root=tmp_path,
        globs=["ignored.txt", "visible.txt"],
        max_results=1,
    )

    assert [entry["path"] for entry in result["entries"]] == ["visible.txt"]
    assert result["truncated"] is False


def test_fs_list_truncated_is_true_only_when_more_visible_results_remain(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("ignored-*.txt\n", encoding="utf-8")
    for idx in range(1, 4):
        (tmp_path / f"ignored-{idx}.txt").write_text("ignored\n", encoding="utf-8")
    (tmp_path / "visible-a.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "visible-b.txt").write_text("b\n", encoding="utf-8")

    result = fs_list(
        root=tmp_path,
        globs=[
            "ignored-1.txt",
            "ignored-2.txt",
            "ignored-3.txt",
            "visible-a.txt",
            "visible-b.txt",
        ],
        max_results=1,
    )

    assert [entry["path"] for entry in result["entries"]] == ["visible-a.txt"]
    assert result["truncated"] is True


def test_fs_list_default_is_bounded_and_reports_counts(tmp_path: Path) -> None:
    for idx in range(170):
        (tmp_path / f"file-{idx:03d}.txt").write_text("x\n", encoding="utf-8")

    result = fs_list(root=tmp_path)

    assert len(result["entries"]) == 150
    assert result["returned_count"] == 150
    assert result["max_results"] == 150
    assert result["truncated"] is True
