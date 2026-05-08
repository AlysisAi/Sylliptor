from __future__ import annotations

from pathlib import Path

import pytest

from sylliptor_agent_cli.tools.fs import fs_list


def test_gitignore_includes_sylliptor_artifacts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gitignore = (repo_root / ".gitignore").read_text(encoding="utf-8")
    lines = {line.strip() for line in gitignore.splitlines()}
    assert ".sylliptor/" in lines
    assert ".sylliptor_images/" in lines
    assert "sylliptor-feedback/" in lines


def test_fs_list_ignores_sylliptor_dirs_by_default(tmp_path: Path) -> None:
    (tmp_path / ".sylliptor").mkdir()
    (tmp_path / ".sylliptor_images").mkdir()
    (tmp_path / "sylliptor-feedback").mkdir()
    (tmp_path / ".sylliptor" / "plan.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".sylliptor_images" / "img.png").write_bytes(b"\x89PNG")
    (tmp_path / "sylliptor-feedback" / "bundle.zip").write_bytes(b"zip")
    (tmp_path / "visible.txt").write_text("ok\n", encoding="utf-8")

    result = fs_list(root=tmp_path)
    paths = {entry["path"] for entry in result["entries"]}

    assert "visible.txt" in paths
    assert ".sylliptor/plan.json" not in paths
    assert ".sylliptor_images/img.png" not in paths
    assert "sylliptor-feedback/bundle.zip" not in paths
    assert result["truncated"] is False


@pytest.mark.parametrize("parent_name", ["build", "dist"])
def test_fs_list_parent_dir_named_like_ignored_dir_does_not_hide_repo_files(
    tmp_path: Path, parent_name: str
) -> None:
    repo = tmp_path / parent_name / "repo"
    repo.mkdir(parents=True)
    (repo / "a.txt").write_text("ok\n", encoding="utf-8")
    (repo / "build").mkdir()
    (repo / "build" / "x.txt").write_text("ignored\n", encoding="utf-8")

    result = fs_list(root=repo)
    paths = {entry["path"] for entry in result["entries"]}

    assert "a.txt" in paths
    assert "build/x.txt" not in paths
    assert result["truncated"] is False
