from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sylliptor_agent_cli.tools.git import GitError, git_apply_patch


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_patch_repo(repo: Path) -> Path:
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    target = repo / "hello.txt"
    target.write_text("old\n", encoding="utf-8")
    _git(repo, "add", "hello.txt")
    _git(repo, "commit", "-m", "init")
    return target


def test_git_apply_patch_normalizes_crlf_and_applies(tmp_path: Path) -> None:
    target = _init_patch_repo(tmp_path)
    patch = (
        "diff --git a/hello.txt b/hello.txt\n"
        "--- a/hello.txt\n"
        "+++ b/hello.txt\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    ).replace("\n", "\r\n")

    result = git_apply_patch(root=tmp_path, patch=patch)

    assert result["applied"] is True
    assert target.read_text(encoding="utf-8") == "new\n"


def test_git_apply_patch_rejects_patch_without_headers(tmp_path: Path) -> None:
    _init_patch_repo(tmp_path)

    with pytest.raises(GitError, match="malformed patch: no file paths found"):
        git_apply_patch(
            root=tmp_path,
            patch="@@ -1 +1 @@\n-old\n+new\n",
        )


def test_git_apply_patch_rejects_placeholder_hunk_header(tmp_path: Path) -> None:
    _init_patch_repo(tmp_path)
    patch = (
        "diff --git a/hello.txt b/hello.txt\n"
        "--- a/hello.txt\n"
        "+++ b/hello.txt\n"
        "@@ ... @@\n"
        "-old\n"
        "+new\n"
    )

    with pytest.raises(GitError, match="placeholder hunk header"):
        git_apply_patch(root=tmp_path, patch=patch)


def test_git_apply_patch_preflight_blocks_invalid_context_without_mutation(
    tmp_path: Path,
) -> None:
    target = _init_patch_repo(tmp_path)
    patch = (
        "diff --git a/hello.txt b/hello.txt\n"
        "--- a/hello.txt\n"
        "+++ b/hello.txt\n"
        "@@ -1 +1 @@\n"
        "-not-old\n"
        "+new\n"
    )

    with pytest.raises(GitError, match="git apply preflight failed"):
        git_apply_patch(root=tmp_path, patch=patch)

    assert target.read_text(encoding="utf-8") == "old\n"
