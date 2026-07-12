from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from ..git_safe import build_git_process_env
from ..runtime_artifacts import RUNTIME_ARTIFACT_DIR_NAMES


class FsError(RuntimeError):
    pass


_DEFAULT_IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "dist",
    "build",
    ".idea",
    ".vscode",
} | set(RUNTIME_ARTIFACT_DIR_NAMES)
_DEFAULT_READ_LINES_MAX_LINES = 200
_FS_EDIT_OPERATIONS = {
    "replace_exact",
    "insert_before_exact",
    "insert_after_exact",
    "replace_lines",
    "insert_before_line",
    "insert_after_line",
    "append",
    "prepend",
}
_FS_EDIT_OPERATION_ALIASES = {
    "replace": "replace_exact",
}
_DEFAULT_FS_READ_MAX_BYTES = 12_000
_DEFAULT_FS_LIST_MAX_RESULTS = 150
_GIT_PROBE_TIMEOUT_S = 2.0


@dataclass(frozen=True)
class PreparedFsEdit:
    path: str
    path_obj: Path
    original_content: str
    updated_content: str
    applied_edits: int


def _resolve_under_root(root: Path, user_path: str) -> Path:
    root_abs = root.resolve()
    p = (root_abs / user_path).resolve()
    try:
        p.relative_to(root_abs)
    except ValueError as e:
        raise FsError(f"Path escapes root: {user_path}") from e
    return p


def fs_read(
    *, root: Path, path: str, max_bytes: int = _DEFAULT_FS_READ_MAX_BYTES
) -> dict[str, Any]:
    p = _resolve_under_root(root, path)
    if not p.exists():
        raise FsError(f"Not found: {path}")
    if p.is_dir():
        raise FsError(f"Is a directory: {path}")

    # Read only what we need (+1 byte lookahead for truncation detection).
    with p.open("rb") as fh:
        data = fh.read(max_bytes + 1)
    truncated = len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]
    text = data.decode("utf-8", errors="replace")
    return {
        "path": path,
        "content": text,
        "truncated": truncated,
        "bytes_read": len(data),
        "max_bytes": max_bytes,
    }


def fs_read_lines(
    *,
    root: Path,
    path: str,
    start_line: int,
    end_line: int | None = None,
    max_lines: int = _DEFAULT_READ_LINES_MAX_LINES,
    include_line_numbers: bool = True,
) -> dict[str, Any]:
    if start_line < 1:
        raise FsError(f"Invalid start_line: {start_line} (must be >= 1)")
    if end_line is not None and end_line < start_line:
        raise FsError(
            f"Invalid line range: end_line ({end_line}) must be >= start_line ({start_line})"
        )
    if max_lines < 1:
        raise FsError(f"Invalid max_lines: {max_lines} (must be >= 1)")

    p = _resolve_under_root(root, path)
    if not p.exists():
        raise FsError(f"Not found: {path}")
    if p.is_dir():
        raise FsError(f"Is a directory: {path}")

    requested_end_line = end_line
    effective_end_line = start_line + max_lines - 1
    if requested_end_line is not None:
        effective_end_line = min(effective_end_line, requested_end_line)

    content_lines: list[str] = []
    actual_end_line = start_line - 1
    total_lines: int | None = None
    lines_seen = 0
    truncated = False

    # Stream forward to the requested window and only report total_lines when
    # we naturally reach EOF, so focused range reads stay cheap on large files.
    with p.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            lines_seen = lineno
            if lineno < start_line:
                continue
            if lineno > effective_end_line:
                truncated = requested_end_line is None or lineno <= requested_end_line
                break

            actual_end_line = lineno
            if include_line_numbers:
                content_lines.append(f"{lineno}: {raw_line}")
            else:
                content_lines.append(raw_line)
        else:
            total_lines = lines_seen

    if lines_seen < start_line:
        raise FsError(f"Start line {start_line} is beyond end of file ({lines_seen} lines): {path}")

    return {
        "path": path,
        "start_line": start_line,
        "end_line": actual_end_line,
        "total_lines": total_lines,
        "content": "".join(content_lines),
        "truncated": truncated,
    }


def fs_write(*, root: Path, path: str, content: str) -> dict[str, Any]:
    p = _resolve_under_root(root, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {"path": path, "bytes": len(content.encode("utf-8"))}


def fs_mkdir(
    *,
    root: Path,
    path: str,
    parents: bool = True,
    exist_ok: bool = True,
) -> dict[str, Any]:
    p = _resolve_under_root(root, path)
    if p.exists():
        if not p.is_dir():
            raise FsError(f"Target exists as a file: {path}")
        if not exist_ok:
            raise FsError(f"Directory already exists and exist_ok is false: {path}")
        return {
            "path": path,
            "created": False,
            "already_exists": True,
            "parents": bool(parents),
            "exist_ok": bool(exist_ok),
        }

    try:
        p.mkdir(parents=bool(parents), exist_ok=bool(exist_ok))
    except FileExistsError as e:
        raise FsError(f"Directory already exists and exist_ok is false: {path}") from e
    except OSError as e:
        raise FsError(str(e)) from e

    return {
        "path": path,
        "created": True,
        "already_exists": False,
        "parents": bool(parents),
        "exist_ok": bool(exist_ok),
    }


def _read_text_preserving_newlines(path_obj: Path) -> str:
    with path_obj.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return fh.read()


def _write_text_preserving_newlines(path_obj: Path, content: str) -> None:
    with path_obj.open("w", encoding="utf-8", newline="") as fh:
        fh.write(content)


def _require_edit_string(edit: dict[str, Any], key: str, *, index: int, op: str) -> str:
    value = edit.get(key)
    if not isinstance(value, str):
        raise FsError(f"Edit {index} ({op}) requires string field: {key}")
    return value


def _require_edit_line_number(edit: dict[str, Any], key: str, *, index: int, op: str) -> int:
    value = edit.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FsError(f"Edit {index} ({op}) requires integer field: {key}")
    if value < 1:
        raise FsError(f"Edit {index} ({op}) {key} must be >= 1")
    return value


def _optional_expected_match_count(edit: dict[str, Any], *, index: int, op: str) -> int | None:
    value = edit.get("expected_match_count")
    if value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise FsError(f"Edit {index} ({op}) expected_match_count must be a non-negative integer")
    return value


def _count_matches(content: str, target: str, *, index: int, op: str) -> int:
    if not target:
        raise FsError(f"Edit {index} ({op}) requires a non-empty target")
    return content.count(target)


def _validate_match_count(
    *,
    count: int,
    expected_count: int | None,
    index: int,
    op: str,
) -> None:
    if expected_count is None:
        if count == 1:
            return
        if count == 0:
            raise FsError(f"Edit {index} ({op}) target matched 0 times; expected exactly 1")
        raise FsError(
            f"Edit {index} ({op}) target matched {count} times; expected exactly 1. "
            "Set expected_match_count to allow this."
        )
    if count != expected_count:
        raise FsError(
            f"Edit {index} ({op}) target matched {count} times; expected {expected_count}"
        )


def _content_lines(content: str) -> list[str]:
    return content.splitlines(keepends=True)


def _validate_line_range(
    *,
    lines: list[str],
    start_line: int,
    end_line: int,
    index: int,
    op: str,
) -> None:
    total_lines = len(lines)
    if end_line < start_line:
        raise FsError(
            f"Edit {index} ({op}) end_line ({end_line}) must be >= start_line ({start_line})"
        )
    if start_line > total_lines:
        raise FsError(
            f"Edit {index} ({op}) start_line {start_line} is beyond end of file "
            f"({total_lines} lines)"
        )
    if end_line > total_lines:
        raise FsError(
            f"Edit {index} ({op}) end_line {end_line} is beyond end of file ({total_lines} lines)"
        )


def _line_selection(lines: list[str], *, start_line: int, end_line: int) -> str:
    return "".join(lines[start_line - 1 : end_line])


def _canonical_line_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _validate_expected_old(
    *,
    selected: str,
    edit: dict[str, Any],
    index: int,
    op: str,
) -> None:
    expected_old = edit.get("expected_old")
    if expected_old is None:
        return
    if not isinstance(expected_old, str):
        raise FsError(f"Edit {index} ({op}) expected_old must be a string when provided")
    if selected != expected_old and _canonical_line_text(selected) != _canonical_line_text(
        expected_old
    ):
        selected_preview = selected[:500].replace("\n", "\\n")
        expected_preview = expected_old[:500].replace("\n", "\\n")
        raise FsError(
            f"Edit {index} ({op}) selected line text did not match expected_old. "
            f"selected={selected_preview!r} expected={expected_preview!r}"
        )


def _apply_line_edit(content: str, edit: dict[str, Any], *, index: int, op: str) -> str:
    lines = _content_lines(content)
    total_lines = len(lines)

    if op == "replace_lines":
        start_line = _require_edit_line_number(edit, "start_line", index=index, op=op)
        end_line = _require_edit_line_number(edit, "end_line", index=index, op=op)
        _validate_line_range(
            lines=lines,
            start_line=start_line,
            end_line=end_line,
            index=index,
            op=op,
        )
        selected = _line_selection(lines, start_line=start_line, end_line=end_line)
        _validate_expected_old(selected=selected, edit=edit, index=index, op=op)
        replacement = _require_edit_string(edit, "replacement", index=index, op=op)
        return "".join(lines[: start_line - 1]) + replacement + "".join(lines[end_line:])

    line = _require_edit_line_number(edit, "line", index=index, op=op)
    if line > total_lines:
        raise FsError(
            f"Edit {index} ({op}) line {line} is beyond end of file ({total_lines} lines)"
        )
    insert_content = _require_edit_string(edit, "content", index=index, op=op)
    if op == "insert_before_line":
        return "".join(lines[: line - 1]) + insert_content + "".join(lines[line - 1 :])
    return "".join(lines[:line]) + insert_content + "".join(lines[line:])


def _apply_single_fs_edit(content: str, edit: dict[str, Any], *, index: int) -> str:
    raw_op = edit.get("op")
    if not isinstance(raw_op, str):
        raise FsError(f"Edit {index} is missing required string field: op")
    op = _FS_EDIT_OPERATION_ALIASES.get(raw_op.strip(), raw_op.strip())
    if op not in _FS_EDIT_OPERATIONS:
        allowed = ", ".join(sorted(_FS_EDIT_OPERATIONS))
        raise FsError(f"Edit {index} has unsupported op: {op!r}. Expected one of: {allowed}")

    if op == "append":
        return content + _require_edit_string(edit, "content", index=index, op=op)
    if op == "prepend":
        return _require_edit_string(edit, "content", index=index, op=op) + content
    if op in {"replace_lines", "insert_before_line", "insert_after_line"}:
        return _apply_line_edit(content, edit, index=index, op=op)

    target = _require_edit_string(edit, "target", index=index, op=op)
    replacement: str | None = None
    insert_content: str | None = None
    if op == "replace_exact":
        replacement = _require_edit_string(edit, "replacement", index=index, op=op)
    else:
        insert_content = _require_edit_string(edit, "content", index=index, op=op)
    expected_count = _optional_expected_match_count(edit, index=index, op=op)
    count = _count_matches(content, target, index=index, op=op)
    _validate_match_count(count=count, expected_count=expected_count, index=index, op=op)
    if count == 0:
        return content

    if op == "replace_exact":
        assert replacement is not None
        return content.replace(target, replacement)

    if op == "insert_before_exact":
        assert insert_content is not None
        return content.replace(target, insert_content + target)
    assert insert_content is not None
    return content.replace(target, target + insert_content)


def prepare_fs_edit(*, root: Path, path: str, edits: list[dict[str, Any]]) -> PreparedFsEdit:
    if not isinstance(edits, list) or not edits:
        raise FsError("edits must be a non-empty array of edit objects")

    path_obj = _resolve_under_root(root, path)
    if not path_obj.exists():
        raise FsError(f"Not found: {path}")
    if path_obj.is_dir():
        raise FsError(f"Is a directory: {path}")

    original_content = _read_text_preserving_newlines(path_obj)
    updated_content = original_content
    for index, raw_edit in enumerate(edits, start=1):
        if not isinstance(raw_edit, dict):
            raise FsError(f"Edit {index} must be an object")
        updated_content = _apply_single_fs_edit(updated_content, raw_edit, index=index)

    return PreparedFsEdit(
        path=path,
        path_obj=path_obj,
        original_content=original_content,
        updated_content=updated_content,
        applied_edits=len(edits),
    )


def write_prepared_fs_edit(prepared: PreparedFsEdit) -> dict[str, Any]:
    _write_text_preserving_newlines(prepared.path_obj, prepared.updated_content)
    return {
        "path": prepared.path,
        "applied_edits": prepared.applied_edits,
        "changed": prepared.updated_content != prepared.original_content,
        "bytes": len(prepared.updated_content.encode("utf-8")),
    }


def fs_edit(*, root: Path, path: str, edits: list[dict[str, Any]]) -> dict[str, Any]:
    prepared = prepare_fs_edit(root=root, path=path, edits=edits)
    return write_prepared_fs_edit(prepared)


def _require_existing_file(path_obj: Path, user_path: str) -> None:
    if not path_obj.exists():
        raise FsError(f"Not found: {user_path}")
    if path_obj.is_dir():
        raise FsError(f"Is a directory: {user_path}")


def _prepare_destination_file(
    destination_obj: Path, destination_path: str, *, overwrite: bool
) -> bool:
    overwritten = False
    if destination_obj.exists():
        if destination_obj.is_dir():
            raise FsError(f"Destination is a directory: {destination_path}")
        if not overwrite:
            raise FsError(f"Destination exists and overwrite is false: {destination_path}")
        overwritten = True
    destination_obj.parent.mkdir(parents=True, exist_ok=True)
    return overwritten


def fs_move(
    *,
    root: Path,
    source_path: str,
    destination_path: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_obj = _resolve_under_root(root, source_path)
    destination_obj = _resolve_under_root(root, destination_path)
    if source_obj == destination_obj:
        raise FsError(f"Source and destination are the same: {source_path}")
    _require_existing_file(source_obj, source_path)
    overwritten = _prepare_destination_file(
        destination_obj,
        destination_path,
        overwrite=overwrite,
    )
    size = source_obj.stat().st_size
    if overwritten:
        destination_obj.unlink()
    shutil.move(os.fspath(source_obj), os.fspath(destination_obj))
    return {
        "source_path": source_path,
        "destination_path": destination_path,
        "moved": True,
        "overwritten": overwritten,
        "bytes": size,
    }


def fs_copy(
    *,
    root: Path,
    source_path: str,
    destination_path: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_obj = _resolve_under_root(root, source_path)
    destination_obj = _resolve_under_root(root, destination_path)
    if source_obj == destination_obj:
        raise FsError(f"Source and destination are the same: {source_path}")
    _require_existing_file(source_obj, source_path)
    overwritten = _prepare_destination_file(
        destination_obj,
        destination_path,
        overwrite=overwrite,
    )
    shutil.copy2(source_obj, destination_obj)
    return {
        "source_path": source_path,
        "destination_path": destination_path,
        "copied": True,
        "overwritten": overwritten,
        "bytes": source_obj.stat().st_size,
    }


def fs_delete(*, root: Path, path: str) -> dict[str, Any]:
    path_obj = _resolve_under_root(root, path)
    _require_existing_file(path_obj, path)
    size = path_obj.stat().st_size
    path_obj.unlink()
    return {
        "path": path,
        "deleted": True,
        "bytes": size,
    }


def _find_git_marker_root(path: Path, *, boundary: Path) -> Path | None:
    boundary_abs = boundary.resolve()
    for candidate in (path, *path.parents):
        candidate_abs = candidate.resolve()
        try:
            candidate_abs.relative_to(boundary_abs)
        except ValueError:
            return None
        if (candidate_abs / ".git").exists():
            return candidate_abs
    return None


def _git_repo_root(root: Path, *, boundary: Path) -> Path | None:
    marker_root = _find_git_marker_root(root, boundary=boundary)
    if marker_root is None:
        return None
    try:
        cp = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            env=build_git_process_env(),
            timeout=_GIT_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if cp.returncode != 0:
        return None
    raw_root = cp.stdout.strip()
    if not raw_root:
        return marker_root
    try:
        repo_root = Path(raw_root).resolve()
        repo_root.relative_to(boundary.resolve())
    except (OSError, ValueError):
        return marker_root
    return repo_root


def _git_check_ignored(repo_root: Path, rel_paths: list[str]) -> set[str]:
    if not rel_paths:
        return set()
    try:
        cp = subprocess.run(
            ["git", "-C", str(repo_root), "check-ignore", "--stdin"],
            input="\n".join(rel_paths) + "\n",
            check=False,
            capture_output=True,
            text=True,
            env=build_git_process_env(),
            timeout=_GIT_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if cp.returncode not in (0, 1):  # 1 means "no matches"
        return _fallback_gitignore_ignored_untracked(repo_root, rel_paths)
    ignored = {Path(line.strip()).as_posix() for line in cp.stdout.splitlines() if line.strip()}
    if not ignored:
        ignored = _fallback_gitignore_ignored_untracked(repo_root, rel_paths)
    return ignored


def _git_tracked_paths(repo_root: Path, rel_paths: list[str]) -> set[str]:
    if not rel_paths:
        return set()
    try:
        cp = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-z", "--", *rel_paths],
            check=False,
            capture_output=True,
            text=True,
            env=build_git_process_env(),
            timeout=_GIT_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if cp.returncode != 0:
        return set()
    return {Path(item).as_posix() for item in cp.stdout.split("\0") if item}


def _fallback_gitignore_ignored_untracked(repo_root: Path, rel_paths: list[str]) -> set[str]:
    ignored = _fallback_gitignore_ignored(repo_root, rel_paths)
    if ignored:
        ignored.difference_update(_git_tracked_paths(repo_root, rel_paths))
    return ignored


def _fallback_gitignore_ignored(repo_root: Path, rel_paths: list[str]) -> set[str]:
    gitignore = repo_root / ".gitignore"
    try:
        raw_patterns = gitignore.read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()

    ignored: set[str] = set()
    normalized_paths = [Path(rel).as_posix() for rel in rel_paths]
    for raw in raw_patterns:
        pattern = raw.strip()
        if not pattern or pattern.startswith("#"):
            continue
        negated = pattern.startswith("!")
        if negated:
            pattern = pattern[1:].strip()
        if not pattern or pattern.endswith("/"):
            continue
        pattern = pattern.lstrip("/").replace("\\", "/")
        for rel in normalized_paths:
            name = rel.rsplit("/", 1)[-1]
            matched = fnmatch(rel, pattern) or ("/" not in pattern and fnmatch(name, pattern))
            if not matched:
                continue
            if negated:
                ignored.discard(rel)
            else:
                ignored.add(rel)
    return ignored


def fs_list(
    *,
    root: Path,
    root_path: str = ".",
    globs: list[str] | None = None,
    ignore: list[str] | None = None,
    max_results: int = _DEFAULT_FS_LIST_MAX_RESULTS,
) -> dict[str, Any]:
    base = _resolve_under_root(root, root_path)
    patterns = globs or ["**/*"]
    ignore_set = set(ignore or [])

    repo_root = _git_repo_root(base, boundary=root)
    entries: list[dict[str, Any]] = []
    truncated = False
    batch_size = max(256, max_results or 0)
    pending: list[tuple[Path, str, str | None]] = []

    def _flush_pending() -> bool:
        nonlocal truncated
        ignored_by_git: set[str] = set()
        if repo_root:
            rels = [rel_git for _, _, rel_git in pending if rel_git]
            ignored_by_git = _git_check_ignored(repo_root, rels)

        # Count visibility after gitignore filtering so returned entries fill the
        # visible result window and `truncated` only reflects hidden extra visible files.
        for path_obj, rel, rel_git in pending:
            if rel_git and rel_git in ignored_by_git:
                continue
            if len(entries) >= max_results:
                truncated = True
                return True
            try:
                size = path_obj.stat().st_size
            except OSError:
                size = None
            entries.append({"path": rel, "size": size})
        return False

    for pat in patterns:
        for p in base.glob(pat):
            rel_path = p.relative_to(base)
            rel_parts = rel_path.parts
            if set(rel_parts) & _DEFAULT_IGNORE_DIRS:
                continue
            if any(seg in ignore_set for seg in rel_parts):
                continue
            if p.is_dir():
                continue

            rel = rel_path.as_posix()
            rel_git: str | None = None
            if repo_root:
                try:
                    rel_git_path = os.path.relpath(p, repo_root)
                except ValueError:
                    rel_git = None
                else:
                    if rel_git_path in {os.curdir, os.pardir} or rel_git_path.startswith(
                        os.pardir + os.sep
                    ):
                        rel_git = None
                    else:
                        rel_git = Path(rel_git_path).as_posix()

            pending.append((p, rel, rel_git))
            if len(pending) >= batch_size:
                if _flush_pending():
                    pending.clear()
                    break
                pending.clear()
        if truncated:
            break

    if not truncated and pending:
        _flush_pending()

    return {
        "root": os.fspath(base),
        "entries": entries,
        "truncated": truncated,
        "returned_count": len(entries),
        "max_results": max_results,
    }
