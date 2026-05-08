from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .availability import is_tool_unavailable_result

ToolFormatter = Callable[[dict[str, Any]], str]
_PATH_BASE_ENUM = ["active_workdir", "workspace_root"]


def _path_base_property(*, description: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "string",
        "enum": list(_PATH_BASE_ENUM),
        "default": "active_workdir",
    }
    if description:
        payload["description"] = description
    return payload


def _cwd_base_property() -> dict[str, Any]:
    return _path_base_property(
        description=(
            "Resolve a relative cwd from the live active_workdir (default) or from the immutable "
            "workspace_root."
        )
    )


def _truncate_inline(text: str, *, max_chars: int = 96) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 3:
        return normalized[:max_chars]
    return normalized[: max_chars - 3] + "..."


def _preview_shell_run(args: dict[str, Any]) -> str:
    cmd = str(args.get("cmd") or "").strip()
    return _truncate_inline(cmd, max_chars=120) or "-"


def _preview_shell_output(args: dict[str, Any]) -> str:
    process_id = str(args.get("process_id") or "").strip()
    since = args.get("since")
    if since is None:
        return _truncate_inline(process_id, max_chars=120) or "-"
    return _truncate_inline(f"{process_id} since={since}", max_chars=120) or "-"


def _preview_shell_kill(args: dict[str, Any]) -> str:
    return _truncate_inline(str(args.get("process_id") or "").strip(), max_chars=120) or "-"


def _preview_fs_read_lines(args: dict[str, Any]) -> str:
    path = str(args.get("path") or "").strip()
    start_line = args.get("start_line")
    end_line = args.get("end_line")
    max_lines = args.get("max_lines")
    if end_line is None:
        line_part = f"{start_line}"
    else:
        line_part = f"{start_line}-{end_line}"
    preview = f"{path}:{line_part}"
    if max_lines is not None:
        preview += f" (max {max_lines})"
    return _truncate_inline(preview, max_chars=120) or "-"


def _preview_verify_run(args: dict[str, Any]) -> str:
    commands = args.get("commands")
    if not isinstance(commands, list) or not commands:
        return "configured commands"
    first = str(commands[0]).strip()
    if len(commands) == 1:
        return _truncate_inline(first, max_chars=120) or "-"
    return _truncate_inline(f"{first} (+{len(commands) - 1} more)", max_chars=120) or "-"


def _preview_git_history(args: dict[str, Any]) -> str:
    mode = str(args.get("mode") or "").strip()
    if mode == "log":
        ref = str(args.get("ref") or "HEAD").strip() or "HEAD"
        path = str(args.get("path") or "").strip()
        grep = str(args.get("grep") or "").strip()
        preview = f"log {ref}"
        if path:
            preview += f" -- {path}"
        if grep:
            preview += f" grep={grep}"
        return _truncate_inline(preview, max_chars=120) or "-"
    if mode == "show":
        commit = str(args.get("commit") or "").strip()
        path = str(args.get("path") or "").strip()
        preview = f"show {commit}"
        if path:
            preview += f" -- {path}"
        return _truncate_inline(preview, max_chars=120) or "-"
    if mode == "blame":
        path = str(args.get("path") or "").strip()
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        preview = f"blame {path}:{start_line}-{end_line}"
        return _truncate_inline(preview, max_chars=120) or "-"
    return _truncate_inline(mode, max_chars=120) or "-"


def _preview_symbol_search(args: dict[str, Any]) -> str:
    query = str(args.get("query") or "").strip()
    kind = str(args.get("kind") or "").strip()
    exact = bool(args.get("exact", False))
    preview = query
    if kind:
        preview += f" kind={kind}"
    if exact:
        preview += " exact"
    return _truncate_inline(preview, max_chars=120) or "-"


def _preview_source_destination(args: dict[str, Any]) -> str:
    source_path = str(args.get("source_path") or "").strip()
    destination_path = str(args.get("destination_path") or "").strip()
    preview = f"{source_path} -> {destination_path}"
    return _truncate_inline(preview, max_chars=120) or "-"


def _preview_single_path(args: dict[str, Any]) -> str:
    path = str(args.get("path") or "").strip()
    return _truncate_inline(path, max_chars=120) or "-"


def _preview_pattern(args: dict[str, Any]) -> str:
    pattern = str(args.get("pattern") or "").strip()
    return _truncate_inline(pattern, max_chars=120) or "-"


def _preview_web_fetch(args: dict[str, Any]) -> str:
    url = str(args.get("url") or "").strip()
    max_chars = args.get("max_chars")
    if max_chars is None:
        return _truncate_inline(url, max_chars=120) or "-"
    return _truncate_inline(f"{url} (max_chars={max_chars})", max_chars=120) or "-"


def _preview_web_search(args: dict[str, Any]) -> str:
    query = str(args.get("query") or "").strip()
    allowed_domains = args.get("allowed_domains")
    if isinstance(allowed_domains, list) and allowed_domains:
        preview = f"{query} domains={','.join(str(item).strip() for item in allowed_domains[:3])}"
        if len(allowed_domains) > 3:
            preview += f" (+{len(allowed_domains) - 3})"
        return _truncate_inline(preview, max_chars=120) or "-"
    return _truncate_inline(query, max_chars=120) or "-"


def _preview_skill_read(args: dict[str, Any]) -> str:
    name = str(args.get("name") or "").strip()
    path = str(args.get("path") or "").strip()
    preview = name
    if path:
        preview += f" :: {path}"
    return _truncate_inline(preview, max_chars=120) or "-"


def _summary_subagent_run(parsed: dict[str, Any]) -> str:
    subagent_name = str(parsed.get("subagent") or parsed.get("name") or "?")
    sandbox_obj = parsed.get("sandbox")
    sandbox = sandbox_obj if isinstance(sandbox_obj, dict) else {}
    mode = str(sandbox.get("mode") or "-")
    tools_obj = sandbox.get("tools")
    tool_count = len(tools_obj) if isinstance(tools_obj, list) else 0
    result_text = str(parsed.get("result") or parsed.get("final_text") or "")
    result_len = len(result_text)
    if "error" in parsed:
        msg = _truncate_inline(str(parsed.get("error") or ""), max_chars=130)
        return f'Subagent "{subagent_name}" failed: {msg}'
    details: list[str] = []
    if tool_count > 0:
        details.append(f"tools={tool_count}")
    if result_len > 0:
        details.append(f"result={result_len} chars")
    suffix = f" ({', '.join(details)})" if details else ""
    return f'Subagent "{subagent_name}" mode={mode}{suffix}.'


def _summary_fs_read(parsed: dict[str, Any]) -> str:
    path = str(parsed.get("path") or "?")
    content = str(parsed.get("content") or "")
    truncated = bool(parsed.get("truncated"))
    trunc_note = ", truncated" if truncated else ""
    return f'Loaded "{path}" ({len(content)} chars{trunc_note}).'


def _summary_fs_read_lines(parsed: dict[str, Any]) -> str:
    path = str(parsed.get("path") or "?")
    start_line = parsed.get("start_line")
    end_line = parsed.get("end_line")
    truncated = bool(parsed.get("truncated"))
    trunc_note = ", truncated" if truncated else ""
    if isinstance(start_line, int) and isinstance(end_line, int) and end_line >= start_line:
        count = end_line - start_line + 1
        if count == 1:
            return f'Loaded "{path}" line {start_line} (1 line{trunc_note}).'
        return f'Loaded "{path}" lines {start_line}-{end_line} ({count} lines{trunc_note}).'
    content = str(parsed.get("content") or "")
    return f'Loaded "{path}" ({len(content)} chars{trunc_note}).'


def _summary_fs_edit(parsed: dict[str, Any]) -> str:
    path = str(parsed.get("path") or "?")
    applied_edits = parsed.get("applied_edits")
    changed = bool(parsed.get("changed"))
    if changed:
        size = parsed.get("bytes")
        return f'Edited "{path}" ({applied_edits} edit(s), {size} bytes).'
    return f'Edited "{path}" ({applied_edits} edit(s), no content change).'


def _summary_fs_move(parsed: dict[str, Any]) -> str:
    source_path = str(parsed.get("source_path") or "?")
    destination_path = str(parsed.get("destination_path") or "?")
    size = parsed.get("bytes")
    overwritten = bool(parsed.get("overwritten"))
    overwrite_note = ", replaced existing destination" if overwritten else ""
    return f'Moved "{source_path}" -> "{destination_path}" ({size} bytes{overwrite_note}).'


def _summary_fs_copy(parsed: dict[str, Any]) -> str:
    source_path = str(parsed.get("source_path") or "?")
    destination_path = str(parsed.get("destination_path") or "?")
    size = parsed.get("bytes")
    overwritten = bool(parsed.get("overwritten"))
    overwrite_note = ", replaced existing destination" if overwritten else ""
    return f'Copied "{source_path}" -> "{destination_path}" ({size} bytes{overwrite_note}).'


def _summary_fs_delete(parsed: dict[str, Any]) -> str:
    path = str(parsed.get("path") or "?")
    size = parsed.get("bytes")
    return f'Deleted "{path}" ({size} bytes).'


def _summary_fs_write(parsed: dict[str, Any]) -> str:
    path = str(parsed.get("path") or "?")
    size = parsed.get("bytes")
    return f'Updated "{path}" ({size} bytes).'


def _summary_fs_mkdir(parsed: dict[str, Any]) -> str:
    path = str(parsed.get("path") or "?")
    if bool(parsed.get("already_exists")):
        return f'Directory "{path}" already exists.'
    return f'Created directory "{path}".'


def _summary_fs_list(parsed: dict[str, Any]) -> str:
    entries = parsed.get("entries")
    count = len(entries) if isinstance(entries, list) else 0
    truncated = bool(parsed.get("truncated"))
    trunc_note = ", truncated" if truncated else ""
    return f"Found {count} file(s){trunc_note}."


def _summary_symbol_search(parsed: dict[str, Any]) -> str:
    matches = parsed.get("matches")
    count = len(matches) if isinstance(matches, list) else 0
    query = _truncate_inline(str(parsed.get("query") or ""), max_chars=44)
    truncated = bool(parsed.get("truncated"))
    trunc_note = ", truncated" if truncated else ""
    return f'Found {count} symbol match(es) for "{query}"{trunc_note}.'


def _summary_search_rg(parsed: dict[str, Any]) -> str:
    matches = parsed.get("matches")
    count = len(matches) if isinstance(matches, list) else 0
    pattern = _truncate_inline(str(parsed.get("pattern") or ""), max_chars=44)
    return f'Found {count} matches for "{pattern}".'


def _summary_history_search(parsed: dict[str, Any]) -> str:
    matches = parsed.get("matches")
    count = len(matches) if isinstance(matches, list) else 0
    pattern = _truncate_inline(str(parsed.get("pattern") or ""), max_chars=44)
    truncated = bool(parsed.get("truncated"))
    trunc_note = ", truncated" if truncated else ""
    return f'Found {count} history match(es) for "{pattern}"{trunc_note}.'


def _summary_skill_read(parsed: dict[str, Any]) -> str:
    name = str(parsed.get("name") or parsed.get("bundle_name") or "?")
    path = str(parsed.get("path") or "SKILL.md")
    content = str(parsed.get("content") or "")
    return f'Loaded skill "{name}" file "{path}" ({len(content)} chars).'


def _summary_web_fetch(parsed: dict[str, Any]) -> str:
    final_url = _truncate_inline(
        str(parsed.get("final_url") or parsed.get("url") or ""), max_chars=84
    )
    status_code = parsed.get("status_code")
    content_type = _truncate_inline(str(parsed.get("content_type") or ""), max_chars=28)
    content = str(parsed.get("content") or "")
    truncated = bool(parsed.get("truncated"))
    trunc_note = ", truncated" if truncated else ""
    return (
        f"Fetched {final_url} status={status_code} type={content_type} "
        f"content={len(content)} chars{trunc_note}."
    )


def _summary_web_search(parsed: dict[str, Any]) -> str:
    answer = str(parsed.get("answer") or "")
    sources = parsed.get("sources")
    source_count = len(sources) if isinstance(sources, list) else 0
    truncated = bool(parsed.get("sources_truncated"))
    trunc_note = ", truncated" if truncated else ""
    backend = _truncate_inline(str(parsed.get("backend") or ""), max_chars=24)
    model = _truncate_inline(str(parsed.get("model") or ""), max_chars=32)
    details = []
    if backend:
        details.append(f"backend={backend}")
    if model:
        details.append(f"model={model}")
    details_note = f" {' '.join(details)}" if details else ""
    return (
        f"Web search returned {source_count} source(s){trunc_note}; "
        f"answer={len(answer)} chars.{details_note}"
    )


def _summary_verify_run(parsed: dict[str, Any]) -> str:
    summary = _truncate_inline(str(parsed.get("summary") or ""), max_chars=120)
    primary_failure = parsed.get("primary_failure")
    hint = ""
    if isinstance(primary_failure, dict):
        snippet = str(primary_failure.get("snippet") or "").strip()
        if snippet:
            hint = f"Hint: {_truncate_inline(snippet, max_chars=110)}"
    artifact_path = str(parsed.get("artifact_path") or "").strip()
    if artifact_path:
        prefix = " ".join(part for part in (summary, hint) if part)
        if prefix:
            return f"{prefix} Artifact: {artifact_path}"
        return f"Artifact: {artifact_path}"
    artifact_saved = bool(parsed.get("artifact_saved"))
    artifact_readable_via_fs = bool(parsed.get("artifact_readable_via_fs"))
    artifact_location = str(parsed.get("artifact_location") or "").strip()
    if artifact_saved and not artifact_readable_via_fs:
        if artifact_location == "external_session_store":
            artifact_note = "Artifact saved externally (not readable via fs)."
        else:
            artifact_note = "Artifact saved outside the workspace (not readable via fs)."
        prefix = " ".join(part for part in (summary, hint) if part)
        if prefix:
            return f"{prefix} {artifact_note}"
        return artifact_note
    if artifact_saved:
        prefix = " ".join(part for part in (summary, hint) if part)
        if prefix:
            return f"{prefix} Artifact saved."
        return "Artifact saved."
    return " ".join(part for part in (summary, hint) if part) or "Verification finished."


def _summary_shell_run(parsed: dict[str, Any]) -> str:
    exit_code = parsed.get("exit_code")
    stdout = str(parsed.get("stdout") or "")
    stderr = str(parsed.get("stderr") or "")
    summary = f"Command exited with code {exit_code}."
    summary += f" stdout={len(stdout)} chars, stderr={len(stderr)} chars."
    if stderr.strip():
        first_err = _truncate_inline(stderr.strip().splitlines()[0], max_chars=90)
        summary += f" stderr preview: {first_err}"
    return summary


def _summary_shell_background(parsed: dict[str, Any]) -> str:
    process_id = str(parsed.get("process_id") or "?")
    status = str(parsed.get("status") or "?")
    return f'Started background process "{process_id}" (status={status}).'


def _summary_shell_output(parsed: dict[str, Any]) -> str:
    process_id = str(parsed.get("process_id") or "?")
    status = str(parsed.get("status") or "?")
    lines = parsed.get("lines")
    line_count = len(lines) if isinstance(lines, list) else 0
    dropped = int(parsed.get("dropped_lines") or 0)
    drop_note = f", {dropped} dropped" if dropped > 0 else ""
    return f'Read {line_count} new line(s) from "{process_id}" (status={status}{drop_note}).'


def _summary_shell_kill(parsed: dict[str, Any]) -> str:
    process_id = str(parsed.get("process_id") or "?")
    status = str(parsed.get("status") or "?")
    exit_code = parsed.get("exit_code")
    code_note = f", exit_code={exit_code}" if exit_code is not None else ""
    return f'Terminated background process "{process_id}" (status={status}{code_note}).'


def _summary_shell_list(parsed: dict[str, Any]) -> str:
    processes = parsed.get("processes")
    count = len(processes) if isinstance(processes, list) else 0
    if count == 0:
        return "No background processes."
    running = sum(
        1
        for process in (processes or [])
        if isinstance(process, dict) and process.get("status") == "running"
    )
    return f"Listed {count} background process(es) ({running} running)."


def _summary_session_set_workdir(parsed: dict[str, Any]) -> str:
    relpath = str(parsed.get("active_workdir_relpath") or ".").strip() or "."
    return f"Active workdir set to {relpath}."


def _summary_git_status(parsed: dict[str, Any]) -> str:
    status_text = str(parsed.get("status") or "")
    lines = [line for line in status_text.splitlines() if line.strip()]
    return f"Collected git status ({len(lines)} non-empty lines)."


def _summary_git_history(parsed: dict[str, Any]) -> str:
    mode = str(parsed.get("mode") or "")
    if mode == "log":
        commits = parsed.get("commits")
        count = len(commits) if isinstance(commits, list) else 0
        truncated = bool(parsed.get("truncated"))
        trunc_note = ", truncated" if truncated else ""
        return f"Loaded git history ({count} commit(s){trunc_note})."
    if mode == "show":
        commit_obj = parsed.get("commit")
        short_commit = ""
        if isinstance(commit_obj, dict):
            short_commit = str(commit_obj.get("short_commit") or "")
        patch_excerpt = str(parsed.get("patch_excerpt") or "")
        truncated = bool(parsed.get("patch_truncated"))
        trunc_note = ", truncated" if truncated else ""
        label = short_commit or "commit"
        return f"Loaded commit {label} ({len(patch_excerpt)} chars{trunc_note})."
    if mode == "blame":
        path = str(parsed.get("path") or "?")
        start_line = parsed.get("start_line")
        end_line = parsed.get("end_line")
        lines = parsed.get("lines")
        count = len(lines) if isinstance(lines, list) else 0
        return f'Loaded blame for "{path}" lines {start_line}-{end_line} ({count} line(s)).'
    return "Loaded git history."


def _summary_git_diff(parsed: dict[str, Any]) -> str:
    diff_text = str(parsed.get("diff") or "")
    files = diff_text.count("diff --git")
    return f"Collected git diff ({len(diff_text)} chars, about {files} file(s))."


def _summary_git_apply_patch(parsed: dict[str, Any]) -> str:
    if parsed.get("applied") is True:
        return "Patch applied successfully."
    keys = ", ".join(sorted(str(k) for k in parsed.keys())[:6])
    return f"Output keys: {keys or '-'}."


@dataclass(frozen=True)
class RichToolMetadata:
    display_name: str
    reasoning_hint: str
    action_hint: str
    fallback_hint: str
    input_preview_formatter: ToolFormatter | None = None
    output_summary_formatter: ToolFormatter | None = None


@dataclass(frozen=True)
class BuiltinToolMetadata:
    name: str
    description: str
    parameters: dict[str, Any]
    categories: tuple[str, ...]
    rich: RichToolMetadata
    built_in_subagent_exposure: str = "hidden"
    optional: bool = False
    optional_unavailable_reason: str | None = None

    def copied_parameters(self) -> dict[str, Any]:
        return copy.deepcopy(self.parameters)


_BUILTIN_TOOL_METADATA: tuple[BuiltinToolMetadata, ...] = (
    BuiltinToolMetadata(
        name="fs_read",
        description="Read a UTF-8 text file under the working root. Prefer after symbol_search or search_rg for exact file contents.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "path_base": _path_base_property(
                    description=(
                        "Resolve a relative path from the live active_workdir (default) or from the "
                        "immutable workspace_root."
                    )
                ),
                "max_bytes": {"type": "integer", "default": 12000},
            },
            "required": ["path"],
        },
        categories=("read", "fs"),
        rich=RichToolMetadata(
            display_name="Read File",
            reasoning_hint="Need exact file content before suggesting edits.",
            action_hint="Read file text from the current workspace.",
            fallback_hint="If read fails, adjust path or list files first.",
            input_preview_formatter=_preview_single_path,
            output_summary_formatter=_summary_fs_read,
        ),
        built_in_subagent_exposure="readonly",
    ),
    BuiltinToolMetadata(
        name="fs_read_lines",
        description="Read a precise 1-indexed line range from a UTF-8 text file. Prefer this for a narrow confirmed range.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "path_base": _path_base_property(),
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
                "max_lines": {"type": "integer", "default": 200},
                "include_line_numbers": {"type": "boolean", "default": True},
            },
            "required": ["path", "start_line"],
        },
        categories=("read", "fs"),
        rich=RichToolMetadata(
            display_name="Read File Lines",
            reasoning_hint="Inspect a precise file range without rereading the whole file.",
            action_hint="Read a narrow 1-indexed line window from the current workspace.",
            fallback_hint="If the range is wrong, adjust start/end lines or fall back to fs_read.",
            input_preview_formatter=_preview_fs_read_lines,
            output_summary_formatter=_summary_fs_read_lines,
        ),
        built_in_subagent_exposure="readonly",
    ),
    BuiltinToolMetadata(
        name="fs_edit",
        description="Apply deterministic exact-text edits to one UTF-8 text file. Prefer for localized edits to an existing file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "path_base": _path_base_property(),
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {
                                "type": "string",
                                "enum": [
                                    "replace",
                                    "replace_exact",
                                    "insert_before_exact",
                                    "insert_after_exact",
                                    "append",
                                    "prepend",
                                ],
                            },
                            "target": {"type": "string"},
                            "replacement": {"type": "string"},
                            "content": {"type": "string"},
                            "expected_match_count": {"type": "integer", "minimum": 0},
                        },
                        "required": ["op"],
                    },
                },
            },
            "required": ["path", "edits"],
        },
        categories=("write", "fs"),
        rich=RichToolMetadata(
            display_name="Edit File",
            reasoning_hint="Apply deterministic exact-text edits to one file.",
            action_hint=(
                "Edit a localized file region with exact-match operations and review the diff preview."
            ),
            fallback_hint="If a target is ambiguous or missing, narrow the edit or use git_apply_patch.",
            input_preview_formatter=_preview_single_path,
            output_summary_formatter=_summary_fs_edit,
        ),
    ),
    BuiltinToolMetadata(
        name="fs_move",
        description="Move or rename one file under the working root. Prefer over shell commands for routine file moves.",
        parameters={
            "type": "object",
            "properties": {
                "source_path": {"type": "string"},
                "source_path_base": _path_base_property(),
                "destination_path": {"type": "string"},
                "destination_path_base": _path_base_property(),
                "overwrite": {"type": "boolean", "default": False},
            },
            "required": ["source_path", "destination_path"],
        },
        categories=("write", "fs"),
        rich=RichToolMetadata(
            display_name="Move File",
            reasoning_hint="Rename or relocate one file without shell commands.",
            action_hint=(
                "Move a single file under the workspace root and confirm the source/destination preview."
            ),
            fallback_hint=(
                "If the destination exists or the path is wrong, adjust the target or enable overwrite explicitly."
            ),
            input_preview_formatter=_preview_source_destination,
            output_summary_formatter=_summary_fs_move,
        ),
    ),
    BuiltinToolMetadata(
        name="fs_copy",
        description="Copy one file under the working root. Prefer over shell commands for routine file copies.",
        parameters={
            "type": "object",
            "properties": {
                "source_path": {"type": "string"},
                "source_path_base": _path_base_property(),
                "destination_path": {"type": "string"},
                "destination_path_base": _path_base_property(),
                "overwrite": {"type": "boolean", "default": False},
            },
            "required": ["source_path", "destination_path"],
        },
        categories=("write", "fs"),
        rich=RichToolMetadata(
            display_name="Copy File",
            reasoning_hint="Duplicate one file without shell commands.",
            action_hint=(
                "Copy a single file under the workspace root and confirm the destination preview."
            ),
            fallback_hint=(
                "If the destination exists or the path is wrong, adjust the target or enable overwrite explicitly."
            ),
            input_preview_formatter=_preview_source_destination,
            output_summary_formatter=_summary_fs_copy,
        ),
    ),
    BuiltinToolMetadata(
        name="fs_delete",
        description="Delete one file under the working root. Prefer over shell commands for routine file deletes.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "path_base": _path_base_property(),
            },
            "required": ["path"],
        },
        categories=("write", "fs"),
        rich=RichToolMetadata(
            display_name="Delete File",
            reasoning_hint="Remove one file without shell commands.",
            action_hint=(
                "Delete a single file under the workspace root and confirm the preview before continuing."
            ),
            fallback_hint="If the path is wrong or protected, adjust the target instead of forcing the delete.",
            input_preview_formatter=_preview_single_path,
            output_summary_formatter=_summary_fs_delete,
        ),
    ),
    BuiltinToolMetadata(
        name="fs_write",
        description="Write a UTF-8 text file under the working root. Prefer for new/generated files or full-file replacements.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "path_base": _path_base_property(),
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        categories=("write", "fs"),
        rich=RichToolMetadata(
            display_name="Write File",
            reasoning_hint="Apply a concrete code/content update.",
            action_hint="Write new file content and verify patch preview.",
            fallback_hint="If blocked, ask for approval or reduce scope.",
            input_preview_formatter=_preview_single_path,
            output_summary_formatter=_summary_fs_write,
        ),
    ),
    BuiltinToolMetadata(
        name="fs_mkdir",
        description="Create one directory under the working root.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "path_base": _path_base_property(),
                "parents": {"type": "boolean", "default": True},
                "exist_ok": {"type": "boolean", "default": True},
            },
            "required": ["path"],
        },
        categories=("write", "fs"),
        rich=RichToolMetadata(
            display_name="Create Directory",
            reasoning_hint="Create empty directories or explicit scaffolding without shell commands.",
            action_hint="Create a workspace-bounded directory path with optional parent creation.",
            fallback_hint=(
                "If the target collides with a file or the path is protected, adjust the path or use fs_write for files."
            ),
            input_preview_formatter=_preview_single_path,
            output_summary_formatter=_summary_fs_mkdir,
        ),
    ),
    BuiltinToolMetadata(
        name="fs_list",
        description="List files under root_path (best-effort .gitignore support).",
        parameters={
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "path_base": _path_base_property(
                    description=(
                        "Resolve a relative root_path from the live active_workdir (default) or from "
                        "the immutable workspace_root."
                    )
                ),
                "globs": {"type": "array", "items": {"type": "string"}},
                "ignore": {"type": "array", "items": {"type": "string"}},
            },
            "required": [],
        },
        categories=("read", "fs"),
        rich=RichToolMetadata(
            display_name="List Files",
            reasoning_hint="Discover relevant files for the task.",
            action_hint="List workspace paths with optional filters.",
            fallback_hint="If results are noisy, narrow globs and retry.",
            output_summary_formatter=_summary_fs_list,
        ),
        built_in_subagent_exposure="readonly",
    ),
    BuiltinToolMetadata(
        name="web_fetch",
        description=(
            "Fetch one specific known HTTP(S) URL with SSRF-style safety checks and return readable text. "
            "Prefer it only for a user-provided URL or one returned by web_search; the runtime rejects guessed "
            "URLs. Do not use it for discovery and do not guess or invent URLs."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_chars": {"type": "integer", "default": 20000, "minimum": 1, "maximum": 50000},
            },
            "required": ["url"],
        },
        categories=("read", "web"),
        rich=RichToolMetadata(
            display_name="Fetch Web Page",
            reasoning_hint="Read targeted external docs/spec pages without shelling out.",
            action_hint="Fetch one URL and inspect extracted readable text and metadata.",
            fallback_hint=(
                "If blocked or unsupported, use a different public URL or request manual input."
            ),
            input_preview_formatter=_preview_web_fetch,
            output_summary_formatter=_summary_web_fetch,
        ),
        built_in_subagent_exposure="hidden",
    ),
    BuiltinToolMetadata(
        name="web_search",
        description=(
            "Search the public web for external docs, APIs, release notes, or error research and return a "
            "concise answer with cited sources. Prefer this when the task requires live/current/latest external discovery or explicit web browsing. Use web_fetch only after you have a URL the user provided or "
            "web_search returned. `external_web_access=false` is only supported by the OpenAI Responses backend."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "allowed_domains": {"type": "array", "items": {"type": "string"}},
                "max_sources": {"type": "integer", "default": 8, "minimum": 1, "maximum": 20},
                "external_web_access": {"type": "boolean", "default": True},
            },
            "required": ["query"],
        },
        categories=("read", "web", "search"),
        rich=RichToolMetadata(
            display_name="Search Web",
            reasoning_hint="Discover the right external docs/page/source before fetching a specific URL.",
            action_hint="Run a bounded web search and return an answer with citations and source URLs.",
            fallback_hint=(
                "If unavailable, use a user-provided direct public URL with web_fetch or ask the user for a "
                "target page."
            ),
            input_preview_formatter=_preview_web_search,
            output_summary_formatter=_summary_web_search,
        ),
        built_in_subagent_exposure="hidden",
    ),
    BuiltinToolMetadata(
        name="symbol_search",
        description=(
            "Search Python (AST) and JavaScript/TypeScript (heuristic) symbols "
            "(functions, classes, methods, constants) under the working root. Prefer this before broad regex search when locating definitions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["function", "class", "method", "constant"],
                },
                "root_path": {"type": "string"},
                "path_base": _path_base_property(),
                "globs": {"type": "array", "items": {"type": "string"}},
                "max_results": {"type": "integer", "default": 100},
                "exact": {"type": "boolean", "default": False},
            },
            "required": ["query"],
        },
        categories=("read", "search", "symbol"),
        rich=RichToolMetadata(
            display_name="Symbol Search",
            reasoning_hint="Navigate Python or JS/TS definitions before broad regex search.",
            action_hint=(
                "Search parsed Python symbols plus pragmatic JS/TS symbols (class/function/method/constant)."
            ),
            fallback_hint="If results are sparse, relax exact/kind filters or fall back to search_rg.",
            input_preview_formatter=_preview_symbol_search,
            output_summary_formatter=_summary_symbol_search,
        ),
        built_in_subagent_exposure="readonly",
    ),
    BuiltinToolMetadata(
        name="search_rg",
        description="Search for a regex pattern under root_path using ripgrep when available. Prefer this for fast text/code lookup before reading or patching files.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "root_path": {"type": "string"},
                "path_base": _path_base_property(),
                "globs": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["pattern"],
        },
        categories=("read", "search"),
        rich=RichToolMetadata(
            display_name="Search Workspace",
            reasoning_hint="Locate exact code/text matches fast.",
            action_hint="Run pattern search and return matching lines.",
            fallback_hint="If no matches, broaden pattern and search scope.",
            input_preview_formatter=_preview_pattern,
            output_summary_formatter=_summary_search_rg,
        ),
        built_in_subagent_exposure="readonly",
    ),
    BuiltinToolMetadata(
        name="history_search",
        description=(
            "Search current session artifacts (history chunks, tool outputs, and memory files) for a regex pattern."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "max_results": {"type": "integer", "default": 50},
                "max_file_bytes": {"type": "integer", "default": 200000},
                "include_history": {"type": "boolean", "default": True},
                "include_tool_outputs": {"type": "boolean", "default": True},
                "include_memory": {"type": "boolean", "default": True},
            },
            "required": ["pattern"],
        },
        categories=("read", "history"),
        rich=RichToolMetadata(
            display_name="Search Session History",
            reasoning_hint="Inspect current session artifacts without rereading every history file.",
            action_hint="Search stored history chunks, tool outputs, and memory summaries for a pattern.",
            fallback_hint="If results are sparse, widen the regex or include more artifact types.",
            input_preview_formatter=_preview_pattern,
            output_summary_formatter=_summary_history_search,
        ),
        built_in_subagent_exposure="readonly",
    ),
    BuiltinToolMetadata(
        name="knowledge_capture_json",
        description=(
            "Optional host-observed structured knowledge capture marker. This is not a "
            "callable runtime tool; the host parses a final assistant fenced block when present."
        ),
        parameters={
            "type": "object",
            "properties": {
                "schema_version": {"type": "integer"},
                "facts": {"type": "array"},
                "decisions": {"type": "array"},
                "open_questions": {"type": "array"},
            },
            "required": [],
        },
        categories=("knowledge", "optional"),
        rich=RichToolMetadata(
            display_name="Knowledge Capture",
            reasoning_hint=(
                "Record reusable repo knowledge only through the final assistant fenced block."
            ),
            action_hint=(
                "Do not call this as a runtime tool; append the bounded fenced JSON block when useful."
            ),
            fallback_hint=(
                "If unavailable, continue with the normal final response; missing capture is non-fatal."
            ),
            output_summary_formatter=lambda parsed: (
                "Optional knowledge capture tool unavailable."
                if is_tool_unavailable_result(parsed)
                else "Knowledge capture metadata result."
            ),
        ),
        optional=True,
        optional_unavailable_reason=(
            "not registered in active tool registry; knowledge capture is a final assistant "
            "fenced block parsed by host"
        ),
    ),
    BuiltinToolMetadata(
        name="skill_read",
        description=(
            "Read a discovered skill bundle entrypoint or one specific file within that skill bundle. "
            "Use this to inspect SKILL.md or targeted references/scripts/assets before applying a skill."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["name"],
        },
        categories=("read", "skills"),
        rich=RichToolMetadata(
            display_name="Read Skill",
            reasoning_hint="Inspect a discovered skill bundle before relying on its instructions.",
            action_hint="Read SKILL.md or one bundle file by name from the discovered skills registry.",
            fallback_hint="If the skill name or file path is wrong, list skills or retry with a bundle-relative path.",
            input_preview_formatter=_preview_skill_read,
            output_summary_formatter=_summary_skill_read,
        ),
        built_in_subagent_exposure="readonly",
    ),
    BuiltinToolMetadata(
        name="verify_run",
        description=(
            "Run configured verification commands. Prefer for tests/lint/build. "
            "Do not pipe/filter, run list/build-only checks, or swap build systems."
        ),
        parameters={
            "type": "object",
            "properties": {
                "commands": {"type": "array", "items": {"type": "string"}},
            },
            "required": [],
        },
        categories=("verify",),
        rich=RichToolMetadata(
            display_name="Run Verification",
            reasoning_hint="Run structured verification before relying on raw shell commands.",
            action_hint="Execute configured or targeted verification commands and inspect pass/fail results.",
            fallback_hint="If using shell_run, run the same unfiltered verifier.",
            input_preview_formatter=_preview_verify_run,
            output_summary_formatter=_summary_verify_run,
        ),
    ),
    BuiltinToolMetadata(
        name="shell_run",
        description="Run a shell command under the working root (policy-checked).",
        parameters={
            "type": "object",
            "properties": {
                "cmd": {"type": "string"},
                "cwd": {"type": "string"},
                "cwd_base": _cwd_base_property(),
            },
            "required": ["cmd"],
        },
        categories=("shell",),
        rich=RichToolMetadata(
            display_name="Run Command",
            reasoning_hint="Validate assumptions with project commands.",
            action_hint="Run command and inspect exit code/stdout/stderr.",
            fallback_hint="If denied/failing, use safer command or ask approval.",
            input_preview_formatter=_preview_shell_run,
            output_summary_formatter=_summary_shell_run,
        ),
    ),
    BuiltinToolMetadata(
        name="shell_background",
        description=(
            "Start a long-running shell command in the background under the working root "
            "(policy-checked). Returns a process_id you can use with shell_output, shell_kill, "
            "and shell_list. Use this for dev servers, file watchers, log tailers, or any "
            "command that should not block the agent loop."
        ),
        parameters={
            "type": "object",
            "properties": {
                "cmd": {"type": "string"},
                "cwd": {"type": "string"},
                "cwd_base": _cwd_base_property(),
            },
            "required": ["cmd"],
        },
        categories=("shell", "background"),
        rich=RichToolMetadata(
            display_name="Run Background Command",
            reasoning_hint=(
                "Spawn a non-blocking process for long-running work without holding the agent loop."
            ),
            action_hint="Start command and capture process_id; later read incrementally with shell_output.",
            fallback_hint="If the command can complete fast, prefer shell_run for direct stdout.",
            input_preview_formatter=_preview_shell_run,
            output_summary_formatter=_summary_shell_background,
        ),
    ),
    BuiltinToolMetadata(
        name="shell_output",
        description=(
            "Read accumulated stdout/stderr from a background process started with shell_background. "
            "Pass since=<next_seq> from the previous read to get only new lines. Output is "
            "ring-buffered; very chatty processes may report dropped_lines > 0."
        ),
        parameters={
            "type": "object",
            "properties": {
                "process_id": {"type": "string"},
                "since": {"type": "integer", "default": 0},
            },
            "required": ["process_id"],
        },
        categories=("shell", "background", "read"),
        rich=RichToolMetadata(
            display_name="Read Background Output",
            reasoning_hint="Inspect what a background process has emitted since the last read.",
            action_hint="Fetch new output lines plus current status, exit_code, and runtime.",
            fallback_hint="If process_id is unknown, list active processes with shell_list first.",
            input_preview_formatter=_preview_shell_output,
            output_summary_formatter=_summary_shell_output,
        ),
    ),
    BuiltinToolMetadata(
        name="shell_kill",
        description=(
            "Terminate a background process by process_id. Sends SIGTERM (or platform equivalent), "
            "escalates to SIGKILL after the configured grace period. Idempotent - calling on an "
            "already-terminated process returns the existing status."
        ),
        parameters={
            "type": "object",
            "properties": {
                "process_id": {"type": "string"},
            },
            "required": ["process_id"],
        },
        categories=("shell", "background"),
        rich=RichToolMetadata(
            display_name="Kill Background Process",
            reasoning_hint="Stop a background process when no longer needed or to free a slot.",
            action_hint="Signal the process; output remains readable after termination.",
            fallback_hint="If kill fails, the session lifecycle reaps remaining processes on close.",
            input_preview_formatter=_preview_shell_kill,
            output_summary_formatter=_summary_shell_kill,
        ),
    ),
    BuiltinToolMetadata(
        name="shell_list",
        description=(
            "List all background processes from this session with their status, exit code, "
            "command preview (truncated), cwd, and runtime. Includes both running and recently-"
            "terminated processes that have not yet been pruned."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        categories=("shell", "background", "read"),
        rich=RichToolMetadata(
            display_name="List Background Processes",
            reasoning_hint="Audit current background activity before starting more work.",
            action_hint="Enumerate active and recently-terminated bg processes.",
            fallback_hint="If empty, no bg processes are tracked in this session.",
            output_summary_formatter=_summary_shell_list,
        ),
    ),
    BuiltinToolMetadata(
        name="session_set_workdir",
        description=(
            "Change the live session active_workdir inside the bound workspace_root so later relative "
            "file, search, and shell calls default there. Use this when the user says things like "
            "'go to packages/app', 'work in apps/web', or 'switch to server/api and inspect package.json'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Target directory inside the current workspace_root. Relative values resolve "
                        "from the current active_workdir."
                    ),
                },
            },
            "required": ["path"],
        },
        categories=("session", "navigation"),
        rich=RichToolMetadata(
            display_name="Set Session Workdir",
            reasoning_hint=(
                "Move the session's live default workdir before more file, search, or shell calls."
            ),
            action_hint=(
                "Change the active workdir inside the current workspace_root when the user asks to "
                "go to packages/app, work in apps/web, or switch to another directory."
            ),
            fallback_hint="If the path escapes the workspace or does not exist, report that blocker clearly.",
            input_preview_formatter=_preview_single_path,
            output_summary_formatter=_summary_session_set_workdir,
        ),
    ),
    BuiltinToolMetadata(
        name="git_status",
        description="Run git status (porcelain) in the working root. Prefer before/after edits to inspect repo state.",
        parameters={"type": "object", "properties": {}, "required": []},
        categories=("read", "git"),
        rich=RichToolMetadata(
            display_name="Git Status",
            reasoning_hint="Check repository state before/after edits.",
            action_hint="Inspect tracked/untracked and dirty changes.",
            fallback_hint="If unavailable, continue with file-based checks.",
            output_summary_formatter=_summary_git_status,
        ),
        built_in_subagent_exposure="readonly",
    ),
    BuiltinToolMetadata(
        name="git_diff",
        description="Run git diff in the working root. Prefer to review current repo changes before the final response.",
        parameters={"type": "object", "properties": {}, "required": []},
        categories=("read", "git"),
        rich=RichToolMetadata(
            display_name="Git Diff",
            reasoning_hint="Review change impact before final response.",
            action_hint="Collect current diff for inspection.",
            fallback_hint="If unavailable, rely on patch/tool summaries.",
            output_summary_formatter=_summary_git_diff,
        ),
        built_in_subagent_exposure="readonly",
    ),
    BuiltinToolMetadata(
        name="git_history",
        description=(
            "Inspect repository history with one tool: mode=log for commit metadata, mode=show for one "
            "commit excerpt, and mode=blame for line ownership. Prefer this over raw shell log/show/blame inspection when available."
        ),
        parameters={
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["log", "show", "blame"]},
                "path": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
                "ref": {"type": "string"},
                "grep": {"type": "string"},
                "author": {"type": "string"},
                "commit": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["mode"],
        },
        categories=("read", "git", "history"),
        rich=RichToolMetadata(
            display_name="Git History",
            reasoning_hint="Inspect repository history without dropping to shell commands.",
            action_hint="Use one tool for commit logs, commit excerpts, or blame ranges.",
            fallback_hint="If the commit/path/range is wrong, narrow the request and retry.",
            input_preview_formatter=_preview_git_history,
            output_summary_formatter=_summary_git_history,
        ),
        built_in_subagent_exposure="readonly",
    ),
    BuiltinToolMetadata(
        name="git_apply_patch",
        description="Apply a unified diff patch using git apply. Prefer for broader, multi-file, or context-heavy edits where unified diff context matters.",
        parameters={
            "type": "object",
            "properties": {"patch": {"type": "string"}},
            "required": ["patch"],
        },
        categories=("write", "git"),
        rich=RichToolMetadata(
            display_name="Apply Patch",
            reasoning_hint="Apply multi-file edits atomically.",
            action_hint="Run patch application and validate outcome.",
            fallback_hint="If patch fails, retry with smaller focused patch.",
            output_summary_formatter=_summary_git_apply_patch,
        ),
    ),
    BuiltinToolMetadata(
        name="subagent_run",
        description=(
            "Run a registered subagent in an isolated nested session and return its single "
            "final report. Each call spawns a fresh subagent with its own system prompt, "
            "tool sandbox, step budget, and message history; the subagent does not see this "
            "conversation's transcript and you do not see its intermediate steps -- only the "
            "final text it produced plus structured metadata.\n"
            "\n"
            "Strategic guidance for when to delegate, parallelism, and trust-but-verify is "
            "covered in the `Subagent delegation` section of the system prompt; this "
            "description focuses on the tool contract.\n"
            "\n"
            "Parameters\n"
            "- name (required, string): registered subagent name. Built-in names include "
            "`explorer`, `general-purpose`, `reviewer`, and `test-strategist`. "
            "Project-level custom subagents "
            "from `.sylliptor_agents/*.md` and user-level ones from the user config dir are "
            "also resolvable. Names are case-insensitive and a small alias table is applied "
            "(e.g. `explore` -> `explorer`). Unknown names return an `error` field plus the "
            "list of available names.\n"
            "- task (required, string): the self-contained brief for the subagent. Treat the "
            "subagent like a smart colleague who just walked into the room: it has no memory "
            "of this conversation and has read no files yet. A good `task` includes (1) the "
            "goal in one sentence, (2) exact repo-root-relative paths or symbols to start "
            "from when known, (3) what you already learned or ruled out, and (4) the shape "
            "of answer you want (e.g. `list 3-5 candidate files`, `verdict + blocking "
            "issues`, `under 250 words`). Terse command-style prompts produce shallow output.\n"
            "- mode (optional, string): one of `readonly`, `review`, `auto`, `fullaccess`. "
            "Defaults to the subagent's declared mode. The effective mode is clamped to be "
            "no more permissive than the parent session's mode, and is never raised above "
            "`auto` unless the parent session itself is `fullaccess`. You cannot use this to "
            "escalate privileges.\n"
            "- max_steps (optional, integer): hard cap on the subagent's tool-step budget. "
            "If omitted, the budget is resolved from session policy and the subagent's name. "
            "Set this only when you have a specific reason; the default is usually correct.\n"
            "\n"
            "Output shape on success\n"
            "Returns an object with: `subagent` (resolved name), `subagent_session_id`, "
            "`result` (the subagent's final assistant text -- this is your primary signal), "
            "`usage` (token/cost totals already merged into the parent session's usage), "
            "and `sandbox` (the effective mode and the list of tools the "
            "subagent had after allow/deny filtering).\n"
            "\n"
            "Output shape on failure\n"
            "Returns an object with `error` (string) and, when applicable, `available_subagents`, "
            "`subagent_session_id`, `exit_code`, `usage`, and `final_text` (best-effort partial "
            "output). Common causes: unknown name, subagents disabled for the session, nested "
            "delegation attempted (subagents cannot themselves call `subagent_run`), no tools "
            "remained after sandbox filtering, or the nested session raised.\n"
            "\n"
            "Sandboxing facts\n"
            "- Each subagent definition declares an allow-list and/or deny-list of tools; "
            "the resulting toolset may be smaller than your own. Do not assume the subagent "
            "has every tool you have.\n"
            "- Subagents cannot recursively spawn subagents (depth is capped at 1).\n"
            "- A subagent may run on a different model and temperature than this session, "
            "controlled by its `model` / `model_role` definition fields.\n"
            "\n"
            "Examples of `task`\n"
            'Good: "Map how API authentication flows from request ingress to user '
            "resolution. Start from src/api/server.py and src/auth/. I have already "
            "confirmed the JWT lib is PyJWT. Report: the call chain (file:line for each "
            "step), where session state is stored, and any auth checks that look "
            'inconsistent. Under 250 words."\n'
            'Bad: "look at the auth code"'
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Registered subagent name. Built-in: explorer, "
                        "reviewer, test-strategist, general-purpose. "
                        "Project-defined custom names from "
                        ".sylliptor_agents/ are also valid. "
                        "Case-insensitive."
                    ),
                },
                "task": {
                    "type": "string",
                    "description": (
                        "Self-contained brief for the subagent. The "
                        "subagent has no memory of this conversation "
                        "and has read no files yet -- include the "
                        "goal, exact paths or symbols to start from, "
                        "what you have ruled out, and the form of "
                        "answer you want. See tool description for "
                        "full guidance."
                    ),
                },
                "mode": {
                    "type": "string",
                    "description": (
                        "Optional mode override "
                        "(readonly|review|auto|fullaccess). Clamped "
                        "to the parent session's mode and never "
                        "escalated above auto unless the parent is "
                        "fullaccess."
                    ),
                },
                "max_steps": {
                    "type": "integer",
                    "description": (
                        "Optional cap on the subagent's tool-step "
                        "budget. If omitted, resolved from session "
                        "policy and the subagent's name."
                    ),
                },
            },
            "required": ["name", "task"],
        },
        categories=("subagent",),
        rich=RichToolMetadata(
            display_name="Run Subagent",
            reasoning_hint="Delegate focused repository analysis to a specialized subagent.",
            action_hint="Run nested subagent session and consume the final summarized result.",
            fallback_hint="If unclear or low confidence, verify claims with direct tools before continuing.",
            output_summary_formatter=_summary_subagent_run,
        ),
    ),
)

_BUILTIN_TOOL_METADATA_BY_NAME = {spec.name: spec for spec in _BUILTIN_TOOL_METADATA}
if len(_BUILTIN_TOOL_METADATA_BY_NAME) != len(_BUILTIN_TOOL_METADATA):
    raise RuntimeError("Duplicate built-in tool name detected in registry.")


def iter_builtin_tool_metadata() -> tuple[BuiltinToolMetadata, ...]:
    return _BUILTIN_TOOL_METADATA


def get_builtin_tool_metadata(name: str) -> BuiltinToolMetadata | None:
    return _BUILTIN_TOOL_METADATA_BY_NAME.get(name)


def require_builtin_tool_metadata(name: str) -> BuiltinToolMetadata:
    spec = get_builtin_tool_metadata(name)
    if spec is None:
        raise KeyError(f"Unknown built-in tool: {name}")
    return spec


def copied_tool_parameters(name: str) -> dict[str, Any]:
    return require_builtin_tool_metadata(name).copied_parameters()


def builtin_tool_names_with_category(category: str) -> tuple[str, ...]:
    normalized = str(category or "").strip().lower()
    return tuple(
        spec.name
        for spec in _BUILTIN_TOOL_METADATA
        if normalized in {tag.lower() for tag in spec.categories}
    )


def built_in_subagent_tool_names(*, exposure: str = "readonly") -> tuple[str, ...]:
    normalized = str(exposure or "").strip().lower()
    return tuple(
        spec.name
        for spec in _BUILTIN_TOOL_METADATA
        if spec.built_in_subagent_exposure.strip().lower() == normalized
    )


def tool_display_name(tool_name: str) -> str:
    spec = get_builtin_tool_metadata(tool_name)
    if spec is None:
        return tool_name
    return spec.rich.display_name


def tool_reasoning_hints(tool_name: str) -> tuple[str, str, str]:
    spec = get_builtin_tool_metadata(tool_name)
    if spec is None:
        return (
            "Execute a targeted helper action.",
            "Run tool and collect structured output.",
            "If it fails, use a narrower fallback strategy.",
        )
    return (
        spec.rich.reasoning_hint,
        spec.rich.action_hint,
        spec.rich.fallback_hint,
    )


def tool_input_preview(tool_name: str, args: dict[str, Any]) -> str:
    spec = get_builtin_tool_metadata(tool_name)
    if spec is None or spec.rich.input_preview_formatter is None:
        return "-"
    return spec.rich.input_preview_formatter(args)


def summarize_tool_output_chunk(tool_name: str, chunk: str) -> str:
    try:
        parsed = json.loads(chunk)
    except json.JSONDecodeError:
        return _truncate_inline(chunk, max_chars=180)

    if not isinstance(parsed, dict):
        compact = json.dumps(parsed, ensure_ascii=True)
        return _truncate_inline(compact, max_chars=180)

    summary_text = _truncate_inline(str(parsed.get("summary") or ""), max_chars=160)
    preview = _truncate_inline(str(parsed.get("preview") or ""), max_chars=96)

    if is_tool_unavailable_result(parsed):
        tool = _truncate_inline(parsed["tool"], max_chars=64)
        reason = _truncate_inline(parsed["reason"], max_chars=140)
        return f"Tool unavailable: {tool}: {reason}"

    if parsed.get("transcript_shaped") is True:
        if summary_text:
            return summary_text + (f" Preview: {preview}" if preview else "")
        if preview:
            return f"Transcript retained a bounded preview. Preview: {preview}"
        return "Tool output was summarized for transcript retention."

    if parsed.get("offloaded") is True:
        artifact_ref = str(
            parsed.get("artifact_locator") or parsed.get("artifact_path") or ""
        ).strip()
        if artifact_ref:
            if summary_text:
                return f"{summary_text} Artifact: {artifact_ref}." + (
                    f" Preview: {preview}" if preview else ""
                )
            return f"Output offloaded to {artifact_ref}." + (
                f" Preview: {preview}" if preview else ""
            )
        if summary_text:
            return summary_text + (f" Preview: {preview}" if preview else "")
        return "Output offloaded into session artifacts."

    spec = get_builtin_tool_metadata(tool_name)
    if tool_name != "subagent_run" and "error" in parsed:
        msg = _truncate_inline(str(parsed.get("error") or ""), max_chars=140)
        return f"Error: {msg}"
    if spec is not None and spec.rich.output_summary_formatter is not None:
        return spec.rich.output_summary_formatter(parsed)
    keys = ", ".join(sorted(str(k) for k in parsed.keys())[:6])
    return f"Output keys: {keys or '-'}."
