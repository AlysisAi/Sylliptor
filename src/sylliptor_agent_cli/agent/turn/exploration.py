from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ...tools.registry import get_builtin_tool_metadata
from ...turn_intent import contains_any_normalized_marker as _contains_any_normalized_marker
from ...turn_intent import normalize_turn_intent_text as _normalize_marker_text
from ..prompt_context import (
    MAX_POST_EXPLORE_ANCHOR_PATHS,
    _extract_repo_relative_paths_from_text,
    _normalize_repo_relative_hint_path,
)
from ..verification import _runtime_message

MAX_RECENT_EXPLORATION_PATHS = 12
_ONE_SHOT_NON_FINAL_PROGRESS_MARKERS = (
    "i will",
    "i'll",
    "ill",
    "next",
    "then i will",
    "let me",
    "plan:",
    "next steps:",
    "i will proceed",
    "θα προχωρησω",
    "θα υλοποιησω",
    "θα ενημερωσω",
    "θα προσθεσω",
    "στη συνεχεια θα",
    "επομενο βημα",
    "σχεδιο:",
    "πρωτα θα",
    "μετα θα",
)
_ONE_SHOT_COMPLETION_MARKERS = (
    "implemented",
    "updated",
    "ran tests",
    "tests pass",
    "changed files",
    "i added",
    "i updated",
    "i ran",
    "completed",
    "finished",
    "υλοποιησα",
    "προσθεσα",
    "ενημερωσα",
    "διορθωσα",
    "ετρεξα τα tests",
    "ετρεξα tests",
    "ολοκληρωθηκε",
    "ολοκληρωσα",
    "τελειωσα",
)
_ONE_SHOT_BLOCKER_MARKERS = (
    "blocked",
    "cannot proceed",
    "can't proceed",
    "need approval",
    "missing info",
    "missing information",
    "need more info",
    "need more information",
    "permission denied",
    "requires approval",
    "δεν μπορω να προχωρησω",
    "δεν μπορω να συνεχισω",
    "χρειαζομαι εγκριση",
    "απαιτειται εγκριση",
    "λειπουν πληροφοριες",
    "δεν εχω αρκετες πληροφοριες",
    "δεν εχω αρκετη πληροφορια",
    "εχω μπλοκαρει",
    "ειμαι μπλοκαρισμενος",
    "ειμαι μπλοκαρισμενη",
    "δεν εχω προσβαση",
)
_STRUCTURED_BLOCKER_MARKER_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\[?\s*)?"
    r"(?:blocked|blocker|needs[_\s-]?user|needs[_\s-]?input|need[_\s-]?user[_\s-]?input)"
    r"(?:\s*\]?)?\s*:",
    re.IGNORECASE,
)
_STRUCTURED_BLOCKER_CATEGORY_RE = re.compile(
    r"\b(?:category|type|reason)\s*:\s*"
    r"(?:approval|ambiguous(?:_requirement)?|credentials?|docker|environment|"
    r"missing(?:_information|_dependency|_toolchain)?|network|permission|policy|"
    r"sandbox|toolchain|unavailable)\b",
    re.IGNORECASE,
)
_BLOCKER_OBSTACLE_RE = re.compile(
    r"\b(?:approval|permission|permissions|sandbox|docker|network|offline|toolchain|"
    r"missing|unavailable|not\s+available|not\s+installed|command\s+not\s+found|"
    r"no\s+such\s+file|ambiguous|clarif(?:y|ication)|user\s+input|"
    r"cannot\s+resolve|can't\s+resolve|environment|host|credential|credentials|"
    r"secret|secrets|api\s+key|access|policy|blocked)\b",
    re.IGNORECASE,
)
_EXPLORATION_FALLBACK_TOOL_NAMES = {
    "fs_read",
    "fs_read_lines",
    "fs_list",
    "symbol_search",
    "search_rg",
    "history_search",
    "git_status",
    "git_diff",
    "git_history",
}
_ACTION_PROGRESS_FALLBACK_TOOL_NAMES = {
    "fs_edit",
    "fs_move",
    "fs_copy",
    "fs_delete",
    "fs_mkdir",
    "fs_write",
    "git_apply_patch",
    "verify_run",
    "shell_run",
    "subagent_run",
}
_ACTION_PROGRESS_TOOL_CATEGORIES = {"write", "verify", "shell", "subagent"}
_EXPLORATION_TOOL_CATEGORIES = {"read", "search", "history"}
_FAILED_EDIT_STAGNATION_TOOL_NAMES = {"fs_edit", "git_apply_patch", "fs_write"}
_UNEXECUTED_TOOL_CALL_MARKUP_MARKERS = (
    "<tool_call",
    "</tool_call",
    "tool_calls",
    "<function_calls",
    "</function_calls",
)


def _assistant_text_contains_progress_intent(text: str) -> bool:
    normalized = _normalize_marker_text(text)
    return _contains_any_normalized_marker(normalized, _ONE_SHOT_NON_FINAL_PROGRESS_MARKERS)


def _assistant_text_has_completion_marker(text: str) -> bool:
    normalized = _normalize_marker_text(text)
    return _contains_any_normalized_marker(normalized, _ONE_SHOT_COMPLETION_MARKERS)


def _assistant_text_has_blocker_marker(text: str) -> bool:
    normalized = _normalize_marker_text(text)
    return _contains_any_normalized_marker(normalized, _ONE_SHOT_BLOCKER_MARKERS)


def _assistant_text_has_structured_blocker_marker(text: str) -> bool:
    return _STRUCTURED_BLOCKER_MARKER_RE.search(str(text or "")) is not None


def _structured_blocker_has_concrete_detail(text: str) -> bool:
    match = _STRUCTURED_BLOCKER_MARKER_RE.search(str(text or ""))
    if match is None:
        return False
    detail = str(text or "")[match.end() :].strip()
    if len(detail.split()) < 3:
        return False
    if _STRUCTURED_BLOCKER_CATEGORY_RE.search(detail):
        return True
    return _BLOCKER_OBSTACLE_RE.search(detail) is not None


def _assistant_text_has_well_formed_blocker(text: str) -> bool:
    raw_text = str(text or "").strip()
    if not raw_text:
        return False
    if _assistant_text_has_structured_blocker_marker(raw_text):
        return _structured_blocker_has_concrete_detail(raw_text)
    if not _assistant_text_has_blocker_marker(raw_text):
        return False
    return _BLOCKER_OBSTACLE_RE.search(raw_text) is not None


def _exploration_attempt_outcome(success_count: int, failed_count: int) -> str:
    if success_count > 0 and failed_count > 0:
        return "mixed"
    if failed_count > 0:
        return "failed"
    if success_count > 0:
        return "successful"
    return "none"


def _one_shot_progress_fingerprint(text: str) -> str:
    normalized = _normalize_marker_text(text)
    return re.sub(r"[^\w ]+", "", normalized, flags=re.UNICODE)


def _tool_call_retry_key(name: str, arguments: dict[str, Any]) -> str:
    try:
        payload = json.dumps(arguments, ensure_ascii=True, sort_keys=True)
    except TypeError:
        payload = json.dumps(str(arguments), ensure_ascii=True)
    return f"{name}:{payload}"


def _exploration_similarity_key(name: str, arguments: dict[str, Any]) -> str:
    relevant_fields = (
        "path",
        "root_path",
        "query",
        "pattern",
        "ref",
        "commit",
    )
    parts = [name]
    for key_name in relevant_fields:
        value = arguments.get(key_name)
        if value is None:
            continue
        parts.append(f"{key_name}={value}")
    return "|".join(parts)


def _edit_similarity_key(name: str, arguments: dict[str, Any]) -> str:
    path = str(arguments.get("path") or "").strip()
    source_path = str(arguments.get("source_path") or "").strip()
    destination_path = str(arguments.get("destination_path") or "").strip()
    ops = arguments.get("edits")
    op_names: list[str] = []
    if isinstance(ops, list):
        for raw in ops:
            if isinstance(raw, dict):
                op = raw.get("op")
                if isinstance(op, str):
                    op_names.append(op.strip())
    op_signature = ",".join(op_names[:5])
    return "|".join(
        [
            name,
            f"path={path}",
            f"source={source_path}",
            f"destination={destination_path}",
            f"ops={op_signature}",
        ]
    )


def _append_recent_exploration_path(
    *,
    paths: list[str],
    candidate: str | None,
    max_items: int = MAX_RECENT_EXPLORATION_PATHS,
) -> None:
    if not candidate:
        return
    normalized = candidate.strip()
    if not normalized or normalized == ".":
        return
    existing_idx = None
    for idx, item in enumerate(paths):
        if item.casefold() == normalized.casefold():
            existing_idx = idx
            break
    if existing_idx is not None:
        paths.pop(existing_idx)
    paths.append(normalized)
    if len(paths) > max_items:
        del paths[0 : len(paths) - max_items]


def _looks_like_unexecuted_tool_call_markup(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    if "dsml" in normalized and ("tool_calls" in normalized or "invoke" in normalized):
        return True
    return any(marker in normalized for marker in _UNEXECUTED_TOOL_CALL_MARKUP_MARKERS)


def _extract_successful_exploration_paths(
    *,
    root: Path,
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    max_items: int = MAX_POST_EXPLORE_ANCHOR_PATHS,
) -> list[str]:
    normalized_tool = tool_name.strip().lower()
    out: list[str] = []

    for key in ("path", "root_path"):
        value = arguments.get(key)
        if isinstance(value, str):
            normalized = _normalize_repo_relative_hint_path(root=root, raw=value)
            if normalized and not any(
                existing.casefold() == normalized.casefold() for existing in out
            ):
                out.append(normalized)
                if len(out) >= max_items:
                    return out

    if normalized_tool == "fs_list":
        result_root = _normalize_repo_relative_hint_path(
            root=root, raw=str(result.get("root") or "")
        )
        entries = result.get("entries")
        if isinstance(entries, list):
            for entry in entries[: max_items * 3]:
                if not isinstance(entry, dict):
                    continue
                rel = str(entry.get("path") or "")
                if not rel:
                    continue
                if result_root and result_root != ".":
                    combined = f"{result_root.rstrip('/')}/{rel.lstrip('./')}"
                else:
                    combined = rel
                normalized = _normalize_repo_relative_hint_path(root=root, raw=combined)
                if normalized and not any(
                    existing.casefold() == normalized.casefold() for existing in out
                ):
                    out.append(normalized)
                    if len(out) >= max_items:
                        return out

    if normalized_tool == "search_rg":
        matches = result.get("matches")
        if isinstance(matches, list):
            for match in matches[: max_items * 3]:
                if not isinstance(match, dict):
                    continue
                normalized = _normalize_repo_relative_hint_path(
                    root=root,
                    raw=str(match.get("path") or ""),
                )
                if normalized and not any(
                    existing.casefold() == normalized.casefold() for existing in out
                ):
                    out.append(normalized)
                    if len(out) >= max_items:
                        return out

    if normalized_tool == "subagent_run":
        subagent_result = str(result.get("result") or "")
        for candidate in _extract_repo_relative_paths_from_text(
            root=root,
            text=subagent_result,
            max_items=max_items,
        ):
            if any(existing.casefold() == candidate.casefold() for existing in out):
                continue
            out.append(candidate)
            if len(out) >= max_items:
                break

    return out


def _is_successful_subagent_run(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    status: str,
    result: dict[str, Any],
) -> bool:
    if tool_name.strip().lower() != "subagent_run":
        return False
    if status == "failed":
        return False
    return bool(str(result.get("subagent") or arguments.get("name") or "").strip())


def _build_post_explore_bootstrap_nudge(
    *,
    anchor_paths: list[str],
    language: str = "",
    explicit_language_override: bool = False,
) -> str:
    message = _runtime_message(
        "one_shot_post_explore_bootstrap_nudge",
        language=language,
        explicit_language_override=explicit_language_override,
    )
    if anchor_paths:
        joined = ", ".join(anchor_paths[:MAX_POST_EXPLORE_ANCHOR_PATHS])
        message += " " + _runtime_message(
            "one_shot_post_explore_bootstrap_targets",
            language=language,
            explicit_language_override=explicit_language_override,
            joined=joined,
        )
    return message


def _tool_categories(tool_name: str) -> set[str]:
    metadata = get_builtin_tool_metadata(tool_name)
    if metadata is None:
        return set()
    return {str(category).strip().lower() for category in metadata.categories}


def _is_action_progress_tool(tool_name: str) -> bool:
    categories = _tool_categories(tool_name)
    if categories:
        return bool(categories & _ACTION_PROGRESS_TOOL_CATEGORIES)
    return tool_name.strip().lower() in _ACTION_PROGRESS_FALLBACK_TOOL_NAMES


def _is_exploration_only_tool(tool_name: str) -> bool:
    categories = _tool_categories(tool_name)
    if categories:
        if categories & _ACTION_PROGRESS_TOOL_CATEGORIES:
            return False
        return bool(categories & _EXPLORATION_TOOL_CATEGORIES)
    return tool_name.strip().lower() in _EXPLORATION_FALLBACK_TOOL_NAMES


def _is_failed_edit_stagnation_tool(tool_name: str) -> bool:
    return tool_name.strip().lower() in _FAILED_EDIT_STAGNATION_TOOL_NAMES
