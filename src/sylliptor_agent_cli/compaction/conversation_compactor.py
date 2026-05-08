from __future__ import annotations

import json
import re
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sylliptor_agent_cli.llm.openai_compat import (
    LLMError,
    OpenAICompatClient,
    strip_provider_metadata_from_message,
)
from sylliptor_agent_cli.model_registry import ModelRegistry
from sylliptor_agent_cli.request_estimation import (
    estimate_request_tokens,
    sanitize_messages_for_estimation,
)
from sylliptor_agent_cli.session_artifacts import SessionArtifactLayout
from sylliptor_agent_cli.session_store import SessionStore
from sylliptor_agent_cli.token_budget import (
    compute_input_budget,
    estimate_tokens,
    trim_text_to_budget,
)
from sylliptor_agent_cli.usage_tracker import UsageSummary, build_usage_record

from .importance import ScoredTurn, estimate_turn_tokens, extract_text, score_turn
from .settings import CompactionSettings

MEMORY_MARKER = "<<<SYLLIPTOR_CONVERSATION_MEMORY_JSON>>>"
PINS_MARKER = "<<<SYLLIPTOR_CONVERSATION_PINS_JSON>>>"
CompactionProfileName = Literal["chat", "execution"]


@dataclass(frozen=True)
class _ChunkPlan:
    start: int
    end: int
    scored_turns: list[ScoredTurn]
    strategy: str


@dataclass(frozen=True)
class _ExecutionBundle:
    start: int
    end: int


@dataclass(frozen=True)
class _ExecutionCompactionPreview:
    updated_messages: list[dict[str, Any]]
    summary: dict[str, Any]
    pins: list[dict[str, Any]]
    memory_message_index: int | None
    pins_message_index: int | None
    predicted_used_tokens: int
    dropped_without_summary: bool


@dataclass(frozen=True)
class _ExecutionArtifactCommit:
    history_path: Path
    history_chunk_index: int


@dataclass(frozen=True)
class _CompactionProfile:
    name: CompactionProfileName
    selection_mode: str
    preserve_first_user_turn: bool = False
    recent_raw_tail_messages: int = 0


def _resolve_compaction_profile(
    *,
    profile: CompactionProfileName,
    settings: CompactionSettings,
) -> _CompactionProfile:
    if profile == "execution":
        recent_tail = min(max(8, settings.max_chunk_messages // 3), 24)
        return _CompactionProfile(
            name="execution",
            selection_mode="execution_activity",
            preserve_first_user_turn=True,
            recent_raw_tail_messages=recent_tail,
        )
    return _CompactionProfile(
        name="chat",
        selection_mode="user_turns",
        preserve_first_user_turn=False,
        recent_raw_tail_messages=0,
    )


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx, ch in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _normalize_string_list(value: Any, *, max_items: int = 10) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= max_items:
            break
    return out


def _normalize_work_done(value: Any, *, max_items: int = 10) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            entry = {
                "summary": str(item.get("summary") or "").strip(),
                "files": _normalize_string_list(item.get("files"), max_items=10),
                "commands": _normalize_string_list(item.get("commands"), max_items=10),
                "results": str(item.get("results") or "").strip(),
            }
        else:
            summary = str(item).strip()
            if not summary:
                continue
            entry = {"summary": summary, "files": [], "commands": [], "results": ""}
        if not entry["summary"]:
            continue
        out.append(entry)
        if len(out) >= max_items:
            break
    return out


def _merge_string_lists(
    new_value: Any,
    prev_value: Any,
    *,
    max_items: int = 10,
) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    candidates = [
        *_normalize_string_list(new_value, max_items=max_items),
        *_normalize_string_list(prev_value, max_items=max_items),
    ]
    for item in candidates:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= max_items:
            break
    return merged


def _merge_work_done(
    new_value: Any,
    prev_value: Any,
    *,
    max_items: int = 10,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    candidates = [
        *_normalize_work_done(new_value, max_items=max_items),
        *_normalize_work_done(prev_value, max_items=max_items),
    ]
    for item in candidates:
        summary = str(item.get("summary") or "").strip()
        if not summary:
            continue
        key = summary.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= max_items:
            break
    return merged


def _normalize_summary(summary: Any, previous: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(summary, dict):
        return None

    goal = str(summary.get("goal") or "").strip()
    if not goal:
        goal = str(previous.get("goal") or "").strip()

    normalized = {
        "goal": goal,
        "constraints": _merge_string_lists(
            summary.get("constraints"),
            previous.get("constraints"),
            max_items=10,
        ),
        "decisions": _merge_string_lists(
            summary.get("decisions"),
            previous.get("decisions"),
            max_items=10,
        ),
        "work_done": _merge_work_done(
            summary.get("work_done"),
            previous.get("work_done"),
            max_items=10,
        ),
        "open_threads": _merge_string_lists(
            summary.get("open_threads"),
            previous.get("open_threads"),
            max_items=10,
        ),
        "next_steps": _merge_string_lists(
            summary.get("next_steps"),
            previous.get("next_steps"),
            max_items=10,
        ),
    }
    return normalized


def _normalize_pin_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def _trim_snippet(text: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    if limit <= 0:
        return ""
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


@dataclass
class CompactionState:
    summary: dict[str, Any]
    history_chunk_index: int
    memory_message_index: int | None
    pinned_prefix_len: int
    pins: list[dict[str, Any]]
    pins_message_index: int | None


class ConversationCompactor:
    def __init__(
        self,
        *,
        root: Path,
        artifact_layout: SessionArtifactLayout,
        store: SessionStore,
        settings: CompactionSettings,
        compactor_client: OpenAICompatClient,
        model_registry: ModelRegistry,
        usage_summary: UsageSummary,
        usage_role: str,
        pinned_prefix_len: int,
        profile: CompactionProfileName = "chat",
    ) -> None:
        self._root = root.resolve()
        self._store = store
        self._settings = settings
        self.compactor_client = compactor_client
        self._model_registry = model_registry
        self._usage_summary = usage_summary
        self._usage_role = usage_role
        self._artifact_layout = artifact_layout
        self._profile = _resolve_compaction_profile(profile=profile, settings=settings)
        self._history_dir = self._artifact_layout.artifact_fs_path("history")
        self._memory_dir = self._artifact_layout.artifact_fs_path("memory")
        self._summary_path = self._memory_dir / "summary.json"
        self._pins_path = self._memory_dir / "pins.json"
        self.state = CompactionState(
            summary={},
            history_chunk_index=0,
            memory_message_index=None,
            pinned_prefix_len=max(0, int(pinned_prefix_len)),
            pins=[],
            pins_message_index=None,
        )

    def _is_memory_message(self, msg: dict[str, Any]) -> bool:
        if str(msg.get("role") or "") != "user":
            return False
        content = msg.get("content")
        if not isinstance(content, str):
            return False
        return content.startswith(MEMORY_MARKER)

    def _is_pins_message(self, msg: dict[str, Any]) -> bool:
        if str(msg.get("role") or "") != "user":
            return False
        content = msg.get("content")
        if not isinstance(content, str):
            return False
        return content.startswith(PINS_MARKER)

    def _history_path(self, chunk_idx: int) -> Path:
        candidate = (self._history_dir / f"chunk_{chunk_idx:04d}.jsonl").resolve()
        candidate.relative_to(self._artifact_layout.filesystem_root.resolve())
        return candidate

    @property
    def profile_name(self) -> CompactionProfileName:
        return self._profile.name

    @property
    def history_dir(self) -> Path:
        return self._history_dir

    @property
    def memory_dir(self) -> Path:
        return self._memory_dir

    @property
    def summary_path(self) -> Path:
        return self._summary_path

    @property
    def pins_path(self) -> Path:
        return self._pins_path

    def artifact_display_reference(self, artifact_path: Path) -> str:
        return self._artifact_layout.display_reference_for_path(
            artifact_path=artifact_path,
            workspace_root=self._root,
        )

    def _write_history_chunk(
        self,
        *,
        chunk_messages: list[dict[str, Any]],
        first_idx: int,
    ) -> Path | None:
        chunk_idx = self.state.history_chunk_index + 1
        history_path = self._history_path(chunk_idx)
        try:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with history_path.open("w", encoding="utf-8") as fh:
                for offset, msg in enumerate(chunk_messages):
                    payload = {
                        "idx": first_idx + offset,
                        "message": strip_provider_metadata_from_message(msg),
                    }
                    fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as exc:
            self._store.append(
                "compaction_warning",
                {
                    "warning": "history_chunk_write_failed",
                    "error": str(exc),
                    "first_idx": first_idx,
                    "count": len(chunk_messages),
                },
            )
            return None

        self.state.history_chunk_index = chunk_idx
        self._store.append(
            "history_chunk_written",
            {
                "path": self.artifact_display_reference(history_path),
                "count": len(chunk_messages),
                "first_idx": first_idx,
                "last_idx": first_idx + max(0, len(chunk_messages) - 1),
                "history_chunk_index": chunk_idx,
            },
        )
        return history_path

    def _serialize_history_chunk(
        self,
        *,
        chunk_messages: list[dict[str, Any]],
        first_idx: int,
    ) -> str:
        rows: list[str] = []
        for offset, msg in enumerate(chunk_messages):
            payload = {
                "idx": first_idx + offset,
                "message": strip_provider_metadata_from_message(msg),
            }
            rows.append(json.dumps(payload, ensure_ascii=False))
        return "\n".join(rows) + ("\n" if rows else "")

    def _stage_artifact_text(
        self,
        *,
        artifact_path: Path,
        contents: str,
        warning: str,
        warning_payload: dict[str, Any] | None = None,
    ) -> Path | None:
        extra_payload = dict(warning_payload or {})
        try:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=artifact_path.parent,
                delete=False,
                prefix=f".{artifact_path.name}.",
                suffix=".tmp",
            ) as fh:
                fh.write(contents)
                staged_path = Path(fh.name)
        except OSError as exc:
            payload = {
                "warning": warning,
                "error": str(exc),
                "path": self.artifact_display_reference(artifact_path),
            }
            payload.update(extra_payload)
            self._store.append("compaction_warning", payload)
            return None
        return staged_path

    def _publish_staged_artifact(
        self,
        *,
        staged_path: Path,
        artifact_path: Path,
        warning: str,
    ) -> bool:
        try:
            staged_path.replace(artifact_path)
        except OSError as exc:
            self._store.append(
                "compaction_warning",
                {
                    "warning": warning,
                    "error": str(exc),
                    "path": self.artifact_display_reference(artifact_path),
                },
            )
            return False
        return True

    def _restore_execution_artifact(
        self,
        *,
        artifact_path: Path,
        previous_bytes: bytes | None,
    ) -> None:
        try:
            if previous_bytes is None:
                artifact_path.unlink(missing_ok=True)
                return
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(previous_bytes)
        except OSError as exc:
            self._store.append(
                "compaction_warning",
                {
                    "warning": "execution_compaction_artifact_rollback_failed",
                    "error": str(exc),
                    "path": self.artifact_display_reference(artifact_path),
                },
            )

    def _cleanup_staged_artifact(self, staged_path: Path | None) -> None:
        if staged_path is None:
            return
        try:
            staged_path.unlink(missing_ok=True)
        except OSError as exc:
            self._store.append(
                "compaction_warning",
                {
                    "warning": "execution_compaction_temp_cleanup_failed",
                    "error": str(exc),
                    "path": self.artifact_display_reference(staged_path),
                },
            )

    def _read_existing_artifact_bytes(
        self,
        *,
        artifact_path: Path,
        warning: str,
    ) -> tuple[bool, bytes | None]:
        try:
            if not artifact_path.exists():
                return True, None
            return True, artifact_path.read_bytes()
        except OSError as exc:
            self._store.append(
                "compaction_warning",
                {
                    "warning": warning,
                    "error": str(exc),
                    "path": self.artifact_display_reference(artifact_path),
                },
            )
            return False, None

    def _commit_execution_artifacts(
        self,
        *,
        chunk_messages: list[dict[str, Any]],
        first_idx: int,
        summary: dict[str, Any],
        pins: list[dict[str, Any]],
    ) -> _ExecutionArtifactCommit | None:
        next_chunk_index = self.state.history_chunk_index + 1
        history_path = self._history_path(next_chunk_index)
        history_text = self._serialize_history_chunk(
            chunk_messages=chunk_messages,
            first_idx=first_idx,
        )
        summary_text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        pins_text = None
        if pins:
            payload = {"pins": [self._public_pin(pin) for pin in pins]}
            pins_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

        staged_history = self._stage_artifact_text(
            artifact_path=history_path,
            contents=history_text,
            warning="history_chunk_write_failed",
            warning_payload={
                "first_idx": first_idx,
                "count": len(chunk_messages),
            },
        )
        if staged_history is None:
            return None
        staged_summary = self._stage_artifact_text(
            artifact_path=self._summary_path,
            contents=summary_text,
            warning="summary_write_failed",
        )
        if staged_summary is None:
            self._cleanup_staged_artifact(staged_history)
            return None

        staged_pins: Path | None = None
        if pins_text is not None:
            staged_pins = self._stage_artifact_text(
                artifact_path=self._pins_path,
                contents=pins_text,
                warning="pins_write_failed",
            )
            if staged_pins is None:
                self._cleanup_staged_artifact(staged_history)
                self._cleanup_staged_artifact(staged_summary)
                return None

        summary_ok, previous_summary = self._read_existing_artifact_bytes(
            artifact_path=self._summary_path,
            warning="summary_write_failed",
        )
        if not summary_ok:
            self._cleanup_staged_artifact(staged_history)
            self._cleanup_staged_artifact(staged_summary)
            self._cleanup_staged_artifact(staged_pins)
            return None
        pins_ok, previous_pins = self._read_existing_artifact_bytes(
            artifact_path=self._pins_path,
            warning="pins_write_failed",
        )
        if not pins_ok:
            self._cleanup_staged_artifact(staged_history)
            self._cleanup_staged_artifact(staged_summary)
            self._cleanup_staged_artifact(staged_pins)
            return None
        published_history = False
        published_summary = False
        published_pins = False

        # Execution compaction publishes artifacts as a single transaction.
        # Safety-critical: if any publish step fails, restore prior files and
        # discard the new history chunk so the committed artifacts stay aligned
        # with the active in-memory conversation state.
        try:
            if not self._publish_staged_artifact(
                staged_path=staged_history,
                artifact_path=history_path,
                warning="history_chunk_write_failed",
            ):
                return None
            published_history = True
            staged_history = None

            if not self._publish_staged_artifact(
                staged_path=staged_summary,
                artifact_path=self._summary_path,
                warning="summary_write_failed",
            ):
                self._restore_execution_artifact(
                    artifact_path=history_path,
                    previous_bytes=None,
                )
                return None
            published_summary = True
            staged_summary = None

            if staged_pins is not None:
                if not self._publish_staged_artifact(
                    staged_path=staged_pins,
                    artifact_path=self._pins_path,
                    warning="pins_write_failed",
                ):
                    self._restore_execution_artifact(
                        artifact_path=history_path,
                        previous_bytes=None,
                    )
                    self._restore_execution_artifact(
                        artifact_path=self._summary_path,
                        previous_bytes=previous_summary,
                    )
                    return None
                published_pins = True
                staged_pins = None
        finally:
            self._cleanup_staged_artifact(staged_history)
            self._cleanup_staged_artifact(staged_summary)
            self._cleanup_staged_artifact(staged_pins)

        if (
            not published_history
            or not published_summary
            or (pins_text is not None and not published_pins)
        ):
            self._restore_execution_artifact(
                artifact_path=history_path,
                previous_bytes=None,
            )
            self._restore_execution_artifact(
                artifact_path=self._summary_path,
                previous_bytes=previous_summary,
            )
            if pins_text is not None:
                self._restore_execution_artifact(
                    artifact_path=self._pins_path,
                    previous_bytes=previous_pins,
                )
            self._store.append(
                "compaction_warning",
                {
                    "warning": "execution_compaction_artifact_commit_failed",
                    "history_path": self.artifact_display_reference(history_path),
                },
            )
            return None

        self._store.append(
            "history_chunk_written",
            {
                "path": self.artifact_display_reference(history_path),
                "count": len(chunk_messages),
                "first_idx": first_idx,
                "last_idx": first_idx + max(0, len(chunk_messages) - 1),
                "history_chunk_index": next_chunk_index,
            },
        )
        return _ExecutionArtifactCommit(
            history_path=history_path,
            history_chunk_index=next_chunk_index,
        )

    def _write_summary_file(self, summary: dict[str, Any]) -> None:
        try:
            self._memory_dir.mkdir(parents=True, exist_ok=True)
            self._summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            self._store.append(
                "compaction_warning",
                {
                    "warning": "summary_write_failed",
                    "error": str(exc),
                    "path": self.artifact_display_reference(self._summary_path),
                },
            )

    def _write_pins_file(self, pins: list[dict[str, Any]]) -> None:
        payload = {"pins": [self._public_pin(pin) for pin in pins]}
        try:
            self._memory_dir.mkdir(parents=True, exist_ok=True)
            self._pins_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            self._store.append(
                "compaction_warning",
                {
                    "warning": "pins_write_failed",
                    "error": str(exc),
                    "path": self.artifact_display_reference(self._pins_path),
                },
            )

    def _build_compactor_messages(
        self,
        *,
        existing_summary: dict[str, Any],
        prepared_chunk_messages: list[dict[str, Any]],
        focus: str | None,
    ) -> list[dict[str, str]]:
        payload = {
            "existing_summary": existing_summary,
            "new_messages": prepared_chunk_messages,
            "focus": (focus or "").strip() or None,
        }
        return [
            {
                "role": "system",
                "content": (
                    "You maintain compact conversation memory for a coding agent. "
                    "Return STRICT JSON only (no markdown). "
                    "Schema keys: goal, constraints, decisions, work_done, "
                    "open_threads, next_steps. "
                    "Keep entries concise, deduplicated, and high-signal only. "
                    "Preserve existing summary details unless contradicted."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]

    def _compactor_request_budget(self, *, ratio: float = 0.85) -> int:
        model_meta = self._model_registry.get(self.compactor_client.model)
        budget = compute_input_budget(
            model_meta,
            safety_margin=self._settings.safety_margin_tokens,
        )
        safe_ratio = min(0.95, max(0.1, float(ratio)))
        return max(256, int(budget * safe_ratio))

    def _trim_message_content_to_tokens(self, content: Any, token_budget: int) -> Any:
        if token_budget <= 0:
            return ""
        if isinstance(content, str):
            trimmed, _ = trim_text_to_budget(content, token_budget)
            return trimmed
        if isinstance(content, list):
            out: list[Any] = []
            parts = [part for part in content]
            per_part_budget = max(16, token_budget // max(1, len(parts)))
            for part in parts:
                if not isinstance(part, dict):
                    out.append(part)
                    continue
                copied = dict(part)
                if copied.get("type") == "text":
                    text = str(copied.get("text") or "")
                    trimmed_text, _ = trim_text_to_budget(text, per_part_budget)
                    copied["text"] = trimmed_text
                elif copied.get("type") == "image_url":
                    copied["image_url"] = {"url": "<image>"}
                out.append(copied)
            return out
        return content

    def _prepare_chunk_messages_for_compactor(
        self,
        chunk_messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        prepared = sanitize_messages_for_estimation(chunk_messages)
        if max_tokens <= 0:
            return prepared

        payload = json.dumps(prepared, ensure_ascii=False, sort_keys=True)
        if estimate_tokens(payload) <= max_tokens:
            return prepared

        working = [dict(msg) for msg in prepared]
        rounds = 3
        for _ in range(rounds):
            current = json.dumps(working, ensure_ascii=False, sort_keys=True)
            current_tokens = estimate_tokens(current)
            if current_tokens <= max_tokens:
                break
            per_message_budget = max(32, max_tokens // max(1, len(working)))
            next_messages: list[dict[str, Any]] = []
            for msg in working:
                copied = dict(msg)
                copied["content"] = self._trim_message_content_to_tokens(
                    copied.get("content"),
                    per_message_budget,
                )
                next_messages.append(copied)
            working = next_messages
        return working

    def _apply_llm_importance_overrides(
        self,
        scored_turns: list[ScoredTurn],
    ) -> list[ScoredTurn]:
        if not self._settings.importance_use_llm:
            return scored_turns
        if len(scored_turns) > self._settings.importance_llm_max_turns:
            self._store.append(
                "compaction_warning",
                {
                    "warning": "importance_llm_skipped_too_many_turns",
                    "turn_count": len(scored_turns),
                    "max_turns": self._settings.importance_llm_max_turns,
                },
            )
            return scored_turns

        request_payload = {
            "turns": [
                {
                    "start": turn.start,
                    "end": turn.end,
                    "token_estimate": turn.token_estimate,
                    "score": turn.score,
                    "reasons": turn.reasons,
                    "preview": turn.user_preview,
                }
                for turn in scored_turns
            ]
        }
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "Rescore turn importance. Return JSON only as "
                    '{"scores":[{"start":int,"end":int,"score":float}]}. '
                    "Higher means more important."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(request_payload, ensure_ascii=False),
            },
        ]

        try:
            response = self.compactor_client.chat(
                messages=prompt_messages,
                tools=None,
                stream=False,
            )
        except LLMError as exc:
            self._store.append(
                "compaction_warning",
                {"warning": "importance_llm_error", "error": str(exc)},
            )
            return scored_turns

        usage = response.usage
        usage_record = build_usage_record(
            role=f"{self._usage_role}:compactor",
            requested_model=self.compactor_client.model,
            response_model=response.response_model,
            messages=prompt_messages,
            response_content=response.content or "",
            response_tool_calls=[],
            api_prompt_tokens=(usage.prompt_tokens if usage else None),
            api_completion_tokens=(usage.completion_tokens if usage else None),
            api_total_tokens=(usage.total_tokens if usage else None),
            api_cached_prompt_tokens=(usage.cached_prompt_tokens if usage else None),
            registry=self._model_registry,
        )
        self._usage_summary.add_record(usage_record)
        self._store.append("llm_usage", usage_record.to_payload())

        text = (response.content or "").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            extracted = _extract_first_json_object(text)
            if extracted is None:
                self._store.append(
                    "compaction_warning",
                    {"warning": "importance_llm_invalid_json"},
                )
                return scored_turns
            try:
                parsed = json.loads(extracted)
            except json.JSONDecodeError:
                self._store.append(
                    "compaction_warning",
                    {"warning": "importance_llm_invalid_json"},
                )
                return scored_turns

        if not isinstance(parsed, dict):
            return scored_turns
        raw_scores = parsed.get("scores")
        if not isinstance(raw_scores, list):
            return scored_turns

        overrides: dict[tuple[int, int], float] = {}
        for row in raw_scores:
            if not isinstance(row, dict):
                continue
            try:
                start = int(row.get("start"))
                end = int(row.get("end"))
                score = float(row.get("score"))
            except (TypeError, ValueError):
                continue
            if score < 0:
                continue
            overrides[(start, end)] = score

        if not overrides:
            return scored_turns

        updated: list[ScoredTurn] = []
        for turn in scored_turns:
            override_score = overrides.get((turn.start, turn.end))
            if override_score is None:
                updated.append(turn)
                continue
            reasons = list(turn.reasons)
            if "llm_importance_override" not in reasons:
                reasons.append("llm_importance_override")
            updated.append(
                ScoredTurn(
                    start=turn.start,
                    end=turn.end,
                    token_estimate=turn.token_estimate,
                    score=override_score,
                    density=override_score / max(1, turn.token_estimate),
                    reasons=reasons,
                    user_preview=turn.user_preview,
                )
            )
        return updated

    def _call_compactor(
        self,
        *,
        prompt_messages: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        try:
            response = self.compactor_client.chat(
                messages=prompt_messages,
                tools=None,
                stream=False,
            )
        except LLMError as exc:
            self._store.append(
                "compaction_warning",
                {"warning": "compactor_llm_error", "error": str(exc)},
            )
            return None

        usage = response.usage
        usage_record = build_usage_record(
            role=f"{self._usage_role}:compactor",
            requested_model=self.compactor_client.model,
            response_model=response.response_model,
            messages=prompt_messages,
            response_content=response.content or "",
            response_tool_calls=[],
            api_prompt_tokens=(usage.prompt_tokens if usage else None),
            api_completion_tokens=(usage.completion_tokens if usage else None),
            api_total_tokens=(usage.total_tokens if usage else None),
            api_cached_prompt_tokens=(usage.cached_prompt_tokens if usage else None),
            registry=self._model_registry,
        )
        self._usage_summary.add_record(usage_record)
        self._store.append("llm_usage", usage_record.to_payload())

        text = (response.content or "").strip()
        parsed: Any
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            extracted = _extract_first_json_object(text)
            if extracted is None:
                self._store.append(
                    "compaction_warning",
                    {"warning": "compactor_invalid_json", "preview": text[:4000]},
                )
                return None
            try:
                parsed = json.loads(extracted)
            except json.JSONDecodeError:
                self._store.append(
                    "compaction_warning",
                    {"warning": "compactor_invalid_json", "preview": text[:4000]},
                )
                return None

        normalized = _normalize_summary(parsed, self.state.summary)
        if normalized is None:
            self._store.append(
                "compaction_warning",
                {"warning": "compactor_json_not_object"},
            )
            return None
        return normalized

    def _memory_message(self, summary: dict[str, Any]) -> dict[str, str]:
        payload = json.dumps(summary, separators=(",", ":"), ensure_ascii=False)
        return {"role": "user", "content": f"{MEMORY_MARKER}\n{payload}"}

    def _public_pin(self, pin: dict[str, Any]) -> dict[str, Any]:
        return {
            "kind": str(pin.get("kind") or "context"),
            "text": str(pin.get("text") or ""),
            "reasons": [str(r) for r in pin.get("reasons", []) if str(r).strip()],
            "source": pin.get("source") if isinstance(pin.get("source"), dict) else {},
        }

    def _pins_message(self, pins: list[dict[str, Any]]) -> dict[str, str]:
        payload = {"pins": [self._public_pin(pin) for pin in pins]}
        compact = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        return {"role": "user", "content": f"{PINS_MARKER}\n{compact}"}

    def _upsert_context_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        summary: dict[str, Any],
        pins: list[dict[str, Any]],
        state: CompactionState | None = None,
    ) -> list[dict[str, Any]]:
        target_state = self.state if state is None else state
        filtered: list[dict[str, Any]] = []
        for msg in messages:
            if self._is_memory_message(msg):
                continue
            if self._is_pins_message(msg):
                continue
            filtered.append(msg)

        insert_at = min(target_state.pinned_prefix_len, len(filtered))
        target_state.pins_message_index = None
        target_state.memory_message_index = None

        if pins:
            filtered.insert(insert_at, self._pins_message(pins))
            target_state.pins_message_index = insert_at
            insert_at += 1
        filtered.insert(insert_at, self._memory_message(summary))
        target_state.memory_message_index = insert_at
        return filtered

    def _turn_ranges(self, messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
        turn_ranges: list[tuple[int, int]] = []
        start_idx: int | None = None
        scan_start = min(self.state.pinned_prefix_len, len(messages))
        for idx in range(scan_start, len(messages)):
            msg = messages[idx]
            if self._is_memory_message(msg) or self._is_pins_message(msg):
                continue
            if str(msg.get("role") or "") != "user":
                continue
            if start_idx is not None:
                turn_ranges.append((start_idx, idx))
            start_idx = idx
        if start_idx is not None:
            turn_ranges.append((start_idx, len(messages)))
        return turn_ranges

    def _range_preview(
        self, range_messages: list[dict[str, Any]], fallback_limit: int = 200
    ) -> str:
        for msg in range_messages:
            preview = _trim_snippet(extract_text(msg), fallback_limit)
            if preview:
                return preview
        return ""

    def _score_ranges(
        self,
        messages: list[dict[str, Any]],
        ranges: list[tuple[int, int]],
    ) -> list[ScoredTurn]:
        scored: list[ScoredTurn] = []
        for start, end in ranges:
            range_messages = messages[start:end]
            score, reasons, preview = score_turn(range_messages)
            if not preview:
                preview = self._range_preview(range_messages)
            token_estimate = estimate_turn_tokens(range_messages)
            density = score / max(1, token_estimate)
            scored.append(
                ScoredTurn(
                    start=start,
                    end=end,
                    token_estimate=token_estimate,
                    score=score,
                    density=density,
                    reasons=reasons,
                    user_preview=preview,
                )
            )
        return self._apply_llm_importance_overrides(scored)

    def _assistant_has_tool_calls(self, msg: dict[str, Any]) -> bool:
        tool_calls = msg.get("tool_calls")
        return isinstance(tool_calls, list) and len(tool_calls) > 0

    def _first_execution_user_idx(self, messages: list[dict[str, Any]]) -> int | None:
        scan_start = min(self.state.pinned_prefix_len, len(messages))
        for idx in range(scan_start, len(messages)):
            msg = messages[idx]
            if self._is_memory_message(msg) or self._is_pins_message(msg):
                continue
            if str(msg.get("role") or "") == "user":
                return idx
        return None

    def _assistant_tool_call_ids(self, msg: dict[str, Any]) -> list[str]:
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            return []
        ids: list[str] = []
        seen: set[str] = set()
        for item in tool_calls:
            if not isinstance(item, dict):
                return []
            call_id = str(item.get("id") or "").strip()
            if not call_id or call_id in seen:
                return []
            seen.add(call_id)
            ids.append(call_id)
        return ids

    def _execution_sequence_end(self, messages: list[dict[str, Any]], *, scan_start: int) -> int:
        for idx in range(scan_start, len(messages)):
            msg = messages[idx]
            if self._is_memory_message(msg) or self._is_pins_message(msg):
                continue
            if str(msg.get("role") or "") == "user":
                return idx
        return len(messages)

    def _build_execution_tool_bundle(
        self,
        messages: list[dict[str, Any]],
        *,
        start: int,
        sequence_end: int,
    ) -> _ExecutionBundle | None:
        assistant_msg = messages[start]
        expected_tool_call_ids = set(self._assistant_tool_call_ids(assistant_msg))
        if not expected_tool_call_ids:
            return None

        seen_tool_call_ids: set[str] = set()
        cursor = start + 1
        # Execution compaction must treat an assistant tool call, its tool results,
        # and the immediate consuming assistant response as one atomic bundle.
        while cursor < sequence_end:
            msg = messages[cursor]
            if self._is_memory_message(msg) or self._is_pins_message(msg):
                return None
            if str(msg.get("role") or "") != "tool":
                break
            tool_call_id = str(msg.get("tool_call_id") or "").strip()
            if (
                not tool_call_id
                or tool_call_id not in expected_tool_call_ids
                or tool_call_id in seen_tool_call_ids
            ):
                return None
            seen_tool_call_ids.add(tool_call_id)
            cursor += 1

        if seen_tool_call_ids != expected_tool_call_ids:
            return None

        if cursor < sequence_end:
            follow_up = messages[cursor]
            if self._is_memory_message(follow_up) or self._is_pins_message(follow_up):
                return None
            if str(
                follow_up.get("role") or ""
            ) == "assistant" and not self._assistant_has_tool_calls(follow_up):
                cursor += 1

        return _ExecutionBundle(start=start, end=cursor)

    def _build_execution_bundles(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[_ExecutionBundle], int, int]:
        first_user_idx = self._first_execution_user_idx(messages)
        if first_user_idx is None:
            return [], len(messages), len(messages)

        scan_start = (
            first_user_idx + 1 if self._profile.preserve_first_user_turn else first_user_idx
        )
        sequence_end = self._execution_sequence_end(messages, scan_start=scan_start)
        bundles: list[_ExecutionBundle] = []
        idx = scan_start
        while idx < sequence_end:
            msg = messages[idx]
            if self._is_memory_message(msg) or self._is_pins_message(msg):
                idx += 1
                continue

            role = str(msg.get("role") or "")
            if role == "tool":
                return bundles, idx, sequence_end
            if role == "assistant" and self._assistant_has_tool_calls(msg):
                bundle = self._build_execution_tool_bundle(
                    messages,
                    start=idx,
                    sequence_end=sequence_end,
                )
                if bundle is None:
                    return bundles, idx, sequence_end
                bundles.append(bundle)
                idx = bundle.end
                continue
            bundles.append(_ExecutionBundle(start=idx, end=idx + 1))
            idx += 1
        return bundles, sequence_end, sequence_end

    def _execution_tail_bundle_index(self, bundles: list[_ExecutionBundle]) -> int:
        keep_messages = self._profile.recent_raw_tail_messages
        if keep_messages <= 0:
            return len(bundles)
        kept = 0
        for idx in range(len(bundles) - 1, -1, -1):
            kept += max(0, bundles[idx].end - bundles[idx].start)
            if kept >= keep_messages:
                return idx
        return 0

    def _has_valid_tool_transcript(self, messages: list[dict[str, Any]]) -> bool:
        open_tool_call_ids: set[str] = set()
        for msg in messages:
            if self._is_memory_message(msg) or self._is_pins_message(msg):
                continue
            role = str(msg.get("role") or "")
            if role == "assistant":
                if self._assistant_has_tool_calls(msg):
                    tool_call_ids = self._assistant_tool_call_ids(msg)
                    if open_tool_call_ids or not tool_call_ids:
                        return False
                    open_tool_call_ids = set(tool_call_ids)
                    continue
                if open_tool_call_ids:
                    return False
                continue
            if role == "tool":
                tool_call_id = str(msg.get("tool_call_id") or "").strip()
                if not tool_call_id or tool_call_id not in open_tool_call_ids:
                    return False
                open_tool_call_ids.remove(tool_call_id)
                continue
            if open_tool_call_ids:
                return False
        return not open_tool_call_ids

    def _clone_state(self) -> CompactionState:
        return CompactionState(
            summary=deepcopy(self.state.summary),
            history_chunk_index=self.state.history_chunk_index,
            memory_message_index=self.state.memory_message_index,
            pinned_prefix_len=self.state.pinned_prefix_len,
            pins=deepcopy(self.state.pins),
            pins_message_index=self.state.pins_message_index,
        )

    def _execution_candidate_stats(
        self,
        *,
        bundles: list[_ExecutionBundle],
        scored_bundles: list[ScoredTurn],
        start_idx: int,
        end_idx: int,
    ) -> tuple[int, int]:
        message_count = 0
        token_estimate = 0
        for idx in range(start_idx, end_idx + 1):
            bundle = bundles[idx]
            message_count += max(0, bundle.end - bundle.start)
            token_estimate += int(scored_bundles[idx].token_estimate)
        return message_count, token_estimate

    def _execution_meets_minimum_removal_thresholds(
        self,
        *,
        message_count: int,
        token_estimate: int,
    ) -> bool:
        return (
            message_count >= self._settings.execution_min_removable_messages
            and token_estimate >= self._settings.execution_min_removable_tokens
        )

    def _expand_execution_window_to_minimums(
        self,
        *,
        bundles: list[_ExecutionBundle],
        scored_bundles: list[ScoredTurn],
        start_idx: int,
        end_idx: int,
    ) -> tuple[int, int] | None:
        left = start_idx
        right = end_idx
        while True:
            message_count, token_estimate = self._execution_candidate_stats(
                bundles=bundles,
                scored_bundles=scored_bundles,
                start_idx=left,
                end_idx=right,
            )
            if self._execution_meets_minimum_removal_thresholds(
                message_count=message_count,
                token_estimate=token_estimate,
            ):
                return left, right
            if left > 0:
                left -= 1
                continue
            if right + 1 < len(bundles):
                right += 1
                continue
            return None

    def _oldest_execution_window(
        self,
        *,
        bundles: list[_ExecutionBundle],
        scored_bundles: list[ScoredTurn],
    ) -> tuple[int, int] | None:
        if not bundles:
            return None
        right = 0
        while True:
            message_count, token_estimate = self._execution_candidate_stats(
                bundles=bundles,
                scored_bundles=scored_bundles,
                start_idx=0,
                end_idx=right,
            )
            if self._execution_meets_minimum_removal_thresholds(
                message_count=message_count,
                token_estimate=token_estimate,
            ):
                break
            if right + 1 >= len(bundles):
                return None
            right += 1

        while right + 1 < len(bundles):
            next_message_count, _ = self._execution_candidate_stats(
                bundles=bundles,
                scored_bundles=scored_bundles,
                start_idx=0,
                end_idx=right + 1,
            )
            if next_message_count > self._settings.max_chunk_messages:
                break
            right += 1
        return 0, right

    def _stage_execution_compaction_preview(
        self,
        *,
        working: list[dict[str, Any]],
        chunk_plan: _ChunkPlan,
        chunk_messages: list[dict[str, Any]],
        tool_list: list[dict[str, Any]] | None,
        used_tokens: int,
        focus: str | None,
        history_rel_path: str,
    ) -> _ExecutionCompactionPreview | None:
        compactor_budget = self._compactor_request_budget(ratio=0.85)
        prepared_chunk = self._prepare_chunk_messages_for_compactor(
            chunk_messages,
            compactor_budget,
        )
        prompt_messages = self._build_compactor_messages(
            existing_summary=self.state.summary,
            prepared_chunk_messages=prepared_chunk,
            focus=focus,
        )
        new_summary = self._call_compactor(prompt_messages=prompt_messages)

        if new_summary is None:
            retry_budget = self._compactor_request_budget(ratio=0.60)
            retry_chunk = self._prepare_chunk_messages_for_compactor(
                chunk_messages,
                retry_budget,
            )
            retry_prompt_messages = self._build_compactor_messages(
                existing_summary=self.state.summary,
                prepared_chunk_messages=retry_chunk,
                focus=focus,
            )
            new_summary = self._call_compactor(prompt_messages=retry_prompt_messages)

        dropped_without_summary = False
        if new_summary is None:
            dropped_without_summary = True
            new_summary = deepcopy(self.state.summary)

        extracted_pins = self._extract_pins_for_chunk(
            history_rel_path=history_rel_path,
            scored_turns=chunk_plan.scored_turns,
        )
        staged_pins = deepcopy(self.state.pins)
        if extracted_pins:
            staged_pins = self._bounded_pins([*staged_pins, *extracted_pins])

        # Stage the execution compaction in memory first. Safety-critical:
        # do not mutate the active state or write artifacts until we prove the
        # new request is smaller than the current one.
        trial_state = self._clone_state()
        trial_state.summary = deepcopy(new_summary)
        trial_state.pins = deepcopy(staged_pins)
        reduced_messages = working[: chunk_plan.start] + working[chunk_plan.end :]
        updated_messages = self._upsert_context_messages(
            reduced_messages,
            summary=trial_state.summary,
            pins=trial_state.pins,
            state=trial_state,
        )
        predicted_used_tokens = estimate_request_tokens(updated_messages, tool_list)
        if predicted_used_tokens >= used_tokens:
            removable_tokens = estimate_turn_tokens(chunk_messages)
            self._store.append(
                "compaction_warning",
                {
                    "warning": "compaction_no_progress",
                    "used_tokens_before": used_tokens,
                    "used_tokens_after": predicted_used_tokens,
                    "chunk_strategy": chunk_plan.strategy,
                    "removable_messages": len(chunk_messages),
                    "removable_estimated_tokens": removable_tokens,
                    "history_path": history_rel_path,
                },
            )
            return None

        return _ExecutionCompactionPreview(
            updated_messages=updated_messages,
            summary=trial_state.summary,
            pins=trial_state.pins,
            memory_message_index=trial_state.memory_message_index,
            pins_message_index=trial_state.pins_message_index,
            predicted_used_tokens=predicted_used_tokens,
            dropped_without_summary=dropped_without_summary,
        )

    def _build_chat_chunk_plan(self, messages: list[dict[str, Any]]) -> _ChunkPlan | None:
        turns = self._turn_ranges(messages)
        keep_recent = self._settings.recent_user_turns_to_keep
        if len(turns) <= keep_recent:
            return None
        eligible_turns = turns[:-keep_recent] if keep_recent > 0 else turns
        if not eligible_turns:
            return None

        scored = self._score_ranges(messages, eligible_turns)
        if scored:
            min_density = min(turn.density for turn in scored)
            max_density = max(turn.density for turn in scored)
            self._store.append(
                "importance_scored",
                {
                    "eligible_turns": len(scored),
                    "min_density": min_density,
                    "max_density": max_density,
                    "strategy": self._settings.importance_strategy,
                },
            )

        if (
            self._settings.importance_enabled
            and self._settings.importance_strategy == "lowest_density"
        ):
            if not scored:
                return None
            selected = min(scored, key=lambda turn: (turn.density, turn.start))
            return _ChunkPlan(
                start=selected.start,
                end=selected.end,
                scored_turns=[selected],
                strategy="lowest_density",
            )

        chunk_start = eligible_turns[0][0]
        chunk_end = chunk_start
        consumed = 0
        selected_ranges: list[tuple[int, int]] = []
        for turn_start, turn_end in eligible_turns:
            turn_size = max(0, turn_end - turn_start)
            if consumed > 0 and consumed + turn_size > self._settings.max_chunk_messages:
                break
            chunk_end = turn_end
            consumed += turn_size
            selected_ranges.append((turn_start, turn_end))
            if consumed >= self._settings.max_chunk_messages:
                break
        if chunk_end <= chunk_start:
            return None

        selected_scored: list[ScoredTurn] = []
        selected_set = {(start, end) for start, end in selected_ranges}
        for turn in scored:
            if (turn.start, turn.end) in selected_set:
                selected_scored.append(turn)

        return _ChunkPlan(
            start=chunk_start,
            end=chunk_end,
            scored_turns=selected_scored,
            strategy="oldest",
        )

    def _build_execution_chunk_plan_for_window(
        self,
        *,
        messages: list[dict[str, Any]],
        bundles: list[_ExecutionBundle],
        scored_bundles: list[ScoredTurn],
        start_idx: int,
        end_idx: int,
        strategy: str,
    ) -> _ChunkPlan | None:
        selected_bundles = bundles[start_idx : end_idx + 1]
        selected_scored = scored_bundles[start_idx : end_idx + 1]
        chunk_plan = _ChunkPlan(
            start=selected_bundles[0].start,
            end=selected_bundles[-1].end,
            scored_turns=selected_scored,
            strategy=strategy,
        )
        reduced_messages = messages[: chunk_plan.start] + messages[chunk_plan.end :]
        if not self._has_valid_tool_transcript(reduced_messages):
            self._store.append(
                "compaction_warning",
                {
                    "warning": "execution_compaction_rejected_invalid_transcript",
                    "chunk_start": chunk_plan.start,
                    "chunk_end": chunk_plan.end,
                    "strategy": chunk_plan.strategy,
                },
            )
            return None
        return chunk_plan

    def _build_execution_chunk_plans(self, messages: list[dict[str, Any]]) -> list[_ChunkPlan]:
        bundles, safe_boundary, sequence_end = self._build_execution_bundles(messages)
        if not bundles:
            return []
        tail_bundle_idx = self._execution_tail_bundle_index(bundles)
        # Preserve the recent execution tail on whole-bundle boundaries so the
        # active conversation never starts in the middle of a tool exchange.
        tail_boundary = (
            bundles[tail_bundle_idx].start if tail_bundle_idx < len(bundles) else safe_boundary
        )
        removable_boundary = min(safe_boundary, tail_boundary)
        eligible_bundles = [bundle for bundle in bundles if bundle.end <= removable_boundary]
        if not eligible_bundles:
            return []

        if safe_boundary < sequence_end:
            self._store.append(
                "compaction_warning",
                {
                    "warning": "execution_compaction_stopped_at_unsafe_bundle_boundary",
                    "blocked_start": safe_boundary,
                },
            )

        ranges = [(bundle.start, bundle.end) for bundle in eligible_bundles]
        scored = self._score_ranges(messages, ranges)
        scored_by_range = {(turn.start, turn.end): turn for turn in scored}
        scored_bundles = [
            scored_by_range[(bundle.start, bundle.end)]
            for bundle in eligible_bundles
            if (bundle.start, bundle.end) in scored_by_range
        ]
        if scored:
            min_density = min(turn.density for turn in scored)
            max_density = max(turn.density for turn in scored)
            self._store.append(
                "importance_scored",
                {
                    "eligible_turns": len(scored),
                    "min_density": min_density,
                    "max_density": max_density,
                    "strategy": f"{self._settings.importance_strategy}:execution",
                },
            )

        candidate_windows: list[tuple[int, int, str]] = []
        if (
            self._settings.importance_enabled
            and self._settings.importance_strategy == "lowest_density"
            and scored
        ):
            anchor_idx = min(
                range(len(scored_bundles)),
                key=lambda idx: (scored_bundles[idx].density, scored_bundles[idx].start),
            )
            selected_window = self._expand_execution_window_to_minimums(
                bundles=eligible_bundles,
                scored_bundles=scored_bundles,
                start_idx=anchor_idx,
                end_idx=anchor_idx,
            )
            if selected_window is not None:
                candidate_windows.append(
                    (selected_window[0], selected_window[1], "execution_lowest_density")
                )
            oldest_window = self._oldest_execution_window(
                bundles=eligible_bundles,
                scored_bundles=scored_bundles,
            )
            if oldest_window is None and not candidate_windows:
                available_messages, available_tokens = self._execution_candidate_stats(
                    bundles=eligible_bundles,
                    scored_bundles=scored_bundles,
                    start_idx=0,
                    end_idx=len(eligible_bundles) - 1,
                )
                self._store.append(
                    "compaction_warning",
                    {
                        "warning": "execution_compaction_below_minimum_removal_threshold",
                        "available_messages": available_messages,
                        "available_estimated_tokens": available_tokens,
                        "required_messages": self._settings.execution_min_removable_messages,
                        "required_tokens": self._settings.execution_min_removable_tokens,
                    },
                )
                return []
            if oldest_window is not None and oldest_window != selected_window:
                candidate_windows.append(
                    (oldest_window[0], oldest_window[1], "execution_oldest_activity")
                )
        else:
            selected_window = self._oldest_execution_window(
                bundles=eligible_bundles,
                scored_bundles=scored_bundles,
            )
            if selected_window is None:
                available_messages, available_tokens = self._execution_candidate_stats(
                    bundles=eligible_bundles,
                    scored_bundles=scored_bundles,
                    start_idx=0,
                    end_idx=len(eligible_bundles) - 1,
                )
                self._store.append(
                    "compaction_warning",
                    {
                        "warning": "execution_compaction_below_minimum_removal_threshold",
                        "available_messages": available_messages,
                        "available_estimated_tokens": available_tokens,
                        "required_messages": self._settings.execution_min_removable_messages,
                        "required_tokens": self._settings.execution_min_removable_tokens,
                    },
                )
                return []
            candidate_windows.append(
                (selected_window[0], selected_window[1], "execution_oldest_activity")
            )

        plans: list[_ChunkPlan] = []
        for start_idx, end_idx, strategy in candidate_windows:
            plan = self._build_execution_chunk_plan_for_window(
                messages=messages,
                bundles=eligible_bundles,
                scored_bundles=scored_bundles,
                start_idx=start_idx,
                end_idx=end_idx,
                strategy=strategy,
            )
            if plan is not None:
                plans.append(plan)
        return plans

    def _build_execution_chunk_plan(self, messages: list[dict[str, Any]]) -> _ChunkPlan | None:
        plans = self._build_execution_chunk_plans(messages)
        if plans:
            return plans[0]
        return None

    def _build_chunk_plan(self, messages: list[dict[str, Any]]) -> _ChunkPlan | None:
        if self._profile.selection_mode == "execution_activity":
            return self._build_execution_chunk_plan(messages)
        return self._build_chat_chunk_plan(messages)

    def _pin_kind(self, turn: ScoredTurn) -> str:
        reasons_cf = {reason.casefold() for reason in turn.reasons}
        preview_cf = turn.user_preview.casefold()
        if "errors_or_failures" in reasons_cf:
            return "error"
        if (
            "requirements_or_constraints" in reasons_cf
            or "acceptance_criteria" in reasons_cf
            or "must" in preview_cf
            or "do not" in preview_cf
            or "never" in preview_cf
        ):
            return "constraint"
        if "shell_or_git_commands" in reasons_cf or "verification_commands" in reasons_cf:
            return "command"
        if "requirements_or_constraints" in reasons_cf:
            return "requirement"
        return "context"

    def _extract_pins_for_chunk(
        self,
        *,
        history_rel_path: str,
        scored_turns: list[ScoredTurn],
    ) -> list[dict[str, Any]]:
        pins: list[dict[str, Any]] = []
        threshold = self._settings.pin_score_threshold
        for turn in scored_turns:
            if turn.score < threshold:
                continue
            snippet = _trim_snippet(turn.user_preview, self._settings.pin_snippet_chars)
            if not snippet:
                continue
            pin = {
                "kind": self._pin_kind(turn),
                "text": snippet,
                "reasons": list(turn.reasons),
                "score": float(turn.score),
                "source": {
                    "history_path": history_rel_path,
                    "idx_range": [turn.start, max(turn.start, turn.end - 1)],
                },
            }
            pins.append(pin)
        return pins

    def _bounded_pins(self, pins: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_text: dict[str, dict[str, Any]] = {}
        for pin in pins:
            text = str(pin.get("text") or "")
            key = _normalize_pin_text(text)
            if not key:
                continue
            existing = by_text.get(key)
            if existing is None:
                by_text[key] = pin
                continue
            existing_score = float(existing.get("score") or 0.0)
            next_score = float(pin.get("score") or 0.0)
            if next_score > existing_score:
                by_text[key] = pin

        ordered = sorted(
            by_text.values(),
            key=lambda pin: float(pin.get("score") or 0.0),
            reverse=True,
        )
        if len(ordered) > self._settings.max_pins:
            ordered = ordered[: self._settings.max_pins]

        while ordered:
            payload = {"pins": [self._public_pin(pin) for pin in ordered]}
            size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            if size <= self._settings.max_pins_chars:
                break
            ordered.pop()
        return ordered

    def _compact_loop(
        self,
        *,
        messages: list[dict[str, Any]],
        tool_list: list[dict[str, Any]] | None,
        main_model: str,
        focus: str | None = None,
        force: bool = False,
    ) -> tuple[list[dict[str, Any]], bool]:
        working = list(messages)
        changed = False
        forced_once = False

        while True:
            model_meta = self._model_registry.get(main_model)
            budget = compute_input_budget(
                model_meta,
                safety_margin=self._settings.safety_margin_tokens,
            )
            used_tokens = estimate_request_tokens(working, tool_list)
            trigger_tokens = int(budget * self._settings.trigger_ratio)
            target_tokens = int(budget * self._settings.target_ratio)

            self._store.append(
                "compaction_check",
                {
                    "used_tokens": used_tokens,
                    "budget_tokens": budget,
                    "trigger_ratio": self._settings.trigger_ratio,
                    "target_ratio": self._settings.target_ratio,
                    "trigger_tokens": trigger_tokens,
                    "target_tokens": target_tokens,
                    "main_model": main_model,
                },
            )

            if not force and used_tokens <= trigger_tokens:
                return working, changed
            if force and forced_once and used_tokens <= target_tokens:
                return working, changed

            execution_chunk_plans: list[_ChunkPlan] | None = None
            chunk_plan: _ChunkPlan | None
            if self._profile.name == "execution":
                execution_chunk_plans = self._build_execution_chunk_plans(working)
                chunk_plan = execution_chunk_plans[0] if execution_chunk_plans else None
            else:
                chunk_plan = self._build_chunk_plan(working)

            if chunk_plan is None:
                self._store.append(
                    "compaction_warning",
                    {
                        "warning": "no_compaction_chunk_available",
                        "used_tokens": used_tokens,
                        "trigger_tokens": trigger_tokens,
                    },
                )
                return working, changed

            if self._profile.name == "execution":
                assert execution_chunk_plans is not None
                next_history_path = self._history_path(self.state.history_chunk_index + 1)
                predicted_history_rel_path = self.artifact_display_reference(next_history_path)
                preview: _ExecutionCompactionPreview | None = None
                selected_plan: _ChunkPlan | None = None
                selected_chunk_messages: list[dict[str, Any]] = []
                for candidate_plan in execution_chunk_plans:
                    candidate_chunk_messages = working[candidate_plan.start : candidate_plan.end]
                    preview = self._stage_execution_compaction_preview(
                        working=working,
                        chunk_plan=candidate_plan,
                        chunk_messages=candidate_chunk_messages,
                        tool_list=tool_list,
                        used_tokens=used_tokens,
                        focus=focus,
                        history_rel_path=predicted_history_rel_path,
                    )
                    if preview is not None:
                        selected_plan = candidate_plan
                        selected_chunk_messages = candidate_chunk_messages
                        break
                if preview is None or selected_plan is None:
                    return working, changed

                artifact_commit = self._commit_execution_artifacts(
                    chunk_messages=selected_chunk_messages,
                    first_idx=selected_plan.start,
                    summary=preview.summary,
                    pins=preview.pins,
                )
                if artifact_commit is None:
                    return working, changed
                history_rel_path = self.artifact_display_reference(artifact_commit.history_path)
                if preview.dropped_without_summary:
                    self._store.append(
                        "compaction_warning",
                        {
                            "warning": "compactor_failed_drop_chunk",
                            "history_path": history_rel_path,
                        },
                    )
                self.state.history_chunk_index = artifact_commit.history_chunk_index
                self.state.summary = preview.summary
                self.state.pins = preview.pins
                self.state.memory_message_index = preview.memory_message_index
                self.state.pins_message_index = preview.pins_message_index
                summary_json = json.dumps(
                    preview.summary,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                self._store.append(
                    "conversation_summary_updated",
                    {
                        "summary_bytes": len(summary_json.encode("utf-8")),
                        "history_chunk_index": self.state.history_chunk_index,
                        "history_path": history_rel_path,
                        "chunk_strategy": selected_plan.strategy,
                        "pins_count": len(self.state.pins),
                        "dropped_without_summary_update": preview.dropped_without_summary,
                    },
                )
                working = preview.updated_messages
                changed = True
                forced_once = True
                new_used_tokens = preview.predicted_used_tokens
                if new_used_tokens <= target_tokens:
                    return working, changed
                continue

            chunk_messages = working[chunk_plan.start : chunk_plan.end]
            next_history_path = self._history_path(self.state.history_chunk_index + 1)
            predicted_history_rel_path = self.artifact_display_reference(next_history_path)
            history_path = self._write_history_chunk(
                chunk_messages=chunk_messages,
                first_idx=chunk_plan.start,
            )
            if history_path is None:
                return working, changed
            history_rel_path = self.artifact_display_reference(history_path)

            compactor_budget = self._compactor_request_budget(ratio=0.85)
            prepared_chunk = self._prepare_chunk_messages_for_compactor(
                chunk_messages,
                compactor_budget,
            )
            prompt_messages = self._build_compactor_messages(
                existing_summary=self.state.summary,
                prepared_chunk_messages=prepared_chunk,
                focus=focus,
            )
            new_summary = self._call_compactor(prompt_messages=prompt_messages)

            if new_summary is None:
                retry_budget = self._compactor_request_budget(ratio=0.60)
                retry_chunk = self._prepare_chunk_messages_for_compactor(
                    chunk_messages,
                    retry_budget,
                )
                retry_prompt_messages = self._build_compactor_messages(
                    existing_summary=self.state.summary,
                    prepared_chunk_messages=retry_chunk,
                    focus=focus,
                )
                new_summary = self._call_compactor(prompt_messages=retry_prompt_messages)

            dropped_without_summary = False
            if new_summary is None:
                dropped_without_summary = True
                self._store.append(
                    "compaction_warning",
                    {
                        "warning": "compactor_failed_drop_chunk",
                        "history_path": history_rel_path,
                    },
                )
                new_summary = self.state.summary

            self.state.summary = new_summary
            self._write_summary_file(new_summary)

            new_pins = self._extract_pins_for_chunk(
                history_rel_path=history_rel_path,
                scored_turns=chunk_plan.scored_turns,
            )
            if new_pins:
                self.state.pins = self._bounded_pins([*self.state.pins, *new_pins])
                self._write_pins_file(self.state.pins)

            reduced_messages = working[: chunk_plan.start] + working[chunk_plan.end :]
            updated_messages = self._upsert_context_messages(
                reduced_messages,
                summary=new_summary,
                pins=self.state.pins,
            )
            summary_json = json.dumps(new_summary, separators=(",", ":"), ensure_ascii=False)
            self._store.append(
                "conversation_summary_updated",
                {
                    "summary_bytes": len(summary_json.encode("utf-8")),
                    "history_chunk_index": self.state.history_chunk_index,
                    "history_path": history_rel_path,
                    "chunk_strategy": chunk_plan.strategy,
                    "pins_count": len(self.state.pins),
                    "dropped_without_summary_update": dropped_without_summary,
                },
            )

            new_used_tokens = estimate_request_tokens(updated_messages, tool_list)
            working = updated_messages
            changed = True
            forced_once = True

            if new_used_tokens >= used_tokens:
                self._store.append(
                    "compaction_warning",
                    {
                        "warning": "compaction_no_progress",
                        "used_tokens_before": used_tokens,
                        "used_tokens_after": new_used_tokens,
                    },
                )
                return working, changed

            if new_used_tokens <= target_tokens:
                return working, changed

    def maybe_compact(
        self,
        *,
        messages: list[dict[str, Any]],
        tool_list: list[dict[str, Any]] | None,
        main_model: str,
        focus: str | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        return self._compact_loop(
            messages=messages,
            tool_list=tool_list,
            main_model=main_model,
            focus=focus,
            force=False,
        )

    def compact_now(
        self,
        *,
        messages: list[dict[str, Any]],
        tool_list: list[dict[str, Any]] | None,
        main_model: str,
        focus: str | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        self._store.append("compaction_forced", {"focus": (focus or "").strip()})
        return self._compact_loop(
            messages=messages,
            tool_list=tool_list,
            main_model=main_model,
            focus=focus,
            force=True,
        )
