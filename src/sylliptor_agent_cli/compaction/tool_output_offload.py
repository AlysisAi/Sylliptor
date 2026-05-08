from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..session_artifacts import SessionArtifactLayout
from ..tools.registry import summarize_tool_output_chunk


@dataclass(frozen=True)
class OffloadResult:
    content_for_message: str
    offloaded: bool
    transcript_shaped: bool
    artifact_locator: str | None
    artifact_fs_path: str | None
    artifact_readable_via_fs: bool
    artifact_location: str | None
    original_chars: int
    preview_chars: int
    message_chars: int
    error: str | None = None


def _safe_component(raw: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", raw.strip())
    return clean or "x"


def _truncated_preview(content: str, limit: int) -> tuple[str, int]:
    if limit <= 0:
        return ("", 0)
    if len(content) <= limit:
        return (content, len(content))
    return (content[:limit] + "...(truncated)", limit)


class ToolOutputOffloader:
    def __init__(
        self,
        *,
        artifact_layout: SessionArtifactLayout,
        workspace_root: Path | None,
        threshold_chars: int,
        preview_chars: int,
    ) -> None:
        self._artifact_layout = artifact_layout
        self._session_artifact_root = artifact_layout.filesystem_root.resolve()
        self._workspace_root = workspace_root.resolve() if workspace_root is not None else None
        self._threshold_chars = max(1, int(threshold_chars))
        self._preview_chars = max(1, int(preview_chars))
        self._base_dir = self._session_artifact_root / "tool_outputs"

    def _transcript_shape_threshold_chars(self) -> int:
        return min(
            self._threshold_chars,
            max((self._preview_chars * 3), self._preview_chars + 256),
        )

    def _build_transcript_stub(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        step: int,
        content_json: str,
        original_chars: int,
        preview_text: str,
        preview_chars: int,
        offloaded: bool,
        transcript_shaped: bool,
        artifact_locator: str | None = None,
        artifact_saved: bool = False,
        artifact_readable_via_fs: bool = False,
        artifact_location: str | None = None,
        error: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "tool": tool_name,
            "tool_call_id": tool_call_id,
            "step": step,
            "offloaded": bool(offloaded),
            "summary": summarize_tool_output_chunk(tool_name, content_json),
            "preview": preview_text,
            "preview_chars": preview_chars,
            "original_chars": original_chars,
            "content_truncated": original_chars > preview_chars,
            "raw_saved_in_session_log": True,
        }
        if transcript_shaped and not offloaded:
            payload["transcript_shaped"] = True
        if offloaded:
            payload["artifact_locator"] = artifact_locator
            payload["artifact_saved"] = artifact_saved
            payload["artifact_readable_via_fs"] = artifact_readable_via_fs
            payload["artifact_location"] = artifact_location
        if error:
            payload["error"] = error
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

    def _artifact_path(self, *, tool_name: str, tool_call_id: str, step: int) -> Path:
        safe_tool = _safe_component(tool_name)
        safe_call_id = _safe_component(tool_call_id)
        filename = f"step{max(0, int(step))}_{safe_tool}_{safe_call_id}.json"
        candidate = self._artifact_layout.artifact_fs_path("tool_outputs", filename).resolve()
        candidate.relative_to(self._session_artifact_root)
        return candidate

    def maybe_offload(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        step: int,
        result: Any,
        content_json: str,
    ) -> OffloadResult:
        original_chars = len(content_json)
        preview_text, preview_chars = _truncated_preview(content_json, self._preview_chars)
        shape_threshold_chars = self._transcript_shape_threshold_chars()
        if original_chars < shape_threshold_chars:
            return OffloadResult(
                content_for_message=content_json,
                offloaded=False,
                transcript_shaped=False,
                artifact_locator=None,
                artifact_fs_path=None,
                artifact_readable_via_fs=False,
                artifact_location=None,
                original_chars=original_chars,
                preview_chars=preview_chars,
                message_chars=original_chars,
                error=None,
            )

        if original_chars < self._threshold_chars:
            content_for_message = self._build_transcript_stub(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                step=step,
                content_json=content_json,
                original_chars=original_chars,
                preview_text=preview_text,
                preview_chars=preview_chars,
                offloaded=False,
                transcript_shaped=True,
            )
            return OffloadResult(
                content_for_message=content_for_message,
                offloaded=False,
                transcript_shaped=True,
                artifact_locator=None,
                artifact_fs_path=None,
                artifact_readable_via_fs=False,
                artifact_location=None,
                original_chars=original_chars,
                preview_chars=preview_chars,
                message_chars=len(content_for_message),
                error=None,
            )

        try:
            artifact_abs = self._artifact_path(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                step=step,
            )
            artifact_abs.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "step": step,
                "result": result,
                "content_json": content_json,
            }
            artifact_abs.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            artifact_ref = self._artifact_layout.model_reference_for_path(
                artifact_path=artifact_abs,
                workspace_root=self._workspace_root,
            )
            content_for_message = self._build_transcript_stub(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                step=step,
                content_json=content_json,
                original_chars=original_chars,
                preview_text=preview_text,
                preview_chars=preview_chars,
                offloaded=True,
                transcript_shaped=True,
                artifact_locator=artifact_ref.locator,
                artifact_saved=True,
                artifact_readable_via_fs=artifact_ref.artifact_readable_via_fs,
                artifact_location=artifact_ref.artifact_location,
            )
            return OffloadResult(
                content_for_message=content_for_message,
                offloaded=True,
                transcript_shaped=True,
                artifact_locator=artifact_ref.locator,
                artifact_fs_path=str(artifact_abs),
                artifact_readable_via_fs=artifact_ref.artifact_readable_via_fs,
                artifact_location=artifact_ref.artifact_location,
                original_chars=original_chars,
                preview_chars=preview_chars,
                message_chars=len(content_for_message),
                error=None,
            )
        except Exception as exc:
            fallback_preview, fallback_preview_chars = _truncated_preview(
                content_json,
                self._preview_chars,
            )
            content_for_message = self._build_transcript_stub(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                step=step,
                content_json=content_json,
                original_chars=original_chars,
                preview_text=fallback_preview,
                preview_chars=fallback_preview_chars,
                offloaded=False,
                transcript_shaped=True,
                error=str(exc),
            )
            return OffloadResult(
                content_for_message=content_for_message,
                offloaded=False,
                transcript_shaped=True,
                artifact_locator=None,
                artifact_fs_path=None,
                artifact_readable_via_fs=False,
                artifact_location=None,
                original_chars=original_chars,
                preview_chars=fallback_preview_chars,
                message_chars=len(content_for_message),
                error=str(exc),
            )
