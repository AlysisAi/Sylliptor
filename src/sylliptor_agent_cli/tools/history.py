from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..session_artifacts import SessionArtifactLayout


class HistorySearchError(RuntimeError):
    pass


_SNIPPET_MAX_CHARS = 240


def _safe_component(raw: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", str(raw).strip())
    return clean or "x"


def _clip_text(text: str, limit: int = _SNIPPET_MAX_CHARS) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


@dataclass(frozen=True)
class _ArtifactSearchRoot:
    path: Path
    layout: SessionArtifactLayout | None = None


def _search_roots(
    *,
    root: Path,
    session_id: str,
    session_artifact_root: Path | None,
) -> list[_ArtifactSearchRoot]:
    roots: list[_ArtifactSearchRoot] = []
    seen: set[Path] = set()

    if session_artifact_root is not None:
        artifact_root = session_artifact_root.resolve()
        if artifact_root not in seen:
            roots.append(
                _ArtifactSearchRoot(
                    path=artifact_root,
                    layout=SessionArtifactLayout(filesystem_root=artifact_root),
                )
            )
            seen.add(artifact_root)

    history_root = (root / ".sylliptor" / "sessions" / _safe_component(session_id)).resolve()
    if history_root not in seen:
        roots.append(_ArtifactSearchRoot(path=history_root))
        seen.add(history_root)
    return roots


def _iter_target_files(
    *,
    roots: list[_ArtifactSearchRoot],
    include_history: bool,
    include_tool_outputs: bool,
    include_memory: bool,
) -> list[tuple[str, Path, SessionArtifactLayout | None]]:
    targets: list[tuple[str, Path, SessionArtifactLayout | None]] = []
    for artifact_root in roots:
        base = artifact_root.path
        if include_history:
            for path in sorted((base / "history").glob("chunk_*.jsonl")):
                targets.append(("history", path, artifact_root.layout))
        if include_tool_outputs:
            for path in sorted((base / "tool_outputs").glob("*.json")):
                targets.append(("tool_output", path, artifact_root.layout))
        if include_memory:
            for filename in ("summary.json", "pins.json"):
                path = base / "memory" / filename
                if path.exists() and path.is_file():
                    targets.append(("memory", path, artifact_root.layout))
    return targets


def _display_artifact_path(
    *,
    file_path: Path,
    workspace_root: Path,
    layout: SessionArtifactLayout | None,
) -> str:
    if layout is not None:
        return layout.display_reference_for_path(
            artifact_path=file_path,
            workspace_root=workspace_root,
        )
    try:
        return os.fspath(file_path.resolve().relative_to(workspace_root)).replace("\\", "/")
    except ValueError:
        return file_path.name


def history_search(
    *,
    root: Path,
    session_id: str,
    session_artifact_root: Path | None = None,
    pattern: str,
    max_results: int = 50,
    max_file_bytes: int = 200_000,
    include_history: bool = True,
    include_tool_outputs: bool = True,
    include_memory: bool = True,
) -> dict[str, Any]:
    root_abs = root.resolve()

    if max_results <= 0:
        max_results = 1
    if max_file_bytes <= 0:
        max_file_bytes = 1

    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise HistorySearchError(f"Invalid regex pattern: {exc}") from exc

    matches: list[dict[str, Any]] = []
    truncated = False

    search_roots = _search_roots(
        root=root_abs,
        session_id=session_id,
        session_artifact_root=session_artifact_root,
    )
    if not any(search_root.path.exists() for search_root in search_roots):
        return {"pattern": pattern, "matches": matches, "truncated": truncated}

    targets = _iter_target_files(
        roots=search_roots,
        include_history=include_history,
        include_tool_outputs=include_tool_outputs,
        include_memory=include_memory,
    )

    for kind, file_path, layout in targets:
        try:
            with file_path.open("rb") as fh:
                data = fh.read(max_file_bytes + 1)
        except OSError:
            continue
        if len(data) > max_file_bytes:
            data = data[:max_file_bytes]
        text = data.decode("utf-8", errors="replace")

        for line_no, line in enumerate(text.splitlines(), start=1):
            if not compiled.search(line):
                continue
            matches.append(
                {
                    "kind": kind,
                    "path": _display_artifact_path(
                        file_path=file_path,
                        workspace_root=root_abs,
                        layout=layout,
                    ),
                    "line": line_no,
                    "text": _clip_text(line),
                }
            )
            if len(matches) >= max_results:
                truncated = True
                break
        if truncated:
            break

    return {"pattern": pattern, "matches": matches, "truncated": truncated}
