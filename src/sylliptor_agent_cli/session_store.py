from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import AppConfig, default_sessions_dir
from .session_artifacts import SessionArtifactLayout
from .web_research import SessionWebResearchTracker


def _now_ts() -> str:
    return datetime.now(UTC).isoformat()


def make_session_id() -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    # 8 chars is enough for local uniqueness.
    import uuid

    return f"{ts}_{uuid.uuid4().hex[:8]}"


def sanitize_session_id(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", session_id.strip()) or "session"


def resolve_sessions_dir(cfg: AppConfig) -> Path:
    if cfg.session_log_dir:
        return Path(cfg.session_log_dir)
    return default_sessions_dir()


@dataclass
class SessionInfo:
    session_id: str
    path: Path
    mtime: float


class SessionStore:
    def __init__(
        self,
        *,
        enabled: bool,
        artifact_persistence_enabled: bool | None = None,
        sessions_dir: Path,
        session_id: str,
        cwd: str,
        repo_root: str | None,
        workspace_root: str | None = None,
        focus_dir: str | None = None,
        git_root: str | None = None,
        workspace_kind: str | None = None,
        binding_source: str | None = None,
        binding_requested_path: str | None = None,
        binding_risk_level: str | None = None,
        binding_created_path: bool | None = None,
        runtime_kind: str | None = None,
        active_workdir: str | None = None,
        active_workdir_relpath: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.sessions_dir = sessions_dir
        self.session_id = session_id
        self.cwd = cwd
        self.repo_root = repo_root
        self.workspace_root = workspace_root
        self.focus_dir = focus_dir
        self.git_root = git_root
        self.workspace_kind = workspace_kind
        self.binding_source = binding_source
        self.binding_requested_path = binding_requested_path
        self.binding_risk_level = binding_risk_level
        self.binding_created_path = binding_created_path
        self.runtime_kind = runtime_kind
        self.active_workdir = active_workdir or cwd
        self.active_workdir_relpath = active_workdir_relpath
        self.path = sessions_dir / f"{session_id}.jsonl"
        self.artifact_persistence_enabled = (
            enabled if artifact_persistence_enabled is None else bool(artifact_persistence_enabled)
        )
        self._events: list[dict[str, Any]] = []
        self._web_research = SessionWebResearchTracker()
        self._hydrate_existing_state()

        self._fh = None
        if self.enabled:
            try:
                self.sessions_dir.mkdir(parents=True, exist_ok=True)
                self._fh = self.path.open("a", encoding="utf-8")
            except OSError:
                self.enabled = False
                self.artifact_persistence_enabled = False

    @property
    def session_artifact_root(self) -> Path:
        return self.sessions_dir / sanitize_session_id(self.session_id)

    @property
    def session_artifact_layout(self) -> SessionArtifactLayout:
        return SessionArtifactLayout(filesystem_root=self.session_artifact_root)

    def runtime_artifact_path(self, *parts: str) -> Path:
        return self.session_artifact_layout.artifact_fs_path(*parts)

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        event = self._build_event(event_type=event_type, payload=payload)
        self._events.append(event)
        observed_web_research = self._web_research.observe_event(
            event_type=event_type,
            payload=payload,
            ts=str(event.get("ts") or "").strip() or None,
        )
        if self.enabled and self._fh:
            self._fh.write(json.dumps(event, ensure_ascii=True) + "\n")
            self._fh.flush()
        if observed_web_research and self.artifact_persistence_enabled:
            self._persist_web_research_artifact()

    def _build_event(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        event = {
            "type": event_type,
            "ts": _now_ts(),
            "session_id": self.session_id,
            "cwd": self.cwd,
            "repo_root": self.repo_root,
            "pid": os.getpid(),
            "payload": payload,
        }
        if self.workspace_root is not None:
            event["workspace_root"] = self.workspace_root
        if self.focus_dir is not None:
            event["focus_dir"] = self.focus_dir
        if self.git_root is not None:
            event["git_root"] = self.git_root
        if self.workspace_kind is not None:
            event["workspace_kind"] = self.workspace_kind
        if self.binding_source is not None:
            event["binding_source"] = self.binding_source
        if self.binding_requested_path is not None:
            event["binding_requested_path"] = self.binding_requested_path
        if self.binding_risk_level is not None:
            event["binding_risk_level"] = self.binding_risk_level
        if self.binding_created_path is not None:
            event["binding_created_path"] = self.binding_created_path
        if self.runtime_kind is not None:
            event["runtime_kind"] = self.runtime_kind
        if self.active_workdir is not None:
            event["active_workdir"] = self.active_workdir
        if self.active_workdir_relpath is not None:
            event["active_workdir_relpath"] = self.active_workdir_relpath
        return event

    def update_active_workdir(self, *, cwd: str, active_workdir_relpath: str) -> None:
        self.cwd = cwd
        self.active_workdir = cwd
        self.active_workdir_relpath = active_workdir_relpath

    def append_artifact_jsonl(self, *parts: str, payload: dict[str, Any]) -> Path | None:
        if not self.enabled:
            return None
        path = self.runtime_artifact_path(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
        return path

    def events_snapshot(self) -> list[dict[str, Any]]:
        return json.loads(json.dumps(self._events, ensure_ascii=True))

    def classify_web_fetch_url(self, raw_url: Any) -> str | None:
        return self._web_research.classify_fetch_url(raw_url)

    def resolve_web_fetch_url(self, raw_url: Any) -> tuple[str | None, str | None]:
        return self._web_research.resolve_fetch_url(raw_url)

    def web_research_artifact_payload(self) -> dict[str, Any]:
        return self._web_research.artifact_payload()

    def web_research_metrics_payload(self) -> dict[str, int]:
        return self._web_research.metrics_payload()

    def has_web_research_activity(self) -> bool:
        return self._web_research.has_activity()

    def _persist_web_research_artifact(self) -> Path | None:
        if not self.artifact_persistence_enabled or not self._web_research.has_activity():
            return None
        path = self.runtime_artifact_path("web_research_sources.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                self._web_research.artifact_payload(),
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def _hydrate_existing_state(self) -> None:
        self._hydrate_from_existing_log()
        self._hydrate_from_existing_web_research_artifact()
        self._web_research.clear_pending()

    def _hydrate_from_existing_log(self) -> bool:
        try:
            if not self.path.exists():
                return False
            events = list(read_session_events(self.path))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        if not events:
            return False
        self._events = json.loads(json.dumps(events, ensure_ascii=True))
        for event in self._events:
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            self._web_research.observe_event(
                event_type=str(event.get("type") or "").strip(),
                payload=payload,
                ts=str(event.get("ts") or "").strip() or None,
            )
        return True

    def _hydrate_from_existing_web_research_artifact(self) -> bool:
        artifact_path = self.runtime_artifact_path("web_research_sources.json")
        try:
            if not artifact_path.exists():
                return False
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        return self._web_research.merge_from_artifact_payload(payload)

    def close(self) -> None:
        if self.artifact_persistence_enabled and self._web_research.has_activity():
            self._persist_web_research_artifact()
        if self._fh:
            self._fh.close()
            self._fh = None


def list_sessions(sessions_dir: Path) -> list[SessionInfo]:
    if not sessions_dir.exists():
        return []
    out: list[SessionInfo] = []
    for p in sessions_dir.glob("*.jsonl"):
        try:
            st = p.stat()
        except OSError:
            continue
        out.append(SessionInfo(session_id=p.stem, path=p, mtime=st.st_mtime))
    out.sort(key=lambda x: x.mtime, reverse=True)
    return out


def read_session_events(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj
