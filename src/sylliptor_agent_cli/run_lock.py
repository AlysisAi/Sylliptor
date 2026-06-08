from __future__ import annotations

import errno
import json
import os
import socket
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json
from .forge import ForgeError, now_iso

LOCK_SCHEMA_VERSION = 1
_LOCK_FILE_NAME = "active_execution.lock.json"
_RECOVERY_FILE_NAME = "active_execution.recovering.json"
_EVENTS_FILE_NAME = "active_execution.events.jsonl"
_WAITING_FILE_PREFIX = "active_execution.waiting."
_WAITING_FILE_SUFFIX = ".json"
_DEFAULT_POLL_INTERVAL_S = 1.0
_WINDOWS_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_WINDOWS_STILL_ACTIVE = 259
_WINDOWS_ERROR_INVALID_PARAMETER = 87


class RunMutationConflictError(ForgeError):
    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "active_workspace_execution",
        metadata: dict[str, Any] | None = None,
        diagnostic: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.metadata = metadata
        self.diagnostic = diagnostic or message


@dataclass(frozen=True)
class RunMutationGuard:
    run_id: str
    mode: str
    run_dir: Path
    workspace_root: Path
    owner_token: str
    lock_path: Path
    recovery_path: Path
    acquired_after_wait: bool = False
    wait_started_at: str | None = None
    wait_finished_at: str | None = None
    wait_record_path: Path | None = None

    def __enter__(self) -> RunMutationGuard:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        _ = exc_type
        _ = exc
        _ = tb
        self.release()

    def __del__(self) -> None:
        with suppress(Exception):
            self.release()

    def release(self) -> None:
        current = _load_metadata(self.lock_path)
        if current is None:
            return
        if str(current.get("owner_token") or "").strip() != self.owner_token:
            return
        self.refresh_heartbeat()
        with suppress(FileNotFoundError):
            self.lock_path.unlink()
        if self.wait_record_path is not None:
            with suppress(FileNotFoundError):
                self.wait_record_path.unlink()
        current_recovery = _load_metadata(self.recovery_path)
        if current_recovery is None:
            return
        if str(current_recovery.get("owner_token") or "").strip() != self.owner_token:
            return
        with suppress(FileNotFoundError):
            self.recovery_path.unlink()

    def refresh_heartbeat(self) -> None:
        current = _load_metadata(self.lock_path)
        if current is None:
            return
        if str(current.get("owner_token") or "").strip() != self.owner_token:
            return
        current["last_heartbeat_at"] = now_iso()
        atomic_write_json(self.lock_path, current)


def acquire_run_mutation_guard(
    *,
    run_id: str,
    mode: str,
    run_dir: Path,
    workspace_root: Path,
    wait: bool = False,
    wait_timeout_s: float | None = None,
    poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
    on_wait: Callable[[dict[str, Any]], None] | None = None,
    owner_session_id: str | None = None,
) -> RunMutationGuard:
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / _LOCK_FILE_NAME
    recovery_path = run_dir / _RECOVERY_FILE_NAME
    events_path = run_dir / _EVENTS_FILE_NAME
    owner_token = uuid.uuid4().hex
    owner_id = _owner_id(owner_token=owner_token, owner_session_id=owner_session_id)
    wait_started_monotonic = time.monotonic()
    wait_started_at: str | None = None
    wait_finished_at: str | None = None
    wait_record_path: Path | None = None
    acquired_after_wait = False
    wait_notice_emitted = False

    while True:
        _clear_stale_recovery_claim(
            recovery_path=recovery_path,
            run_id=run_id,
            mode=mode,
        )
        metadata = _build_metadata(
            run_id=run_id,
            mode=mode,
            workspace_root=workspace_root,
            run_dir=run_dir,
            owner_token=owner_token,
            owner_id=owner_id,
            owner_session_id=owner_session_id,
            kind="lock",
        )
        metadata_text = _metadata_text(metadata)
        try:
            _write_exclusive(lock_path, metadata_text)
        except FileExistsError:
            existing_metadata, existing_text = _load_metadata_with_text(lock_path)
            if existing_metadata is None or existing_text is None:
                raise _conflict_error(
                    run_id=run_id,
                    mode=mode,
                    metadata=None,
                    note="the active run lock exists but its metadata is unreadable, so recovery is blocked",
                ) from None
            stale_reason = _definitely_stale_reason(existing_metadata)
            if stale_reason is None:
                conflict = _conflict_error(
                    run_id=run_id,
                    mode=mode,
                    metadata=existing_metadata,
                )
                if not wait:
                    raise conflict from None
                now_monotonic = time.monotonic()
                if wait_timeout_s is not None and now_monotonic - wait_started_monotonic >= max(
                    0.0, wait_timeout_s
                ):
                    if wait_record_path is not None:
                        _append_event(
                            events_path,
                            {
                                "schema_version": LOCK_SCHEMA_VERSION,
                                "event": "queued_wait_timed_out",
                                "reason_code": "active_execution_wait_timeout",
                                "run_id": run_id,
                                "mode": mode,
                                "owner_id": owner_id,
                                "workspace_root": _safe_resolved_path(workspace_root),
                                "workspace_identity": normalize_workspace_identity_path(
                                    workspace_root
                                ),
                                "run_dir": _safe_resolved_path(run_dir),
                                "wait_started_at": wait_started_at,
                            },
                        )
                        with suppress(FileNotFoundError):
                            wait_record_path.unlink()
                    raise _conflict_error(
                        run_id=run_id,
                        mode=mode,
                        metadata=existing_metadata,
                        note=(
                            "queued execution timed out waiting for the active mutation guard "
                            "to finish"
                        ),
                        reason_code="active_execution_wait_timeout",
                    ) from None
                if wait_started_at is None:
                    wait_started_at = now_iso()
                    acquired_after_wait = True
                    wait_record_path = run_dir / (
                        f"{_WAITING_FILE_PREFIX}{owner_token}{_WAITING_FILE_SUFFIX}"
                    )
                    wait_payload = _build_wait_metadata(
                        run_id=run_id,
                        mode=mode,
                        workspace_root=workspace_root,
                        run_dir=run_dir,
                        owner_token=owner_token,
                        owner_id=owner_id,
                        owner_session_id=owner_session_id,
                        blocked_by=existing_metadata,
                        started_at=wait_started_at,
                        diagnostic=conflict.diagnostic,
                    )
                    atomic_write_json(wait_record_path, wait_payload)
                    _append_event(events_path, {**wait_payload, "event": "queued_wait_started"})
                if on_wait is not None and not wait_notice_emitted:
                    on_wait(
                        {
                            "reason_code": conflict.reason_code,
                            "diagnostic": conflict.diagnostic,
                            "blocked_by": _public_lock_metadata(existing_metadata),
                            "run_id": run_id,
                            "mode": mode,
                            "wait_started_at": wait_started_at,
                        }
                    )
                    wait_notice_emitted = True
                time.sleep(max(0.05, min(float(poll_interval_s), 5.0)))
                continue
            recovery_metadata = _build_metadata(
                run_id=run_id,
                mode=mode,
                workspace_root=workspace_root,
                run_dir=run_dir,
                owner_token=owner_token,
                owner_id=owner_id,
                owner_session_id=owner_session_id,
                kind="recovery",
                recovery_reason=stale_reason,
            )
            try:
                _write_exclusive(recovery_path, _metadata_text(recovery_metadata))
            except FileExistsError:
                _clear_stale_recovery_claim(
                    recovery_path=recovery_path,
                    run_id=run_id,
                    mode=mode,
                )
                raise _conflict_error(
                    run_id=run_id,
                    mode=mode,
                    metadata=existing_metadata,
                    note="another execution is already recovering the stale run lock",
                ) from None
            try:
                current_metadata, current_text = _load_metadata_with_text(lock_path)
                if current_metadata is None or current_text is None:
                    raise _conflict_error(
                        run_id=run_id,
                        mode=mode,
                        metadata=None,
                        note="the active run lock disappeared or became unreadable during recovery",
                    )
                if current_text != existing_text:
                    raise _conflict_error(
                        run_id=run_id,
                        mode=mode,
                        metadata=current_metadata,
                        note="the active run lock changed while recovery was in progress",
                    )
                if _definitely_stale_reason(current_metadata) is None:
                    raise _conflict_error(
                        run_id=run_id,
                        mode=mode,
                        metadata=current_metadata,
                        note="the active run lock is no longer definitely stale",
                    )
                lock_path.unlink()
                _append_event(
                    events_path,
                    {
                        **recovery_metadata,
                        "event": "stale_lock_recovered",
                        "recovered_owner": _public_lock_metadata(current_metadata),
                    },
                )
                try:
                    metadata = _build_metadata(
                        run_id=run_id,
                        mode=mode,
                        workspace_root=workspace_root,
                        run_dir=run_dir,
                        owner_token=owner_token,
                        owner_id=owner_id,
                        owner_session_id=owner_session_id,
                        kind="lock",
                    )
                    metadata_text = _metadata_text(metadata)
                    _write_exclusive(lock_path, metadata_text)
                except FileExistsError:
                    replacement_metadata = _load_metadata(lock_path)
                    raise _conflict_error(
                        run_id=run_id,
                        mode=mode,
                        metadata=replacement_metadata,
                        note="another execution claimed the run while stale-lock recovery was finalizing",
                    ) from None
            finally:
                current_recovery = _load_metadata(recovery_path)
                if (
                    current_recovery is not None
                    and str(current_recovery.get("owner_token") or "").strip() == owner_token
                ):
                    with suppress(FileNotFoundError):
                        recovery_path.unlink()
        break
    if acquired_after_wait:
        wait_finished_at = now_iso()
        _append_event(
            events_path,
            {
                "schema_version": LOCK_SCHEMA_VERSION,
                "event": "queued_wait_finished",
                "reason_code": "queued_execution_acquired_lock",
                "run_id": run_id,
                "mode": mode,
                "owner_id": owner_id,
                "workspace_root": _safe_resolved_path(workspace_root),
                "workspace_identity": normalize_workspace_identity_path(workspace_root),
                "run_dir": _safe_resolved_path(run_dir),
                "wait_started_at": wait_started_at,
                "wait_finished_at": wait_finished_at,
            },
        )
    return RunMutationGuard(
        run_id=run_id,
        mode=mode,
        run_dir=run_dir,
        workspace_root=workspace_root,
        owner_token=owner_token,
        lock_path=lock_path,
        recovery_path=recovery_path,
        acquired_after_wait=acquired_after_wait,
        wait_started_at=wait_started_at,
        wait_finished_at=wait_finished_at,
        wait_record_path=wait_record_path,
    )


def inspect_run_mutation_lock(run_dir: Path) -> dict[str, Any] | None:
    return _load_metadata(run_dir / _LOCK_FILE_NAME)


def workspace_mutation_run_id(workspace_root: Path | str) -> str:
    return f"workspace:{normalize_workspace_identity_path(workspace_root)}"


def normalize_workspace_identity_path(workspace_root: Path | str) -> str:
    raw = os.fspath(workspace_root).strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    raw = raw.rstrip("/") or raw
    if _looks_like_windows_drive_path(raw):
        drive = raw[:2].lower()
        rest = "/".join(part for part in raw[2:].split("/") if part)
        return f"{drive}/{rest}".rstrip("/").casefold()
    if raw.startswith("/mnt/") and len(raw) >= 7 and raw[6:7] == "/":
        drive = raw[5:6].lower()
        tail = "/".join(part for part in raw[7:].split("/") if part)
        return f"{drive}:/{tail}".rstrip("/").casefold()
    try:
        return os.fspath(Path(raw).expanduser().resolve()).replace("\\", "/").rstrip("/")
    except OSError:
        return raw


def write_run_mutation_lock_metadata(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def lock_diagnostic(metadata: dict[str, Any] | None, *, run_id: str, mode: str) -> dict[str, Any]:
    error = _conflict_error(run_id=run_id, mode=mode, metadata=metadata)
    return {
        "reason_code": error.reason_code,
        "diagnostic": error.diagnostic,
        "blocked_by": _public_lock_metadata(metadata),
    }


def _owner_id(*, owner_token: str, owner_session_id: str | None) -> str:
    session = str(owner_session_id or "").strip()
    if session:
        return session
    return f"{socket.gethostname()}:{os.getpid()}:{owner_token[:12]}"


def _safe_resolved_path(path: Path | str) -> str:
    try:
        return os.fspath(Path(path).expanduser().resolve(strict=False)).replace("\\", "/")
    except (OSError, RuntimeError, ValueError):
        return os.fspath(path).replace("\\", "/")


def _looks_like_windows_drive_path(raw: str) -> bool:
    return len(raw) >= 2 and raw[0].isalpha() and raw[1] == ":"


def _public_lock_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if metadata is None:
        return None
    public_keys = {
        "schema_version",
        "run_id",
        "mode",
        "kind",
        "pid",
        "hostname",
        "acquired_at",
        "started_at",
        "last_heartbeat_at",
        "owner_id",
        "owner_session_id",
        "workspace_root",
        "workspace_identity",
        "run_dir",
        "stale_policy",
        "reason_code",
        "diagnostic",
        "recovery_reason",
    }
    return {key: value for key, value in metadata.items() if key in public_keys}


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event_payload = dict(payload)
    event_payload.pop("owner_token", None)
    event_payload.setdefault("event_at", now_iso())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event_payload, sort_keys=True) + "\n")


def _build_metadata(
    *,
    run_id: str,
    mode: str,
    workspace_root: Path,
    run_dir: Path,
    owner_token: str,
    owner_id: str,
    owner_session_id: str | None,
    kind: str,
    recovery_reason: str | None = None,
) -> dict[str, Any]:
    acquired_at = now_iso()
    payload: dict[str, Any] = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "run_id": run_id,
        "mode": mode,
        "kind": kind,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "acquired_at": acquired_at,
        "started_at": acquired_at,
        "last_heartbeat_at": acquired_at,
        "owner_id": owner_id,
        "owner_session_id": owner_session_id,
        "owner_token": owner_token,
        "workspace_root": _safe_resolved_path(workspace_root),
        "workspace_identity": normalize_workspace_identity_path(workspace_root),
        "run_dir": _safe_resolved_path(run_dir),
        "stale_policy": "same-host owner pid must be absent; ambiguous locks stay active",
        "reason_code": "stale_lock_recovery" if kind == "recovery" else "active_execution_lock",
        "diagnostic": (
            "Recovering a definitely stale Forge mutation lock."
            if kind == "recovery"
            else "Forge execution is mutating this workspace/run."
        ),
    }
    if recovery_reason:
        payload["recovery_reason"] = recovery_reason
    return payload


def _build_wait_metadata(
    *,
    run_id: str,
    mode: str,
    workspace_root: Path,
    run_dir: Path,
    owner_token: str,
    owner_id: str,
    owner_session_id: str | None,
    blocked_by: dict[str, Any],
    started_at: str,
    diagnostic: str,
) -> dict[str, Any]:
    return {
        "schema_version": LOCK_SCHEMA_VERSION,
        "kind": "queued_wait",
        "reason_code": "blocked_by_active_workspace_execution",
        "run_id": run_id,
        "mode": mode,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at": started_at,
        "last_heartbeat_at": started_at,
        "owner_id": owner_id,
        "owner_session_id": owner_session_id,
        "owner_token": owner_token,
        "workspace_root": _safe_resolved_path(workspace_root),
        "workspace_identity": normalize_workspace_identity_path(workspace_root),
        "run_dir": _safe_resolved_path(run_dir),
        "blocked_by": _public_lock_metadata(blocked_by),
        "stale_policy": "queued wait rechecks the lock and only recovers definitely stale locks",
        "diagnostic": diagnostic,
    }


def _metadata_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _write_exclusive(path: Path, text: str) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(path, flags, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
    except Exception:
        with suppress(FileNotFoundError):
            path.unlink()
        raise


def _load_metadata(path: Path) -> dict[str, Any] | None:
    payload, _ = _load_metadata_with_text(path)
    return payload


def _load_metadata_with_text(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, None
    except OSError:
        return None, None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None, text
    if not isinstance(payload, dict):
        return None, text
    return payload, text


def _clear_stale_recovery_claim(
    *,
    recovery_path: Path,
    run_id: str,
    mode: str,
) -> None:
    recovery_metadata = _load_metadata(recovery_path)
    if recovery_metadata is None:
        return
    stale_reason = _definitely_stale_reason(recovery_metadata)
    if stale_reason is not None:
        with suppress(FileNotFoundError):
            recovery_path.unlink()
        return
    raise _conflict_error(
        run_id=run_id,
        mode=mode,
        metadata=recovery_metadata,
        note="another execution is already recovering this run",
    )


def _definitely_stale_reason(metadata: dict[str, Any]) -> str | None:
    if str(metadata.get("hostname") or "").strip() != socket.gethostname():
        return None
    raw_pid = metadata.get("pid")
    try:
        pid = int(raw_pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    is_running = _process_is_running(pid)
    if is_running is False:
        return "owner process is no longer running"
    return None


def _process_is_running(pid: int) -> bool | None:
    if os.name == "nt":
        return _windows_process_is_running(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as e:
        if getattr(e, "errno", None) == errno.ESRCH:
            return False
        return None
    return True


def _windows_process_is_running(pid: int) -> bool | None:
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.OpenProcess(
            _WINDOWS_PROCESS_QUERY_LIMITED_INFORMATION,
            False,
            pid,
        )
        if not handle:
            error = ctypes.get_last_error()
            if error == _WINDOWS_ERROR_INVALID_PARAMETER:
                return False
            return None
        exit_code = wintypes.DWORD()
        try:
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return None
            return exit_code.value == _WINDOWS_STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return None


def _conflict_error(
    *,
    run_id: str,
    mode: str,
    metadata: dict[str, Any] | None,
    note: str | None = None,
    reason_code: str = "blocked_by_active_workspace_execution",
) -> RunMutationConflictError:
    details: list[str] = []
    if metadata is not None:
        owner_mode = str(metadata.get("mode") or "").strip()
        if owner_mode:
            details.append(f"owner mode={owner_mode}")
        pid = str(metadata.get("pid") or "").strip()
        if pid:
            details.append(f"pid={pid}")
        hostname = str(metadata.get("hostname") or "").strip()
        if hostname:
            details.append(f"host={hostname}")
        acquired_at = str(metadata.get("acquired_at") or "").strip()
        if acquired_at:
            details.append(f"acquired_at={acquired_at}")
    detail_suffix = f" ({', '.join(details)})" if details else ""
    note_suffix = f" {note}." if note else ""
    target = "workspace" if run_id.startswith("workspace:") else "run"
    message = (
        f"Another Forge execution is already mutating this {target} "
        f"(run_id={run_id}, requested_mode={mode}){detail_suffix}.{note_suffix} "
        "Wait for the active execution to finish, or clear the lock only if it is definitely stale."
    )
    return RunMutationConflictError(
        message,
        reason_code=reason_code,
        metadata=_public_lock_metadata(metadata),
        diagnostic=message,
    )
