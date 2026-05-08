from __future__ import annotations

import errno
import json
import os
import socket
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json
from .forge import ForgeError, now_iso

LOCK_SCHEMA_VERSION = 1
_LOCK_FILE_NAME = "active_execution.lock.json"
_RECOVERY_FILE_NAME = "active_execution.recovering.json"
_WINDOWS_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_WINDOWS_STILL_ACTIVE = 259
_WINDOWS_ERROR_INVALID_PARAMETER = 87


class RunMutationConflictError(ForgeError):
    pass


@dataclass(frozen=True)
class RunMutationGuard:
    run_id: str
    mode: str
    run_dir: Path
    workspace_root: Path
    owner_token: str
    lock_path: Path
    recovery_path: Path

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
        with suppress(FileNotFoundError):
            self.lock_path.unlink()
        current_recovery = _load_metadata(self.recovery_path)
        if current_recovery is None:
            return
        if str(current_recovery.get("owner_token") or "").strip() != self.owner_token:
            return
        with suppress(FileNotFoundError):
            self.recovery_path.unlink()


def acquire_run_mutation_guard(
    *,
    run_id: str,
    mode: str,
    run_dir: Path,
    workspace_root: Path,
) -> RunMutationGuard:
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / _LOCK_FILE_NAME
    recovery_path = run_dir / _RECOVERY_FILE_NAME
    owner_token = uuid.uuid4().hex
    metadata = _build_metadata(
        run_id=run_id,
        mode=mode,
        workspace_root=workspace_root,
        run_dir=run_dir,
        owner_token=owner_token,
        kind="lock",
    )
    metadata_text = _metadata_text(metadata)

    while True:
        _clear_stale_recovery_claim(
            recovery_path=recovery_path,
            run_id=run_id,
            mode=mode,
        )
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
                raise _conflict_error(
                    run_id=run_id,
                    mode=mode,
                    metadata=existing_metadata,
                ) from None
            recovery_metadata = _build_metadata(
                run_id=run_id,
                mode=mode,
                workspace_root=workspace_root,
                run_dir=run_dir,
                owner_token=owner_token,
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
                try:
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
        else:
            return RunMutationGuard(
                run_id=run_id,
                mode=mode,
                run_dir=run_dir,
                workspace_root=workspace_root,
                owner_token=owner_token,
                lock_path=lock_path,
                recovery_path=recovery_path,
            )
        return RunMutationGuard(
            run_id=run_id,
            mode=mode,
            run_dir=run_dir,
            workspace_root=workspace_root,
            owner_token=owner_token,
            lock_path=lock_path,
            recovery_path=recovery_path,
        )


def inspect_run_mutation_lock(run_dir: Path) -> dict[str, Any] | None:
    return _load_metadata(run_dir / _LOCK_FILE_NAME)


def write_run_mutation_lock_metadata(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _build_metadata(
    *,
    run_id: str,
    mode: str,
    workspace_root: Path,
    run_dir: Path,
    owner_token: str,
    kind: str,
    recovery_reason: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "run_id": run_id,
        "mode": mode,
        "kind": kind,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "acquired_at": now_iso(),
        "owner_token": owner_token,
        "workspace_root": os.fspath(workspace_root.resolve()),
        "run_dir": os.fspath(run_dir.resolve()),
    }
    if recovery_reason:
        payload["recovery_reason"] = recovery_reason
    return payload


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
    return RunMutationConflictError(
        "Another Forge execution is already mutating this run "
        f"(run_id={run_id}, requested_mode={mode}){detail_suffix}.{note_suffix} "
        "Wait for the active execution to finish, or clear the lock only if it is definitely stale."
    )
