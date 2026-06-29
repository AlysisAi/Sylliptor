from __future__ import annotations

import errno
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .background_runner import _apply_env_overrides
from .branding import default_sandbox_docker_image
from .sandbox_runner import (
    _SENSITIVE_ENV_KEYS,
    _build_bwrap_argv,
    _build_docker_argv,
    _docker_cleanup_container,
    _supports_bwrap_unshare_cgroup,
)
from .sandbox_settings import ShellSandboxSettings

SERVICE_SCHEMA_VERSION = 1
SERVICE_STDOUT_LOG = "stdout.log"
SERVICE_STDERR_LOG = "stderr.log"
SERVICE_METADATA = "service.json"
_STOP_TIMEOUT_S = 3.0
_KILL_TIMEOUT_S = 1.0


class ProcessOwnership(StrEnum):
    SESSION = "SESSION"
    DURABLE_SERVICE = "DURABLE_SERVICE"


class DurableServiceStatus(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    EXITED = "exited"
    FAILED = "failed"
    STOPPED = "stopped"
    STALE = "stale"
    UNKNOWN = "unknown"


class ReadinessStatus(StrEnum):
    READY = "ready"
    NOT_READY = "not_ready"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class DurableServiceStart:
    service_id: str
    payload: dict[str, Any]


class DurableServiceManager:
    def __init__(
        self,
        *,
        root: Path,
        state_dir: Path,
        settings: ShellSandboxSettings,
    ) -> None:
        self.root = root.resolve()
        self.state_dir = state_dir.resolve()
        self.settings = settings
        self._popens: dict[str, subprocess.Popen[bytes]] = {}

    def start(
        self,
        *,
        cmd: str,
        cwd: Path,
        readiness: dict[str, Any] | None = None,
    ) -> DurableServiceStart:
        cleaned_cmd = str(cmd or "").strip()
        if not cleaned_cmd:
            raise ValueError("cmd cannot be empty")
        cwd_abs = cwd.resolve()
        try:
            cwd_abs.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"cwd escapes root: {cwd}") from exc

        service_id = f"svc_{uuid.uuid4().hex[:16]}"
        service_dir = self._service_dir(service_id)
        service_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = service_dir / SERVICE_STDOUT_LOG
        stderr_path = service_dir / SERVICE_STDERR_LOG
        metadata_path = service_dir / SERVICE_METADATA
        readiness_spec = normalize_readiness_spec(readiness)
        launch = self._build_launch(cmd=cleaned_cmd, cwd=cwd_abs, service_id=service_id)

        started_at = time.time()
        try:
            with (
                stdout_path.open("ab", buffering=0) as stdout_fh,
                stderr_path.open("ab", buffering=0) as stderr_fh,
            ):
                popen = subprocess.Popen(
                    launch["popen_args"],
                    shell=bool(launch["shell"]),
                    cwd=str(launch["popen_cwd"]),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_fh,
                    stderr=stderr_fh,
                    env=launch["env"],
                    start_new_session=(os.name != "nt"),
                    creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
                    close_fds=True,
                )
        except BaseException:
            self._safe_unlink(metadata_path)
            raise

        self._popens[service_id] = popen
        metadata = {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "service_id": service_id,
            "ownership": ProcessOwnership.DURABLE_SERVICE.value,
            "status": DurableServiceStatus.RUNNING.value,
            "pid": popen.pid,
            "pgid": _process_group_id(popen.pid),
            "pid_start_token": _pid_start_token(popen.pid),
            "started_at_wall": started_at,
            "root": os.fspath(self.root),
            "cwd": os.fspath(cwd_abs),
            "backend": str(launch["backend"]),
            "container_name": launch.get("container_name"),
            "cmd_sha256": hashlib.sha256(cleaned_cmd.encode("utf-8")).hexdigest(),
            "readiness": readiness_spec,
            "stdout_log_path": os.fspath(stdout_path),
            "stderr_log_path": os.fspath(stderr_path),
        }
        self._write_metadata(metadata)
        readiness_payload = self._check_readiness(
            metadata=metadata,
            timeout_s=_readiness_timeout(readiness_spec),
        )
        if readiness_payload["status"] != ReadinessStatus.READY.value:
            self._terminate_loaded_metadata(metadata, remove_metadata=False)
            status_payload = self.status(service_id)
            status_payload["failure_category"] = "readiness_failed"
            status_payload["readiness"] = readiness_payload
            return DurableServiceStart(service_id=service_id, payload=status_payload)
        return DurableServiceStart(service_id=service_id, payload=self.status(service_id))

    def status(self, service_id: str) -> dict[str, Any]:
        metadata = self._read_metadata(service_id)
        if metadata is None:
            return {
                "service_id": service_id,
                "ownership": ProcessOwnership.DURABLE_SERVICE.value,
                "status": DurableServiceStatus.UNKNOWN.value,
                "alive": False,
                "readiness": {
                    "type": "process_alive",
                    "status": ReadinessStatus.FAILED.value,
                    "detail": "metadata not found",
                },
                "failure_category": "unknown_service",
            }
        alive, identity_valid = self._metadata_process_alive(metadata)
        status = (
            DurableServiceStatus.RUNNING.value
            if alive
            else DurableServiceStatus.STALE.value
            if not identity_valid
            else DurableServiceStatus.EXITED.value
        )
        metadata["status"] = status
        self._write_metadata(metadata)
        readiness = self._check_readiness(
            metadata=metadata,
            timeout_s=_readiness_timeout(dict(metadata.get("readiness") or {})),
        )
        return _metadata_public_payload(
            metadata,
            status=status,
            alive=alive,
            identity_valid=identity_valid,
            readiness=readiness,
        )

    def stop(self, service_id: str) -> dict[str, Any]:
        metadata = self._read_metadata(service_id)
        if metadata is None:
            return {
                "service_id": service_id,
                "ownership": ProcessOwnership.DURABLE_SERVICE.value,
                "status": DurableServiceStatus.UNKNOWN.value,
                "stopped": False,
                "failure_category": "unknown_service",
            }
        alive, identity_valid = self._metadata_process_alive(metadata)
        if alive and not identity_valid:
            payload = _metadata_public_payload(
                metadata,
                status=DurableServiceStatus.STALE.value,
                alive=False,
                identity_valid=False,
                readiness={
                    "type": "process_alive",
                    "status": ReadinessStatus.FAILED.value,
                    "detail": "pid identity does not match durable service metadata",
                },
            )
            payload["stopped"] = False
            payload["failure_category"] = "stale_metadata_identity_mismatch"
            metadata["status"] = DurableServiceStatus.STALE.value
            self._write_metadata(metadata)
            return payload
        if alive:
            self._terminate_loaded_metadata(metadata, remove_metadata=True)
        else:
            self._remove_metadata(metadata)
        payload = _metadata_public_payload(
            metadata,
            status=DurableServiceStatus.STOPPED.value,
            alive=False,
            identity_valid=identity_valid,
            readiness={
                "type": str(dict(metadata.get("readiness") or {}).get("type") or "process_alive"),
                "status": ReadinessStatus.FAILED.value,
                "detail": "service stopped",
            },
        )
        payload["stopped"] = True
        return payload

    def list_active(self) -> list[dict[str, Any]]:
        if not self.state_dir.exists():
            return []
        services: list[dict[str, Any]] = []
        for metadata_path in sorted(self.state_dir.glob(f"*/{SERVICE_METADATA}")):
            service_id = metadata_path.parent.name
            payload = self.status(service_id)
            if payload.get("status") == DurableServiceStatus.RUNNING.value:
                services.append(payload)
        return services

    def _build_launch(self, *, cmd: str, cwd: Path, service_id: str) -> dict[str, Any]:
        if self.settings.mode == "off":
            return {
                "backend": "host",
                "popen_args": cmd,
                "popen_cwd": cwd,
                "shell": True,
                "env": _safe_parent_env(),
            }

        backend = self.settings.backend
        has_bwrap = platform.system().lower() == "linux" and shutil.which("bwrap") is not None
        has_docker = shutil.which("docker") is not None
        if backend in {"auto", "bwrap"} and has_bwrap:
            argv, env = _build_bwrap_argv(
                root=self.root,
                cwd=cwd,
                cmd=cmd,
                network=self.settings.network,
                clear_env=self.settings.clear_env,
                profile=self.settings.bwrap_profile,
                unshare_cgroup=_supports_bwrap_unshare_cgroup(),
            )
            argv = [item for item in argv if item != "--die-with-parent"]
            return {
                "backend": "bwrap",
                "popen_args": argv,
                "popen_cwd": self.root,
                "shell": False,
                "env": env,
            }
        if backend in {"auto", "docker"} and has_docker:
            container_name = f"sylliptor-svc-{service_id[-12:]}"
            argv, env = _build_docker_argv(
                root=self.root,
                cwd=cwd,
                cmd=cmd,
                container_name=container_name,
                network=self.settings.network,
                docker_image=self.settings.docker_image or default_sandbox_docker_image("dev"),
                clear_env=self.settings.clear_env,
                pids_limit=self.settings.docker_pids_limit,
                memory_limit=self.settings.docker_memory,
                cpus=self.settings.docker_cpus,
                read_only_rootfs=self.settings.docker_read_only,
                protect_repo_meta=self.settings.protect_repo_meta,
                env_allowlist=self.settings.docker_env_allowlist,
            )
            return {
                "backend": "docker",
                "container_name": container_name,
                "popen_args": argv,
                "popen_cwd": self.root,
                "shell": False,
                "env": env,
            }
        if self.settings.mode == "warn":
            raise RuntimeError(
                "Durable service sandbox unavailable; host fallback is disabled for durable services."
            )
        raise RuntimeError(
            "Shell sandbox strict mode is enabled, but no usable durable service backend is available."
        )

    def _check_readiness(
        self,
        *,
        metadata: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        readiness = dict(metadata.get("readiness") or {})
        readiness_type = str(readiness.get("type") or "process_alive")
        deadline = time.monotonic() + max(0.0, timeout_s)
        interval_s = max(0.02, min(float(readiness.get("interval_s") or 0.1), 1.0))
        last_detail = ""
        while True:
            alive, identity_valid = self._metadata_process_alive(metadata)
            if readiness_type == "process_alive":
                status = (
                    ReadinessStatus.READY if alive and identity_valid else ReadinessStatus.FAILED
                )
                return {
                    "type": readiness_type,
                    "status": status.value,
                    "detail": "process is alive"
                    if status == ReadinessStatus.READY
                    else "process is not alive",
                }
            if not alive or not identity_valid:
                return {
                    "type": readiness_type,
                    "status": ReadinessStatus.FAILED.value,
                    "detail": "process is not alive",
                }
            if readiness_type == "tcp":
                host = str(readiness.get("host") or "127.0.0.1")
                port = int(readiness.get("port") or 0)
                try:
                    with socket.create_connection((host, port), timeout=min(interval_s, 0.5)):
                        return {
                            "type": readiness_type,
                            "status": ReadinessStatus.READY.value,
                            "host": host,
                            "port": port,
                            "detail": "tcp connection succeeded",
                        }
                except OSError as exc:
                    last_detail = str(exc)
            elif readiness_type == "unix_socket":
                path = Path(str(readiness.get("path") or ""))
                if path.exists():
                    return {
                        "type": readiness_type,
                        "status": ReadinessStatus.READY.value,
                        "path": os.fspath(path),
                        "detail": "unix socket exists",
                    }
                last_detail = "unix socket does not exist"
            elif readiness_type == "command":
                command = str(readiness.get("command") or "").strip()
                if not command:
                    return {
                        "type": readiness_type,
                        "status": ReadinessStatus.FAILED.value,
                        "detail": "readiness command is empty",
                    }
                try:
                    result = subprocess.run(
                        command,
                        shell=True,
                        cwd=str(metadata.get("cwd") or self.root),
                        env=_safe_parent_env(),
                        capture_output=True,
                        text=True,
                        timeout=min(max(0.1, timeout_s), 5.0),
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    last_detail = "readiness command timed out"
                else:
                    if result.returncode == 0:
                        return {
                            "type": readiness_type,
                            "status": ReadinessStatus.READY.value,
                            "exit_code": result.returncode,
                            "detail": "readiness command passed",
                        }
                    last_detail = f"readiness command exited {result.returncode}"
            else:
                return {
                    "type": readiness_type,
                    "status": ReadinessStatus.UNSUPPORTED.value,
                    "detail": f"unsupported readiness type: {readiness_type}",
                }

            if time.monotonic() >= deadline:
                return {
                    "type": readiness_type,
                    "status": ReadinessStatus.FAILED.value,
                    "detail": last_detail or "readiness probe timed out",
                }
            time.sleep(interval_s)

    def _metadata_process_alive(self, metadata: dict[str, Any]) -> tuple[bool, bool]:
        pid = int(metadata.get("pid") or 0)
        if pid <= 0:
            return False, False
        expected_token = str(metadata.get("pid_start_token") or "")
        if expected_token:
            current_token = _pid_start_token(pid)
            if current_token and current_token != expected_token:
                return True, False
        popen = self._popens.get(str(metadata.get("service_id") or ""))
        if popen is not None and popen.poll() is not None:
            return False, True
        return _pid_exists(pid), True

    def _terminate_loaded_metadata(
        self,
        metadata: dict[str, Any],
        *,
        remove_metadata: bool,
    ) -> None:
        service_id = str(metadata.get("service_id") or "")
        popen = self._popens.get(service_id)
        if str(metadata.get("backend") or "") == "docker" and metadata.get("container_name"):
            _docker_cleanup_container(
                str(metadata["container_name"]),
                cwd=os.fspath(self.root),
                env=_safe_parent_env(),
                warning_callback=None,
                reason="durable service stop",
                quiet=True,
            )
        if popen is not None and popen.poll() is None:
            _terminate_popen_tree(popen, timeout_s=_STOP_TIMEOUT_S)
        else:
            pid = int(metadata.get("pid") or 0)
            pgid = metadata.get("pgid")
            _terminate_pid_or_group(pid=pid, pgid=pgid, timeout_s=_STOP_TIMEOUT_S)
        metadata["status"] = DurableServiceStatus.STOPPED.value
        if remove_metadata:
            self._remove_metadata(metadata)
        else:
            self._write_metadata(metadata)

    def _read_metadata(self, service_id: str) -> dict[str, Any] | None:
        normalized = _normalize_service_id(service_id)
        if not normalized:
            return None
        path = self._service_dir(normalized) / SERVICE_METADATA
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict):
            return None
        if raw.get("service_id") != normalized:
            return None
        return raw

    def _write_metadata(self, metadata: dict[str, Any]) -> None:
        service_id = str(metadata.get("service_id") or "")
        service_dir = self._service_dir(service_id)
        service_dir.mkdir(parents=True, exist_ok=True)
        path = service_dir / SERVICE_METADATA
        path.write_text(
            json.dumps(_sanitize_metadata(metadata), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _remove_metadata(self, metadata: dict[str, Any]) -> None:
        service_id = str(metadata.get("service_id") or "")
        self._safe_unlink(self._service_dir(service_id) / SERVICE_METADATA)

    def _service_dir(self, service_id: str) -> Path:
        normalized = _normalize_service_id(service_id)
        if not normalized:
            raise ValueError("invalid service_id")
        return self.state_dir / normalized

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return


def normalize_readiness_spec(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not raw:
        return {"type": "process_alive", "timeout_s": 1.0, "interval_s": 0.1}
    if not isinstance(raw, dict):
        return {"type": "process_alive", "timeout_s": 1.0, "interval_s": 0.1}
    readiness_type = str(raw.get("type") or "process_alive").strip().lower()
    out: dict[str, Any] = {
        "type": readiness_type,
        "timeout_s": _bounded_float(raw.get("timeout_s"), default=5.0, low=0.0, high=30.0),
        "interval_s": _bounded_float(raw.get("interval_s"), default=0.1, low=0.02, high=2.0),
    }
    if readiness_type == "tcp":
        out["host"] = str(raw.get("host") or "127.0.0.1")
        out["port"] = int(raw.get("port") or 0)
    elif readiness_type == "unix_socket":
        out["path"] = str(raw.get("path") or "")
    elif readiness_type == "command":
        out["command"] = str(raw.get("command") or "")
    return out


def _readiness_timeout(readiness: dict[str, Any]) -> float:
    return _bounded_float(readiness.get("timeout_s"), default=5.0, low=0.0, high=30.0)


def _bounded_float(value: Any, *, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _safe_parent_env() -> dict[str, str]:
    blocked = {key.upper() for key in _SENSITIVE_ENV_KEYS}
    return _apply_env_overrides(
        {key: value for key, value in os.environ.items() if key.upper() not in blocked},
        None,
    )


def _normalize_service_id(service_id: str) -> str:
    cleaned = str(service_id or "").strip()
    if not cleaned or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_" for ch in cleaned):
        return ""
    return cleaned


def _metadata_public_payload(
    metadata: dict[str, Any],
    *,
    status: str,
    alive: bool,
    identity_valid: bool,
    readiness: dict[str, Any],
) -> dict[str, Any]:
    return {
        "service_id": metadata.get("service_id"),
        "ownership": metadata.get("ownership") or ProcessOwnership.DURABLE_SERVICE.value,
        "status": status,
        "alive": bool(alive),
        "identity_valid": bool(identity_valid),
        "pid": metadata.get("pid"),
        "pgid": metadata.get("pgid"),
        "backend": metadata.get("backend"),
        "readiness": readiness,
        "start_timestamp": metadata.get("started_at_wall"),
        "stdout_log_path": metadata.get("stdout_log_path"),
        "stderr_log_path": metadata.get("stderr_log_path"),
        "log_paths": {
            "stdout": metadata.get("stdout_log_path"),
            "stderr": metadata.get("stderr_log_path"),
        },
        "failure_category": None
        if status == DurableServiceStatus.RUNNING.value
        and readiness.get("status") == ReadinessStatus.READY.value
        else "service_not_ready",
    }


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "schema_version",
        "service_id",
        "ownership",
        "status",
        "pid",
        "pgid",
        "pid_start_token",
        "started_at_wall",
        "root",
        "cwd",
        "backend",
        "container_name",
        "cmd_sha256",
        "readiness",
        "stdout_log_path",
        "stderr_log_path",
    }
    return {key: metadata[key] for key in sorted(allowed_keys) if key in metadata}


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def _pid_start_token(pid: int) -> str | None:
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        text = stat_path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        after_comm = text.rsplit(") ", 1)[1]
        fields = after_comm.split()
        return f"linux:{fields[19]}"
    except (IndexError, ValueError):
        return None


def _process_group_id(pid: int) -> int | None:
    if os.name == "nt":
        return None
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def _terminate_popen_tree(popen: subprocess.Popen[bytes], *, timeout_s: float) -> None:
    if popen.poll() is not None:
        return
    if os.name == "nt":  # pragma: no cover - exercised on Windows
        popen.terminate()
    else:
        pgid = _process_group_id(popen.pid)
        if pgid is not None:
            with _ignore_process_lookup():
                os.killpg(pgid, signal.SIGTERM)
        else:
            with _ignore_process_lookup():
                popen.terminate()
    try:
        popen.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "nt":  # pragma: no cover - exercised on Windows
        popen.kill()
    else:
        pgid = _process_group_id(popen.pid)
        if pgid is not None:
            with _ignore_process_lookup():
                os.killpg(pgid, signal.SIGKILL)
        else:
            with _ignore_process_lookup():
                popen.kill()
    with _ignore_timeout():
        popen.wait(timeout=_KILL_TIMEOUT_S)


def _terminate_pid_or_group(*, pid: int, pgid: Any, timeout_s: float) -> None:
    if pid <= 0:
        return
    if os.name != "nt" and isinstance(pgid, int) and pgid > 0:
        with _ignore_process_lookup():
            os.killpg(pgid, signal.SIGTERM)
    else:
        with _ignore_process_lookup():
            os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return
        time.sleep(0.05)
    if os.name != "nt" and isinstance(pgid, int) and pgid > 0:
        with _ignore_process_lookup():
            os.killpg(pgid, signal.SIGKILL)
    else:
        with _ignore_process_lookup():
            os.kill(pid, signal.SIGKILL)


class _ignore_process_lookup:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return isinstance(exc, ProcessLookupError)


class _ignore_timeout:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return isinstance(exc, subprocess.TimeoutExpired)
