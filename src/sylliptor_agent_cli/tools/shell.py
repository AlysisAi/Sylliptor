from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any


class ShellError(RuntimeError):
    pass


_PYTHON_CMD_PATTERN = re.compile(r"(^|(?:&&|\|\||;)\s*)python(?=\s)")


def _python3_fallback_cmd(cmd: str) -> str | None:
    match = _PYTHON_CMD_PATTERN.search(cmd)
    if not match:
        return None
    start, end = match.span()
    prefix = match.group(1)
    replacement = f"{prefix}python3"
    return f"{cmd[:start]}{replacement}{cmd[end:]}"


def _is_python_permission_denied(stderr: str) -> bool:
    lowered = stderr.lower()
    return "python: permission denied" in lowered


def shell_run(
    *,
    root: Path,
    cmd: str,
    cwd: str | None = None,
    timeout_s: int = 60,
    runner: Any | None = None,
) -> dict[str, Any]:
    base = root.resolve()
    if cwd:
        cwd_path = (base / cwd).resolve()
        try:
            cwd_path.relative_to(base)
        except ValueError as e:
            raise ShellError(f"cwd escapes root: {cwd}") from e
    else:
        cwd_path = base

    if runner is None:
        raise ShellError("Shell runner is required; implicit host execution is disabled.")

    try:
        cp = runner.run(root=base, cwd=cwd_path, cmd=cmd, timeout_s=timeout_s)
    except subprocess.TimeoutExpired as e:
        raise ShellError(f"Command timed out after {timeout_s}s") from e
    except Exception as e:  # noqa: BLE001
        raise ShellError(f"Failed to run command: {e}") from e

    effective_cmd = cmd
    fallback_cmd = _python3_fallback_cmd(cmd)
    if fallback_cmd and cp.returncode != 0 and _is_python_permission_denied(cp.stderr or ""):
        try:
            cp = runner.run(root=base, cwd=cwd_path, cmd=fallback_cmd, timeout_s=timeout_s)
            effective_cmd = fallback_cmd
        except subprocess.TimeoutExpired as e:
            raise ShellError(f"Command timed out after {timeout_s}s") from e
        except Exception as e:  # noqa: BLE001
            raise ShellError(f"Failed to run command: {e}") from e

    stdout = cp.stdout or ""
    stderr = cp.stderr or ""
    truncated = False
    limit = 20000
    if len(stdout) > limit:
        stdout = stdout[:limit] + "...(truncated)"
        truncated = True
    if len(stderr) > limit:
        stderr = stderr[:limit] + "...(truncated)"
        truncated = True

    return {
        "cmd": cmd,
        "effective_cmd": effective_cmd,
        "cwd": str(cwd_path),
        "exit_code": cp.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "truncated": truncated,
    }
