from __future__ import annotations

import fnmatch
import json
import os
import queue
import re
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


class SearchError(RuntimeError):
    pass


_RG_NO_CONFIG_SUPPORTED: bool | None = None
_SEARCH_RG_MAX_MATCHES_PER_FILE = 8
_SEARCH_RG_MAX_MATCH_TEXT_CHARS = 240
_SEARCH_RG_MAX_OUTPUT_CHARS = 12_000
_RG_PROBE_TIMEOUT_S = 2.0
_SEARCH_RG_TIMEOUT_S = 15.0


def _rg_available() -> bool:
    try:
        cp = subprocess.run(
            ["rg", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=_RG_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return cp.returncode == 0


def _rg_supports_no_config() -> bool:
    global _RG_NO_CONFIG_SUPPORTED
    if _RG_NO_CONFIG_SUPPORTED is not None:
        return _RG_NO_CONFIG_SUPPORTED
    try:
        cp = subprocess.run(
            ["rg", "--no-config", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=_RG_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        _RG_NO_CONFIG_SUPPORTED = False
        return False
    _RG_NO_CONFIG_SUPPORTED = cp.returncode == 0
    return _RG_NO_CONFIG_SUPPORTED


def _clip_match_text(
    text: str, *, max_chars: int = _SEARCH_RG_MAX_MATCH_TEXT_CHARS
) -> tuple[str, bool]:
    normalized = text.rstrip("\n")
    if len(normalized) <= max_chars:
        return normalized, False
    return normalized[:max_chars].rstrip() + "...(truncated)", True


def _rel_path(root: Path, path: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return os.fspath(path).replace("\\", "/")
    return os.fspath(rel).replace("\\", "/")


def _clean_globs(globs: list[str] | None) -> list[str] | None:
    cleaned = [str(pattern) for pattern in (globs or []) if str(pattern).strip()]
    return cleaned or None


def _matches_globs(rel_path: str, globs: list[str] | None) -> bool:
    if not globs:
        return True
    return any(fnmatch.fnmatchcase(rel_path, pattern) for pattern in globs)


def _iter_candidate_files(*, root: Path, base: Path, globs: list[str] | None):
    if base.is_file():
        # ripgrep does not apply --glob filtering to an explicitly targeted file path
        yield base
        return

    for path in base.rglob("*"):
        if not path.is_file():
            continue
        rel = _rel_path(base, path)
        if _matches_globs(rel, globs):
            yield path


def _queue_text_stream_lines(
    stream: Any,
    output_queue: queue.Queue[str | object],
    done_sentinel: object,
) -> None:
    try:
        for line in iter(stream.readline, ""):
            output_queue.put(line)
    finally:
        try:
            stream.close()
        except Exception:
            pass
        output_queue.put(done_sentinel)


def _collect_text_stream(stream: Any, collector: list[str]) -> None:
    try:
        for line in iter(stream.readline, ""):
            collector.append(line)
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _stop_process(process: Any) -> int:
    try:
        process.terminate()
    except OSError:
        pass
    try:
        return int(process.wait(timeout=0.5))
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        return int(process.wait())


def search_rg(
    *,
    root: Path,
    pattern: str,
    root_path: str = ".",
    globs: list[str] | None = None,
    max_results: int = 200,
) -> dict[str, Any]:
    root_abs = root.resolve()
    base = (root_abs / root_path).resolve()
    try:
        base.relative_to(root_abs)
    except ValueError as e:
        raise SearchError(f"root_path escapes root: {root_path}") from e
    if pattern == "":
        raise SearchError("pattern must be a non-empty string")

    safe_max_results = max(1, int(max_results))
    cleaned_globs = _clean_globs(globs)
    matches: list[dict[str, Any]] = []
    per_file_match_counts: dict[str, int] = defaultdict(int)
    output_chars = 0
    truncated = False
    per_file_truncated = False
    match_text_truncated = False

    def _timeout_message(*, backend: str) -> str:
        return (
            f"{backend} search timed out after {_SEARCH_RG_TIMEOUT_S:g}s; "
            "try a narrower root_path, globs, or pattern"
        )

    def _append_match(*, path: str, line_number: int, line_text: str) -> str:
        nonlocal output_chars, truncated, per_file_truncated, match_text_truncated
        if per_file_match_counts[path] >= _SEARCH_RG_MAX_MATCHES_PER_FILE:
            truncated = True
            per_file_truncated = True
            return "per_file_limit"
        clipped_text, clipped = _clip_match_text(line_text)
        entry = {
            "path": path,
            "line": int(line_number),
            "text": clipped_text,
        }
        if clipped:
            entry["text_truncated"] = True
            match_text_truncated = True
        entry_chars = len(json.dumps(entry, ensure_ascii=True, separators=(",", ":")))
        if len(matches) >= safe_max_results:
            truncated = True
            return "max_results"
        if matches and (output_chars + entry_chars) > _SEARCH_RG_MAX_OUTPUT_CHARS:
            truncated = True
            return "output_cap"
        matches.append(entry)
        per_file_match_counts[path] += 1
        output_chars += entry_chars
        return "appended"

    if _rg_available():
        cmd = ["rg"]
        if _rg_supports_no_config():
            cmd.append("--no-config")
        cmd.extend(
            [
                "--json",
                "--line-buffered",
                "-e",
                pattern,
            ]
        )
        if cleaned_globs:
            for g in cleaned_globs:
                cmd.extend(["--glob", g])
        cmd.append(os.fspath(base))
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as e:
            raise SearchError("Failed to execute ripgrep") from e

        if proc.stdout is None or proc.stderr is None:
            raise SearchError("Failed to capture ripgrep output")

        stdout_queue: queue.Queue[str | object] = queue.Queue()
        stdout_done = object()
        stderr_lines: list[str] = []
        stdout_thread = threading.Thread(
            target=_queue_text_stream_lines,
            args=(proc.stdout, stdout_queue, stdout_done),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_collect_text_stream,
            args=(proc.stderr, stderr_lines),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        deadline = time.monotonic() + _SEARCH_RG_TIMEOUT_S
        stdout_finished = False
        terminated_early = False

        while not stdout_finished:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _stop_process(proc)
                stdout_thread.join(timeout=0.5)
                stderr_thread.join(timeout=0.5)
                raise SearchError(_timeout_message(backend="ripgrep"))
            try:
                line = stdout_queue.get(timeout=min(0.1, remaining))
            except queue.Empty:
                if proc.poll() is not None and not stdout_thread.is_alive():
                    stdout_finished = True
                continue
            if line is stdout_done:
                stdout_finished = True
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") != "match":
                continue
            data = obj.get("data") or {}
            path = ((data.get("path") or {}).get("text")) or ""
            line_number = data.get("line_number")
            lines = ((data.get("lines") or {}).get("text")) or ""
            if not path or not line_number:
                continue
            try:
                rel = os.fspath(Path(path).resolve().relative_to(root_abs))
            except ValueError:
                rel = path
            append_status = _append_match(path=rel, line_number=int(line_number), line_text=lines)
            if append_status == "max_results":
                terminated_early = True
                break
            if append_status == "output_cap":
                terminated_early = True
                break

        returncode = _stop_process(proc) if terminated_early else int(proc.wait())
        stdout_thread.join(timeout=0.5)
        stderr_thread.join(timeout=0.5)
        stderr_text = "".join(stderr_lines).strip()
        if not terminated_early and returncode not in (0, 1):  # 1 means no matches
            raise SearchError(stderr_text or f"rg failed with exit code {returncode}")

        return {
            "pattern": pattern,
            "matches": matches,
            "backend": "rg",
            "truncated": truncated,
            "per_file_truncated": per_file_truncated,
            "match_text_truncated": match_text_truncated,
            "returned_matches": len(matches),
        }

    # Python fallback (regex).
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise SearchError(f"Invalid regex: {e}") from e

    deadline = time.monotonic() + _SEARCH_RG_TIMEOUT_S

    def _raise_if_python_timeout() -> None:
        if time.monotonic() >= deadline:
            raise SearchError(_timeout_message(backend="python fallback"))

    for p in _iter_candidate_files(root=root_abs, base=base, globs=cleaned_globs):
        _raise_if_python_timeout()
        if len(matches) >= safe_max_results:
            truncated = True
            break
        # Skip large files.
        try:
            if p.stat().st_size > 512 * 1024:
                continue
        except OSError:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        stop_search = False
        for i, line in enumerate(text.splitlines(), start=1):
            _raise_if_python_timeout()
            if len(matches) >= safe_max_results:
                truncated = True
                stop_search = True
                break
            if rx.search(line):
                rel = _rel_path(root_abs, p)
                append_status = _append_match(path=rel, line_number=i, line_text=line)
                if append_status == "max_results":
                    stop_search = True
                    break
                if append_status == "output_cap":
                    stop_search = True
                    break
                if append_status == "per_file_limit":
                    break
        if stop_search:
            break

    return {
        "pattern": pattern,
        "matches": matches,
        "backend": "python",
        "truncated": truncated,
        "per_file_truncated": per_file_truncated,
        "match_text_truncated": match_text_truncated,
        "returned_matches": len(matches),
    }
