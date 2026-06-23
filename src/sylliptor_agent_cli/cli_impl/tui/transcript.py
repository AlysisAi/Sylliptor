"""Thread-safe conversation model for the full-screen TUI.

The agent turn runs on a worker thread and pushes content here (streamed
assistant tokens, tool-trace lines, errors); the prompt_toolkit UI thread reads
snapshots of it on every redraw. All mutation goes through a lock and triggers a
(thread-safe) ``invalidate`` so the next redraw picks up the change.

Kept free of any agent imports so it can be unit-tested in isolation.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

# (role, text) — role drives styling in the app: "user" | "assistant" | "reasoning"
# | "trace" | "error" | "warn" | "info" | "system".
Entry = tuple[str, str]


class TuiTranscript:
    def __init__(self, *, invalidate: Callable[[], None] | None = None) -> None:
        self.entries: list[Entry] = []
        self._lock = threading.RLock()
        self._status: str | None = None
        self._assistant_index: int | None = None
        # Live model reasoning ("thinking") block: index of the open reasoning
        # entry, when it started, and the elapsed seconds of each closed block
        # (keyed by entry index) so the renderer can collapse it to "thought for
        # Ns" once the answer (or a tool) follows.
        self._reasoning_index: int | None = None
        self._reasoning_start: float | None = None
        self._reasoning_secs: dict[int, int] = {}
        # Live Forge execution view: a task table + phase/spinner rendered at the
        # bottom of the transcript while ``/execute plan`` runs the swarm on a
        # worker thread. ``None`` when no run is active. Shape:
        #   {"run_id", "tasks": [{"id","title","status"}], "active": str|None,
        #    "phase": str, "message": str, "started": float, "done": bool}
        self._forge: dict[str, object] | None = None
        self._invalidate: Callable[[], None] = invalidate or (lambda: None)

    def set_invalidate(self, fn: Callable[[], None]) -> None:
        self._invalidate = fn

    def _touch(self) -> None:
        try:
            self._invalidate()
        except Exception:
            pass

    # ---- reasoning ("thinking") block ----
    def _close_reasoning_locked(self) -> None:
        """Close the open reasoning block, recording its elapsed seconds. Caller
        holds the lock. Any content/tool/turn boundary collapses thinking."""
        if self._reasoning_index is not None:
            if self._reasoning_start is not None:
                self._reasoning_secs[self._reasoning_index] = max(
                    0, int(time.monotonic() - self._reasoning_start)
                )
            self._reasoning_index = None
            self._reasoning_start = None

    def end_reasoning(self) -> None:
        """Collapse the open reasoning block (e.g. when a tool starts) without
        appending anything — records its elapsed seconds for the summary."""
        with self._lock:
            self._close_reasoning_locked()
        self._touch()

    def stream_reasoning(self, delta: str) -> None:
        if not delta:
            return
        with self._lock:
            if self._reasoning_index is None:
                self.entries.append(("reasoning", ""))
                self._reasoning_index = len(self.entries) - 1
                self._reasoning_start = time.monotonic()
            role, current = self.entries[self._reasoning_index]
            self.entries[self._reasoning_index] = (role, current + delta)
        self._touch()

    # ---- generic appends ----
    def append(self, role: str, text: str) -> None:
        with self._lock:
            self._close_reasoning_locked()
            self.entries.append((role, text))
            # Any non-streamed line closes the open assistant block.
            if role != "assistant":
                self._assistant_index = None
        self._touch()

    def append_trace(self, line: str) -> None:
        """Append a trace line, coalescing into the previous trace block.

        Consecutive trace lines (e.g. a stream of swarm-progress events) join into
        one multi-line entry instead of becoming separate blocks — the renderer
        puts a blank spacer *between* entries, so without coalescing a busy run
        would be sprayed across the pane."""
        with self._lock:
            self._close_reasoning_locked()
            if self.entries and self.entries[-1][0] == "trace":
                role, current = self.entries[-1]
                self.entries[-1] = (role, f"{current}\n{line}")
            else:
                self.entries.append(("trace", line))
            self._assistant_index = None
        self._touch()

    def append_user(self, text: str) -> None:
        with self._lock:
            self._close_reasoning_locked()
            self.entries.append(("user", text))
            self._assistant_index = None
            # A fresh submission ends the prior live Forge view (its final table
            # stays in scrollback only while it is the latest thing shown).
            self._forge = None
        self._touch()

    # ---- streaming assistant block ----
    def begin_turn(self) -> None:
        with self._lock:
            self._close_reasoning_locked()
            self._assistant_index = None
            self._status = None
        self._touch()

    def stream_assistant(self, delta: str) -> None:
        if not delta:
            return
        with self._lock:
            # The first answer token collapses the reasoning block.
            self._close_reasoning_locked()
            if self._assistant_index is None:
                self.entries.append(("assistant", ""))
                self._assistant_index = len(self.entries) - 1
            role, current = self.entries[self._assistant_index]
            self.entries[self._assistant_index] = (role, current + delta)
            self._status = None
        self._touch()

    def finish_assistant(self, text: str = "") -> None:
        with self._lock:
            self._close_reasoning_locked()
            if self._assistant_index is not None:
                role, current = self.entries[self._assistant_index]
                if not current and text:
                    self.entries[self._assistant_index] = (role, text)
            elif text.strip():
                self.entries.append(("assistant", text))
            self._assistant_index = None
            self._status = None
        self._touch()

    # ---- transient status (working line) ----
    def set_status(self, status: str | None) -> None:
        with self._lock:
            self._status = (status or None) if status is None else (status.strip() or None)
        self._touch()

    @property
    def status(self) -> str | None:
        with self._lock:
            return self._status

    # ---- live Forge execution view ----
    def forge_begin(self, run_id: str, tasks: list[tuple[str, str, str]]) -> None:
        """Open the live Forge execution view with the initial task list.

        ``tasks`` is a list of ``(id, title, status)``; statuses then refresh as
        the swarm runs (``forge_update_statuses``) and on completion
        (``forge_finish``)."""
        with self._lock:
            self._forge = {
                "run_id": str(run_id or ""),
                "tasks": [
                    {"id": str(tid), "title": str(title), "status": str(status or "planned")}
                    for tid, title, status in tasks
                ],
                "active": None,
                "phase": "execute",
                "message": "Starting…",
                "started": time.monotonic(),
                "done": False,
                "ok": True,
            }
        self._touch()

    def forge_update_statuses(self, statuses: dict[str, str]) -> None:
        """Update task statuses (keyed by task id) from the latest plan snapshot."""
        with self._lock:
            if self._forge is None:
                return
            for task in self._forge["tasks"]:  # type: ignore[index]
                new_status = statuses.get(task["id"])
                if new_status:
                    task["status"] = str(new_status)
        self._touch()

    def forge_sync_tasks(self, tasks: list[tuple[str, str, str]]) -> None:
        """Reconcile the live view with the plan on disk: update existing task
        statuses AND append any new tasks (e.g. ones plan enrichment added mid-run),
        preserving order so the table never silently drops a real task."""
        with self._lock:
            if self._forge is None:
                return
            existing = {task["id"]: task for task in self._forge["tasks"]}  # type: ignore[index]
            for tid, title, status in tasks:
                row = existing.get(str(tid))
                if row is not None:
                    if status:
                        row["status"] = str(status)
                else:
                    new_row = {
                        "id": str(tid),
                        "title": str(title),
                        "status": str(status or "planned"),
                    }
                    self._forge["tasks"].append(new_row)  # type: ignore[union-attr]
                    existing[str(tid)] = new_row
        self._touch()

    def forge_set_active(self, task_id: str | None, phase: str = "", message: str = "") -> None:
        """Mark the task the swarm is currently working on, plus the phase/message
        shown on the live status line."""
        with self._lock:
            if self._forge is None:
                return
            if task_id is not None:
                self._forge["active"] = str(task_id)
            if phase:
                self._forge["phase"] = str(phase)
            if message:
                self._forge["message"] = str(message)
        self._touch()

    def forge_finish(self, statuses: dict[str, str], summary: str = "", ok: bool = True) -> None:
        """Apply final task statuses and mark the run done (table stays visible).

        ``ok`` is the overall outcome (no failed/remaining tasks) — it colours the
        summary line green vs red so it never contradicts the per-task glyphs."""
        with self._lock:
            if self._forge is None:
                return
            for task in self._forge["tasks"]:  # type: ignore[index]
                new_status = statuses.get(task["id"])
                if new_status:
                    task["status"] = str(new_status)
            self._forge["active"] = None
            self._forge["done"] = True
            self._forge["ok"] = bool(ok)
            self._forge["phase"] = "done"
            if summary:
                self._forge["message"] = str(summary)
        self._touch()

    def forge_clear(self) -> None:
        with self._lock:
            self._forge = None
        self._touch()

    def forge_snapshot(self) -> dict[str, object] | None:
        """Return a copy of the live Forge view, or ``None`` when inactive."""
        with self._lock:
            if self._forge is None:
                return None
            return {
                "run_id": self._forge["run_id"],
                "tasks": [dict(task) for task in self._forge["tasks"]],  # type: ignore[union-attr]
                "active": self._forge["active"],
                "phase": self._forge["phase"],
                "message": self._forge["message"],
                "started": self._forge["started"],
                "done": self._forge["done"],
                "ok": self._forge.get("ok", True),
            }

    def clear(self) -> None:
        with self._lock:
            self.entries.clear()
            self._assistant_index = None
            self._reasoning_index = None
            self._reasoning_start = None
            self._reasoning_secs.clear()
            self._status = None
            self._forge = None
        self._touch()

    def load_history(self, messages: list[dict[str, object]]) -> None:
        """Replace the transcript with a resumed conversation's history.

        Clears every transient bit (like :meth:`clear`) and repopulates the pane
        from ``messages`` (the loaded history of a ``/resume`` target) so the user
        sees the prior conversation after switching sessions. Only ``user`` and
        ``assistant`` turns with text content are shown — tool calls/results are
        omitted so the reloaded view reads as the clean conversation. Because no
        assistant block stays "open" (``_assistant_index`` is reset), every
        assistant entry renders as a completed (markdown) reply.
        """
        with self._lock:
            self.entries.clear()
            self._assistant_index = None
            self._reasoning_index = None
            self._reasoning_start = None
            self._reasoning_secs.clear()
            self._status = None
            self._forge = None
            for message in messages or []:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role") or "").strip().lower()
                if role not in ("user", "assistant"):
                    continue
                content = message.get("content")
                if not isinstance(content, str):
                    continue
                text = content.rstrip()
                if not text:
                    continue
                self.entries.append((role, text))
        self._touch()

    def reasoning_snapshot(self) -> tuple[int | None, dict[int, int]]:
        """Return ``(live_reasoning_index, {entry_index: elapsed_seconds})``.

        ``live_reasoning_index`` is the still-streaming reasoning entry (rendered
        in full, dim); the map gives the duration of each closed block so the
        renderer can collapse it to a one-line "thought for Ns" summary.
        """
        with self._lock:
            return self._reasoning_index, dict(self._reasoning_secs)

    def snapshot(self) -> tuple[list[Entry], str | None, int | None]:
        """Return ``(entries, status, streaming_index)`` under the lock.

        ``streaming_index`` is the index of the still-streaming assistant entry
        (``None`` when no block is open), so the renderer keeps that one plain and
        only markdown-renders completed replies — read atomically here to avoid a
        race with a worker finishing the block mid-redraw.
        """
        with self._lock:
            return list(self.entries), self._status, self._assistant_index


__all__ = ["TuiTranscript", "Entry"]
