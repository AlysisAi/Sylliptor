"""``TuiSurface`` — renders agent events into the full-screen TUI transcript.

The agent runtime drives surfaces through the legacy ``on_*`` callbacks (same
path ``RichSurface`` uses), calling them synchronously from whatever thread runs
``session.run_turn``. In the TUI that is a worker thread, so every handler just
mutates the thread-safe :class:`TuiTranscript` (which schedules a redraw); none
of them touch prompt_toolkit widgets directly.

Message/tool ``emit_*`` events are additive duplicates of the ``on_*`` calls and
are intentionally left undefined (defining no-op versions would double-render).
Error/warning/info are EITHER/OR in the runtime, so we provide real delegating
``emit_error``/``emit_warning``/``emit_info`` that route to the ``on_*``
renderers — otherwise the runtime's capability probe would drop them.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from ...surface.types import (
    ApprovalDecision,
    ApprovalRequest,
    PatchEvent,
    StatusEvent,
    SubagentEndEvent,
    SubagentStartEvent,
    ToolEndEvent,
    ToolOutputEvent,
    ToolStartEvent,
)
from .transcript import TuiTranscript

# Per-worker-thread cancellation token. The TUI sets this at the top of each turn
# worker; once the user soft-interrupts, that worker's token flips cancelled and
# every output handler below drops its events — so an abandoned turn (still blocked
# waiting on a slow model) can never paint into the transcript after the interrupt.
# It is thread-local, so a *new* turn on a fresh worker is unaffected.
_worker_ctx = threading.local()


def set_active_cancellation(token: object | None) -> None:
    _worker_ctx.token = token


def _worker_cancelled() -> bool:
    token = getattr(_worker_ctx, "token", None)
    return bool(token is not None and getattr(token, "is_cancelled", False))


def _tool_label(name: str) -> str:
    try:
        from ...tools.registry import tool_display_name

        return tool_display_name(name)
    except Exception:
        return name


def _format_ms(elapsed_ms: int) -> str:
    try:
        elapsed_ms = int(elapsed_ms)
    except (TypeError, ValueError):
        return ""
    if elapsed_ms < 1000:
        return f"{elapsed_ms}ms"
    return f"{elapsed_ms / 1000.0:.1f}s"


def _truncate(text: str, *, limit: int = 160) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "…"


# Task-status buckets for the end-of-run summary — kept in sync with
# cli_common._forge_task_status_counts (the authority).
_FORGE_FAIL_SET = {
    "failed",
    "verify_failed",
    "candidate_rejected",
    "changes_requested",
    "merge_conflict",
    "blocked_integration",
    "blocked",
}
_FORGE_OBSOLETE_SET = {"superseded", "invalidated"}


class TuiSurface:
    """Surface implementation backed by a :class:`TuiTranscript`."""

    renders_error_panel = True

    def __init__(
        self,
        transcript: TuiTranscript,
        *,
        auto_approve: Callable[[], bool],
        request_approval_ui: Callable[[ApprovalRequest], ApprovalDecision] | None = None,
        on_hud_refresh: Callable[[], None] | None = None,
    ) -> None:
        self._t = transcript
        self._auto_approve = auto_approve
        self._approval_ui = request_approval_ui
        # Live footer HUD (context/tokens/cost) refresh. The agent runtime drives
        # the surface from the SAME worker thread that runs ``run_turn``, so this
        # callback reads session state at quiescent points (tool-end / message-done)
        # with no concurrent mutation. Throttled so a tool-heavy turn does not
        # re-estimate the whole transcript on every step. Without it the footer would
        # only update once the entire (possibly long, multi-step) turn completes,
        # leaving the bottom-right numbers frozen while the agent is visibly working.
        self._hud_refresh = on_hud_refresh
        self._hud_last_refresh: float = 0.0
        # Live Forge execution context (set for the duration of /execute plan): the
        # run paths (to refresh task statuses from plan.json), a debounce stamp, the
        # worker cancellation token, and the run id (to drop stale/superseded events).
        self._forge_paths: object | None = None
        self._forge_last_refresh: float = 0.0
        self._forge_token: object | None = None
        self._forge_run_id: str = ""

    def _maybe_refresh_hud(self) -> None:
        """Refresh the footer HUD mid-turn (throttled), so context/tokens/cost tick
        up while a multi-step turn runs instead of only at the end. Runs on the
        worker thread; the callback (the caller's end-of-turn refresher) mutates the
        shared ``TuiState`` and the next transcript redraw repaints the footer."""
        fn = self._hud_refresh
        if fn is None or _worker_cancelled():
            return
        import time

        now = time.monotonic()
        if now - self._hud_last_refresh < 0.4:
            return
        self._hud_last_refresh = now
        try:
            fn()
        except Exception:
            pass

    # ------------------------------------------------------------------ content
    def on_user_message(self, text: str) -> None:
        # The app echoes the user's line instantly on submit; here we just open a
        # fresh assistant block for the turn that is starting.
        self._t.begin_turn()

    def on_assistant_token(self, delta: str) -> None:
        if _worker_cancelled():
            return
        self._t.stream_assistant(delta)

    def on_reasoning_token(self, delta: str) -> None:
        # Opt-in live reasoning channel (the agent runtime calls this only when
        # the surface defines it). Streams the model's thinking into a dim block
        # that collapses once the answer arrives.
        if _worker_cancelled():
            return
        self._t.stream_reasoning(delta)

    def on_assistant_message_done(self, text: str) -> None:
        if _worker_cancelled():
            return
        self._t.finish_assistant(text or "")
        # A model response just landed (usage recorded) — refresh the live HUD so a
        # multi-step turn's bottom-right numbers advance, not just at the very end.
        self._maybe_refresh_hud()

    def on_progress_update(self, message: str) -> None:
        if _worker_cancelled():
            return
        self._t.set_status(_truncate(message, limit=80) or None)

    def on_status_update(self, status: StatusEvent) -> None:
        # Static workspace/model/mode line — the footer already carries this.
        return

    # ------------------------------------------------------------- Forge swarm
    def begin_forge(self, paths: object, token: object | None = None) -> None:
        """Open the live Forge execution view from the current plan on disk.

        Called by the worker just before ``run_swarm`` so the task table shows the
        initial statuses; the swarm then refreshes them through
        :meth:`on_swarm_event` and the final pass in :meth:`end_forge`. ``token`` is
        the worker's cancellation token — checked in :meth:`on_swarm_event` (which
        runs on the trace sink's OWN thread, where the thread-local
        ``_worker_cancelled`` would not see it) so a soft-interrupt stops painting."""
        self._forge_paths = paths
        self._forge_token = token
        self._forge_last_refresh = 0.0
        self._forge_run_id = str(getattr(paths, "run_id", "") or "")
        self._t.forge_begin(self._forge_run_id, list(self._read_plan_tasks()))

    def _forge_run_cancelled(self) -> bool:
        token = self._forge_token
        return bool(token is not None and getattr(token, "is_cancelled", False))

    def on_swarm_event(self, event: object) -> None:
        """Receive a structured swarm trace event (preferred by the serialized
        sink over :meth:`on_progress_update`): stream it as a trace/error line,
        highlight the in-flight task, and refresh statuses from plan.json.

        Runs on the trace sink's daemon thread, so it guards on the worker's token
        (not the thread-local cancel flag), on the view still being open (a new turn
        clears it), and on the event's run id matching — so a soft-interrupted or
        superseded run can never paint stale progress."""
        if self._forge_run_cancelled():
            return
        if self._t.forge_snapshot() is None:
            return  # the view was cleared (e.g. a new turn started)
        event_run_id = str(getattr(event, "run_id", "") or "")
        if event_run_id and self._forge_run_id and event_run_id != self._forge_run_id:
            return  # stale event from a previous run
        phase = str(getattr(event, "phase", "") or "")
        message = str(getattr(event, "message", "") or "")
        task_id = getattr(event, "task_id", None)
        task_id = str(task_id) if task_id else None
        # Keep the run CLEAN: the live task table + the single moving phase line tell
        # the story, so progress events are NOT spilled as a wall of trace lines.
        # Only errors are surfaced as persistent lines (they matter and are rare).
        if "error" in phase.lower():
            prefix = f"[{task_id}] " if task_id else ""
            self._t.append(
                "error", f"✗ {_truncate(f'{prefix}{message}'.strip() or phase, limit=200)}"
            )
        self._t.forge_set_active(task_id, phase=phase, message=message)
        self._maybe_refresh_forge_statuses()

    def end_forge(self, summary: str = "") -> None:
        """Apply the final task statuses from disk (adding any tasks enrichment added
        mid-run) and freeze the view as done, with a concise outcome summary."""
        tasks = self._read_plan_tasks()
        self._t.forge_sync_tasks(tasks)
        done = failed = remaining = 0
        for _tid, _title, status in tasks:
            sl = status.strip().lower()
            if sl == "done":
                done += 1
            elif sl in _FORGE_FAIL_SET:
                failed += 1
            elif sl in _FORGE_OBSOLETE_SET:
                continue
            else:
                remaining += 1
        ok = failed == 0 and remaining == 0
        if not summary:
            if not tasks:
                summary = "No execution-ready tasks."
            else:
                parts = [f"{done} done"]
                if failed:
                    parts.append(f"{failed} failed")
                if remaining:
                    parts.append(f"{remaining} remaining")
                summary = "Done · " + " · ".join(parts) if ok else "Finished · " + " · ".join(parts)
        self._t.forge_finish({tid: status for tid, _title, status in tasks}, summary, ok=ok)
        self._forge_paths = None
        self._forge_token = None
        self._forge_run_id = ""

    def append_system(self, text: str) -> None:
        """Append captured handler output (warnings / summary) as one system block."""
        clean = str(text or "").rstrip()
        if clean:
            self._t.append("system", clean)

    def append_note(self, text: str, *, role: str = "system") -> None:
        """Append a single status line with an explicit role.

        Used for the ``/resume`` outcome line so the caller can pick a role that
        flips the welcome→chat pane when needed (a resumed session with no visible
        history leaves the transcript empty, where a ``system`` line stays hidden
        behind the landing screen but an ``assistant`` line surfaces it)."""
        clean = str(text or "").rstrip()
        if clean:
            self._t.append(str(role or "system"), clean)

    def replace_history(self, messages: list[dict[str, object]]) -> None:
        """Reload the transcript with a resumed conversation's history.

        Called (on the UI thread, no turn in flight) after ``/resume`` swaps the
        live session: the prior conversation's user/assistant turns replace the
        current pane so the user actually "enters" the resumed session instead of
        keeping the previous transcript. Tool calls/results are dropped by
        :meth:`TuiTranscript.load_history` so the view stays clean."""
        self._t.load_history(messages or [])

    def _maybe_refresh_forge_statuses(self) -> None:
        import time

        now = time.monotonic()
        if now - self._forge_last_refresh < 0.4:
            return
        self._forge_last_refresh = now
        tasks = self._read_plan_tasks()
        if tasks:
            # Sync (not just update) so tasks added by plan enrichment mid-run show up.
            self._t.forge_sync_tasks(tasks)

    def _read_plan_tasks(self) -> list[tuple[str, str, str]]:
        """Read ``(id, title, status)`` for each task from plan.json (best-effort).

        Reads the raw JSON rather than the validating ``load_plan`` so a mid-run
        refresh stays cheap and never raises while the swarm rewrites the file."""
        paths = self._forge_paths
        path = getattr(paths, "plan_json_path", None)
        if path is None:
            return []
        try:
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        out: list[tuple[str, str, str]] = []
        for task in data.get("tasks") or []:
            if isinstance(task, dict) and task.get("id"):
                out.append(
                    (
                        str(task["id"]),
                        str(task.get("title") or ""),
                        str(task.get("status") or "planned"),
                    )
                )
        return out

    # -------------------------------------------------------------------- tools
    def on_tool_start(self, event: ToolStartEvent) -> None:
        # No "⚙ start" line — the live activity indicator under the question shows
        # the running tool (with a spinner + elapsed); it commits to a "✓ …" line
        # on completion. Collapse any open thinking first.
        if _worker_cancelled():
            return
        label = _tool_label(event.name)
        self._t.end_reasoning()
        self._t.set_status(f"{label}…")

    def on_tool_output(self, event: ToolOutputEvent) -> None:
        return

    def on_tool_end(self, event: ToolEndEvent) -> None:
        if _worker_cancelled():
            return
        label = _tool_label(event.name)
        elapsed = _format_ms(event.elapsed_ms)
        suffix = f" ({elapsed})" if elapsed else ""
        if event.status == "done":
            self._t.append("trace", f"✓ {label}{suffix}")
        else:
            err = ""
            if isinstance(event.meta, dict):
                err = str(event.meta.get("error") or "")
            detail = f": {_truncate(err)}" if err else ""
            self._t.append("error", f"✗ {label} failed{suffix}{detail}")
        self._t.set_status(None)
        # Between steps (no concurrent message mutation on this worker thread) is a
        # safe point to advance the footer HUD as the turn progresses.
        self._maybe_refresh_hud()

    # --------------------------------------------------------------- subagents
    def on_subagent_start(self, event: SubagentStartEvent) -> None:
        if _worker_cancelled():
            return
        self._t.append("trace", f"↪ {event.name} · {event.mode}")

    def on_subagent_end(self, event: SubagentEndEvent) -> None:
        if _worker_cancelled():
            return
        status = "finished" if event.status == "success" else event.status
        self._t.append("trace", f"↩ {event.name} · {status} · {event.steps_completed} steps")

    def on_patch_generated(self, event: PatchEvent) -> None:
        if _worker_cancelled():
            return
        files = ", ".join(event.files[:5]) or "patch"
        self._t.append("trace", f"✎ patch · {files}")

    # ---------------------------------------------------------------- diagnostics
    def on_error(self, err: str) -> None:
        if _worker_cancelled():
            return
        self._t.append("error", _truncate(str(err), limit=480) or "error")
        self._t.set_status(None)

    def on_warning(self, warning: str) -> None:
        if _worker_cancelled():
            return
        self._t.append("warn", _truncate(str(warning), limit=480) or "warning")

    # ------------------------------------------------------------------ approval
    def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        if _worker_cancelled():
            # The user interrupted this (now-abandoned) turn; deny without prompting.
            return ApprovalDecision(allow=False)
        try:
            auto = bool(self._auto_approve())
        except Exception:
            auto = False
        if auto:
            return ApprovalDecision(allow=True, allow_for_session=True)
        if self._approval_ui is not None:
            try:
                decision = self._approval_ui(request)
            except Exception:
                decision = None
            if isinstance(decision, ApprovalDecision):
                return decision
        # No interactive approver reachable → fail closed.
        self._t.append("warn", f"Denied {request.kind}: approval required (auto-approve is off).")
        return ApprovalDecision(allow=False)

    # ----------------------------------------------------------------- emit path
    # The agent dual-emits: message/tool events go through additive ``emit_*``
    # helpers (skipped when absent) AND the legacy ``on_*`` calls we implement
    # above — so we deliberately do NOT define ``emit_message_*`` /
    # ``emit_tool_call_*`` (that would double-render). Error/warning/info, by
    # contrast, are EITHER/OR: the runtime prefers a real (non-Noop) ``emit_*``
    # and only falls back to ``on_*`` otherwise. We therefore define real
    # delegating handlers so those never get silently dropped.
    #
    # Note: do not add a catch-all ``__getattr__`` here — a synthesized no-op
    # ``emit_error``/``emit_warning`` looks "real" to the runtime's capability
    # probe and would swallow every error/warning in the TUI.
    def emit_error(
        self,
        code: str,
        message: str,
        recoverable: bool,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        prefix = f"{code}: " if code else ""
        suffix = "" if recoverable else " (not recoverable)"
        self.on_error(f"{prefix}{message}{suffix}")

    def emit_warning(
        self,
        message: str,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        self.on_warning(str(message))

    def emit_info(
        self,
        message: str,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        self.on_progress_update(str(message))


__all__ = ["TuiSurface", "set_active_cancellation"]
