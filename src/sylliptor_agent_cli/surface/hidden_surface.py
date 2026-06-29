from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any

from .base import Surface
from .events import Event
from .noop_surface import NoopSurface
from .types import (
    ApprovalDecision,
    ApprovalRequest,
    SubagentEndEvent,
    SubagentStartEvent,
    ToolEndEvent,
    ToolOutputEvent,
    ToolStartEvent,
)

_PARENT_SURFACE_FORWARD_LOCK = threading.RLock()


class HiddenApprovalSurface(NoopSurface):
    def __init__(self, parent_surface: Surface | None) -> None:
        self._parent_surface = parent_surface or NoopSurface()

    def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        return self._parent_surface.request_approval(request)

    def emit_message_delta(
        self,
        text: str,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        _ = (text, worker_id, role)

    def emit_message_end(
        self,
        text: str = "",
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        _ = (text, worker_id, role)

    def emit_tool_call_started(
        self,
        call_id: str,
        name: str,
        arguments_preview: str,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        _ = (call_id, name, arguments_preview, worker_id, role)

    def emit_tool_call_progress(
        self,
        call_id: str,
        text: str,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        _ = (call_id, text, worker_id, role)

    def emit_tool_call_completed(
        self,
        call_id: str,
        success: bool,
        result_preview: str,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        _ = (call_id, success, result_preview, worker_id, role)

    def emit_status_update(
        self,
        *,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        cached_tokens: int | None = None,
        cost_usd: float | None = None,
        mode: str | None = None,
        model: str | None = None,
        step: int | None = None,
        step_budget: int | None = None,
    ) -> None:
        _ = (
            tokens_in,
            tokens_out,
            cached_tokens,
            cost_usd,
            mode,
            model,
            step,
            step_budget,
        )

    def emit_mode_changed(self, mode: str) -> None:
        _ = mode

    def emit_plan_node_updated(
        self,
        node_id: str,
        state: str,
        summary: str | None = None,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        _ = (node_id, state, summary, worker_id, role)

    def emit_swarm_worker_state_changed(
        self,
        worker_id: str,
        state: str,
        *,
        role: str | None = None,
    ) -> None:
        _ = (worker_id, state, role)

    def emit_verify_gate_result(
        self,
        command: str,
        success: bool,
        summary: str,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        _ = (command, success, summary, worker_id, role)

    def emit_review_gate_decision(
        self,
        decision: str,
        summary: str,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        _ = (decision, summary, worker_id, role)

    def emit_error(
        self,
        code: str,
        message: str,
        recoverable: bool,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        _ = (code, message, recoverable, worker_id, role)

    def emit_warning(
        self,
        message: str,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        _ = (message, worker_id, role)

    def emit_info(
        self,
        message: str,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        _ = (message, worker_id, role)

    def emit_prompt_for_input(self, prompt_id: str, prompt_text: str, kind: str) -> None:
        _ = (prompt_id, prompt_text, kind)

    def emit_config_form_request(self, form_id: str, schema: dict[str, Any]) -> None:
        _ = (form_id, schema)

    def emit(self, event: Event) -> None:
        _ = event


class NestedSubagentSurface(HiddenApprovalSurface):
    def __init__(
        self,
        parent_surface: Surface | None,
        *,
        subagent_name: str,
        subagent_mode: str,
    ) -> None:
        super().__init__(parent_surface)
        self._subagent_name = subagent_name
        self._subagent_mode = subagent_mode
        self._tool_call_prefix = f"subagent:{subagent_name}:"
        self._steps_completed = 0
        self._assistant_messages_done: list[str] = []

    @property
    def steps_completed(self) -> int:
        return self._steps_completed

    @property
    def last_assistant_message_done(self) -> str:
        return self._assistant_messages_done[-1] if self._assistant_messages_done else ""

    def _scoped_worker_id(self, worker_id: str | None) -> str:
        return worker_id or self._subagent_name

    def _scoped_role(self, role: str | None) -> str:
        return role or self._subagent_mode

    def _scoped_tool_call_id(self, call_id: str) -> str:
        return f"{self._tool_call_prefix}{call_id}"

    def _call_parent_emit(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        handler = getattr(self._parent_surface, method_name, None)
        if callable(handler):
            with _PARENT_SURFACE_FORWARD_LOCK:
                handler(*args, **kwargs)

    def on_assistant_message_done(self, text: str) -> None:
        clean = str(text or "").strip()
        if clean:
            self._assistant_messages_done.append(clean)

    def on_subagent_start(self, event: SubagentStartEvent) -> None:
        handler = getattr(self._parent_surface, "on_subagent_start", None)
        if callable(handler):
            with _PARENT_SURFACE_FORWARD_LOCK:
                handler(event)

    def on_subagent_end(self, event: SubagentEndEvent) -> None:
        handler = getattr(self._parent_surface, "on_subagent_end", None)
        if callable(handler):
            with _PARENT_SURFACE_FORWARD_LOCK:
                handler(event)

    def emit_message_delta(
        self,
        text: str,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        self._call_parent_emit(
            "emit_message_delta",
            text,
            worker_id=self._scoped_worker_id(worker_id),
            role=self._scoped_role(role),
        )

    def emit_message_end(
        self,
        text: str = "",
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        self._call_parent_emit(
            "emit_message_end",
            text,
            worker_id=self._scoped_worker_id(worker_id),
            role=self._scoped_role(role),
        )

    def emit_tool_call_started(
        self,
        call_id: str,
        name: str,
        arguments_preview: str,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        self._call_parent_emit(
            "emit_tool_call_started",
            self._scoped_tool_call_id(call_id),
            name,
            arguments_preview,
            worker_id=self._scoped_worker_id(worker_id),
            role=self._scoped_role(role),
        )

    def emit_tool_call_progress(
        self,
        call_id: str,
        text: str,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        self._call_parent_emit(
            "emit_tool_call_progress",
            self._scoped_tool_call_id(call_id),
            text,
            worker_id=self._scoped_worker_id(worker_id),
            role=self._scoped_role(role),
        )

    def emit_tool_call_completed(
        self,
        call_id: str,
        success: bool,
        result_preview: str,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        self._call_parent_emit(
            "emit_tool_call_completed",
            self._scoped_tool_call_id(call_id),
            success,
            result_preview,
            worker_id=self._scoped_worker_id(worker_id),
            role=self._scoped_role(role),
        )

    def emit_status_update(
        self,
        *,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        cached_tokens: int | None = None,
        cost_usd: float | None = None,
        mode: str | None = None,
        model: str | None = None,
        step: int | None = None,
        step_budget: int | None = None,
    ) -> None:
        self._call_parent_emit(
            "emit_status_update",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cached_tokens=cached_tokens,
            cost_usd=cost_usd,
            mode=mode,
            model=model,
            step=step,
            step_budget=step_budget,
        )

    def emit_warning(
        self,
        message: str,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        self._call_parent_emit(
            "emit_warning",
            message,
            worker_id=self._scoped_worker_id(worker_id),
            role=self._scoped_role(role),
        )

    def emit_error(
        self,
        code: str,
        message: str,
        recoverable: bool,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        self._call_parent_emit(
            "emit_error",
            code,
            message,
            recoverable,
            worker_id=self._scoped_worker_id(worker_id),
            role=self._scoped_role(role),
        )

    def emit_info(
        self,
        message: str,
        *,
        worker_id: str | None = None,
        role: str | None = None,
    ) -> None:
        self._call_parent_emit(
            "emit_info",
            message,
            worker_id=self._scoped_worker_id(worker_id),
            role=self._scoped_role(role),
        )

    def on_tool_start(self, event: ToolStartEvent) -> None:
        self._steps_completed = max(self._steps_completed, int(event.step))
        self._parent_surface.on_tool_start(
            replace(
                event,
                tool_call_id=self._scoped_tool_call_id(event.tool_call_id),
                subagent_name=self._subagent_name,
                subagent_mode=self._subagent_mode,
                nesting_depth=max(int(event.nesting_depth), 0) + 1,
            )
        )

    def on_tool_output(self, event: ToolOutputEvent) -> None:
        self._parent_surface.on_tool_output(
            replace(
                event,
                tool_call_id=self._scoped_tool_call_id(event.tool_call_id),
                subagent_name=self._subagent_name,
                subagent_mode=self._subagent_mode,
                nesting_depth=max(int(event.nesting_depth), 0) + 1,
            )
        )

    def on_tool_end(self, event: ToolEndEvent) -> None:
        self._parent_surface.on_tool_end(
            replace(
                event,
                tool_call_id=self._scoped_tool_call_id(event.tool_call_id),
                subagent_name=self._subagent_name,
                subagent_mode=self._subagent_mode,
                nesting_depth=max(int(event.nesting_depth), 0) + 1,
            )
        )
