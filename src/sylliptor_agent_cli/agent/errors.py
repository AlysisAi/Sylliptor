from __future__ import annotations


class AgentRuntimeError(RuntimeError):
    pass


class ApprovalDeclinedError(AgentRuntimeError):
    def __init__(
        self,
        approval_kind: str,
        *,
        message: str | None = None,
    ) -> None:
        self.approval_kind = str(approval_kind or "approval").strip() or "approval"
        super().__init__(message or f"User declined: {self.approval_kind}")


class SessionWorkdirError(AgentRuntimeError):
    pass
