from __future__ import annotations


class AgentRuntimeError(RuntimeError):
    pass


class SessionWorkdirError(AgentRuntimeError):
    pass
