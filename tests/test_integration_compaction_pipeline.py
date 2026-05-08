from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sylliptor_agent_cli.agent_loop import AgentSession, create_session
from sylliptor_agent_cli.compaction.conversation_compactor import MEMORY_MARKER, PINS_MARKER
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.llm.openai_compat import LLMError, LLMResponse, ToolCall
from sylliptor_agent_cli.token_budget import estimate_tokens


class ScriptedClient:
    def __init__(self, *, model: str, responses: list[LLMResponse]) -> None:
        self.model = model
        self.temperature = 1.0
        self._responses = responses
        self.calls = 0

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        on_text_delta=None,  # type: ignore[no-untyped-def]
    ) -> LLMResponse:
        _ = messages, tools, stream, on_text_delta
        if self.calls >= len(self._responses):
            raise AssertionError("ScriptedClient exhausted responses")
        response = self._responses[self.calls]
        self.calls += 1
        return response


class RetrySizingCompactor:
    def __init__(self) -> None:
        self.model = "compactor-model"
        self.temperature = 0.2
        self.calls = 0
        self.first_tokens: int | None = None
        self.last_tokens: int | None = None
        self.retried_with_smaller_payload = False

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        on_text_delta=None,  # type: ignore[no-untyped-def]
    ) -> LLMResponse:
        _ = tools, stream, on_text_delta
        payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        tokens = estimate_tokens(payload)
        self.calls += 1
        if self.calls == 1:
            self.first_tokens = tokens
            raise LLMError("context length exceeded")
        self.last_tokens = tokens
        if self.first_tokens is None:
            raise AssertionError("Expected first_tokens to be set before retry")
        if tokens >= self.first_tokens:
            raise AssertionError("Retry payload was not reduced")
        self.retried_with_smaller_payload = True
        summary = {
            "goal": "integration",
            "constraints": ["do not break public API"],
            "decisions": [],
            "work_done": [],
            "open_threads": [],
            "next_steps": [],
        }
        return LLMResponse(content=json.dumps(summary), tool_calls=[], raw={})


class AlwaysFailCompactor:
    def __init__(self) -> None:
        self.model = "compactor-model"
        self.temperature = 0.2
        self.calls = 0

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        on_text_delta=None,  # type: ignore[no-untyped-def]
    ) -> LLMResponse:
        _ = messages, tools, stream, on_text_delta
        self.calls += 1
        raise LLMError("context length exceeded")


def _build_cfg() -> AppConfig:
    cfg = AppConfig(
        model="test-model",
        stream=False,
        max_steps=6,
        temperature=1.0,
        routing_mode="code_only",
    )
    cfg.extra_fields = {
        "model_metadata_overrides": {
            "models": {
                "test-model": {"context_window_tokens": 4096, "max_output_tokens": 512},
                "compactor-model": {"context_window_tokens": 2048, "max_output_tokens": 256},
            },
            "default": {"context_window_tokens": 4096, "max_output_tokens": 512},
        },
        "compaction": {
            "enabled": True,
            "summarize_conversation": True,
            "offload_tool_outputs": True,
            "tool_output_offload_threshold_chars": 500,
            "tool_output_preview_chars": 120,
            "recent_user_turns_to_keep": 1,
            "trigger_ratio": 0.45,
            "target_ratio": 0.30,
            "max_chunk_messages": 60,
            "safety_margin_tokens": 256,
            "importance_enabled": True,
            "importance_strategy": "lowest_density",
            "pin_score_threshold": 6.0,
            "max_pins": 20,
            "pin_snippet_chars": 200,
        },
    }
    return cfg


def _append_turn(session: AgentSession, *, user_text: str, assistant_text: str) -> None:
    session.messages.append({"role": "user", "content": user_text})
    session.messages.append({"role": "assistant", "content": assistant_text})


def _compaction_session_dir(session: AgentSession) -> Path:
    return session.store.session_artifact_root


def test_run_turn_compaction_and_offload_creates_artifacts_and_pins(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_MODEL_COMPACTOR", "compactor-model")
    big_file = tmp_path / "big.txt"
    big_file.write_text("A" * 24000, encoding="utf-8")

    session = create_session(
        cfg=_build_cfg(),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=6,
        no_log=True,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / ".sylliptor" / "sessions",
        session_id_override="integration-sid",
        enable_compaction=True,
    )
    try:
        assert session.conversation_compactor is not None
        session.client = ScriptedClient(  # type: ignore[assignment]
            model="test-model",
            responses=[
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="tc-read",
                            name="fs_read",
                            arguments={"path": "big.txt", "max_bytes": 20000},
                        )
                    ],
                    raw={},
                ),
                LLMResponse(content="Final answer.", tool_calls=[], raw={}),
            ],
        )

        compactor_summary = {
            "goal": "integration",
            "constraints": ["do not change public API"],
            "decisions": ["keep review mode defaults"],
            "work_done": [],
            "open_threads": [],
            "next_steps": [],
        }
        session.conversation_compactor.compactor_client = ScriptedClient(  # type: ignore[assignment]
            model="compactor-model",
            responses=[
                LLMResponse(content=json.dumps(compactor_summary), tool_calls=[], raw={})
                for _ in range(12)
            ],
        )

        _append_turn(
            session,
            user_text="ok " + ("l" * 9000),
            assistant_text="ack " + ("m" * 9000),
        )
        _append_turn(
            session,
            user_text=(
                "MUST keep API stable. DO NOT rename commands. "
                "Acceptance criteria: tests pass. Constraint: preserve behavior. " + ("h" * 15000)
            ),
            assistant_text="noted " + ("n" * 4000),
        )
        _append_turn(
            session,
            user_text="recent context " + ("r" * 5000),
            assistant_text="recent ack",
        )

        exit_code = session.run_turn("Please read big.txt and summarize key points.")
        assert exit_code == 0

        compaction_session_dir = _compaction_session_dir(session)
        offload_session_dir = session.store.session_artifact_root
        history_files = sorted((compaction_session_dir / "history").glob("chunk_*.jsonl"))
        assert history_files, "Expected at least one compacted history chunk"
        history_text = "\n".join(path.read_text(encoding="utf-8") for path in history_files)
        assert "MUST keep API stable" in history_text
        summary_path = compaction_session_dir / "memory" / "summary.json"
        assert summary_path.exists()
        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary_data.get("goal") == "integration"
        assert "keep review mode defaults" in (summary_data.get("decisions") or [])
        pins_path = compaction_session_dir / "memory" / "pins.json"
        assert pins_path.exists()
        offload_files = sorted((offload_session_dir / "tool_outputs").glob("*.json"))
        assert offload_files, "Expected offloaded tool output artifact(s)"
        offload_data = json.loads(offload_files[0].read_text(encoding="utf-8"))
        assert offload_data.get("tool_name") == "fs_read"
        assert isinstance(offload_data.get("result"), dict)
        full_content = str((offload_data.get("result") or {}).get("content") or "")
        assert len(full_content) > 10000
        assert "A" * 1000 in full_content

        tool_messages = [m for m in session.messages if str(m.get("role")) == "tool"]
        assert tool_messages, "Expected tool message in conversation"
        tool_stub = json.loads(str(tool_messages[-1].get("content") or "{}"))
        assert tool_stub.get("offloaded") is True
        assert tool_stub.get("artifact_locator") == (
            "session_artifacts/tool_outputs/step1_fs_read_tc-read.json"
        )
        assert tool_stub.get("artifact_readable_via_fs") is True

        pins_data = json.loads(pins_path.read_text(encoding="utf-8"))
        assert any(
            (
                "MUST keep API stable" in str(pin.get("text") or "")
                or "Acceptance criteria" in str(pin.get("text") or "")
            )
            for pin in pins_data.get("pins", [])
        )
        assert any(
            str(m.get("content") or "").startswith(MEMORY_MARKER)
            for m in session.messages
            if str(m.get("role")) == "user"
        )
        assert any(
            str(m.get("content") or "").startswith(PINS_MARKER)
            for m in session.messages
            if str(m.get("role")) == "user"
        )
    finally:
        session.close()


def test_compactor_context_length_retry_then_success_end_to_end(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_MODEL_COMPACTOR", "compactor-model")
    session = create_session(
        cfg=_build_cfg(),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "session-artifacts",
        session_id_override="integration-retry",
        enable_compaction=True,
    )
    try:
        assert session.conversation_compactor is not None
        session.client = ScriptedClient(  # type: ignore[assignment]
            model="test-model",
            responses=[LLMResponse(content="done", tool_calls=[], raw={})],
        )
        retry_compactor = RetrySizingCompactor()
        session.conversation_compactor.compactor_client = retry_compactor  # type: ignore[assignment]

        _append_turn(
            session,
            user_text="very large low signal " + ("x" * 22000),
            assistant_text="assistant filler " + ("y" * 22000),
        )
        _append_turn(
            session,
            user_text="recent keep " + ("z" * 3000),
            assistant_text="recent assistant",
        )

        exit_code = session.run_turn("continue")
        assert exit_code == 0
        assert retry_compactor.calls >= 2
        assert retry_compactor.first_tokens is not None
        assert retry_compactor.last_tokens is not None
        assert retry_compactor.retried_with_smaller_payload is True

        session_dir = _compaction_session_dir(session)
        summary_path = session_dir / "memory" / "summary.json"
        assert summary_path.exists()
        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary_data.get("goal") == "integration"
        assert list((session_dir / "history").glob("chunk_*.jsonl"))
    finally:
        session.close()


def test_compactor_fails_both_attempts_drop_chunk_still_completes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_MODEL_COMPACTOR", "compactor-model")
    session = create_session(
        cfg=_build_cfg(),
        root=tmp_path,
        mode="auto",
        yes=True,
        max_steps=4,
        no_log=True,
        api_key_override="override-key",
        session_log_dir_override=tmp_path / "session-artifacts",
        session_id_override="integration-dropchunk",
        enable_compaction=True,
    )
    try:
        assert session.conversation_compactor is not None
        session.client = ScriptedClient(  # type: ignore[assignment]
            model="test-model",
            responses=[LLMResponse(content="done", tool_calls=[], raw={})],
        )
        failing_compactor = AlwaysFailCompactor()
        session.conversation_compactor.compactor_client = failing_compactor  # type: ignore[assignment]

        high_signal = (
            "MUST preserve this requirement. DO NOT remove approval UX. "
            "Acceptance criteria: no stalls. " + ("q" * 18000)
        )
        _append_turn(
            session,
            user_text=high_signal,
            assistant_text="ack " + ("w" * 8000),
        )
        _append_turn(
            session,
            user_text="recent keep " + ("e" * 3000),
            assistant_text="recent assistant",
        )

        exit_code = session.run_turn("continue")
        assert exit_code == 0
        assert failing_compactor.calls >= 2

        session_dir = _compaction_session_dir(session)
        assert list((session_dir / "history").glob("chunk_*.jsonl"))
        assert (session_dir / "memory" / "summary.json").exists()
        pins_path = session_dir / "memory" / "pins.json"
        assert pins_path.exists()
        pins_data = json.loads(pins_path.read_text(encoding="utf-8"))
        assert any(
            (
                "MUST preserve this requirement" in str(pin.get("text") or "")
                or "Acceptance criteria" in str(pin.get("text") or "")
            )
            for pin in pins_data.get("pins", [])
        )
        assert any(
            str(m.get("content") or "").startswith(PINS_MARKER)
            for m in session.messages
            if str(m.get("role")) == "user"
        )
        assert any(
            str(m.get("content") or "").startswith(MEMORY_MARKER)
            for m in session.messages
            if str(m.get("role")) == "user"
        )
        serialized_messages = json.dumps(session.messages, ensure_ascii=False)
        assert high_signal not in serialized_messages
    finally:
        session.close()
