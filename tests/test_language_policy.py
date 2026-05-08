from __future__ import annotations

import json

import pytest

from sylliptor_agent_cli.agent_loop import (
    _build_turn_language_system_message,
    _detect_turn_language_and_script,
)
from sylliptor_agent_cli.language_policy import (
    DEFAULT_REPLY_LANGUAGE,
    DEFAULT_REPLY_SCRIPT,
    normalize_language_name,
    normalize_script_name,
)
from sylliptor_agent_cli.llm.openai_compat import LLMError, LLMResponse


class _LanguageDecisionClient:
    model = "test-model"
    temperature = 0.0

    def __init__(self, content: str | None = None, *, error: BaseException | None = None) -> None:
        self.content = content
        self.error = error
        self.calls = 0
        self.last_messages: list[dict[str, object]] = []

    def chat(self, *, messages: list[dict[str, object]], **_: object) -> LLMResponse:
        self.calls += 1
        self.last_messages = list(messages)
        if self.error is not None:
            raise self.error
        return LLMResponse(content=str(self.content or ""), tool_calls=[], raw={})


def test_normalizers_keep_model_selected_names_without_alias_mapping() -> None:
    assert normalize_language_name("  Modern   Greek  ") == "Modern Greek"
    assert normalize_script_name("  Greek   alphabet  ") == "Greek alphabet"
    assert normalize_language_name("x" * 100) == "x" * 80
    assert normalize_script_name(None) == ""


def test_turn_language_decision_uses_model_selected_reply_language() -> None:
    client = _LanguageDecisionClient(
        json.dumps(
            {
                "language": "Greek",
                "script": "Greek",
                "explicit_language_override": False,
                "confidence": 0.82,
            }
        )
    )

    decision = _detect_turn_language_and_script(
        client=client,
        instruction="ti kaneis;",
        recent_visible_history=[{"role": "assistant", "content": "previous"}],
    )

    assert client.calls == 1
    assert decision.language == "Greek"
    assert decision.script == "Greek"
    assert decision.explicit_language_override is False
    assert decision.language_source == "model"
    assert decision.confidence == pytest.approx(0.82)
    assert client.last_messages[-1] == {"role": "user", "content": "ti kaneis;"}
    assert any(
        msg.get("role") == "assistant" and msg.get("content") == "previous"
        for msg in client.last_messages
    )


def test_turn_language_decision_preserves_explicit_override_metadata() -> None:
    client = _LanguageDecisionClient(
        json.dumps(
            {
                "language": "Spanish",
                "script": "Latin",
                "explicit_language_override": True,
                "confidence": 1.0,
            }
        )
    )

    decision = _detect_turn_language_and_script(
        client=client,
        instruction="Please answer in Spanish.",
    )

    assert decision.language == "Spanish"
    assert decision.script == "Latin"
    assert decision.explicit_language_override is True
    assert decision.language_source == "explicit_request"


def test_turn_language_decision_defaults_explicitly_on_invalid_model_json() -> None:
    client = _LanguageDecisionClient("not json")

    decision = _detect_turn_language_and_script(client=client, instruction="hello")

    assert decision.language == DEFAULT_REPLY_LANGUAGE
    assert decision.script == DEFAULT_REPLY_SCRIPT
    assert decision.explicit_language_override is False
    assert decision.language_source == "fallback"
    assert decision.failure_reason == "invalid_language_decision_json"


def test_turn_language_decision_propagates_fatal_auth_errors() -> None:
    client = _LanguageDecisionClient(error=LLMError("LLM error 401: invalid_api_key"))

    with pytest.raises(LLMError, match="invalid_api_key"):
        _detect_turn_language_and_script(client=client, instruction="hello")


def test_model_determined_language_directive_is_available_without_explicit_override() -> None:
    directive = _build_turn_language_system_message(
        "Greek",
        "Greek",
        explicit_language_override=False,
    )

    assert directive is not None
    assert "selected reply language/script for this turn is model-determined" in directive
    assert "Respond in Greek using the Greek writing system" in directive
