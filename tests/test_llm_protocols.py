from __future__ import annotations

from sylliptor_agent_cli.config import ConfigError
from sylliptor_agent_cli.llm.protocols import (
    ANTHROPIC_MESSAGES_PROTOCOL,
    GEMINI_GENERATE_CONTENT_PROTOCOL,
    GEMINI_INTERACTIONS_PROTOCOL,
    OPENAI_COMPAT_PROTOCOL,
    OPENAI_RESPONSES_PROTOCOL,
    SUPPORTED_LLM_PROTOCOLS,
    UnsupportedProtocolError,
    get_provider_protocol_capabilities,
)


def test_supported_llm_protocols_include_native_foundation_values() -> None:
    assert SUPPORTED_LLM_PROTOCOLS == {
        OPENAI_COMPAT_PROTOCOL,
        OPENAI_RESPONSES_PROTOCOL,
        ANTHROPIC_MESSAGES_PROTOCOL,
        GEMINI_GENERATE_CONTENT_PROTOCOL,
        GEMINI_INTERACTIONS_PROTOCOL,
    }


def test_unsupported_protocol_error_is_config_error() -> None:
    assert issubclass(UnsupportedProtocolError, ConfigError)


def test_provider_protocol_capabilities_describe_native_web_search_adapters() -> None:
    anthropic_compat = get_provider_protocol_capabilities(
        provider_key="anthropic",
        protocol=OPENAI_COMPAT_PROTOCOL,
    )
    openai_responses = get_provider_protocol_capabilities(
        provider_key="openai",
        protocol=OPENAI_RESPONSES_PROTOCOL,
    )
    anthropic = get_provider_protocol_capabilities(
        provider_key="anthropic",
        protocol=ANTHROPIC_MESSAGES_PROTOCOL,
    )
    gemini = get_provider_protocol_capabilities(
        provider_key="gemini",
        protocol=GEMINI_GENERATE_CONTENT_PROTOCOL,
    )
    interactions = get_provider_protocol_capabilities(
        provider_key="gemini",
        protocol=GEMINI_INTERACTIONS_PROTOCOL,
    )

    assert anthropic_compat is not None
    assert anthropic_compat.supports_native_web_search is False
    assert anthropic_compat.supports_provider_hosted_web_search_adapter is False
    assert anthropic_compat.default_web_search_adapter == "anthropic_messages"

    assert openai_responses is not None
    assert openai_responses.supports_streaming is True
    assert openai_responses.supports_tool_calling is True
    assert openai_responses.supports_provider_hosted_web_search_adapter is True
    assert openai_responses.default_web_search_adapter == "openai_responses"
    assert any("Native OpenAI Responses chat supports" in q for q in openai_responses.quirks)

    assert anthropic is not None
    assert anthropic.supports_streaming is True
    assert anthropic.supports_tool_calling is True
    assert anthropic.supports_native_web_search is True
    assert anthropic.supports_provider_hosted_web_search_adapter is True
    assert anthropic.default_web_search_adapter == "anthropic_messages"
    assert "reasoning_effort" in anthropic.unsupported_parameters
    assert any("Native Anthropic Messages chat supports" in q for q in anthropic.quirks)

    assert gemini is not None
    assert gemini.supports_streaming is True
    assert gemini.supports_tool_calling is True
    assert gemini.supports_native_web_search is True
    assert gemini.supports_provider_hosted_web_search_adapter is True
    assert gemini.default_web_search_adapter == "gemini_grounding"
    assert "reasoning_effort:xhigh" not in gemini.unsupported_parameters
    assert any("Native Gemini GenerateContent chat supports" in q for q in gemini.quirks)

    assert interactions is not None
    assert interactions.supports_streaming is False
    assert interactions.supports_tool_calling is False
    assert interactions.supports_provider_hosted_web_search_adapter is False
    assert interactions.default_web_search_adapter == "gemini_grounding"
    assert "tools" in interactions.unsupported_parameters
    assert any("Experimental Gemini Interactions prototype" in q for q in interactions.quirks)
