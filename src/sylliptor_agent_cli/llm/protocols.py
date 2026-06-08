from __future__ import annotations

from dataclasses import dataclass

from ..config import ConfigError

OPENAI_COMPAT_PROTOCOL = "openai_compat"
OPENAI_RESPONSES_PROTOCOL = "openai_responses"
ANTHROPIC_MESSAGES_PROTOCOL = "anthropic_messages"
GEMINI_GENERATE_CONTENT_PROTOCOL = "gemini_generate_content"
GEMINI_INTERACTIONS_PROTOCOL = "gemini_interactions"

SUPPORTED_LLM_PROTOCOLS: frozenset[str] = frozenset(
    {
        OPENAI_COMPAT_PROTOCOL,
        OPENAI_RESPONSES_PROTOCOL,
        ANTHROPIC_MESSAGES_PROTOCOL,
        GEMINI_GENERATE_CONTENT_PROTOCOL,
        GEMINI_INTERACTIONS_PROTOCOL,
    }
)

_PROVIDER_CAPABILITY_ALIASES: dict[str, str] = {
    "vertex_ai": "gemini",
    "vertex_ai-language-models": "gemini",
    "vertex-ai-language-models": "gemini",
    "google": "gemini",
    "google_ai": "gemini",
    "google-ai": "gemini",
}


@dataclass(frozen=True)
class ProviderProtocolCapabilities:
    """Provider capabilities separated by chat protocol and web-search adapter surface.

    The `supports_streaming`, `supports_tool_calling`, and `supports_structured_outputs`
    fields describe the chat transport named by `protocol`. The web-search fields describe
    the separate `web_search` runtime adapter that may exist even while chat uses
    `openai_compat`.
    """

    provider_key: str
    protocol: str
    supports_streaming: bool
    supports_tool_calling: bool
    supports_structured_outputs: bool
    supports_provider_hosted_web_search_adapter: bool
    default_web_search_adapter: str
    unsupported_parameters: tuple[str, ...] = ()
    quirks: tuple[str, ...] = ()

    @property
    def supports_native_web_search(self) -> bool:
        """Backward-compatible alias for the provider-hosted web-search adapter flag."""
        return self.supports_provider_hosted_web_search_adapter


class UnsupportedProtocolError(ConfigError):
    """Raised when a recognized provider protocol lacks a chat client implementation."""


PROVIDER_PROTOCOL_CAPABILITIES: tuple[ProviderProtocolCapabilities, ...] = (
    ProviderProtocolCapabilities(
        provider_key="openai",
        protocol=OPENAI_COMPAT_PROTOCOL,
        supports_streaming=True,
        supports_tool_calling=True,
        supports_structured_outputs=True,
        supports_provider_hosted_web_search_adapter=False,
        default_web_search_adapter="openai_responses",
        quirks=("Sylliptor's current chat runtime uses Chat Completions-compatible requests.",),
    ),
    ProviderProtocolCapabilities(
        provider_key="openai",
        protocol=OPENAI_RESPONSES_PROTOCOL,
        supports_streaming=True,
        supports_tool_calling=True,
        supports_structured_outputs=True,
        supports_provider_hosted_web_search_adapter=True,
        default_web_search_adapter="openai_responses",
        quirks=("Native OpenAI Responses chat supports buffered and SSE streaming responses.",),
    ),
    ProviderProtocolCapabilities(
        provider_key="anthropic",
        protocol=OPENAI_COMPAT_PROTOCOL,
        supports_streaming=True,
        supports_tool_calling=True,
        supports_structured_outputs=False,
        supports_provider_hosted_web_search_adapter=False,
        default_web_search_adapter="anthropic_messages",
        unsupported_parameters=("reasoning_effort",),
        quirks=(
            "Current Claude chat path uses Anthropic's OpenAI-compatible endpoint.",
            "Advanced Claude features require the native Messages API.",
        ),
    ),
    ProviderProtocolCapabilities(
        provider_key="anthropic",
        protocol=ANTHROPIC_MESSAGES_PROTOCOL,
        supports_streaming=True,
        supports_tool_calling=True,
        supports_structured_outputs=False,
        supports_provider_hosted_web_search_adapter=True,
        default_web_search_adapter="anthropic_messages",
        unsupported_parameters=("reasoning_effort",),
        quirks=("Native Anthropic Messages chat supports buffered and SSE streaming responses.",),
    ),
    ProviderProtocolCapabilities(
        provider_key="gemini",
        protocol=OPENAI_COMPAT_PROTOCOL,
        supports_streaming=True,
        supports_tool_calling=True,
        supports_structured_outputs=True,
        supports_provider_hosted_web_search_adapter=False,
        default_web_search_adapter="gemini_grounding",
        unsupported_parameters=("reasoning_effort:xhigh",),
        quirks=("Current Gemini chat path uses Google's OpenAI-compatible v1beta endpoint.",),
    ),
    ProviderProtocolCapabilities(
        provider_key="gemini",
        protocol=GEMINI_GENERATE_CONTENT_PROTOCOL,
        supports_streaming=True,
        supports_tool_calling=True,
        supports_structured_outputs=True,
        supports_provider_hosted_web_search_adapter=True,
        default_web_search_adapter="gemini_grounding",
        quirks=(
            "Native Gemini GenerateContent chat supports buffered and streamGenerateContent "
            "SSE responses.",
        ),
    ),
    ProviderProtocolCapabilities(
        provider_key="gemini",
        protocol=GEMINI_INTERACTIONS_PROTOCOL,
        supports_streaming=False,
        supports_tool_calling=False,
        supports_structured_outputs=False,
        supports_provider_hosted_web_search_adapter=False,
        default_web_search_adapter="gemini_grounding",
        unsupported_parameters=("stream", "tools", "tool_choice", "response_format", "web_search"),
        quirks=(
            "Experimental Gemini Interactions prototype; gated by "
            "SYLLIPTOR_EXPERIMENTAL_GEMINI_INTERACTIONS=1 or "
            "experimental_gemini_interactions_enabled=true.",
            "Only text-only buffered requests are implemented; Gemini GenerateContent remains "
            "the stable native Gemini protocol.",
        ),
    ),
)


def get_provider_protocol_capabilities(
    *,
    provider_key: str,
    protocol: str,
) -> ProviderProtocolCapabilities | None:
    normalized_provider = str(provider_key or "").strip().lower()
    normalized_provider = _PROVIDER_CAPABILITY_ALIASES.get(
        normalized_provider,
        normalized_provider,
    )
    normalized_protocol = str(protocol or "").strip().lower()
    for capabilities in PROVIDER_PROTOCOL_CAPABILITIES:
        if (
            capabilities.provider_key == normalized_provider
            and capabilities.protocol == normalized_protocol
        ):
            return capabilities
    return None
