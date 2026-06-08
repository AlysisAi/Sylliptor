from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .llm.protocols import (
    ANTHROPIC_MESSAGES_PROTOCOL,
    GEMINI_GENERATE_CONTENT_PROTOCOL,
    GEMINI_INTERACTIONS_PROTOCOL,
    OPENAI_COMPAT_PROTOCOL,
    OPENAI_RESPONSES_PROTOCOL,
)
from .profiles import ProfileSpec
from .web_search_adapters import (
    ANTHROPIC_MESSAGES_ADAPTER,
    AUTO_WEB_SEARCH_ADAPTER,
    DASHSCOPE_CHAT_ADAPTER,
    GEMINI_GROUNDING_ADAPTER,
    GROQ_COMPOUND_ADAPTER,
    MISTRAL_CONVERSATIONS_ADAPTER,
    MOONSHOT_KIMI_ADAPTER,
    OPENAI_RESPONSES_ADAPTER,
    OPENROUTER_WEB_ADAPTER,
    PERPLEXITY_SONAR_ADAPTER,
    VOLCENGINE_WEB_SEARCH_ADAPTER,
    XAI_RESPONSES_ADAPTER,
    ZHIPU_WEB_SEARCH_ADAPTER,
)

NATIVE_PROFILE_PROTOCOLS: frozenset[str] = frozenset(
    {
        OPENAI_RESPONSES_PROTOCOL,
        ANTHROPIC_MESSAGES_PROTOCOL,
        GEMINI_GENERATE_CONTENT_PROTOCOL,
        GEMINI_INTERACTIONS_PROTOCOL,
    }
)
FIRST_PARTY_NATIVE_PRESET_KEYS: tuple[str, ...] = (
    "openai-responses",
    "anthropic",
    "gemini",
)
FIRST_PARTY_COMPATIBILITY_PRESET_KEYS: tuple[str, ...] = (
    "openai",
    "anthropic-compat",
    "gemini-compat",
)
LEGACY_NATIVE_ALIAS_PRESET_KEYS: tuple[str, ...] = ("anthropic-native", "gemini-native")
LOCAL_PROFILE_PRESET_KEYS: tuple[str, ...] = ("ollama", "lm-studio", "vllm")
_CONVERSION_PRESET_BY_FAMILY: dict[str, dict[str, str]] = {
    "openai": {"native": "openai-responses", "compatibility": "openai"},
    "anthropic": {"native": "anthropic", "compatibility": "anthropic-compat"},
    "gemini": {"native": "gemini", "compatibility": "gemini-compat"},
}


@dataclass(frozen=True)
class ProfilePreset:
    key: str
    label: str
    protocol: str
    base_url: str
    api_key_env: str | None
    extra_headers: dict[str, str] = field(default_factory=dict)
    suggested_models: tuple[str, ...] = ()
    suggested_model_descriptions: dict[str, str] = field(default_factory=dict)
    model_aliases: dict[str, str] = field(default_factory=dict)
    validation_model: str = ""
    web_search_adapter: str = AUTO_WEB_SEARCH_ADAPTER
    web_search_model: str = ""
    setup_warning: str = ""
    notes: str = ""


def preset_protocol_kind(preset: ProfilePreset) -> str:
    return "native" if preset.protocol in NATIVE_PROFILE_PROTOCOLS else "compatibility"


def preset_protocol_summary(preset: ProfilePreset) -> str:
    if preset.protocol in NATIVE_PROFILE_PROTOCOLS:
        return (
            f"Native first-party protocol: {preset.protocol} (recommended for first-party API keys)"
        )
    return "Compatibility protocol: OpenAI-compatible chat transport"


def preset_selection_label(preset: ProfilePreset) -> str:
    """Return a setup/config label that keeps protocol details out of the primary choice."""
    if preset.key == "openai-responses":
        return "OpenAI - Recommended native Responses"
    if preset.key in {"anthropic", "anthropic-native"}:
        return "Anthropic Claude - native Messages, recommended"
    if preset.key in {"gemini", "gemini-native"}:
        return "Google Gemini - native GenerateContent, recommended"
    if preset.key == "openai":
        return "OpenAI - Compatibility/gateway Chat Completions"
    if preset.key == "anthropic-compat":
        return "Anthropic Claude compatibility - legacy OpenAI-compatible"
    if preset.key == "gemini-compat":
        return "Google Gemini compatibility - legacy OpenAI-compatible"
    if preset.key in LOCAL_PROFILE_PRESET_KEYS:
        return f"{preset.label} - Local endpoint"
    if preset.key == "custom":
        return "Custom OpenAI-compatible endpoint"
    return preset.label


def provider_selection_presets() -> list[ProfilePreset]:
    """Order setup/config presets around user-facing provider choices.

    Native first-party providers are first for new users. Compatibility/gateway, local, custom, and
    one-release alias presets remain available through the advanced/legacy picker instead of the
    normal first screen.
    """
    by_key = PRESET_BY_KEY
    return [by_key[key] for key in FIRST_PARTY_NATIVE_PRESET_KEYS if key in by_key]


def advanced_provider_selection_presets() -> list[ProfilePreset]:
    """Return compatibility, gateway, local, custom, and legacy alias presets."""
    by_key = PRESET_BY_KEY
    first_party_compat = [
        by_key[key] for key in FIRST_PARTY_COMPATIBILITY_PRESET_KEYS if key in by_key
    ]
    local = [by_key[key] for key in LOCAL_PROFILE_PRESET_KEYS if key in by_key]
    custom = [by_key["custom"]] if "custom" in by_key else []
    used = {
        preset.key
        for preset in (
            *first_party_compat,
            *local,
            *custom,
        )
    }
    gateways = [
        preset
        for preset in PROFILE_PRESETS
        if preset.key not in used and preset.protocol == OPENAI_COMPAT_PROTOCOL
    ]
    aliases = [by_key[key] for key in LEGACY_NATIVE_ALIAS_PRESET_KEYS if key in by_key]
    return [*first_party_compat, *gateways, *local, *custom, *aliases]


PROFILE_PRESETS: tuple[ProfilePreset, ...] = (
    ProfilePreset(
        key="openai",
        label="OpenAI",
        protocol="openai_compat",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        suggested_models=("gpt-5.5", "gpt-5.5-pro", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4"),
        suggested_model_descriptions={
            "gpt-5.5": "default - flagship model for complex coding and reasoning",
            "gpt-5.5-pro": "advanced - strongest GPT-5.5 model for long-running reasoning",
            "gpt-5.4-mini": "fast - lower-latency/lower-cost GPT-5.4 model",
            "gpt-5.4-nano": "economy - lowest-cost GPT-5.4 model for lightweight checks",
            "gpt-5.4": "coding - more capable GPT-5.4 model with lower cost than gpt-5.5",
        },
        model_aliases={
            "gpt-5-nano": "gpt-5.4-nano",
        },
        validation_model="gpt-5.4-mini",
        web_search_adapter=OPENAI_RESPONSES_ADAPTER,
    ),
    ProfilePreset(
        key="openai-responses",
        label="OpenAI Responses",
        protocol="openai_responses",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        suggested_models=("gpt-5.5", "gpt-5.5-pro", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4"),
        suggested_model_descriptions={
            "gpt-5.5": "default - flagship model for complex coding and reasoning",
            "gpt-5.5-pro": "advanced - strongest GPT-5.5 model for long-running reasoning",
            "gpt-5.4-mini": "fast - lower-latency/lower-cost Responses model",
            "gpt-5.4-nano": "economy - lowest-cost Responses model for lightweight checks",
            "gpt-5.4": "coding - more capable GPT-5.4 model with lower cost than gpt-5.5",
        },
        model_aliases={
            "gpt-5-nano": "gpt-5.4-nano",
        },
        validation_model="gpt-5.4-mini",
        web_search_adapter=OPENAI_RESPONSES_ADAPTER,
        notes=(
            "Native OpenAI Responses API chat with SSE streaming support. Use the OpenAI compat "
            "preset to keep Chat Completions-compatible behavior."
        ),
    ),
    ProfilePreset(
        key="anthropic",
        label="Anthropic Claude",
        protocol="anthropic_messages",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        suggested_models=(
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-8",
        ),
        suggested_model_descriptions={
            "claude-sonnet-4-6": "default - best speed/intelligence balance for coding",
            "claude-haiku-4-5-20251001": "fast - lowest-latency Claude option",
            "claude-opus-4-8": "advanced - strongest Claude model for complex agentic tasks",
        },
        model_aliases={
            "claude-sonnet-4": "claude-sonnet-4-6",
            "claude-4-sonnet": "claude-sonnet-4-6",
            "claude-opus-4.7": "claude-opus-4-7",
            "claude-opus-4.8": "claude-opus-4-8",
            "claude-3-5-haiku-latest": "claude-3-5-haiku-20241022",
        },
        validation_model="claude-sonnet-4-6",
        web_search_adapter=ANTHROPIC_MESSAGES_ADAPTER,
        notes=(
            "Native Anthropic Messages API chat with SSE streaming support. Compatibility mode "
            "remains available as anthropic-compat for legacy OpenAI-compatible fallback."
        ),
    ),
    ProfilePreset(
        key="anthropic-compat",
        label="Anthropic Claude compatibility",
        protocol="openai_compat",
        base_url="https://api.anthropic.com/v1/",
        api_key_env="ANTHROPIC_API_KEY",
        suggested_models=(
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-8",
        ),
        suggested_model_descriptions={
            "claude-sonnet-4-6": "default - best speed/intelligence balance for coding",
            "claude-haiku-4-5-20251001": "fast - lowest-latency Claude option",
            "claude-opus-4-8": "advanced - strongest Claude model for complex agentic tasks",
        },
        model_aliases={
            "claude-sonnet-4": "claude-sonnet-4-6",
            "claude-4-sonnet": "claude-sonnet-4-6",
            "claude-opus-4.7": "claude-opus-4-7",
            "claude-opus-4.8": "claude-opus-4-8",
            "claude-3-5-haiku-latest": "claude-3-5-haiku-20241022",
        },
        validation_model="claude-sonnet-4-6",
        web_search_adapter=ANTHROPIC_MESSAGES_ADAPTER,
        setup_warning=(
            "Anthropic labels the OpenAI SDK compatibility layer as a test path; "
            "use the anthropic preset for native Messages API behavior."
        ),
        notes=(
            "Chat uses Anthropic OpenAI-compat at /v1; web_search uses the native "
            "Anthropic Messages web_search adapter when the model/account supports it."
        ),
    ),
    ProfilePreset(
        key="anthropic-native",
        label="Anthropic Claude (native alias)",
        protocol="anthropic_messages",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        suggested_models=(
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-8",
        ),
        suggested_model_descriptions={
            "claude-sonnet-4-6": "default - best speed/intelligence balance for coding",
            "claude-haiku-4-5-20251001": "fast - lowest-latency Claude option",
            "claude-opus-4-8": "advanced - strongest Claude model for complex agentic tasks",
        },
        model_aliases={
            "claude-sonnet-4": "claude-sonnet-4-6",
            "claude-4-sonnet": "claude-sonnet-4-6",
            "claude-opus-4.7": "claude-opus-4-7",
            "claude-opus-4.8": "claude-opus-4-8",
            "claude-3-5-haiku-latest": "claude-3-5-haiku-20241022",
        },
        validation_model="claude-sonnet-4-6",
        web_search_adapter=ANTHROPIC_MESSAGES_ADAPTER,
        notes=(
            "Legacy alias for the native anthropic preset. Prefer the anthropic preset for new "
            "first-party Claude profiles."
        ),
    ),
    ProfilePreset(
        key="gemini",
        label="Google Gemini",
        protocol="gemini_generate_content",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key_env="GEMINI_API_KEY",
        suggested_models=(
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-3.1-pro-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
        ),
        suggested_model_descriptions={
            "gemini-3.5-flash": "default - stable frontier Flash model for agentic tasks",
            "gemini-3.1-flash-lite": "fast - stable low-cost high-volume model",
            "gemini-3.1-pro-preview": "advanced - current Gemini Pro preview for reasoning",
            "gemini-2.5-pro": "fallback - stable advanced reasoning model",
            "gemini-2.5-flash-lite": "economy - stable low-cost Gemini 2.5 model",
            "gemini-2.5-flash": "fallback - broad price-performance model",
        },
        model_aliases={
            "gemini-3.1-preview": "gemini-3.1-pro-preview",
            "gemini-3-pro-preview": "gemini-3.1-pro-preview",
            "gemini-3-flash-preview": "gemini-3.5-flash",
            "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
            "gemini-flash-latest": "gemini-3.5-flash",
            "gemini-flash-lite-latest": "gemini-3.1-flash-lite",
            "gemini-pro-latest": "gemini-3.1-pro-preview",
            "gemini-2.5-flash-latest": "gemini-2.5-flash",
        },
        validation_model="gemini-2.5-flash",
        web_search_adapter=GEMINI_GROUNDING_ADAPTER,
        setup_warning=(
            "Gemini native GenerateContent uses the Google Gemini API v1beta surface and "
            "model availability can vary by account, region, and provider rollout."
        ),
        notes=(
            "Native Gemini GenerateContent API chat with streamGenerateContent SSE support. "
            "Compatibility mode remains available as gemini-compat for legacy OpenAI-compatible "
            "fallback."
        ),
    ),
    ProfilePreset(
        key="gemini-compat",
        label="Google Gemini compatibility",
        protocol="openai_compat",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        suggested_models=(
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-3.1-pro-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
        ),
        suggested_model_descriptions={
            "gemini-3.5-flash": "default - stable frontier Flash model for agentic tasks",
            "gemini-3.1-flash-lite": "fast - stable low-cost high-volume model",
            "gemini-3.1-pro-preview": "advanced - current Gemini Pro preview for reasoning",
            "gemini-2.5-pro": "fallback - stable advanced reasoning model",
            "gemini-2.5-flash-lite": "economy - stable low-cost Gemini 2.5 model",
            "gemini-2.5-flash": "fallback - broad price-performance model",
        },
        model_aliases={
            "gemini-3.1-preview": "gemini-3.1-pro-preview",
            "gemini-3-pro-preview": "gemini-3.1-pro-preview",
            "gemini-3-flash-preview": "gemini-3.5-flash",
            "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
            "gemini-flash-latest": "gemini-3.5-flash",
            "gemini-flash-lite-latest": "gemini-3.1-flash-lite",
            "gemini-pro-latest": "gemini-3.1-pro-preview",
            "gemini-2.5-flash-latest": "gemini-2.5-flash",
        },
        validation_model="gemini-2.5-flash",
        web_search_adapter=GEMINI_GROUNDING_ADAPTER,
        setup_warning=(
            "Gemini OpenAI compatibility is served from v1beta; use the gemini preset for "
            "native GenerateContent behavior."
        ),
    ),
    ProfilePreset(
        key="gemini-native",
        label="Google Gemini (native alias)",
        protocol="gemini_generate_content",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key_env="GEMINI_API_KEY",
        suggested_models=(
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-3.1-pro-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
        ),
        suggested_model_descriptions={
            "gemini-3.5-flash": "default - stable frontier Flash model for agentic tasks",
            "gemini-3.1-flash-lite": "fast - stable low-cost high-volume model",
            "gemini-3.1-pro-preview": "advanced - current Gemini Pro preview for reasoning",
            "gemini-2.5-pro": "fallback - stable advanced reasoning model",
            "gemini-2.5-flash-lite": "economy - stable low-cost Gemini 2.5 model",
            "gemini-2.5-flash": "fallback - broad price-performance model",
        },
        model_aliases={
            "gemini-3.1-preview": "gemini-3.1-pro-preview",
            "gemini-3-pro-preview": "gemini-3.1-pro-preview",
            "gemini-3-flash-preview": "gemini-3.5-flash",
            "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
            "gemini-flash-latest": "gemini-3.5-flash",
            "gemini-flash-lite-latest": "gemini-3.1-flash-lite",
            "gemini-pro-latest": "gemini-3.1-pro-preview",
            "gemini-2.5-flash-latest": "gemini-2.5-flash",
        },
        validation_model="gemini-2.5-flash",
        web_search_adapter=GEMINI_GROUNDING_ADAPTER,
        setup_warning=(
            "Gemini native GenerateContent uses the Google Gemini API v1beta surface and "
            "model availability can vary by account, region, and provider rollout."
        ),
        notes=(
            "Legacy alias for the native gemini preset. Prefer the gemini preset for new "
            "first-party Gemini profiles."
        ),
    ),
    ProfilePreset(
        key="deepseek",
        label="DeepSeek",
        protocol="openai_compat",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        suggested_models=("deepseek-v4-pro", "deepseek-v4-flash"),
        suggested_model_descriptions={
            "deepseek-v4-pro": "default - stronger V4 agentic coding/reasoning model",
            "deepseek-v4-flash": "fast - economical V4 model with the same endpoint",
        },
        setup_warning=(
            "Do not use retired legacy aliases deepseek-chat or deepseek-reasoner "
            "for production defaults; use the V4 model IDs."
        ),
    ),
    ProfilePreset(
        key="qwen-intl",
        label="Alibaba Qwen / DashScope (Intl)",
        protocol="openai_compat",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        suggested_models=("qwen3.7-plus", "qwen3.7-max", "qwen3.6-flash", "qwen3-coder-plus"),
        suggested_model_descriptions={
            "qwen3.7-plus": "default - current 1M-context Qwen model for agentic coding",
            "qwen3.7-max": "advanced - strongest Qwen 3.7 model for complex reasoning",
            "qwen3.6-flash": "fast - lower-cost 1M-context Qwen 3.6 model",
            "qwen3-coder-plus": "coding - dedicated code-generation model",
        },
        web_search_adapter=DASHSCOPE_CHAT_ADAPTER,
        web_search_model="qwen3.7-plus",
        setup_warning=(
            "DashScope API keys are region-specific; use a key from the Singapore region."
        ),
    ),
    ProfilePreset(
        key="qwen-us",
        label="Alibaba Qwen / DashScope (US)",
        protocol="openai_compat",
        base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        suggested_models=("qwen3.7-plus", "qwen3.7-max", "qwen3.6-flash", "qwen3-coder-plus"),
        suggested_model_descriptions={
            "qwen3.7-plus": "default - current 1M-context Qwen model for agentic coding",
            "qwen3.7-max": "advanced - strongest Qwen 3.7 model for complex reasoning",
            "qwen3.6-flash": "fast - lower-cost 1M-context Qwen 3.6 model",
            "qwen3-coder-plus": "coding - dedicated code-generation model",
        },
        web_search_adapter=DASHSCOPE_CHAT_ADAPTER,
        web_search_model="qwen3.7-plus",
        setup_warning="DashScope API keys are region-specific; use a key from the US region.",
    ),
    ProfilePreset(
        key="qwen-cn",
        label="Alibaba Qwen / DashScope (China)",
        protocol="openai_compat",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        suggested_models=("qwen3.7-plus", "qwen3.7-max", "qwen3.6-flash", "qwen3-coder-plus"),
        suggested_model_descriptions={
            "qwen3.7-plus": "default - current 1M-context Qwen model for agentic coding",
            "qwen3.7-max": "advanced - strongest Qwen 3.7 model for complex reasoning",
            "qwen3.6-flash": "fast - lower-cost 1M-context Qwen 3.6 model",
            "qwen3-coder-plus": "coding - dedicated code-generation model",
        },
        web_search_adapter=DASHSCOPE_CHAT_ADAPTER,
        web_search_model="qwen3.7-plus",
        setup_warning="DashScope API keys are region-specific; use a key from the China region.",
    ),
    ProfilePreset(
        key="zhipu",
        label="Zhipu / GLM",
        protocol="openai_compat",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key_env="ZHIPUAI_API_KEY",
        suggested_models=("glm-5.1", "glm-5", "glm-4.6"),
        suggested_model_descriptions={
            "glm-5.1": "default - latest GLM flagship model for agentic coding",
            "glm-5": "coding - GLM-5 family model with broad tool-use support",
            "glm-4.6": "fallback - stable GLM 4.6 model",
        },
        web_search_adapter=ZHIPU_WEB_SEARCH_ADAPTER,
        web_search_model="glm-5.1",
    ),
    ProfilePreset(
        key="moonshot",
        label="Moonshot / Kimi",
        protocol="openai_compat",
        base_url="https://api.moonshot.cn/v1",
        api_key_env="MOONSHOT_API_KEY",
        suggested_models=("kimi-k2.6", "kimi-k2"),
        web_search_adapter=MOONSHOT_KIMI_ADAPTER,
        web_search_model="kimi-k2.6",
    ),
    ProfilePreset(
        key="minimax",
        label="MiniMax",
        protocol="openai_compat",
        base_url="https://api.minimax.io/v1",
        api_key_env="MINIMAX_API_KEY",
        suggested_models=("MiniMax-M2.7", "MiniMax-M2.7-highspeed", "MiniMax-M2"),
        suggested_model_descriptions={
            "MiniMax-M2.7": "default - latest MiniMax coding and agentic model",
            "MiniMax-M2.7-highspeed": "fast - lower-latency M2.7 variant",
            "MiniMax-M2": "fallback - previous-generation M2 model",
        },
    ),
    ProfilePreset(
        key="bytedance",
        label="ByteDance Doubao",
        protocol="openai_compat",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key_env="ARK_API_KEY",
        suggested_models=(
            "doubao-seed-2-0-pro-260215",
            "doubao-seed-2-0-lite-260215",
            "doubao-seed-1-6-250615",
        ),
        suggested_model_descriptions={
            "doubao-seed-2-0-pro-260215": "default - latest Doubao Seed 2.0 Pro model",
            "doubao-seed-2-0-lite-260215": "fast - lower-cost Doubao Seed 2.0 model",
            "doubao-seed-1-6-250615": "fallback - stable Doubao Seed 1.6 model",
        },
        web_search_adapter=VOLCENGINE_WEB_SEARCH_ADAPTER,
        web_search_model="doubao-seed-1-6-250615",
    ),
    ProfilePreset(
        key="01ai",
        label="01.AI / Yi",
        protocol="openai_compat",
        base_url="https://api.lingyiwanwu.com/v1",
        api_key_env="YI_API_KEY",
        suggested_models=("yi-large",),
    ),
    ProfilePreset(
        key="groq",
        label="Groq",
        protocol="openai_compat",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        suggested_models=(
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
        ),
        suggested_model_descriptions={
            "openai/gpt-oss-120b": "default - production reasoning/coding model on Groq",
            "openai/gpt-oss-20b": "fast - cheaper agentic reasoning/coding model",
            "llama-3.3-70b-versatile": "fast - stable general text model",
        },
        web_search_adapter=GROQ_COMPOUND_ADAPTER,
        web_search_model="groq/compound-mini",
        setup_warning=(
            "Groq is mostly OpenAI-compatible; avoid preview-only models as production defaults."
        ),
    ),
    ProfilePreset(
        key="cerebras",
        label="Cerebras",
        protocol="openai_compat",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        suggested_models=("llama3.3-70b",),
    ),
    ProfilePreset(
        key="mistral",
        label="Mistral AI",
        protocol="openai_compat",
        base_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
        suggested_models=("mistral-medium-3-5", "devstral-2512", "mistral-small-2603"),
        suggested_model_descriptions={
            "mistral-medium-3-5": "default - agentic/coding model with Chat Completions",
            "devstral-2512": "coding - dedicated software engineering agent model",
            "mistral-small-2603": "fast - efficient hybrid instruct/reasoning/coding model",
        },
        web_search_adapter=MISTRAL_CONVERSATIONS_ADAPTER,
        web_search_model="mistral-medium-latest",
    ),
    ProfilePreset(
        key="xai",
        label="xAI Grok",
        protocol="openai_compat",
        base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
        suggested_models=(
            "grok-4.3",
            "grok-4.20-0309-reasoning",
            "grok-code-fast-1",
        ),
        suggested_model_descriptions={
            "grok-4.3": "default - xAI's recommended model for Chat API callers",
            "grok-4.20-0309-reasoning": "reasoning - 2M-context reasoning chat model",
            "grok-code-fast-1": "coding - fast code-oriented model",
        },
        model_aliases={
            "grok-4": "grok-4.3",
            "grok-4-fast": "grok-4.3",
            "grok-4.3-latest": "grok-4.3",
        },
        web_search_adapter=XAI_RESPONSES_ADAPTER,
        setup_warning=(
            "Older grok-4, grok-4-fast, and grok-3 IDs are retiring; use Grok 4.3 for "
            "general chat or grok-code-fast-1 for code-focused workflows."
        ),
    ),
    ProfilePreset(
        key="cohere",
        label="Cohere (compat)",
        protocol="openai_compat",
        base_url="https://api.cohere.ai/compatibility/v1",
        api_key_env="COHERE_API_KEY",
        suggested_models=(
            "command-a-plus-05-2026",
            "command-a-reasoning-08-2025",
            "command-a-03-2025",
        ),
        suggested_model_descriptions={
            "command-a-plus-05-2026": "default - newest Command A+ model",
            "command-a-reasoning-08-2025": "reasoning - Command A reasoning model",
            "command-a-03-2025": "fallback - stable Command A model",
        },
    ),
    ProfilePreset(
        key="openrouter",
        label="OpenRouter (gateway)",
        protocol="openai_compat",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        suggested_models=(
            "openai/gpt-5.5",
            "anthropic/claude-opus-4.8",
            "google/gemini-3.5-flash",
            "qwen/qwen3.7-plus",
            "deepseek/deepseek-v4-pro",
        ),
        suggested_model_descriptions={
            "openai/gpt-5.5": "default - OpenAI GPT-5.5 through OpenRouter",
            "anthropic/claude-opus-4.8": "advanced - Claude Opus 4.8 through OpenRouter",
            "google/gemini-3.5-flash": "fast - Gemini 3.5 Flash through OpenRouter",
            "qwen/qwen3.7-plus": "coding - Qwen 3.7 Plus through OpenRouter",
            "deepseek/deepseek-v4-pro": "reasoning - DeepSeek V4 Pro through OpenRouter",
        },
        web_search_adapter=OPENROUTER_WEB_ADAPTER,
        setup_warning=(
            "OpenRouter routes through upstream providers; availability, pricing, privacy, "
            "and parameter support can vary by route."
        ),
        notes="Single API to many providers' models.",
    ),
    ProfilePreset(
        key="perplexity",
        label="Perplexity Sonar",
        protocol="openai_compat",
        base_url="https://api.perplexity.ai",
        api_key_env="PERPLEXITY_API_KEY",
        suggested_models=("sonar-pro", "sonar"),
        web_search_adapter=PERPLEXITY_SONAR_ADAPTER,
        web_search_model="sonar",
        notes="Sonar models include web-grounded answers and citations.",
    ),
    ProfilePreset(
        key="together",
        label="Together AI",
        protocol="openai_compat",
        base_url="https://api.together.ai/v1",
        api_key_env="TOGETHER_API_KEY",
        suggested_models=(
            "zai-org/GLM-5.1",
            "moonshotai/Kimi-K2.6",
            "deepseek-ai/DeepSeek-V4-Pro",
            "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
            "openai/gpt-oss-120b",
        ),
        suggested_model_descriptions={
            "zai-org/GLM-5.1": "default - Together recommended model for coding agents",
            "moonshotai/Kimi-K2.6": "agentic - Kimi K2.6 through Together",
            "deepseek-ai/DeepSeek-V4-Pro": "reasoning - DeepSeek V4 Pro through Together",
            "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": "coding - Qwen Coder option",
            "openai/gpt-oss-120b": "reasoning - medium general-purpose reasoning model",
        },
        setup_warning="Verify account access and model limits with Together's Models API.",
    ),
    ProfilePreset(
        key="fireworks",
        label="Fireworks AI",
        protocol="openai_compat",
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env="FIREWORKS_API_KEY",
        suggested_models=(
            "accounts/fireworks/models/deepseek-v4-pro",
            "accounts/fireworks/models/kimi-k2p6",
            "accounts/fireworks/models/glm-5p1",
            "accounts/fireworks/models/qwen2p5-coder-32b-instruct",
        ),
        suggested_model_descriptions={
            "accounts/fireworks/models/deepseek-v4-pro": "default - DeepSeek V4 Pro on Fireworks",
            "accounts/fireworks/models/kimi-k2p6": "agentic - Kimi K2.6 on Fireworks",
            "accounts/fireworks/models/glm-5p1": "coding - GLM 5.1 on Fireworks",
            "accounts/fireworks/models/qwen2p5-coder-32b-instruct": (
                "fallback - Qwen2.5 Coder model"
            ),
        },
    ),
    ProfilePreset(
        key="ollama",
        label="Ollama (local)",
        protocol="openai_compat",
        base_url="http://localhost:11434/v1",
        api_key_env=None,
        suggested_models=("llama3.3",),
        notes="Local Ollama server. No API key required.",
    ),
    ProfilePreset(
        key="lm-studio",
        label="LM Studio (local)",
        protocol="openai_compat",
        base_url="http://localhost:1234/v1",
        api_key_env=None,
        suggested_models=("local-model",),
        notes="Local LM Studio server. No API key required.",
    ),
    ProfilePreset(
        key="vllm",
        label="vLLM (self-hosted)",
        protocol="openai_compat",
        base_url="http://localhost:8000/v1",
        api_key_env=None,
        suggested_models=("local-model",),
    ),
    ProfilePreset(
        key="custom",
        label="Custom (specify URL manually)",
        protocol="openai_compat",
        base_url="",
        api_key_env=None,
        suggested_models=(),
        notes="Use for unlisted endpoints. Type the URL during setup.",
    ),
)

PRESET_BY_KEY: dict[str, ProfilePreset] = {preset.key: preset for preset in PROFILE_PRESETS}


def get_preset(key: str) -> ProfilePreset | None:
    return PRESET_BY_KEY.get(str(key or "").strip().lower())


def model_options_for_preset(preset: ProfilePreset) -> tuple[tuple[str, str, str], ...]:
    """Return picker rows for the models this preset intentionally supports."""
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for model in preset.suggested_models:
        model_id = str(model or "").strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        description = str(preset.suggested_model_descriptions.get(model_id) or "").strip()
        rows.append((model_id, model_id, description or "suggested by provider preset"))
    return tuple(rows)


def canonical_model_alias_for_preset(preset: ProfilePreset, model: str) -> str:
    """Map explicit stale provider aliases to the preset's current model ID."""
    raw = str(model or "").strip()
    if not raw:
        return raw
    for alias, canonical in preset.model_aliases.items():
        if str(alias or "").strip().casefold() == raw.casefold():
            normalized = str(canonical or "").strip()
            return normalized or raw
    return raw


def find_preset_for_profile(profile: ProfileSpec) -> ProfilePreset | None:
    """Best-effort mapping from a persisted profile back to a known provider preset."""
    name_match = get_preset(profile.name)
    if name_match is not None and _profile_matches_preset(profile, name_match):
        return name_match

    for preset in PROFILE_PRESETS:
        if preset.key == "custom":
            continue
        if _profile_matches_preset(profile, preset):
            return preset

    if profile.protocol != OPENAI_COMPAT_PROTOCOL:
        return None
    return find_preset_for_base_url(profile.base_url)


def find_preset_for_base_url(base_url: str) -> ProfilePreset | None:
    normalized = _normalized_base_url(base_url)
    if not normalized:
        return None
    matches: list[ProfilePreset] = []
    for preset in PROFILE_PRESETS:
        if preset.key == "custom":
            continue
        if _normalized_base_url(preset.base_url) == normalized:
            matches.append(preset)
    if not matches:
        return None
    compatibility = next(
        (preset for preset in matches if preset.protocol == OPENAI_COMPAT_PROTOCOL),
        None,
    )
    return compatibility or matches[0]


def _profile_matches_preset(profile: ProfileSpec, preset: ProfilePreset) -> bool:
    if str(profile.protocol or OPENAI_COMPAT_PROTOCOL).strip() != preset.protocol:
        return False
    profile_url = _normalized_base_url(profile.base_url)
    preset_url = _normalized_base_url(preset.base_url)
    if not profile_url or not preset_url or profile_url != preset_url:
        return False
    return True


def profile_provider_family(profile: ProfileSpec) -> str | None:
    """Resolve a profile to a first-party family for protocol conversion and diagnostics."""
    protocol = str(profile.protocol or OPENAI_COMPAT_PROTOCOL).strip()
    if protocol == OPENAI_RESPONSES_PROTOCOL:
        return "openai"
    if protocol == ANTHROPIC_MESSAGES_PROTOCOL:
        return "anthropic"
    if protocol == GEMINI_GENERATE_CONTENT_PROTOCOL:
        return "gemini"
    if protocol == GEMINI_INTERACTIONS_PROTOCOL:
        return "gemini"

    preset = get_preset(profile.name)
    if preset is not None:
        if preset.key in {"openai", "openai-responses"}:
            return "openai"
        if preset.key in {"anthropic", "anthropic-compat", "anthropic-native"}:
            return "anthropic"
        if preset.key in {"gemini", "gemini-compat", "gemini-native"}:
            return "gemini"

    normalized_name = str(profile.name or "").strip().lower()
    if "openai" in normalized_name:
        return "openai"
    if "anthropic" in normalized_name or "claude" in normalized_name:
        return "anthropic"
    if "gemini" in normalized_name or "google" in normalized_name:
        return "gemini"

    parsed = _split_base_url(profile.base_url)
    if parsed[0] == "api.openai.com":
        return "openai"
    if parsed[0] == "api.anthropic.com":
        return "anthropic"
    if parsed[0] == "generativelanguage.googleapis.com":
        return "gemini"
    return None


def target_preset_for_profile_conversion(
    profile: ProfileSpec,
    *,
    target: str,
) -> ProfilePreset | None:
    normalized_target = normalize_conversion_target(target)
    family = profile_provider_family(profile)
    if family is None:
        return None
    preset_key = _CONVERSION_PRESET_BY_FAMILY.get(family, {}).get(normalized_target)
    if preset_key is None:
        return None
    return get_preset(preset_key)


def normalize_conversion_target(value: str) -> str:
    target = str(value or "").strip().lower().replace("_", "-")
    if target in {"native", "first-party", "firstparty"}:
        return "native"
    if target in {"compat", "compatibility", "openai-compatible", "gateway"}:
        return "compatibility"
    raise ValueError("conversion target must be 'native' or 'compatibility'")


def convert_profile_to_preset(profile: ProfileSpec, preset: ProfilePreset) -> ProfileSpec:
    current_model = str(profile.default_model or "").strip()
    default_model = canonical_model_alias_for_preset(preset, current_model)
    target_family = _preset_provider_family(preset)
    if not default_model or _model_known_incompatible_with_family(default_model, target_family):
        default_model = preset.suggested_models[0] if preset.suggested_models else ""
    notes = _converted_profile_notes(profile, preset)

    return ProfileSpec(
        name=profile.name,
        protocol=preset.protocol,
        base_url=preset.base_url,
        api_key_env=profile.api_key_env or preset.api_key_env,
        extra_headers=dict(profile.extra_headers),
        default_model=default_model,
        web_search_adapter=preset.web_search_adapter,
        web_search_model=preset.web_search_model,
        notes=notes,
    )


def _converted_profile_notes(profile: ProfileSpec, preset: ProfilePreset) -> str:
    notes = str(profile.notes or "").strip()
    if not notes:
        return preset.notes
    source_preset = find_preset_for_profile(profile)
    if source_preset is not None and notes == str(source_preset.notes or "").strip():
        return preset.notes

    lowered = notes.lower()
    target_is_native = preset.protocol in NATIVE_PROFILE_PROTOCOLS
    if target_is_native and (
        "openai-compat" in lowered
        or "openai compatible" in lowered
        or "openai-compatible" in lowered
        or "compatibility mode" in lowered
    ):
        return preset.notes
    if not target_is_native and "native" in lowered:
        return preset.notes
    return notes


def _split_base_url(value: str | None) -> tuple[str, str]:
    try:
        parsed = urlsplit(str(value or "").strip())
    except ValueError:
        return "", ""
    path = parsed.path.rstrip("/").lower()
    return (parsed.hostname or "").rstrip(".").lower(), path


def _preset_provider_family(preset: ProfilePreset) -> str | None:
    for family, targets in _CONVERSION_PRESET_BY_FAMILY.items():
        if preset.key in targets.values():
            return family
    return None


def _model_known_incompatible_with_family(model: str, family: str | None) -> bool:
    if family is None:
        return False
    normalized = model.strip().lower()
    model_family = _known_model_family(normalized)
    if model_family is None:
        return False
    if model_family != family:
        return True
    return _has_known_provider_namespace(normalized)


def known_model_family(model: str) -> str | None:
    """Best-effort model-family classifier for static diagnostics.

    This is intentionally conservative. Unknown custom gateway models return None so doctor
    diagnostics do not over-warn on valid provider-specific names Sylliptor cannot know offline.
    """
    return _known_model_family(str(model or "").strip().lower())


def model_known_incompatible_with_family(model: str, family: str | None) -> bool:
    """Public wrapper used by diagnostics and tests."""
    return _model_known_incompatible_with_family(model, family)


def _known_model_family(model: str) -> str | None:
    known_prefixes: dict[str, tuple[str, ...]] = {
        "openai": ("gpt-", "chatgpt-", "o1", "o3", "o4", "o5"),
        "anthropic": ("claude-",),
        "gemini": ("gemini-",),
    }
    known_namespaces: dict[str, tuple[str, ...]] = {
        "openai": ("openai",),
        "anthropic": ("anthropic", "anthropic-ai"),
        "gemini": ("google", "gemini"),
    }
    parts = [part for part in model.split("/") if part]
    for known_family, namespaces in known_namespaces.items():
        if parts and parts[0] in namespaces:
            return known_family
    model_id = parts[-1] if parts else model
    for known_family, prefixes in known_prefixes.items():
        if model_id.startswith(prefixes):
            return known_family
    return None


def _has_known_provider_namespace(model: str) -> bool:
    if "/" not in model:
        return False
    namespace = model.split("/", 1)[0]
    return namespace in {"openai", "anthropic", "anthropic-ai", "google", "gemini"}


def _normalized_base_url(value: str | None) -> str:
    return str(value or "").strip().rstrip("/")


def make_profile_from_preset(
    preset: ProfilePreset,
    *,
    name: str | None = None,
) -> ProfileSpec:
    profile_name = str(name or preset.key).strip().lower()
    return ProfileSpec(
        name=profile_name,
        protocol=preset.protocol,
        base_url=preset.base_url,
        api_key_env=preset.api_key_env,
        extra_headers=dict(preset.extra_headers),
        default_model=preset.suggested_models[0] if preset.suggested_models else "",
        web_search_adapter=preset.web_search_adapter,
        web_search_model=preset.web_search_model,
        notes=preset.notes,
    )
