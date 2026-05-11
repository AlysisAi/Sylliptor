from __future__ import annotations

from dataclasses import dataclass, field

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


PROFILE_PRESETS: tuple[ProfilePreset, ...] = (
    ProfilePreset(
        key="openai",
        label="OpenAI",
        protocol="openai_compat",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        suggested_models=("gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"),
        suggested_model_descriptions={
            "gpt-5.5": "default - flagship model for complex coding and reasoning",
            "gpt-5.4": "coding - strong frontier model with lower cost than gpt-5.5",
            "gpt-5.4-mini": "fast - lower latency/cost for high-volume work",
            "gpt-5.4-nano": "fastest - cheapest GPT-5.4-class option",
        },
        web_search_adapter=OPENAI_RESPONSES_ADAPTER,
    ),
    ProfilePreset(
        key="anthropic",
        label="Anthropic Claude (compat)",
        protocol="openai_compat",
        base_url="https://api.anthropic.com/v1/",
        api_key_env="ANTHROPIC_API_KEY",
        # Keep these IDs aligned with Anthropic's Models overview.
        # TODO: replace the static list with dynamic Models API discovery.
        suggested_models=(
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ),
        suggested_model_descriptions={
            "claude-opus-4-7": "default - most capable Claude model for agentic coding",
            "claude-sonnet-4-6": "coding - best speed/intelligence balance",
            "claude-haiku-4-5-20251001": "fast - lowest-latency Claude option",
        },
        web_search_adapter=ANTHROPIC_MESSAGES_ADAPTER,
        setup_warning=(
            "Anthropic labels the OpenAI SDK compatibility layer as a test path; "
            "advanced Claude features require the native Messages API."
        ),
        notes=(
            "Chat uses Anthropic OpenAI-compat at /v1; web_search uses the native "
            "Anthropic Messages web_search adapter when the model/account supports it."
        ),
    ),
    ProfilePreset(
        key="gemini",
        label="Google Gemini (compat)",
        protocol="openai_compat",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        suggested_models=(
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite",
        ),
        suggested_model_descriptions={
            "gemini-3.1-pro-preview": "default - software engineering and agentic workflows",
            "gemini-3-flash-preview": "coding - fast Gemini 3 model with 1M context",
            "gemini-3.1-flash-lite": "fast - low-cost high-volume agentic tasks",
        },
        model_aliases={
            "gemini-3.1-preview": "gemini-3.1-pro-preview",
            "gemini-3-pro-preview": "gemini-3.1-pro-preview",
            "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
        },
        validation_model="gemini-3-flash-preview",
        web_search_adapter=GEMINI_GROUNDING_ADAPTER,
        setup_warning=(
            "Gemini OpenAI compatibility is served from v1beta and these Gemini 3.x "
            "IDs are preview models; availability can change."
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
        suggested_models=("qwen3.6-plus", "qwen3.6-flash", "qwen3-coder-plus"),
        suggested_model_descriptions={
            "qwen3.6-plus": "default - 1M-context model recommended for large codebases",
            "qwen3.6-flash": "fast - lower-cost 1M-context model",
            "qwen3-coder-plus": "coding - dedicated code-generation model",
        },
        web_search_adapter=DASHSCOPE_CHAT_ADAPTER,
        web_search_model="qwen3.5-plus",
        setup_warning="DashScope API keys are region-specific; use a key from the Singapore region.",
    ),
    ProfilePreset(
        key="qwen-us",
        label="Alibaba Qwen / DashScope (US)",
        protocol="openai_compat",
        base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        suggested_models=("qwen3.6-plus", "qwen3.6-flash", "qwen3-coder-plus"),
        suggested_model_descriptions={
            "qwen3.6-plus": "default - 1M-context model recommended for large codebases",
            "qwen3.6-flash": "fast - lower-cost 1M-context model",
            "qwen3-coder-plus": "coding - dedicated code-generation model",
        },
        web_search_adapter=DASHSCOPE_CHAT_ADAPTER,
        web_search_model="qwen3.5-plus",
        setup_warning="DashScope API keys are region-specific; use a key from the US region.",
    ),
    ProfilePreset(
        key="qwen-cn",
        label="Alibaba Qwen / DashScope (China)",
        protocol="openai_compat",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        suggested_models=("qwen3.6-plus", "qwen3.6-flash", "qwen3-coder-plus"),
        suggested_model_descriptions={
            "qwen3.6-plus": "default - 1M-context model recommended for large codebases",
            "qwen3.6-flash": "fast - lower-cost 1M-context model",
            "qwen3-coder-plus": "coding - dedicated code-generation model",
        },
        web_search_adapter=DASHSCOPE_CHAT_ADAPTER,
        web_search_model="qwen3.5-plus",
        setup_warning="DashScope API keys are region-specific; use a key from the China region.",
    ),
    ProfilePreset(
        key="zhipu",
        label="Zhipu / GLM",
        protocol="openai_compat",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key_env="ZHIPUAI_API_KEY",
        suggested_models=("glm-4.6", "glm-4-flash"),
        web_search_adapter=ZHIPU_WEB_SEARCH_ADAPTER,
        web_search_model="glm-4.6",
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
        suggested_models=("MiniMax-M2",),
    ),
    ProfilePreset(
        key="bytedance",
        label="ByteDance Doubao",
        protocol="openai_compat",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key_env="ARK_API_KEY",
        suggested_models=("doubao-1.5-pro",),
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
            "grok-4.20-0309-non-reasoning",
        ),
        suggested_model_descriptions={
            "grok-4.3": "default - xAI's recommended model for Chat API callers",
            "grok-4.20-0309-reasoning": "reasoning - 2M-context reasoning chat model",
            "grok-4.20-0309-non-reasoning": "fast - 2M-context non-reasoning chat model",
        },
        web_search_adapter=XAI_RESPONSES_ADAPTER,
        setup_warning=(
            "Older grok-4, grok-4-fast, grok-3, and grok-code-fast-1 IDs are retiring; "
            "use Grok 4.3 or the listed 4.20 chat IDs."
        ),
    ),
    ProfilePreset(
        key="cohere",
        label="Cohere (compat)",
        protocol="openai_compat",
        base_url="https://api.cohere.ai/compatibility/v1",
        api_key_env="COHERE_API_KEY",
        suggested_models=("command-a",),
    ),
    ProfilePreset(
        key="openrouter",
        label="OpenRouter (gateway)",
        protocol="openai_compat",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        suggested_models=(
            "openai/gpt-5.5",
            "deepseek/deepseek-v4-pro",
            "mistralai/mistral-medium-3-5",
        ),
        suggested_model_descriptions={
            "openai/gpt-5.5": "default - OpenAI GPT-5.5 through OpenRouter",
            "deepseek/deepseek-v4-pro": "reasoning - DeepSeek V4 Pro through OpenRouter",
            "mistralai/mistral-medium-3-5": "coding - Mistral Medium 3.5 through OpenRouter",
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
            "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
            "openai/gpt-oss-120b",
        ),
        suggested_model_descriptions={
            "zai-org/GLM-5.1": "default - Together recommended model for coding agents",
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
        suggested_models=("accounts/fireworks/models/qwen2p5-coder-32b-instruct",),
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

    return find_preset_for_base_url(profile.base_url)


def find_preset_for_base_url(base_url: str) -> ProfilePreset | None:
    normalized = _normalized_base_url(base_url)
    if not normalized:
        return None
    for preset in PROFILE_PRESETS:
        if preset.key == "custom":
            continue
        if _normalized_base_url(preset.base_url) == normalized:
            return preset
    return None


def _profile_matches_preset(profile: ProfileSpec, preset: ProfilePreset) -> bool:
    profile_url = _normalized_base_url(profile.base_url)
    preset_url = _normalized_base_url(preset.base_url)
    if not profile_url or not preset_url or profile_url != preset_url:
        return False
    return True


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
