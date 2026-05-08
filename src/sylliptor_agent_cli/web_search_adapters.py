from __future__ import annotations

AUTO_WEB_SEARCH_ADAPTER = "auto"
OPENAI_RESPONSES_ADAPTER = "openai_responses"
XAI_RESPONSES_ADAPTER = "xai_responses"
ANTHROPIC_MESSAGES_ADAPTER = "anthropic_messages"
GEMINI_GROUNDING_ADAPTER = "gemini_grounding"
OPENROUTER_WEB_ADAPTER = "openrouter_web"
DASHSCOPE_CHAT_ADAPTER = "dashscope_chat"
MOONSHOT_KIMI_ADAPTER = "moonshot_kimi"
ZHIPU_WEB_SEARCH_ADAPTER = "zhipu_web_search"
VOLCENGINE_WEB_SEARCH_ADAPTER = "volcengine_web_search"
PERPLEXITY_SONAR_ADAPTER = "perplexity_sonar"
GROQ_COMPOUND_ADAPTER = "groq_compound"
MISTRAL_CONVERSATIONS_ADAPTER = "mistral_conversations"
TAVILY_ADAPTER = "tavily"

VALID_WEB_SEARCH_ADAPTERS: frozenset[str] = frozenset(
    {
        AUTO_WEB_SEARCH_ADAPTER,
        OPENAI_RESPONSES_ADAPTER,
        XAI_RESPONSES_ADAPTER,
        ANTHROPIC_MESSAGES_ADAPTER,
        GEMINI_GROUNDING_ADAPTER,
        OPENROUTER_WEB_ADAPTER,
        DASHSCOPE_CHAT_ADAPTER,
        MOONSHOT_KIMI_ADAPTER,
        ZHIPU_WEB_SEARCH_ADAPTER,
        VOLCENGINE_WEB_SEARCH_ADAPTER,
        PERPLEXITY_SONAR_ADAPTER,
        GROQ_COMPOUND_ADAPTER,
        MISTRAL_CONVERSATIONS_ADAPTER,
        TAVILY_ADAPTER,
    }
)
WEB_SEARCH_ADAPTER_CHOICES: tuple[str, ...] = tuple(sorted(VALID_WEB_SEARCH_ADAPTERS))


def normalize_web_search_adapter(raw: object) -> str:
    value = str(raw or "").strip().lower()
    if not value:
        return AUTO_WEB_SEARCH_ADAPTER
    if value in VALID_WEB_SEARCH_ADAPTERS:
        return value
    allowed = ", ".join(WEB_SEARCH_ADAPTER_CHOICES)
    raise ValueError(f"web_search_adapter must be one of: {allowed}")
