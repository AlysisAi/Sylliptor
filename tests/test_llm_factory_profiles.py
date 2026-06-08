from __future__ import annotations

from sylliptor_agent_cli.agent_loop import create_session
from sylliptor_agent_cli.config import AppConfig, set_config_value
from sylliptor_agent_cli.llm.anthropic_messages import AnthropicMessagesClient
from sylliptor_agent_cli.llm.base import ChatClient
from sylliptor_agent_cli.llm.factory import make_llm_client
from sylliptor_agent_cli.llm.gemini_generate_content import GeminiGenerateContentClient
from sylliptor_agent_cli.llm.gemini_interactions import (
    GEMINI_INTERACTIONS_CONFIG_FLAG,
    GEMINI_INTERACTIONS_EXPERIMENT_ENV,
    GeminiInteractionsClient,
)
from sylliptor_agent_cli.llm.openai_compat import OpenAICompatClient
from sylliptor_agent_cli.llm.openai_responses import OpenAIResponsesClient
from sylliptor_agent_cli.llm.protocols import (
    ANTHROPIC_MESSAGES_PROTOCOL,
    GEMINI_GENERATE_CONTENT_PROTOCOL,
    GEMINI_INTERACTIONS_PROTOCOL,
    OPENAI_COMPAT_PROTOCOL,
    OPENAI_RESPONSES_PROTOCOL,
)
from sylliptor_agent_cli.profiles import ProfileSpec, add_profile, set_active_profile
from sylliptor_agent_cli.surface.noop_surface import NoopSurface


def _cfg_with_profile(profile: ProfileSpec, *, active: bool = True) -> AppConfig:
    cfg = AppConfig(model="gpt-test")
    cfg.extra_fields = {"profiles": {}, "active_profile": ""}
    add_profile(cfg, profile)
    if active:
        set_active_profile(cfg, profile.name)
    return cfg


class _WarningSurface(NoopSurface):
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def on_warning(self, warning: str) -> None:
        self.warnings.append(warning)


def test_make_llm_client_uses_active_profile_base_url() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(name="anthropic", base_url="https://api.anthropic.com/v1/openai")
    )
    cfg.base_url = "https://api.openai.com/v1"

    client = make_llm_client(cfg=cfg, api_key="key", model="claude")

    assert client.base_url == "https://api.anthropic.com/v1/openai"


def test_make_llm_client_passes_extra_headers_for_anthropic_profile() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="anthropic",
            base_url="https://api.anthropic.com/v1/openai",
            extra_headers={"anthropic-version": "2023-06-01"},
        )
    )
    cfg.base_url = "https://api.openai.com/v1"

    client = make_llm_client(cfg=cfg, api_key="key", model="claude")

    assert client.extra_headers["anthropic-version"] == "2023-06-01"


def test_make_llm_client_explicit_profile_arg_wins_over_active() -> None:
    cfg = _cfg_with_profile(ProfileSpec(name="openai", base_url="https://api.openai.com/v1"))
    explicit = ProfileSpec(name="custom", base_url="https://custom.example/v1")

    client = make_llm_client(cfg=cfg, api_key="key", model="model", profile=explicit)

    assert client.base_url == "https://custom.example/v1"


def test_active_profile_base_url_overrides_stale_top_level_base_url() -> None:
    cfg = _cfg_with_profile(ProfileSpec(name="openai", base_url="https://api.openai.com/v1"))
    cfg.base_url = "https://legacy.example/v1"

    client = make_llm_client(cfg=cfg, api_key="key", model="model")

    assert client.base_url == "https://api.openai.com/v1"


def test_make_llm_client_uses_configured_reasoning_effort() -> None:
    cfg = _cfg_with_profile(ProfileSpec(name="openai", base_url="https://api.openai.com/v1"))
    cfg.llm_reasoning_effort = "high"

    client = make_llm_client(cfg=cfg, api_key="key", model="gpt-5")

    assert client.reasoning_effort == "high"


def test_make_llm_client_routes_openai_compat_to_existing_client() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="openai",
            protocol="openai_compat",
            base_url="https://api.openai.com/v1",
            extra_headers={"x-test": "yes"},
        )
    )

    client = make_llm_client(cfg=cfg, api_key="key", model="gpt-test")

    assert isinstance(client, ChatClient)
    assert isinstance(client, OpenAICompatClient)
    assert client.base_url == "https://api.openai.com/v1"
    assert client.extra_headers == {"x-test": "yes"}


def test_make_llm_client_routes_openai_responses_to_native_client() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="openai-native",
            protocol=OPENAI_RESPONSES_PROTOCOL,
            base_url="https://api.openai.com/v1",
            extra_headers={"x-test": "yes"},
            web_search_adapter="openai_responses",
        )
    )
    cfg.llm_reasoning_effort = "low"
    cfg.web_search_mode = "native"

    client = make_llm_client(cfg=cfg, api_key="key", model="gpt-test")

    assert isinstance(client, ChatClient)
    assert isinstance(client, OpenAIResponsesClient)
    assert client.base_url == "https://api.openai.com/v1"
    assert client.extra_headers == {"x-test": "yes"}
    assert client.reasoning_effort == "low"
    assert client.web_search_mode == "native"
    assert client.web_search_adapter == "openai_responses"


def test_make_llm_client_routes_anthropic_messages_to_native_client() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="anthropic",
            protocol=ANTHROPIC_MESSAGES_PROTOCOL,
            base_url="https://api.anthropic.com/v1",
            extra_headers={"anthropic-beta": "test-beta"},
            web_search_adapter="anthropic_messages",
        )
    )
    cfg.web_search_mode = "native"

    client = make_llm_client(cfg=cfg, api_key="key", model="claude-sonnet-4-6")

    assert isinstance(client, ChatClient)
    assert isinstance(client, AnthropicMessagesClient)
    assert client.base_url == "https://api.anthropic.com/v1"
    assert client.extra_headers == {"anthropic-beta": "test-beta"}
    assert client.web_search_mode == "native"
    assert client.web_search_adapter == "anthropic_messages"


def test_make_llm_client_routes_gemini_generate_content_to_native_client() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="gemini",
            protocol=GEMINI_GENERATE_CONTENT_PROTOCOL,
            base_url="https://generativelanguage.googleapis.com/v1beta",
            extra_headers={"x-test": "yes"},
            web_search_adapter="gemini_grounding",
        )
    )
    cfg.web_search_mode = "native"
    cfg.llm_reasoning_effort = "low"

    client = make_llm_client(cfg=cfg, api_key="key", model="gemini-3-flash-preview")

    assert isinstance(client, ChatClient)
    assert isinstance(client, GeminiGenerateContentClient)
    assert client.base_url == "https://generativelanguage.googleapis.com/v1beta"
    assert client.extra_headers == {"x-test": "yes"}
    assert client.reasoning_effort == "low"
    assert client.web_search_mode == "native"
    assert client.web_search_adapter == "gemini_grounding"


def test_make_llm_client_rejects_gemini_interactions_without_feature_flag(
    monkeypatch,
) -> None:
    monkeypatch.delenv(GEMINI_INTERACTIONS_EXPERIMENT_ENV, raising=False)
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="gemini-interactions",
            protocol=GEMINI_INTERACTIONS_PROTOCOL,
            base_url="https://generativelanguage.googleapis.com/v1beta",
            extra_headers={"x-test": "yes"},
        )
    )

    try:
        make_llm_client(cfg=cfg, api_key="key", model="gemini-2.5-flash")
    except Exception as exc:
        error = exc
    else:  # pragma: no cover - assertion failure branch.
        raise AssertionError("expected experimental Gemini Interactions profile to be rejected")

    assert type(error).__name__ == "UnsupportedProtocolError"
    assert "experimental and disabled by default" in str(error)
    assert GEMINI_INTERACTIONS_EXPERIMENT_ENV in str(error)


def test_make_llm_client_routes_gemini_interactions_when_env_flag_enabled(
    monkeypatch,
) -> None:
    monkeypatch.setenv(GEMINI_INTERACTIONS_EXPERIMENT_ENV, "1")
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="gemini-interactions",
            protocol=GEMINI_INTERACTIONS_PROTOCOL,
            base_url="https://generativelanguage.googleapis.com/v1beta",
            extra_headers={"x-test": "yes"},
        )
    )

    client = make_llm_client(cfg=cfg, api_key="key", model="gemini-2.5-flash")

    assert isinstance(client, ChatClient)
    assert isinstance(client, GeminiInteractionsClient)
    assert client.base_url == "https://generativelanguage.googleapis.com/v1beta"
    assert client.extra_headers == {"x-test": "yes"}


def test_make_llm_client_routes_gemini_interactions_when_config_flag_enabled(
    monkeypatch,
) -> None:
    monkeypatch.delenv(GEMINI_INTERACTIONS_EXPERIMENT_ENV, raising=False)
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="gemini-interactions",
            protocol=GEMINI_INTERACTIONS_PROTOCOL,
            base_url="https://generativelanguage.googleapis.com/v1beta",
        )
    )
    cfg = set_config_value(cfg, GEMINI_INTERACTIONS_CONFIG_FLAG, "true")

    client = make_llm_client(cfg=cfg, api_key="key", model="gemini-2.5-flash")

    assert isinstance(client, GeminiInteractionsClient)


def test_make_llm_client_return_annotation_is_protocol_neutral() -> None:
    assert make_llm_client.__annotations__["return"] == "ChatClient"


def test_make_llm_client_defaults_legacy_profile_without_protocol_to_openai_compat() -> None:
    profile = ProfileSpec.from_dict(
        "legacy",
        {
            "base_url": "https://legacy.example/v1",
            "default_model": "legacy-model",
        },
    )
    cfg = _cfg_with_profile(profile)

    client = make_llm_client(cfg=cfg, api_key="key", model="legacy-model")

    assert isinstance(client, OpenAICompatClient)
    assert client.base_url == "https://legacy.example/v1"


def test_create_session_accepts_gemini_generate_content_profile(tmp_path) -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="gemini",
            protocol=GEMINI_GENERATE_CONTENT_PROTOCOL,
            base_url="https://generativelanguage.googleapis.com/v1beta",
            web_search_adapter="gemini_grounding",
        )
    )
    cfg.model = "gemini-3-flash-preview"
    cfg.routing_mode = "code_only"
    cfg.web_search_mode = "off"

    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="review",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="key",
        non_interactive=True,
    )

    assert isinstance(session.client, GeminiGenerateContentClient)
    assert session.client.model == "gemini-3-flash-preview"


def test_create_session_accepts_openai_responses_profile(tmp_path) -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="openai-native",
            protocol=OPENAI_RESPONSES_PROTOCOL,
            base_url="https://api.openai.com/v1",
            web_search_adapter="openai_responses",
        )
    )
    cfg.model = "gpt-5.5"
    cfg.routing_mode = "code_only"
    cfg.web_search_mode = "off"

    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="review",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="key",
        non_interactive=True,
    )

    assert isinstance(session.client, OpenAIResponsesClient)
    assert session.client.model == "gpt-5.5"


def test_create_session_accepts_anthropic_messages_profile(tmp_path) -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="anthropic",
            protocol=ANTHROPIC_MESSAGES_PROTOCOL,
            base_url="https://api.anthropic.com/v1",
            web_search_adapter="anthropic_messages",
        )
    )
    cfg.model = "claude-sonnet-4-6"
    cfg.routing_mode = "code_only"
    cfg.web_search_mode = "off"

    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="review",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="key",
        non_interactive=True,
    )

    assert isinstance(session.client, AnthropicMessagesClient)
    assert session.client.model == "claude-sonnet-4-6"


def test_create_session_keeps_streaming_for_openai_responses_profile(tmp_path) -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="openai-native",
            protocol=OPENAI_RESPONSES_PROTOCOL,
            base_url="https://api.openai.com/v1",
            web_search_adapter="openai_responses",
        )
    )
    cfg.model = "gpt-5.5"
    cfg.routing_mode = "code_only"
    cfg.web_search_mode = "off"
    cfg.stream = True
    surface = _WarningSurface()

    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="review",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="key",
        non_interactive=True,
        surface=surface,
    )

    assert isinstance(session.client, OpenAIResponsesClient)
    assert session.stream is True
    assert session.cfg.stream is True
    assert surface.warnings == []


def test_create_session_keeps_streaming_for_anthropic_messages_profile(tmp_path) -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="anthropic",
            protocol=ANTHROPIC_MESSAGES_PROTOCOL,
            base_url="https://api.anthropic.com/v1",
            web_search_adapter="anthropic_messages",
        )
    )
    cfg.model = "claude-sonnet-4-6"
    cfg.routing_mode = "code_only"
    cfg.web_search_mode = "off"
    cfg.stream = True
    surface = _WarningSurface()

    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="review",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="key",
        non_interactive=True,
        surface=surface,
    )

    assert isinstance(session.client, AnthropicMessagesClient)
    assert session.stream is True
    assert session.cfg.stream is True
    assert surface.warnings == []


def test_create_session_keeps_streaming_for_gemini_generate_content_profile(tmp_path) -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="gemini",
            protocol=GEMINI_GENERATE_CONTENT_PROTOCOL,
            base_url="https://generativelanguage.googleapis.com/v1beta",
            web_search_adapter="gemini_grounding",
        )
    )
    cfg.model = "gemini-3-flash-preview"
    cfg.routing_mode = "code_only"
    cfg.web_search_mode = "off"
    cfg.stream = True
    surface = _WarningSurface()

    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="review",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="key",
        non_interactive=True,
        surface=surface,
    )

    assert isinstance(session.client, GeminiGenerateContentClient)
    assert session.stream is True
    assert session.cfg.stream is True
    assert surface.warnings == []


def test_create_session_keeps_streaming_for_openai_compat_profile(tmp_path) -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="openai",
            protocol=OPENAI_COMPAT_PROTOCOL,
            base_url="https://api.openai.com/v1",
        )
    )
    cfg.model = "gpt-test"
    cfg.routing_mode = "code_only"
    cfg.web_search_mode = "off"
    cfg.stream = True
    surface = _WarningSurface()

    session = create_session(
        cfg=cfg,
        root=tmp_path,
        mode="review",
        yes=True,
        max_steps=1,
        no_log=True,
        api_key_override="key",
        non_interactive=True,
        surface=surface,
    )

    assert isinstance(session.client, OpenAICompatClient)
    assert session.stream is True
    assert session.cfg.stream is True
    assert not any("streaming is disabled" in warning for warning in surface.warnings)
