from __future__ import annotations

from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.llm.factory import make_llm_client
from sylliptor_agent_cli.profiles import ProfileSpec, add_profile, set_active_profile


def _cfg_with_profile(profile: ProfileSpec, *, active: bool = True) -> AppConfig:
    cfg = AppConfig(model="gpt-test")
    cfg.extra_fields = {"profiles": {}, "active_profile": ""}
    add_profile(cfg, profile)
    if active:
        set_active_profile(cfg, profile.name)
    return cfg


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
