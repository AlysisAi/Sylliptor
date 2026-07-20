"""Tests for the static reasoning contract table (capability report, Part A)."""

from __future__ import annotations

from sylliptor_agent_cli.profile_presets import PROFILE_PRESETS
from sylliptor_agent_cli.reasoning_contracts import (
    _CONTRACTS,
    ALWAYS_ON,
    NONE,
    OFF_EXPLICIT,
    OFF_IMPOSSIBLE,
    OFF_OMIT,
    OFF_SWAPS_MODEL,
    OFF_UNKNOWN,
    OPTIONAL,
    UNKNOWN,
    UNKNOWN_CONTRACT,
    reasoning_contract_for,
    reasoning_off_hazard,
    reasoning_off_is_safe,
)

_VALID_MODES = {ALWAYS_ON, OPTIONAL, NONE, UNKNOWN}
_VALID_OFF = {OFF_OMIT, OFF_EXPLICIT, OFF_IMPOSSIBLE, OFF_SWAPS_MODEL, OFF_UNKNOWN}


def test_table_integrity() -> None:
    for provider, rules in _CONTRACTS.items():
        assert rules, provider
        for prefix, contract in rules:
            assert contract.mode in _VALID_MODES, (provider, prefix)
            assert contract.off in _VALID_OFF, (provider, prefix)
            # always-on models must never advertise a safe off path.
            if contract.mode == ALWAYS_ON:
                assert contract.off in {OFF_IMPOSSIBLE, OFF_SWAPS_MODEL}, (provider, prefix)
            # models without a knob carry no allowed values.
            if contract.mode == NONE:
                assert contract.values == (), (provider, prefix)
            # a declared default must be an allowed value when values are known.
            if contract.default and contract.values:
                assert contract.default in contract.values, (provider, prefix)


def test_unknown_model_and_provider_fall_back_to_unknown() -> None:
    assert reasoning_contract_for("nope", "whatever") is UNKNOWN_CONTRACT
    assert reasoning_contract_for("openai", "some-future-model") is UNKNOWN_CONTRACT
    assert reasoning_contract_for(None, None) is UNKNOWN_CONTRACT
    assert not reasoning_off_is_safe("nope", "whatever")
    assert "unknown" in reasoning_off_hazard("nope", "whatever")


def test_kimi_code_off_swaps_model() -> None:
    contract = reasoning_contract_for("kimi-code", "k3")
    assert contract.mode == ALWAYS_ON
    assert contract.off == OFF_SWAPS_MODEL
    assert contract.default == "high"  # coding-surface default, not the platform 'max'
    assert not reasoning_off_is_safe("kimi-code", "k3")
    assert "substitutes" in reasoning_off_hazard("kimi-code", "k3")


def test_moonshot_platform_contracts() -> None:
    k27 = reasoning_contract_for("moonshot", "kimi-k2.7-code")
    assert k27.mode == ALWAYS_ON and k27.off == OFF_IMPOSSIBLE
    highspeed = reasoning_contract_for("moonshot", "kimi-k2.7-code-highspeed")
    assert highspeed is k27  # prefix rule covers the variant
    k3 = reasoning_contract_for("moonshot", "kimi-k3")
    assert k3.values == ("low", "high", "max") and k3.default == "max"
    assert reasoning_off_is_safe("moonshot", "kimi-k2.6")


def test_openai_codex_cannot_disable() -> None:
    codex = reasoning_contract_for("openai", "gpt-5.3-codex")
    assert codex.mode == ALWAYS_ON
    assert not codex.allows_value("none")
    terra = reasoning_contract_for("openai", "gpt-5.6-terra")
    assert terra.allows_value("none") and terra.default == "medium"
    assert not terra.allows_value("minimal")  # dead on the 5.x families


def test_anthropic_adaptive_vs_haiku() -> None:
    fable = reasoning_contract_for("anthropic", "claude-fable-5")
    assert fable.mode == ALWAYS_ON and fable.off == OFF_IMPOSSIBLE
    haiku = reasoning_contract_for("anthropic", "claude-haiku-4-5")
    assert haiku.wire == "budget_tokens"
    sonnet = reasoning_contract_for("anthropic", "claude-sonnet-5")
    assert sonnet.off == OFF_OMIT and sonnet.toggleable


def test_gemini_pro_lacks_minimal() -> None:
    pro = reasoning_contract_for("gemini", "gemini-3.1-pro-preview")
    assert not pro.allows_value("minimal")
    flash = reasoning_contract_for("gemini", "gemini-3.5-flash")
    assert flash.allows_value("minimal")
    assert flash.off == OFF_IMPOSSIBLE


def test_catalog_models_resolve_beyond_unknown_where_researched() -> None:
    # Every suggested model on the core presets should hit a real rule (the
    # probe-gated providers legitimately stay unknown).
    probe_gated = {"bytedance", "fireworks", "openrouter", "sylliptor"}
    for preset in PROFILE_PRESETS:
        provider = preset.provider_key or preset.key
        if provider in probe_gated or preset.key in {"ollama", "lm-studio", "vllm", "custom"}:
            continue
        for model in preset.suggested_models:
            contract = reasoning_contract_for(provider, model, preset_key=preset.key)
            assert contract is not UNKNOWN_CONTRACT, (preset.key, model)


def test_preset_key_scopes_the_surface() -> None:
    # Same vendor, different surface, different contract: platform k3 defaults
    # to 'max' and errors on disable; membership k3 defaults to 'high' and
    # silently swaps models on disable.
    platform = reasoning_contract_for("moonshot", "kimi-k3")
    membership = reasoning_contract_for("moonshot", "k3", preset_key="kimi-code")
    assert platform.default == "max" and platform.off == OFF_IMPOSSIBLE
    assert membership.default == "high" and membership.off == OFF_SWAPS_MODEL
