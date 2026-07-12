from __future__ import annotations

import builtins

import sylliptor_agent_cli.litellm_static_provider as provider_mod
import sylliptor_agent_cli.model_registry as model_registry_mod
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.litellm_static_provider import (
    BUNDLED_MODEL_CATALOG_SOURCE,
    LiteLLMStaticMetadata,
    get_bundled_model_catalog_provenance,
    resolve_litellm_static_metadata,
)
from sylliptor_agent_cli.model_registry import ModelRegistry
from sylliptor_agent_cli.token_budget import compute_input_budget
from sylliptor_agent_cli.usage_tracker import compute_context_left


def _bundled_meta(
    *,
    context_window_tokens: int | None,
    max_output_tokens: int | None,
    supports_vision: bool | None = None,
    input_cost_per_token: float | None = None,
    output_cost_per_token: float | None = None,
    error: str | None = None,
) -> LiteLLMStaticMetadata:
    return LiteLLMStaticMetadata(
        model_key="gpt-5-nano",
        context_window_tokens=context_window_tokens,
        max_output_tokens=max_output_tokens,
        supports_vision=supports_vision,
        input_cost_per_token=input_cost_per_token,
        output_cost_per_token=output_cost_per_token,
        raw_metadata={},
        error=error,
    )


def _raise(exc: Exception) -> None:
    raise exc


def test_litellm_static_provider_handles_missing_bundled_catalog(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_mod,
        "_load_bundled_model_catalog",
        lambda: _raise(FileNotFoundError("missing")),
    )
    result = resolve_litellm_static_metadata("gpt-5-nano")
    assert result.error == "bundled model catalog missing"
    assert result.context_window_tokens is None
    assert result.max_output_tokens is None


def test_provider_scoped_catalog_lookup_never_borrows_another_host_route() -> None:
    cases = [
        ("MiniMax-M2.7", "https://api.minimax.io/v1", "minimax", None),
        ("yi-large", "https://api.lingyiwanwu.com/v1", "01ai", None),
        ("llama3.3-70b", "https://api.cerebras.ai/v1", "cerebras", None),
        (
            "sonar-pro",
            "https://api.perplexity.ai",
            "perplexity",
            "perplexity/sonar-pro",
        ),
        ("zai-org/GLM-5.1", "https://api.together.ai/v1", "together", None),
        (
            "accounts/fireworks/models/deepseek-v4-pro",
            "https://api.fireworks.ai/inference/v1",
            "fireworks",
            "fireworks_ai/deepseek-v4-pro",
        ),
        (
            "gpt-4o",
            "https://api.groq.com/openai/v1",
            "groq",
            None,
        ),
    ]

    for model, base_url, provider_hint, expected_key in cases:
        result = resolve_litellm_static_metadata(
            model,
            base_url=base_url,
            provider_hint=provider_hint,
        )
        assert result.model_key == expected_key
        if expected_key is None:
            assert result.error == "model not found in bundled model catalog"
        else:
            assert result.error is None
            assert result.raw_metadata["catalog_provider_hint"] == provider_hint


def test_litellm_static_provider_handles_invalid_bundled_catalog(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_mod,
        "_load_bundled_model_catalog",
        lambda: _raise(ValueError("bad json")),
    )
    result = resolve_litellm_static_metadata("gpt-5-nano")
    assert result.error == "bundled model catalog invalid"
    assert result.context_window_tokens is None
    assert result.max_output_tokens is None


def test_litellm_static_provider_treats_catalog_decode_errors_as_invalid(monkeypatch) -> None:
    provider_mod._load_bundled_model_catalog.cache_clear()

    class _FakeCatalogPath:
        def joinpath(self, _filename: str) -> _FakeCatalogPath:
            return self

        def read_text(self, *, encoding: str) -> str:
            _ = encoding
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(provider_mod.resources, "files", lambda _package: _FakeCatalogPath())
    result = resolve_litellm_static_metadata("gpt-5-nano")
    assert result.error == "bundled model catalog invalid"
    assert result.context_window_tokens is None
    assert result.max_output_tokens is None


def test_bundled_model_catalog_provenance_handles_missing_meta(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_mod,
        "_load_bundled_model_catalog_meta",
        lambda: _raise(FileNotFoundError("missing")),
    )
    provenance = get_bundled_model_catalog_provenance()
    assert provenance.error == "bundled model catalog provenance missing"
    assert provenance.upstream_commit_sha is None


def test_bundled_model_catalog_provenance_handles_invalid_meta(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_mod,
        "_load_bundled_model_catalog_meta",
        lambda: _raise(ValueError("bad meta")),
    )
    provenance = get_bundled_model_catalog_provenance()
    assert provenance.error == "bundled model catalog provenance invalid"
    assert provenance.fetched_at_utc is None


def test_litellm_static_provider_never_imports_litellm(monkeypatch) -> None:
    provider_mod._load_bundled_model_catalog.cache_clear()
    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "litellm":
            raise AssertionError("litellm import attempted")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    result = resolve_litellm_static_metadata("gpt-5-nano")
    assert result.error is None
    assert result.model_key is not None


def test_litellm_static_provider_uses_model_variants(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_mod,
        "_load_bundled_model_catalog",
        lambda: {
            "sample_spec": {"max_tokens": "ignore-me"},
            "openai/gpt-4o-2026-01-01": {
                "max_tokens": 256000,
                "max_output_tokens": 8192,
                "input_cost_per_token": 0.000001,
                "output_cost_per_token": 0.000002,
                "cache_read_input_token_cost": 0.0000001,
                "cache_creation_input_token_cost": 0.00000125,
                "cache_creation_input_token_cost_above_1hr": 0.000002,
                "output_cost_per_reasoning_token": 0.000003,
            },
            "ignored_string": "not-a-model",
        },
    )
    result = resolve_litellm_static_metadata("gpt-4o")
    assert result.error is None
    assert result.model_key == "openai/gpt-4o-2026-01-01"
    assert result.context_window_tokens == 256000
    assert result.max_output_tokens == 8192
    assert result.input_cost_per_token == 0.000001
    assert result.output_cost_per_token == 0.000002
    assert result.cache_read_input_cost_per_token == 0.0000001
    assert result.cache_creation_input_cost_per_token == 0.00000125
    assert result.cache_creation_1h_input_cost_per_token == 0.000002
    assert result.reasoning_output_cost_per_token == 0.000003


def test_bundled_snapshot_resolves_1h_cache_write_rate() -> None:
    provider_mod._load_bundled_model_catalog.cache_clear()
    catalog = provider_mod._load_bundled_model_catalog()

    result = resolve_litellm_static_metadata("claude-opus-4-1")
    assert result.error is None
    raw = catalog[result.model_key]
    assert result.cache_creation_1h_input_cost_per_token is not None
    assert (
        result.cache_creation_1h_input_cost_per_token
        == raw["cache_creation_input_token_cost_above_1hr"]
    )
    assert result.cache_creation_input_cost_per_token == raw["cache_creation_input_token_cost"]
    assert (
        result.cache_creation_1h_input_cost_per_token > result.cache_creation_input_cost_per_token
    )


def test_bundled_snapshot_resolves_reasoning_output_rate() -> None:
    provider_mod._load_bundled_model_catalog.cache_clear()
    catalog = provider_mod._load_bundled_model_catalog()

    candidates = [
        (key, entry)
        for key, entry in catalog.items()
        if isinstance(entry, dict)
        and isinstance(entry.get("output_cost_per_reasoning_token"), (int, float))
        and entry.get("output_cost_per_reasoning_token") != entry.get("output_cost_per_token")
    ]
    assert candidates

    # Variant matching may resolve a dated key to its undated sibling, so only
    # keep candidates that round-trip to their own catalog entry.
    for key, entry in candidates:
        result = resolve_litellm_static_metadata(key)
        if result.model_key != key:
            continue
        assert result.error is None
        assert result.reasoning_output_cost_per_token is not None
        assert result.reasoning_output_cost_per_token == entry["output_cost_per_reasoning_token"]
        assert result.reasoning_output_cost_per_token != result.output_cost_per_token
        return
    raise AssertionError("no reasoning-rate catalog entry resolved to itself")


def test_litellm_static_provider_ignores_sample_spec_entry(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_mod,
        "_load_bundled_model_catalog",
        lambda: {
            "sample_spec": {
                "max_tokens": 123456,
                "max_output_tokens": 7890,
            }
        },
    )
    result = resolve_litellm_static_metadata("sample_spec")
    assert result.error == "model not found in bundled model catalog"
    assert result.model_key is None


def test_litellm_static_provider_derives_total_context_from_input_and_output(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_mod,
        "_load_bundled_model_catalog",
        lambda: {
            "dashscope/qwen3.5-plus": {
                "max_tokens": 65536,
                "max_input_tokens": 991808,
                "max_output_tokens": 65536,
                "supports_vision": True,
            }
        },
    )
    result = resolve_litellm_static_metadata(
        "qwen3.5-plus",
        base_url="https://coding-intl.dashscope.aliyuncs.com/v1",
    )
    assert result.error is None
    assert result.model_key == "dashscope/qwen3.5-plus"
    assert result.context_window_tokens == 1057344
    assert result.max_output_tokens == 65536
    assert result.supports_vision is True


def test_litellm_static_provider_accepts_integral_float_capacity_fields(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_mod,
        "_load_bundled_model_catalog",
        lambda: {
            "xai/grok-4-fast-reasoning": {
                "max_tokens": 2000000.0,
                "max_input_tokens": 1800000.0,
                "max_output_tokens": 200000.0,
            }
        },
    )

    result = resolve_litellm_static_metadata("xai/grok-4-fast-reasoning")

    assert result.error is None
    assert result.context_window_tokens == 2000000
    assert result.max_output_tokens == 200000


def test_litellm_static_provider_uses_max_tokens_when_only_output_cap_exists(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_mod,
        "_load_bundled_model_catalog",
        lambda: {
            "openai/gpt-5-nano": {
                "max_tokens": 128000,
                "input_cost_per_token": 0.1,
                "output_cost_per_token": 0.2,
            }
        },
    )
    result = resolve_litellm_static_metadata("gpt-5-nano")
    assert result.error is None
    assert result.context_window_tokens == 128000
    assert result.max_output_tokens is None


def test_litellm_static_provider_prefers_endpoint_matching_alias(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_mod,
        "_load_bundled_model_catalog",
        lambda: {
            "openrouter/qwen3.5-plus": {
                "max_tokens": 128000,
                "max_output_tokens": 4096,
            },
            "dashscope/qwen3.5-plus": {
                "max_input_tokens": 991808,
                "max_output_tokens": 65536,
            },
        },
    )
    result = resolve_litellm_static_metadata(
        "qwen3.5-plus",
        base_url="https://coding-intl.dashscope.aliyuncs.com/v1",
    )
    assert result.error is None
    assert result.model_key == "dashscope/qwen3.5-plus"
    assert result.context_window_tokens == 1057344
    assert result.max_output_tokens == 65536


def test_litellm_static_provider_prefers_shallower_alias_when_provider_is_ambiguous(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        provider_mod,
        "_load_bundled_model_catalog",
        lambda: {
            "openrouter/z-ai/glm-5": {
                "max_input_tokens": 202752,
                "max_output_tokens": 128000,
            },
            "zai/glm-5": {
                "max_input_tokens": 200000,
                "max_output_tokens": 128000,
            },
        },
    )
    result = resolve_litellm_static_metadata("glm-5")
    assert result.error is None
    assert result.model_key == "zai/glm-5"
    assert result.context_window_tokens == 328000
    assert result.max_output_tokens == 128000


def test_env_overrides_beat_user_and_bundled_catalog(monkeypatch) -> None:
    monkeypatch.setattr(
        model_registry_mod,
        "resolve_litellm_static_metadata",
        lambda _model, *, base_url=None, provider_hint=None: _bundled_meta(
            context_window_tokens=128000,
            max_output_tokens=4096,
            supports_vision=True,
            input_cost_per_token=0.1,
            output_cost_per_token=0.2,
        ),
    )
    monkeypatch.setenv("SYLLIPTOR_CONTEXT_WINDOW", "64000")
    monkeypatch.setenv("SYLLIPTOR_MAX_OUTPUT_TOKENS", "4000")
    monkeypatch.setenv("SYLLIPTOR_SUPPORTS_VISION", "1")
    monkeypatch.setenv("SYLLIPTOR_INPUT_COST_PER_TOKEN", "0.01")
    monkeypatch.setenv("SYLLIPTOR_OUTPUT_COST_PER_TOKEN", "0.02")

    cfg = AppConfig(base_url="https://api.openai.com/v1", model="gpt-5-nano")
    cfg.extra_fields = {
        "model_metadata_overrides": {
            "models": {
                "gpt-5-nano": {
                    "context_window_tokens": 32000,
                    "max_output_tokens": 3000,
                    "supports_vision": False,
                    "input_cost_per_token": 1.0,
                    "output_cost_per_token": 2.0,
                }
            }
        }
    }
    meta = ModelRegistry(cfg=cfg).get("gpt-5-nano")
    assert meta.context_window_tokens == 64000
    assert meta.max_output_tokens == 4000
    assert meta.supports_vision is True
    assert meta.input_cost_per_token == 0.01
    assert meta.output_cost_per_token == 0.02
    assert meta.field_sources["context_window_tokens"] == "env:SYLLIPTOR_CONTEXT_WINDOW"
    assert meta.field_sources["max_output_tokens"] == "env:SYLLIPTOR_MAX_OUTPUT_TOKENS"


def test_user_overrides_beat_bundled_catalog(monkeypatch) -> None:
    monkeypatch.setattr(
        model_registry_mod,
        "resolve_litellm_static_metadata",
        lambda _model, *, base_url=None, provider_hint=None: _bundled_meta(
            context_window_tokens=128000,
            max_output_tokens=4096,
            supports_vision=True,
            input_cost_per_token=0.000001,
            output_cost_per_token=0.000002,
        ),
    )
    cfg = AppConfig(base_url="https://api.openai.com/v1", model="gpt-5-nano")
    cfg.extra_fields = {
        "model_metadata_overrides": {
            "models": {
                "gpt-5-nano": {
                    "context_window_tokens": 99999,
                    "max_output_tokens": 4444,
                    "cache_read_input_cost_per_token": 0.0000001,
                    "cache_creation_input_cost_per_token": 0.0000002,
                    "cache_creation_1h_input_cost_per_token": 0.0000003,
                }
            }
        }
    }
    meta = ModelRegistry(cfg=cfg).get("gpt-5-nano")
    assert meta.context_window_tokens == 99999
    assert meta.max_output_tokens == 4444
    assert meta.input_cost_per_token == 0.000001
    assert meta.output_cost_per_token == 0.000002
    assert meta.cache_read_input_cost_per_token == 0.0000001
    assert meta.cache_creation_input_cost_per_token == 0.0000002
    assert meta.cache_creation_1h_input_cost_per_token == 0.0000003
    assert meta.field_sources["context_window_tokens"] == "user:models['gpt-5-nano']"
    assert meta.field_sources["max_output_tokens"] == "user:models['gpt-5-nano']"
    assert meta.field_sources["cache_read_input_cost_per_token"] == "user:models['gpt-5-nano']"
    assert meta.field_sources["input_cost_per_token"] == BUNDLED_MODEL_CATALOG_SOURCE


def test_reasoning_support_uses_catalog_and_model_override_precedence(monkeypatch) -> None:
    monkeypatch.setattr(
        model_registry_mod,
        "resolve_litellm_static_metadata",
        lambda _model, *, base_url=None, provider_hint=None: LiteLLMStaticMetadata(
            model_key="openai/gpt-test",
            context_window_tokens=128000,
            max_output_tokens=4096,
            supports_vision=False,
            input_cost_per_token=None,
            output_cost_per_token=None,
            raw_metadata={"supports_reasoning": True},
            error=None,
        ),
    )
    cfg = AppConfig(base_url="https://api.openai.com/v1", model="gpt-test")

    catalog_meta = ModelRegistry(cfg=cfg).get("gpt-test")
    assert catalog_meta.supports_reasoning is True
    assert catalog_meta.field_sources["supports_reasoning"] == BUNDLED_MODEL_CATALOG_SOURCE

    cfg.extra_fields = {
        "model_metadata_overrides": {"models": {"gpt-test": {"supports_reasoning": False}}}
    }
    overridden_meta = ModelRegistry(cfg=cfg).get("gpt-test")
    assert overridden_meta.supports_reasoning is False
    assert overridden_meta.field_sources["supports_reasoning"] == "user:models['gpt-test']"


def test_bundled_catalog_beats_fallback_when_available(monkeypatch) -> None:
    monkeypatch.setattr(
        model_registry_mod,
        "resolve_litellm_static_metadata",
        lambda _model, *, base_url=None, provider_hint=None: _bundled_meta(
            context_window_tokens=200000,
            max_output_tokens=8192,
            supports_vision=True,
            input_cost_per_token=0.000001,
            output_cost_per_token=0.000002,
        ),
    )
    cfg = AppConfig(model="gpt-5-nano")
    meta = ModelRegistry(cfg=cfg).get("gpt-5-nano")
    assert meta.context_window_tokens == 200000
    assert meta.max_output_tokens == 8192
    assert meta.supports_vision is True
    assert meta.input_cost_per_token == 0.000001
    assert meta.output_cost_per_token == 0.000002
    assert meta.field_sources["context_window_tokens"] == BUNDLED_MODEL_CATALOG_SOURCE
    assert meta.field_sources["max_output_tokens"] == BUNDLED_MODEL_CATALOG_SOURCE
    assert meta.field_sources["supports_vision"] == BUNDLED_MODEL_CATALOG_SOURCE


def test_built_in_deepseek_v4_metadata_beats_fallback_when_bundled_catalog_lags(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        model_registry_mod,
        "resolve_litellm_static_metadata",
        lambda _model, *, base_url=None, provider_hint=None: _bundled_meta(
            context_window_tokens=None,
            max_output_tokens=None,
            error="model not found in bundled model catalog",
        ),
    )

    cfg = AppConfig(base_url="https://api.deepseek.com", model="deepseek-v4-pro")
    registry = ModelRegistry(cfg=cfg)
    meta = registry.get("deepseek-v4-pro")

    assert meta.model_name == "deepseek-v4-pro"
    assert meta.context_window_tokens == 1_000_000
    assert meta.max_output_tokens == 384_000
    assert meta.input_cost_per_token == 0.000000435
    assert meta.output_cost_per_token == 0.00000087
    assert meta.field_sources["context_window_tokens"] == "built_in"
    assert meta.field_sources["max_output_tokens"] == "built_in"
    assert registry.last_error is None
    assert not any("fallback context/max_output" in warning for warning in meta.warnings)


def test_per_field_mixing_sets_source_to_mixed(monkeypatch) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONTEXT_WINDOW", "64000")
    monkeypatch.setattr(
        model_registry_mod,
        "resolve_litellm_static_metadata",
        lambda _model, *, base_url=None, provider_hint=None: _bundled_meta(
            context_window_tokens=128000,
            max_output_tokens=4096,
            supports_vision=True,
            input_cost_per_token=0.000001,
            output_cost_per_token=0.000002,
        ),
    )
    cfg = AppConfig(model="gpt-5-nano")
    meta = ModelRegistry(cfg=cfg).get("gpt-5-nano")
    assert meta.context_window_tokens == 64000
    assert meta.max_output_tokens == 4096
    assert meta.source == "mixed"


def test_model_registry_uses_bundled_total_context_for_budget(monkeypatch) -> None:
    monkeypatch.setattr(
        model_registry_mod,
        "resolve_litellm_static_metadata",
        lambda _model, *, base_url=None, provider_hint=None: LiteLLMStaticMetadata(
            model_key="dashscope/qwen3.5-plus",
            context_window_tokens=1057344,
            max_output_tokens=65536,
            supports_vision=False,
            input_cost_per_token=None,
            output_cost_per_token=None,
            raw_metadata={
                "max_input_tokens": 991808,
                "max_output_tokens": 65536,
            },
            error=None,
        ),
    )
    cfg = AppConfig(base_url="https://coding-intl.dashscope.aliyuncs.com/v1", model="qwen3.5-plus")
    meta = ModelRegistry(cfg=cfg).get("qwen3.5-plus")
    assert meta.context_window_tokens == 1057344
    assert meta.max_output_tokens == 65536
    assert compute_input_budget(meta) == 991296


def test_endpoint_scoped_overrides_take_precedence(monkeypatch) -> None:
    monkeypatch.setattr(
        model_registry_mod,
        "resolve_litellm_static_metadata",
        lambda _model, *, base_url=None, provider_hint=None: _bundled_meta(
            context_window_tokens=128000,
            max_output_tokens=4096,
        ),
    )
    cfg = AppConfig(base_url="https://example.com/v1", model="gpt-5-nano")
    cfg.extra_fields = {
        "model_metadata_overrides": {
            "default": {"context_window_tokens": 16000},
            "models": {"gpt-5-nano": {"context_window_tokens": 24000}},
            "endpoints": {
                "https://example.com/v1/": {
                    "default": {"context_window_tokens": 32000},
                    "models": {"gpt-5-nano": {"context_window_tokens": 64000}},
                }
            },
        }
    }
    meta = ModelRegistry(cfg=cfg).get("gpt-5-nano")
    assert meta.context_window_tokens == 64000
    assert meta.field_sources["context_window_tokens"] == (
        "user:endpoints['https://example.com/v1/'].models['gpt-5-nano']"
    )


def test_override_alias_matching_supports_provider_and_version_variants(monkeypatch) -> None:
    monkeypatch.setattr(
        model_registry_mod,
        "resolve_litellm_static_metadata",
        lambda _model, *, base_url=None, provider_hint=None: _bundled_meta(
            context_window_tokens=None,
            max_output_tokens=None,
        ),
    )
    cfg = AppConfig(base_url="https://api.openai.com/v1", model="gpt-4o")
    cfg.extra_fields = {
        "model_metadata_overrides": {
            "models": {
                "openai/gpt-4o": {
                    "context_window_tokens": 123456,
                    "max_output_tokens": 3456,
                }
            }
        }
    }
    plain = ModelRegistry(cfg=cfg).get("gpt-4o")
    dated = ModelRegistry(cfg=cfg).get("gpt-4o-2026-01-01")
    assert plain.context_window_tokens == 123456
    assert dated.context_window_tokens == 123456
    assert plain.max_output_tokens == 3456
    assert dated.max_output_tokens == 3456
    assert plain.field_sources["context_window_tokens"] == "user:models['openai/gpt-4o']"
    assert dated.field_sources["context_window_tokens"] == "user:models['openai/gpt-4o']"


def test_registry_records_bundled_catalog_error_and_fallback_warning(monkeypatch) -> None:
    monkeypatch.setattr(
        model_registry_mod,
        "resolve_litellm_static_metadata",
        lambda _model, *, base_url=None, provider_hint=None: _bundled_meta(
            context_window_tokens=None,
            max_output_tokens=None,
            error="bundled model catalog missing",
        ),
    )
    cfg = AppConfig(model="gpt-5-nano")
    registry = ModelRegistry(cfg=cfg)
    meta = registry.get("gpt-5-nano")
    assert meta.context_window_tokens == 128000
    assert meta.max_output_tokens == 8192
    assert registry.last_error == "bundled model catalog missing"
    assert any("fallback context/max_output" in warning for warning in meta.warnings)


def test_unknown_model_fallback_window_keeps_startup_context_gauge_healthy(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        model_registry_mod,
        "resolve_litellm_static_metadata",
        lambda _model, *, base_url=None, provider_hint=None: _bundled_meta(
            context_window_tokens=None,
            max_output_tokens=None,
            error="model not found in bundled model catalog",
        ),
    )
    cfg = AppConfig(model="custom-live-model")
    registry = ModelRegistry(cfg=cfg)

    ctx = compute_context_left(
        messages=[{"role": "system", "content": "startup context " * 20_000}],
        model_name="custom-live-model",
        registry=registry,
    )

    assert ctx.context_window_tokens == 128000
    assert ctx.context_window_percent_left is not None
    assert ctx.context_window_percent_left > 60.0
