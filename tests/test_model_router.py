from __future__ import annotations

import pytest

from sylliptor_agent_cli.config import AppConfig, ConfigError
from sylliptor_agent_cli.model_router import (
    PREFER_CONTEXT_FORGE,
    ROLE_CODING,
    ROLE_COMPACTOR,
    ROLE_COMPREHENSION,
    ROLE_REVIEW,
    ROLE_ROUTER,
    resolve_model_for_role,
)


def test_resolve_model_for_role_env_overrides_plan_and_config(monkeypatch) -> None:
    cfg = AppConfig(model="default-model")
    cfg.extra_fields = {
        "role_models": {
            ROLE_CODING: "config-coding-model",
            ROLE_REVIEW: "config-review-model",
        }
    }
    plan = {"role_models": {ROLE_CODING: "plan-coding-model"}}
    monkeypatch.setenv("SYLLIPTOR_MODEL_CODING", "env-coding-model")

    resolved = resolve_model_for_role(cfg=cfg, role=ROLE_CODING, plan=plan)
    assert resolved == "env-coding-model"


def test_resolve_model_for_role_plan_overrides_config(monkeypatch) -> None:
    cfg = AppConfig(model="default-model")
    cfg.extra_fields = {
        "role_models": {
            ROLE_CODING: "config-coding-model",
        }
    }
    plan = {"role_models": {ROLE_CODING: "plan-coding-model"}}
    monkeypatch.delenv("SYLLIPTOR_MODEL_CODING", raising=False)

    resolved = resolve_model_for_role(cfg=cfg, role=ROLE_CODING, plan=plan)
    assert resolved == "plan-coding-model"


def test_resolve_router_model_prefers_forge_override(monkeypatch) -> None:
    cfg = AppConfig(model="default-model")
    cfg.extra_fields = {
        "role_models": {ROLE_ROUTER: "config-router-model"},
        "forge_role_models": {ROLE_ROUTER: "forge-router-model"},
    }
    monkeypatch.delenv("SYLLIPTOR_MODEL_ROUTER", raising=False)

    resolved = resolve_model_for_role(
        cfg=cfg,
        role=ROLE_ROUTER,
        plan={},
        prefer_context=PREFER_CONTEXT_FORGE,
    )

    assert resolved == "forge-router-model"


def test_resolve_model_for_role_falls_back_to_cfg_model(monkeypatch) -> None:
    cfg = AppConfig(model="default-model")
    cfg.extra_fields = {}
    monkeypatch.delenv("SYLLIPTOR_MODEL_REVIEW", raising=False)

    resolved = resolve_model_for_role(cfg=cfg, role=ROLE_REVIEW, plan=None)
    assert resolved == "default-model"


def test_resolve_model_for_role_raises_when_all_empty(monkeypatch) -> None:
    cfg = AppConfig(model="")
    cfg.extra_fields = {}
    monkeypatch.delenv("SYLLIPTOR_MODEL_CODING", raising=False)

    with pytest.raises(ConfigError, match="Model is not set for role 'coding'"):
        resolve_model_for_role(cfg=cfg, role=ROLE_CODING, plan={})


def test_resolve_model_for_role_can_disable_default_fallback(monkeypatch) -> None:
    cfg = AppConfig(model="default-model")
    cfg.extra_fields = {}
    monkeypatch.delenv("SYLLIPTOR_MODEL_CONFLICT_RESOLVE", raising=False)

    with pytest.raises(ConfigError, match="conflict_resolve"):
        resolve_model_for_role(
            cfg=cfg,
            role="conflict_resolve",
            plan={},
            fallback_to_default=False,
        )


def test_resolve_model_for_role_rejects_unknown_role() -> None:
    cfg = AppConfig(model="default-model")
    with pytest.raises(ConfigError, match="Unknown model role"):
        resolve_model_for_role(cfg=cfg, role="unknown-role", plan=None)


def test_resolve_model_for_compactor_env_overrides_plan_and_config(monkeypatch) -> None:
    cfg = AppConfig(model="default-model")
    cfg.extra_fields = {"role_models": {ROLE_COMPACTOR: "config-compactor-model"}}
    plan = {"role_models": {ROLE_COMPACTOR: "plan-compactor-model"}}
    monkeypatch.setenv("SYLLIPTOR_MODEL_COMPACTOR", "env-compactor-model")

    resolved = resolve_model_for_role(cfg=cfg, role=ROLE_COMPACTOR, plan=plan)
    assert resolved == "env-compactor-model"


def test_resolve_model_for_comprehension_uses_dedicated_env_var(monkeypatch) -> None:
    cfg = AppConfig(model="default-model")
    cfg.extra_fields = {
        "role_models": {
            ROLE_CODING: "config-coding-model",
            ROLE_COMPREHENSION: "config-comprehension-model",
        }
    }
    monkeypatch.setenv("SYLLIPTOR_COMPREHENSION_MODEL", "env-comprehension-model")

    resolved = resolve_model_for_role(cfg=cfg, role=ROLE_COMPREHENSION, plan={})
    assert resolved == "env-comprehension-model"


def test_resolve_model_for_comprehension_falls_back_to_coding_role(monkeypatch) -> None:
    cfg = AppConfig(model="default-model")
    cfg.extra_fields = {"role_models": {ROLE_CODING: "config-coding-model"}}
    monkeypatch.delenv("SYLLIPTOR_COMPREHENSION_MODEL", raising=False)
    monkeypatch.delenv("SYLLIPTOR_MODEL_CODING", raising=False)

    resolved = resolve_model_for_role(cfg=cfg, role=ROLE_COMPREHENSION, plan={})
    assert resolved == "config-coding-model"
