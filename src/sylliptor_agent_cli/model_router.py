from __future__ import annotations

from typing import Any

from .branding import env_get
from .config import AppConfig, ConfigError

ROLE_CODING = "coding"
ROLE_REVIEW = "review"
ROLE_CONFLICT_REVIEW = "conflict_review"
ROLE_CONFLICT_RESOLVE = "conflict_resolve"
ROLE_PLANNER = "planner"
ROLE_COMPREHENSION = "comprehension"
ROLE_COMPACTOR = "compactor"
ROLE_ROUTER = "router"
PREFER_CONTEXT_FORGE = "forge"

ROLE_ENV_VARS: dict[str, str] = {
    ROLE_CODING: "SYLLIPTOR_MODEL_CODING",
    ROLE_REVIEW: "SYLLIPTOR_MODEL_REVIEW",
    ROLE_CONFLICT_REVIEW: "SYLLIPTOR_MODEL_CONFLICT_REVIEW",
    ROLE_CONFLICT_RESOLVE: "SYLLIPTOR_MODEL_CONFLICT_RESOLVE",
    ROLE_PLANNER: "SYLLIPTOR_MODEL_PLANNER",
    ROLE_COMPREHENSION: "SYLLIPTOR_COMPREHENSION_MODEL",
    ROLE_COMPACTOR: "SYLLIPTOR_MODEL_COMPACTOR",
    ROLE_ROUTER: "SYLLIPTOR_MODEL_ROUTER",
}


def _role_models_from_plan(plan: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(plan, dict):
        return {}
    raw = plan.get("role_models")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        role = str(key).strip().lower()
        model = str(value).strip()
        if role and model:
            out[role] = model
    return out


def _role_models_from_cfg(cfg: AppConfig) -> dict[str, str]:
    raw = cfg.extra_fields.get("role_models") if isinstance(cfg.extra_fields, dict) else None
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        role = str(key).strip().lower()
        model = str(value).strip()
        if role and model:
            out[role] = model
    return out


def _forge_role_models_from_cfg(cfg: AppConfig) -> dict[str, str]:
    raw = cfg.extra_fields.get("forge_role_models") if isinstance(cfg.extra_fields, dict) else None
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        role = str(key).strip().lower()
        model = str(value).strip()
        if role and model:
            out[role] = model
    return out


def resolve_model_for_role(
    *,
    cfg: AppConfig,
    role: str,
    plan: dict[str, Any] | None = None,
    fallback_to_default: bool = True,
    prefer_context: str = "default",
) -> str:
    role_key = role.strip().lower()
    if role_key not in ROLE_ENV_VARS:
        raise ConfigError(f"Unknown model role: {role}")

    env_key = ROLE_ENV_VARS[role_key]
    env_value = str(env_get(env_key) or "").strip()
    if env_value:
        return env_value

    plan_models = _role_models_from_plan(plan)
    plan_model = (plan_models.get(role_key) or "").strip()
    if plan_model:
        return plan_model

    if prefer_context == PREFER_CONTEXT_FORGE:
        forge_cfg_models = _forge_role_models_from_cfg(cfg)
        forge_cfg_model = (forge_cfg_models.get(role_key) or "").strip()
        if forge_cfg_model:
            return forge_cfg_model

    cfg_models = _role_models_from_cfg(cfg)
    cfg_model = (cfg_models.get(role_key) or "").strip()
    if cfg_model:
        return cfg_model

    if role_key == ROLE_COMPREHENSION:
        return resolve_model_for_role(
            cfg=cfg,
            role=ROLE_CODING,
            plan=plan,
            fallback_to_default=fallback_to_default,
            prefer_context=prefer_context,
        )

    if fallback_to_default:
        fallback = (cfg.model or "").strip()
        if fallback:
            return fallback

    raise ConfigError(
        "Model is not set for role "
        f"'{role_key}'. Set {env_key}, role_models.{role_key}, or default model."
    )
