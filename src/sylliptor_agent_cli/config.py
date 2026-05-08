from __future__ import annotations

import json
import math
import os
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from .branding import (
    canonical_user_config_dir,
    canonical_user_data_dir,
    env_get,
)
from .llm.provider_limits import (
    DEFAULT_PROVIDER_CONCURRENCY_CAPS,
    DEFAULT_PROVIDER_RETRY_BASE_DELAY_SECONDS,
    DEFAULT_PROVIDER_RETRY_MAX_DELAY_SECONDS,
    DEFAULT_PROVIDER_RETRY_MAX_RETRIES,
)
from .step_budget import (
    DEFAULT_CHAT_MAX_STEPS,
    DEFAULT_SUBAGENT_MAX_STEPS,
    DEFAULT_TASK_MAX_STEPS,
)
from .web_search_adapters import normalize_web_search_adapter


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ApiKeyResolution:
    key: str | None
    source: str


_VALID_TOOLBAR_ITEMS: set[str] = {
    "mode",
    "model",
    "stream",
    "trace",
    "images",
    "temp",
    "ctx",
    "subagents",
    "tokens",
    "cost",
    "forge",
    "plan",
}
_DEFAULT_TOOLBAR_ITEMS: tuple[str, ...] = ("mode", "model", "ctx", "subagents")
DEFAULT_VERIFY_COMMANDS: tuple[str, ...] = ("pytest -q",)
VERIFY_RUNNER_PREFIXES: frozenset[tuple[str, str]] = frozenset(
    {
        ("poetry", "run"),
        ("uv", "run"),
        ("pipenv", "run"),
    }
)
VERIFY_PY_LAUNCHERS: frozenset[str] = frozenset({"py", "py.exe"})
VERIFY_PYTHON_LAUNCHER_RE = re.compile(r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?$")
VERIFY_MODULE_NAMES: frozenset[str] = frozenset({"pytest", "ruff", "unittest"})
_WEB_SEARCH_MODES: set[str] = {"off", "auto"}
_REASONING_EFFORTS: set[str] = {"none", "minimal", "low", "medium", "high", "xhigh"}
DEFAULT_FEEDBACK_GITHUB_REPO = "AlysisAi/Sylliptor"
_GITHUB_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}$")


class AssetsComprehensionConfig(BaseModel):
    role: str = "comprehension"
    vision_fallback_profile: str | None = None
    vision_with_ocr_when_available: bool = True
    ocr_enabled: Literal["auto", "always", "never"] = "auto"
    ocr_provider: str = "tesseract"
    ocr_timeout_seconds: int = 30
    image_max_edge_pixels: int = 2048
    questioning_mode: Literal["assertive", "balanced", "assumption_friendly"] = "balanced"
    schema_version: int = 1


class AssetsPlannerConfig(BaseModel):
    inline_images: bool = True
    max_inline_images: int = 8
    readiness_policy: Literal["soft", "block", "partial"] = "soft"
    readiness_timeout_seconds: float = 60.0
    max_chars_per_asset: int = 2000
    max_primary_per_task: int = 8


class AssetsWorkerConfig(BaseModel):
    inline_images: bool = True
    max_inline_images: int = 8
    fail_on_mirror_error: bool = False
    allocator_role: str = "comprehension"
    allocator_timeout_seconds: int = 30
    max_chars_per_asset_block: int = 4000
    max_focused_extract_chars: int = 4000
    schema_version: int = 1


class AssetsConfig(BaseModel):
    enabled: bool = True
    comprehension: AssetsComprehensionConfig = Field(default_factory=AssetsComprehensionConfig)
    planner: AssetsPlannerConfig = Field(default_factory=AssetsPlannerConfig)
    worker: AssetsWorkerConfig = Field(default_factory=AssetsWorkerConfig)


class AppConfig(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    model: str = ""
    llm_timeout_s: float = 60.0
    llm_enable_thinking: bool | None = None
    llm_reasoning_effort: str | None = None
    provider_concurrency_caps: dict[str, int] = Field(
        default_factory=lambda: dict(DEFAULT_PROVIDER_CONCURRENCY_CAPS),
    )
    provider_retry_max_retries: int = DEFAULT_PROVIDER_RETRY_MAX_RETRIES
    provider_retry_base_delay_seconds: float = DEFAULT_PROVIDER_RETRY_BASE_DELAY_SECONDS
    provider_retry_max_delay_seconds: float = DEFAULT_PROVIDER_RETRY_MAX_DELAY_SECONDS
    model_metadata_policy: str = "warn"
    default_mode: str = "review"  # review|auto|readonly|fullaccess
    max_steps: int = DEFAULT_CHAT_MAX_STEPS
    temperature: float = 0.2  # legacy global override
    coding_temperature: float = 0.2
    review_temperature: float = 0.0
    planner_temperature: float = 0.2
    conflict_review_temperature: float = 0.0
    compactor_temperature: float = 0.2
    chat_temperature: float = 0.7
    stream: bool = False
    routing_mode: str = "auto"  # auto|code_only
    step_budget_policy: str = "adaptive"
    task_max_steps: int = DEFAULT_TASK_MAX_STEPS
    subagent_max_steps: int = DEFAULT_SUBAGENT_MAX_STEPS
    subagents_enabled: bool = True
    skills_enabled: bool = True
    skills_auto_invoke: bool = Field(
        default=True,
        description=(
            "Enable model-decided skill activation from discovered skill descriptions; "
            "explicit false preserves manual/discovery-only behavior."
        ),
    )
    custom_tools_enabled: bool = True
    web_search_mode: str = "auto"
    web_search_adapter: str = "auto"
    web_search_base_url: str | None = None
    web_search_model: str | None = None
    web_search_timeout_s: float = 45.0
    update_check_enabled: bool = True
    update_check_interval_hours: int = 24
    update_check_timeout_s: float = 3.0
    feedback_github_enabled: bool = True
    feedback_github_repo: str = DEFAULT_FEEDBACK_GITHUB_REPO
    feedback_open_browser: bool = True
    session_log_dir: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    verify_commands: list[str] = Field(
        default_factory=lambda: list(DEFAULT_VERIFY_COMMANDS),
    )
    integration_verify_mode: str = "warn"
    integration_verify_commands: list[str] = Field(default_factory=list)
    replanning_mode: str = "off"
    toolbar_items: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_TOOLBAR_ITEMS),
    )
    assets: AssetsConfig = Field(default_factory=AssetsConfig)

    # Internal: allow future keys without crashing older clients.
    extra_fields: dict[str, Any] = Field(default_factory=dict, exclude=True)


_ROLE_TEMPERATURE_FIELDS: dict[str, str] = {
    "coding": "coding_temperature",
    "review": "review_temperature",
    "planner": "planner_temperature",
    "conflict_review": "conflict_review_temperature",
    "compactor": "compactor_temperature",
    "chat": "chat_temperature",
}

_ROLE_TEMPERATURE_DEFAULTS: dict[str, float] = {
    "coding": 0.2,
    "review": 0.0,
    "planner": 0.2,
    "conflict_review": 0.0,
    "compactor": 0.2,
    "chat": 0.7,
}


def clone_cfg(cfg: AppConfig) -> AppConfig:
    """Return a deep copy of AppConfig while preserving extra_fields."""
    return cfg.model_copy(deep=True)


def normalize_verify_command_list(commands: Sequence[str] | None) -> tuple[str, ...]:
    if not commands:
        return ()
    normalized: list[str] = []
    for item in commands:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def is_generic_verify_command_fallback(commands: Sequence[str] | None) -> bool:
    return normalize_verify_command_list(commands) == DEFAULT_VERIFY_COMMANDS


def is_generic_configured_verify_preset(commands: Sequence[str] | None) -> bool:
    normalized = normalize_verify_command_list(commands)
    if not normalized:
        return False
    return all(_is_generic_configured_verify_command(command) for command in normalized)


def split_verify_command_parts(command: str) -> list[str] | None:
    text = str(command or "")
    launcher_path_parts = _split_supported_launcher_path_command_parts(text)
    if launcher_path_parts is not None:
        return launcher_path_parts
    try:
        return shlex.split(text, posix=True)
    except ValueError:
        return None


def strip_verify_runner_prefix(parts: Sequence[str]) -> list[str] | None:
    tokens = list(parts)
    if len(tokens) >= 2 and (tokens[0].casefold(), tokens[1].casefold()) in VERIFY_RUNNER_PREFIXES:
        return tokens[2:] or None
    return tokens


def verify_launcher_basename(token: str) -> str:
    normalized = _strip_matching_shell_quotes(str(token).strip()).replace("\\", "/")
    if not normalized:
        return ""
    return normalized.rsplit("/", 1)[-1].casefold()


def normalize_verify_module_invocation(parts: Sequence[str]) -> list[str]:
    tokens = list(parts)
    if len(tokens) < 3 or tokens[1] != "-m":
        return tokens
    launcher = verify_launcher_basename(tokens[0])
    if launcher not in VERIFY_PY_LAUNCHERS and not VERIFY_PYTHON_LAUNCHER_RE.fullmatch(launcher):
        return tokens
    module = tokens[2].casefold()
    if module not in VERIFY_MODULE_NAMES:
        return tokens
    return [module, *tokens[3:]]


def _is_generic_configured_verify_command(command: str) -> bool:
    tokens = split_verify_command_parts(command)
    if not tokens:
        return False
    if any(token in {"||", "&&", ";", "|", "&"} for token in tokens):
        return False
    if tokens[0] == "env" or _looks_like_env_assignment(tokens[0]):
        return False
    lowered = [token.casefold() for token in tokens]
    if _is_generic_pytest_or_ruff_command(lowered):
        return True
    runner_stripped = strip_verify_runner_prefix(lowered)
    if runner_stripped is None:
        return False
    return runner_stripped != lowered and _is_generic_pytest_or_ruff_command(runner_stripped)


def _is_generic_pytest_or_ruff_command(tokens: list[str]) -> bool:
    normalized = normalize_verify_module_invocation(tokens)
    return _is_generic_pytest_command(normalized) or _is_generic_ruff_check_command(normalized)


def _is_generic_pytest_command(tokens: list[str]) -> bool:
    if tokens in (["pytest"], ["pytest", "-q"], ["py.test"], ["py.test", "-q"]):
        return True
    if len(tokens) in {3, 4} and tokens[0] in {"python", "python3"}:
        if tokens[1:3] == ["-m", "pytest"]:
            return len(tokens) == 3 or tokens[3] == "-q"
    return False


def _is_generic_ruff_check_command(tokens: list[str]) -> bool:
    if tokens[:2] != ["ruff", "check"]:
        return False
    targets = tokens[2:]
    if not targets:
        return True
    if len(targets) != 1:
        return False
    return _normalize_generic_ruff_check_target(targets[0]) in {".", "src"}


def _normalize_generic_ruff_check_target(target: str) -> str:
    normalized = target.strip().replace("\\", "/")
    while normalized.endswith("/") and normalized not in {"/", "./"}:
        normalized = normalized[:-1]
    if normalized == "./":
        return "."
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized or "."


def _looks_like_env_assignment(token: str) -> bool:
    if "=" not in token:
        return False
    name, _value = token.split("=", 1)
    if not name:
        return False
    return all(ch == "_" or ch.isalnum() for ch in name)


def _split_supported_launcher_path_command_parts(command: str) -> list[str] | None:
    try:
        raw_parts = shlex.split(command, posix=False)
    except ValueError:
        return None
    if not raw_parts:
        return None

    parts = [_strip_matching_shell_quotes(token) for token in raw_parts]
    candidate = parts
    runner_stripped = strip_verify_runner_prefix(candidate)
    if runner_stripped is None:
        return None
    if runner_stripped != candidate:
        candidate = runner_stripped

    if len(candidate) < 3 or candidate[1] != "-m":
        return None
    launcher = candidate[0]
    if "\\" not in launcher and "/" not in launcher:
        return None

    launcher_basename = verify_launcher_basename(launcher)
    if launcher_basename not in VERIFY_PY_LAUNCHERS and not VERIFY_PYTHON_LAUNCHER_RE.fullmatch(
        launcher_basename
    ):
        return None
    return parts


def _strip_matching_shell_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        return token[1:-1]
    return token


def _config_dir() -> Path:
    override = os.environ.get("SYLLIPTOR_CONFIG_DIR")
    if override:
        return Path(override)
    return canonical_user_config_dir()


def _data_dir() -> Path:
    override = os.environ.get("SYLLIPTOR_DATA_DIR")
    if override:
        return Path(override)
    return canonical_user_data_dir()


def config_path() -> Path:
    return _config_dir() / "config.json"


def credentials_path() -> Path:
    return _config_dir() / "credentials.json"


def default_sessions_dir() -> Path:
    return _data_dir() / "sessions"


def default_chat_history_path() -> Path:
    return _data_dir() / "chat_history.txt"


def load_config() -> AppConfig:
    path = config_path()
    if not path.exists():
        cfg = AppConfig()
        from .profiles import migrate_legacy_to_profiles, sync_active_profile_to_config

        migrate_legacy_to_profiles(cfg)
        sync_active_profile_to_config(cfg)
        return cfg

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 - user-controlled file
        raise ConfigError(f"Failed to read config: {path}") from e

    if not isinstance(raw, dict):
        raise ConfigError(f"Invalid config format (expected JSON object): {path}")

    # Allow unknown keys; stash them so we can round-trip. Map deprecated
    # web_search keys into the simplified auto|off mode when older configs
    # are loaded.
    known = {k: v for k, v in raw.items() if k in AppConfig.model_fields}
    if "web_search_mode" in raw:
        known["web_search_mode"] = _normalize_web_search_mode(
            raw.get("web_search_mode"),
            allow_legacy_on=True,
        )
    elif "web_search_enabled" in raw:
        known["web_search_mode"] = _coerce_legacy_web_search_mode(raw.get("web_search_enabled"))
    if "web_search_adapter" in raw:
        known["web_search_adapter"] = _normalize_web_search_adapter(raw.get("web_search_adapter"))
    if "temperature" in raw:
        legacy_temperature = raw.get("temperature")
        for field in _ROLE_TEMPERATURE_FIELDS.values():
            if field not in raw:
                known[field] = legacy_temperature
    unknown = {
        k: v
        for k, v in raw.items()
        if k not in AppConfig.model_fields and k != "web_search_enabled"
    }
    cfg = AppConfig(**known)
    cfg.extra_fields = unknown
    from .profiles import migrate_legacy_to_profiles, sync_active_profile_to_config

    migrate_legacy_to_profiles(cfg)
    sync_active_profile_to_config(cfg)
    _canonicalize_active_profile_model(cfg)
    return cfg


def save_config(cfg: AppConfig) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = cfg.model_dump()
    # Preserve unknown keys.
    if cfg.extra_fields:
        extra_fields = dict(cfg.extra_fields)
        extra_fields.pop("web_search_enabled", None)
        data.update(extra_fields)

    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonicalize_active_profile_model(cfg: AppConfig) -> None:
    active_profile = str((cfg.extra_fields or {}).get("active_profile") or "").strip()
    model = str(getattr(cfg, "model", "") or "").strip()
    if not active_profile or not model:
        return

    from .profile_presets import find_preset_for_profile, get_preset
    from .profiles import get_profile, update_active_profile_defaults

    profile = get_profile(cfg, active_profile)
    preset = find_preset_for_profile(profile) if profile is not None else None
    preset = preset or get_preset(active_profile)
    canonical = _canonicalize_model_for_config(cfg, model, active_preset=preset)
    if canonical == model:
        return
    cfg.model = canonical
    update_active_profile_defaults(cfg, default_model=canonical)


def _canonicalize_model_for_config(
    cfg: AppConfig,
    model: str,
    *,
    active_preset: Any | None = None,
) -> str:
    raw = str(model or "").strip()
    if not raw:
        return raw

    active_profile = str((cfg.extra_fields or {}).get("active_profile") or "").strip()
    if active_preset is None and active_profile:
        from .profile_presets import find_preset_for_profile, get_preset
        from .profiles import get_profile

        profile = get_profile(cfg, active_profile)
        active_preset = find_preset_for_profile(profile) if profile is not None else None
        active_preset = active_preset or get_preset(active_profile)

    if active_preset is not None:
        active_model = _canonicalize_model_for_preset(raw, active_preset)
        if active_model is not None:
            return active_model
        if _provider_switch_is_unsafe(active_preset):
            return raw

    matches = _matching_model_presets(raw)
    direct_matches = tuple(
        (preset, canonical)
        for preset, canonical in matches
        if not _provider_switch_is_unsafe(preset)
    )
    if len(direct_matches) == 1:
        preset, canonical = direct_matches[0]
        _align_active_profile_to_preset(cfg, preset)
        return canonical
    if len(matches) == 1:
        preset, canonical = matches[0]
        _align_active_profile_to_preset(cfg, preset)
        return canonical
    return raw


def _canonicalize_model_for_preset(model: str, preset: Any) -> str | None:
    lookup = _build_profile_model_lookup_index(preset)
    for alias in _iter_model_lookup_aliases(model):
        canonical = lookup.get(alias.casefold())
        if canonical:
            return canonical
    return None


def _build_profile_model_lookup_index(preset: Any) -> dict[str, str]:
    index: dict[str, str] = {}
    for model in getattr(preset, "suggested_models", ()):
        canonical = str(model or "").strip()
        if not canonical:
            continue
        for alias in _iter_model_lookup_aliases(canonical):
            index.setdefault(alias.casefold(), canonical)
    return index


def _iter_model_lookup_aliases(raw: str) -> tuple[str, ...]:
    value = str(raw or "").strip()
    if not value:
        return ()

    candidates: list[str] = [value]
    if "/" in value:
        provider_stripped = value.rsplit("/", 1)[-1].strip()
        if provider_stripped:
            candidates.append(provider_stripped)

    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized_separators = re.sub(r"[\s_]+", "-", candidate.strip())
        variants = (
            candidate.strip(),
            normalized_separators,
            re.sub(r"(?<=\d)-(?=\d)", ".", normalized_separators),
            re.sub(r"(?<=\d)\.(?=\d)", "-", normalized_separators),
        )
        for variant in variants:
            clean = variant.strip()
            if not clean:
                continue
            folded = clean.casefold()
            if folded in seen:
                continue
            seen.add(folded)
            out.append(clean)
    return tuple(out)


def _matching_model_presets(model: str) -> tuple[tuple[Any, str], ...]:
    from .profile_presets import PROFILE_PRESETS

    matches: list[tuple[Any, str]] = []
    for preset in PROFILE_PRESETS:
        if preset.key == "custom":
            continue
        canonical = _canonicalize_model_for_preset(model, preset)
        if canonical is not None:
            matches.append((preset, canonical))
    return tuple(matches)


def _provider_switch_is_unsafe(preset: Any) -> bool:
    return str(getattr(preset, "key", "") or "").strip().lower() in {"openrouter", "custom"}


def _align_active_profile_to_preset(
    cfg: AppConfig,
    preset: Any,
    *,
    base_url: str | None = None,
    default_model: str | None = None,
) -> None:
    from .profile_presets import make_profile_from_preset
    from .profiles import (
        add_profile,
        set_active_profile,
        update_active_profile_defaults,
    )

    profile_name = _find_profile_name_for_preset(cfg, preset)
    if profile_name is None:
        profile_name = _next_profile_name_for_preset(cfg, preset)
        add_profile(cfg, make_profile_from_preset(preset, name=profile_name))
    set_active_profile(cfg, profile_name)
    update_active_profile_defaults(
        cfg,
        base_url=base_url,
        default_model=default_model,
    )


def _find_profile_name_for_preset(cfg: AppConfig, preset: Any) -> str | None:
    from .profile_presets import find_preset_for_profile
    from .profiles import list_profiles

    active = str((cfg.extra_fields or {}).get("active_profile") or "").strip()
    profiles = list_profiles(cfg)
    for profile in profiles:
        if profile.name != active:
            continue
        matched = find_preset_for_profile(profile)
        if matched is not None and matched.key == preset.key:
            return profile.name
    for profile in profiles:
        matched = find_preset_for_profile(profile)
        if matched is not None and matched.key == preset.key:
            return profile.name
    return None


def _next_profile_name_for_preset(cfg: AppConfig, preset: Any) -> str:
    from .profiles import list_profiles

    existing = {profile.name for profile in list_profiles(cfg)}
    base = str(getattr(preset, "key", "") or "provider").strip().lower()
    if base not in existing:
        return base
    for suffix in range(2, 100):
        candidate = f"{base}-{suffix}"
        if candidate not in existing:
            return candidate
    raise ConfigError(f"Could not allocate a profile name for provider preset {base!r}.")


def _load_credentials_data() -> dict[str, Any]:
    path = credentials_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 - user-controlled file
        raise ConfigError(f"Failed to read persisted API key: {path}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"Invalid persisted API key format (expected JSON object): {path}")
    return dict(raw)


def _save_credentials_data(data: dict[str, Any]) -> None:
    path = credentials_path()
    clean_data = dict(data)
    if not clean_data:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def load_persisted_api_key() -> str | None:
    raw = _load_credentials_data()
    value = str(raw.get("api_key") or "").strip()
    return value or None


def save_persisted_api_key(api_key: str) -> None:
    normalized = str(api_key or "").strip()
    if not normalized:
        raise ConfigError("API key is empty.")
    data = _load_credentials_data()
    data["api_key"] = normalized
    _save_credentials_data(data)


def clear_persisted_api_key() -> bool:
    data = _load_credentials_data()
    if "api_key" not in data:
        return False
    data.pop("api_key", None)
    _save_credentials_data(data)
    return True


def load_persisted_profile_keys() -> dict[str, str]:
    raw = _load_credentials_data().get("profile_keys")
    if not isinstance(raw, dict):
        return {}
    keys: dict[str, str] = {}
    for name, value in raw.items():
        profile_name = str(name or "").strip().lower()
        key = str(value or "").strip()
        if profile_name and key:
            keys[profile_name] = key
    return keys


def save_persisted_profile_key(profile_name: str, value: str) -> None:
    from .profiles import ProfileSpec

    normalized_name = ProfileSpec(name=str(profile_name or "").strip().lower()).name
    normalized_key = str(value or "").strip()
    if not normalized_key:
        raise ConfigError("API key is empty.")
    data = _load_credentials_data()
    profile_keys = data.get("profile_keys")
    if not isinstance(profile_keys, dict):
        profile_keys = {}
    profile_keys[normalized_name] = normalized_key
    data["profile_keys"] = dict(sorted(profile_keys.items()))
    _save_credentials_data(data)


def clear_persisted_profile_key(profile_name: str) -> bool:
    from .profiles import ProfileSpec

    normalized_name = ProfileSpec(name=str(profile_name or "").strip().lower()).name
    data = _load_credentials_data()
    profile_keys = data.get("profile_keys")
    if not isinstance(profile_keys, dict) or normalized_name not in profile_keys:
        return False
    profile_keys.pop(normalized_name, None)
    if profile_keys:
        data["profile_keys"] = dict(sorted(profile_keys.items()))
    else:
        data.pop("profile_keys", None)
    _save_credentials_data(data)
    return True


def rename_persisted_profile_key(old_name: str, new_name: str) -> bool:
    from .profiles import ProfileSpec

    old_profile = ProfileSpec(name=str(old_name or "").strip().lower()).name
    new_profile = ProfileSpec(name=str(new_name or "").strip().lower()).name
    data = _load_credentials_data()
    profile_keys = data.get("profile_keys")
    if not isinstance(profile_keys, dict) or old_profile not in profile_keys:
        return False
    profile_keys[new_profile] = profile_keys.pop(old_profile)
    data["profile_keys"] = dict(sorted(profile_keys.items()))
    _save_credentials_data(data)
    return True


def resolve_profile_api_key(
    cfg: AppConfig,
    profile_name: str,
) -> ApiKeyResolution:
    from .profiles import get_profile

    profile = get_profile(cfg, profile_name)
    if profile is None:
        return ApiKeyResolution(key=None, source="missing")
    stored_profile_key = _resolve_stored_profile_api_key(cfg, profile_name)
    if stored_profile_key.key:
        return stored_profile_key
    env_name = str(profile.api_key_env or "").strip()
    if env_name:
        env_key = str(os.environ.get(env_name) or "").strip()
        if env_key:
            return ApiKeyResolution(key=env_key, source=f"env:{env_name}")
    active_profile = str((cfg.extra_fields or {}).get("active_profile") or "").strip()
    if profile.name == active_profile:
        legacy_key = load_persisted_api_key()
        if legacy_key:
            return ApiKeyResolution(key=legacy_key, source="stored:legacy")
    if profile.name == "openai":
        openai_key = str(os.environ.get("OPENAI_API_KEY") or "").strip()
        if openai_key:
            return ApiKeyResolution(key=openai_key, source="env:OPENAI_API_KEY")
    return ApiKeyResolution(key=None, source="missing")


def _resolve_stored_profile_api_key(cfg: AppConfig, profile_name: str) -> ApiKeyResolution:
    from .profiles import get_profile

    profile = get_profile(cfg, profile_name)
    if profile is None:
        return ApiKeyResolution(key=None, source="missing")
    profile_key = load_persisted_profile_keys().get(profile.name)
    if profile_key:
        return ApiKeyResolution(key=profile_key, source=f"stored:profile={profile.name}")
    return ApiKeyResolution(key=None, source="missing")


def resolve_api_key(
    cfg: AppConfig | None = None,
    *,
    profile_name: str | None = None,
) -> ApiKeyResolution:
    effective_cfg = cfg or load_config()
    if profile_name is None:
        profile_name = str((effective_cfg.extra_fields or {}).get("active_profile") or "").strip()
    if profile_name:
        stored_profile_key = _resolve_stored_profile_api_key(effective_cfg, profile_name)
        if stored_profile_key.key:
            return stored_profile_key
    prefer_profile_scoped = bool(
        profile_name and _should_prefer_profile_scoped_api_key(effective_cfg, profile_name)
    )
    if profile_name and prefer_profile_scoped:
        resolved = resolve_profile_api_key(effective_cfg, profile_name)
        if resolved.key and _is_profile_scoped_api_key_source(resolved.source):
            return resolved
    sylliptor_key = str(env_get("SYLLIPTOR_API_KEY") or "").strip()
    if sylliptor_key and not prefer_profile_scoped:
        return ApiKeyResolution(key=sylliptor_key, source="env:SYLLIPTOR_API_KEY")
    if profile_name:
        resolved = resolve_profile_api_key(effective_cfg, profile_name)
        if resolved.key and (not prefer_profile_scoped or resolved.source != "env:OPENAI_API_KEY"):
            return resolved
    legacy_key = load_persisted_api_key()
    if legacy_key:
        return ApiKeyResolution(key=legacy_key, source="stored:legacy")
    if prefer_profile_scoped:
        return ApiKeyResolution(key=None, source="missing")
    openai_key = str(os.environ.get("OPENAI_API_KEY") or "").strip()
    if openai_key:
        return ApiKeyResolution(key=openai_key, source="env:OPENAI_API_KEY")
    return ApiKeyResolution(key=None, source="missing")


def _should_prefer_profile_scoped_api_key(cfg: AppConfig, profile_name: str) -> bool:
    from .profiles import DEFAULT_OPENAI_BASE_URL, get_profile, resolve_effective_base_url

    profile = get_profile(cfg, profile_name)
    if profile is None:
        return False
    if profile.name not in {"default", "openai"}:
        return True
    effective_base_url = resolve_effective_base_url(cfg=cfg, profile=profile)
    return effective_base_url.rstrip("/") != DEFAULT_OPENAI_BASE_URL.rstrip("/")


def _is_profile_scoped_api_key_source(source: str) -> bool:
    normalized = str(source or "").strip()
    if normalized.startswith("stored:profile="):
        return True
    if not normalized.startswith("env:"):
        return False
    env_name = normalized.removeprefix("env:")
    return env_name not in {"SYLLIPTOR_API_KEY", "OPENAI_API_KEY"}


def get_api_key(cfg: AppConfig | None = None) -> str:
    resolved = resolve_api_key(cfg)
    if not resolved.key:
        raise ConfigError(_missing_api_key_message(cfg or load_config()))
    return resolved.key


def _missing_api_key_message(cfg: AppConfig) -> str:
    suggestions: list[str] = ["set SYLLIPTOR_API_KEY"]
    try:
        from .profiles import (
            DEFAULT_OPENAI_BASE_URL,
            get_active_profile,
            resolve_effective_base_url,
        )

        profile = get_active_profile(cfg)
        effective_base_url = resolve_effective_base_url(cfg=cfg, profile=profile)
    except ConfigError:
        profile = None
        effective_base_url = ""
    is_openai_endpoint = effective_base_url.rstrip("/") == DEFAULT_OPENAI_BASE_URL.rstrip("/")
    if profile is not None:
        if profile.api_key_env and (profile.api_key_env != "OPENAI_API_KEY" or is_openai_endpoint):
            suggestions.insert(0, f"set {profile.api_key_env}")
        suggestions.append(f"run `sylliptor profile set-key {profile.name} --key <key>`")
        if is_openai_endpoint:
            suggestions.append("set OPENAI_API_KEY")
    suggestions.append("run `sylliptor config set-api-key`")
    sentences = [suggestion[0].upper() + suggestion[1:] for suggestion in suggestions]
    return "Missing API key. " + "; or ".join(sentences) + "."


_SETTABLE_KEYS: set[str] = {
    "base_url",
    "model",
    "llm_timeout_s",
    "llm_enable_thinking",
    "llm_reasoning_effort",
    "provider_concurrency_caps",
    "provider_retry_max_retries",
    "provider_retry_base_delay_seconds",
    "provider_retry_max_delay_seconds",
    "model_metadata_policy",
    "default_mode",
    "max_steps",
    "temperature",
    "coding_temperature",
    "review_temperature",
    "planner_temperature",
    "conflict_review_temperature",
    "compactor_temperature",
    "chat_temperature",
    "stream",
    "routing_mode",
    "step_budget_policy",
    "subagents_enabled",
    "skills_enabled",
    "skills_auto_invoke",
    "custom_tools_enabled",
    "task_max_steps",
    "subagent_max_steps",
    "web_search_mode",
    "web_search_enabled",
    "web_search_adapter",
    "web_search_base_url",
    "web_search_model",
    "web_search_timeout_s",
    "update_check_enabled",
    "update_check_interval_hours",
    "update_check_timeout_s",
    "feedback_github_enabled",
    "feedback_github_repo",
    "feedback_open_browser",
    "session_log_dir",
    "prompt_cache_key",
    "prompt_cache_retention",
    "verify_commands",
    "integration_verify_mode",
    "integration_verify_commands",
    "replanning_mode",
    "toolbar_items",
    "assets.enabled",
    "assets.comprehension.role",
    "assets.comprehension.vision_fallback_profile",
    "assets.comprehension.vision_with_ocr_when_available",
    "assets.comprehension.ocr_enabled",
    "assets.comprehension.ocr_provider",
    "assets.comprehension.ocr_timeout_seconds",
    "assets.comprehension.image_max_edge_pixels",
    "assets.comprehension.questioning_mode",
    "assets.comprehension.schema_version",
    "assets.planner.inline_images",
    "assets.planner.max_inline_images",
    "assets.planner.readiness_policy",
    "assets.planner.readiness_timeout_seconds",
    "assets.planner.max_chars_per_asset",
    "assets.planner.max_primary_per_task",
    "assets.worker.inline_images",
    "assets.worker.max_inline_images",
    "assets.worker.fail_on_mirror_error",
    "assets.worker.allocator_role",
    "assets.worker.allocator_timeout_seconds",
    "assets.worker.max_chars_per_asset_block",
    "assets.worker.max_focused_extract_chars",
    "assets.worker.schema_version",
}

_VERIFY_LIKE_MODES: set[str] = {"off", "warn", "strict"}
_MODEL_METADATA_POLICIES: set[str] = {"warn", "strict"}


def _parse_command_list(value: str, *, key: str, allow_empty: bool) -> list[str]:
    raw = value.strip()
    if not raw:
        raise ConfigError(f"{key} must be a JSON array of command strings")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ConfigError(f"{key} must be valid JSON array") from e
    if not isinstance(parsed, list):
        raise ConfigError(f"{key} must be a JSON array")
    commands: list[str] = []
    for item in parsed:
        cmd = str(item).strip()
        if cmd:
            commands.append(cmd)
    if not commands and not allow_empty:
        raise ConfigError(f"{key} cannot be empty")
    return commands


def _parse_provider_concurrency_caps(value: str, *, key: str) -> dict[str, int]:
    raw = value.strip()
    if not raw:
        raise ConfigError(f"{key} must be a JSON object mapping provider keys to integer caps")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ConfigError(f"{key} must be valid JSON object") from e
    if not isinstance(parsed, dict):
        raise ConfigError(f"{key} must be a JSON object")

    caps: dict[str, int] = {}
    for raw_provider, raw_cap in parsed.items():
        provider = str(raw_provider or "").strip().lower()
        if not provider:
            raise ConfigError(f"{key} provider keys must be non-empty strings")
        try:
            cap = int(raw_cap if raw_cap is not None else 0)
        except (TypeError, ValueError) as e:
            raise ConfigError(f"{key}.{provider} must be an integer >= 0") from e
        if cap < 0:
            raise ConfigError(f"{key}.{provider} must be an integer >= 0")
        caps[provider] = cap
    return caps


def _coerce_non_negative_float(value: str, *, key: str) -> float:
    try:
        parsed = float(value)
    except ValueError as e:
        raise ConfigError(f"{key} must be a number") from e
    if parsed < 0:
        raise ConfigError(f"{key} must be >= 0")
    return parsed


def _coerce_positive_float(value: str, *, key: str) -> float:
    parsed = _coerce_non_negative_float(value, key=key)
    if parsed <= 0 or not math.isfinite(parsed):
        raise ConfigError(f"{key} must be > 0")
    return parsed


def _coerce_optional_bool(value: str, *, key: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"", "auto", "default", "none", "null"}:
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{key} must be true/false or auto")


def _coerce_reasoning_effort(value: Any, *, key: str) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "auto", "default"}:
        return None
    if normalized in _REASONING_EFFORTS:
        return normalized
    allowed = ", ".join(sorted((*_REASONING_EFFORTS, "auto")))
    raise ConfigError(f"{key} must be one of: {allowed}")


def _coerce_bool(value: str, *, key: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{key} must be true/false")


def _coerce_github_repo(value: str, *, key: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ConfigError(f"{key} must be a GitHub repo in owner/name form")
    if raw.startswith("https://") or raw.startswith("http://"):
        try:
            parsed = urlsplit(raw)
        except ValueError as e:
            raise ConfigError(f"{key} must be a GitHub repo or GitHub URL") from e
        hostname = (parsed.hostname or "").rstrip(".").lower()
        if hostname != "github.com":
            raise ConfigError(f"{key} URL must use github.com")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ConfigError(f"{key} GitHub URL must include owner and repo")
        raw = f"{parts[0]}/{parts[1]}"
    raw = raw.removesuffix(".git")
    if not _GITHUB_REPO_RE.fullmatch(raw):
        raise ConfigError(f"{key} must be a GitHub repo in owner/name form")
    return raw


def _coerce_positive_int(value: str, *, key: str) -> int:
    try:
        parsed = int(value)
    except ValueError as e:
        raise ConfigError(f"{key} must be an integer") from e
    if parsed <= 0:
        raise ConfigError(f"{key} must be > 0")
    return parsed


def _coerce_non_negative_int(value: str, *, key: str) -> int:
    try:
        parsed = int(value)
    except ValueError as e:
        raise ConfigError(f"{key} must be an integer") from e
    if parsed < 0:
        raise ConfigError(f"{key} must be >= 0")
    return parsed


def _resolve_positive_timeout(raw: Any) -> float | None:
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return None
    if parsed <= 0 or not math.isfinite(parsed):
        return None
    return parsed


def _is_dashscope_base_url(base_url: str | None) -> bool:
    normalized = str(base_url or "").strip()
    if not normalized:
        return False
    try:
        parsed = urlsplit(normalized)
    except ValueError:
        return False
    if parsed.scheme.lower() != "https":
        return False
    hostname = (parsed.hostname or "").rstrip(".").lower()
    return (
        hostname == "dashscope.aliyuncs.com"
        or hostname == "dashscope-intl.aliyuncs.com"
        or hostname == "dashscope-us.aliyuncs.com"
        or hostname.endswith(".dashscope.aliyuncs.com")
        or hostname.endswith(".dashscope-intl.aliyuncs.com")
        or hostname.endswith(".dashscope-us.aliyuncs.com")
        or (hostname.endswith(".aliyuncs.com") and ".dashscope-" in f".{hostname}")
    )


def _is_qwen_model(model: str | None) -> bool:
    normalized = str(model or "").strip().lower()
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[-1]
    return normalized.startswith("qwen")


def _normalize_web_search_mode(raw: Any, *, allow_legacy_on: bool = False) -> str:
    value = str(raw or "").strip().lower()
    if allow_legacy_on and value == "on":
        return "auto"
    if value in _WEB_SEARCH_MODES:
        return value
    allowed = ", ".join(sorted(_WEB_SEARCH_MODES))
    raise ConfigError(f"web_search_mode must be one of: {allowed}")


def _normalize_web_search_adapter(raw: Any) -> str:
    try:
        return normalize_web_search_adapter(raw)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _coerce_legacy_web_search_mode(raw: Any) -> str:
    if isinstance(raw, bool):
        return "auto" if raw else "off"
    value = str(raw or "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return "auto"
    if value in {"0", "false", "no", "off"}:
        return "off"
    raise ConfigError("web_search_enabled must be true/false")


def _apply_legacy_temperature_override(cfg: AppConfig, temperature: float) -> None:
    cfg.temperature = temperature
    cfg.coding_temperature = temperature
    cfg.review_temperature = temperature
    cfg.planner_temperature = temperature
    cfg.conflict_review_temperature = temperature
    cfg.compactor_temperature = temperature
    cfg.chat_temperature = temperature


def resolve_llm_timeout_s(cfg: AppConfig | None) -> float:
    env_timeout = _resolve_positive_timeout(env_get("SYLLIPTOR_LLM_TIMEOUT_S"))
    if env_timeout is not None:
        return env_timeout
    cfg_timeout = _resolve_positive_timeout(getattr(cfg, "llm_timeout_s", None))
    if cfg_timeout is not None:
        return cfg_timeout
    return 60.0


def resolve_llm_enable_thinking(cfg: AppConfig | None) -> bool | None:
    env_value = env_get("SYLLIPTOR_LLM_ENABLE_THINKING")
    if env_value is not None:
        return _coerce_optional_bool(str(env_value), key="SYLLIPTOR_LLM_ENABLE_THINKING")

    cfg_value = getattr(cfg, "llm_enable_thinking", None)
    if isinstance(cfg_value, bool):
        return cfg_value
    if cfg_value is not None:
        return _coerce_optional_bool(str(cfg_value), key="llm_enable_thinking")

    if cfg is not None and _is_dashscope_base_url(getattr(cfg, "base_url", None)):
        if _is_qwen_model(getattr(cfg, "model", None)):
            return False
    return None


def _legacy_reasoning_effort_hint(cfg: AppConfig | None) -> str | None:
    if cfg is None:
        return None
    extra_fields = getattr(cfg, "extra_fields", None)
    if not isinstance(extra_fields, dict):
        return None
    try:
        return _coerce_reasoning_effort(
            extra_fields.get("llm_thinking_label"),
            key="llm_thinking_label",
        )
    except ConfigError:
        return None


def resolve_llm_reasoning_effort(cfg: AppConfig | None) -> str | None:
    env_value = env_get("SYLLIPTOR_LLM_REASONING_EFFORT")
    if env_value is not None:
        return _coerce_reasoning_effort(env_value, key="SYLLIPTOR_LLM_REASONING_EFFORT")

    cfg_value = getattr(cfg, "llm_reasoning_effort", None)
    if cfg_value is not None:
        return _coerce_reasoning_effort(cfg_value, key="llm_reasoning_effort")

    return _legacy_reasoning_effort_hint(cfg)


def resolve_web_search_mode(cfg: AppConfig | None) -> str:
    if cfg is None:
        return "auto"
    return _normalize_web_search_mode(
        getattr(cfg, "web_search_mode", "auto"),
        allow_legacy_on=True,
    )


def resolve_web_search_enabled(cfg: AppConfig | None) -> bool:
    return resolve_web_search_mode(cfg) != "off"


def resolve_web_search_adapter(cfg: AppConfig | None) -> str:
    env_adapter = str(env_get("SYLLIPTOR_WEB_SEARCH_ADAPTER") or "").strip()
    if env_adapter:
        return _normalize_web_search_adapter(env_adapter)

    cfg_adapter = _normalize_web_search_adapter(getattr(cfg, "web_search_adapter", "auto"))
    if cfg_adapter != "auto":
        return cfg_adapter

    if cfg is not None:
        try:
            from .profiles import get_active_profile

            profile = get_active_profile(cfg)
        except Exception:
            profile = None
        if profile is not None:
            profile_adapter = _normalize_web_search_adapter(profile.web_search_adapter)
            if profile_adapter != "auto":
                return profile_adapter

    return "auto"


def resolve_web_search_api_key(
    cfg: AppConfig | None,
    *,
    api_key_fallback: str | None = None,
) -> str | None:
    _ = cfg
    env_key = str(env_get("SYLLIPTOR_WEB_SEARCH_API_KEY") or "").strip()
    if env_key:
        return env_key
    fallback = str(api_key_fallback or "").strip()
    return fallback or None


def resolve_web_search_explicit_base_url(cfg: AppConfig | None) -> str | None:
    env_base_url = str(env_get("SYLLIPTOR_WEB_SEARCH_BASE_URL") or "").strip()
    if env_base_url:
        return env_base_url.rstrip("/")

    cfg_base_url = str(getattr(cfg, "web_search_base_url", "") or "").strip()
    if cfg_base_url:
        return cfg_base_url.rstrip("/")
    return None


def resolve_web_search_base_url(cfg: AppConfig | None) -> str | None:
    explicit_base_url = resolve_web_search_explicit_base_url(cfg)
    if explicit_base_url:
        return explicit_base_url

    fallback_base_url = str(getattr(cfg, "base_url", "") or "").strip()
    if cfg is not None and not fallback_base_url:
        return None

    if cfg is not None:
        try:
            from .profiles import DEFAULT_OPENAI_BASE_URL, get_active_profile

            profile = get_active_profile(cfg)
        except Exception:
            profile = None
        if profile is not None:
            profile_base_url = str(profile.base_url or "").strip().rstrip("/")
            cfg_base_url = str(getattr(cfg, "base_url", "") or "").strip().rstrip("/")
            default_base_url = DEFAULT_OPENAI_BASE_URL.rstrip("/")
            if cfg_base_url and cfg_base_url not in {profile_base_url, default_base_url}:
                return cfg_base_url
            if profile_base_url:
                return profile_base_url

    if fallback_base_url:
        return fallback_base_url.rstrip("/")
    return None


def is_first_party_openai_base_url(base_url: str | None) -> bool:
    normalized = str(base_url or "").strip()
    if not normalized:
        return False
    try:
        parsed = urlsplit(normalized)
    except ValueError:
        return False
    if parsed.scheme.lower() != "https":
        return False
    hostname = (parsed.hostname or "").rstrip(".").lower()
    return hostname == "api.openai.com"


def resolve_web_search_model(cfg: AppConfig | None) -> str | None:
    env_model = str(env_get("SYLLIPTOR_WEB_SEARCH_MODEL") or "").strip()
    if env_model:
        return env_model
    cfg_model = str(getattr(cfg, "web_search_model", "") or "").strip()
    if cfg_model:
        return cfg_model
    if cfg is not None:
        try:
            from .profiles import get_active_profile

            profile = get_active_profile(cfg)
        except Exception:
            profile = None
        if profile is not None:
            profile_model = str(profile.web_search_model or "").strip()
            if profile_model:
                return profile_model
    fallback_model = str(getattr(cfg, "model", "") or "").strip()
    return fallback_model or None


def resolve_web_search_timeout_s(cfg: AppConfig | None) -> float:
    env_timeout_raw = env_get("SYLLIPTOR_WEB_SEARCH_TIMEOUT_S")
    if env_timeout_raw is not None:
        env_timeout = _resolve_positive_timeout(env_timeout_raw)
        if env_timeout is None:
            raise ConfigError("SYLLIPTOR_WEB_SEARCH_TIMEOUT_S must be > 0")
        return env_timeout

    cfg_timeout = _resolve_positive_timeout(getattr(cfg, "web_search_timeout_s", None))
    if cfg_timeout is None:
        raise ConfigError("web_search_timeout_s must be > 0")
    return cfg_timeout


def resolve_feedback_github_enabled(cfg: AppConfig | None) -> bool:
    env_value = env_get("SYLLIPTOR_FEEDBACK_GITHUB_ENABLED")
    if env_value is not None:
        return _coerce_bool(str(env_value), key="SYLLIPTOR_FEEDBACK_GITHUB_ENABLED")
    return bool(getattr(cfg, "feedback_github_enabled", True))


def resolve_feedback_github_repo(cfg: AppConfig | None) -> str:
    env_value = env_get("SYLLIPTOR_FEEDBACK_GITHUB_REPO")
    if env_value is not None:
        return _coerce_github_repo(str(env_value), key="SYLLIPTOR_FEEDBACK_GITHUB_REPO")
    raw = getattr(cfg, "feedback_github_repo", DEFAULT_FEEDBACK_GITHUB_REPO)
    return _coerce_github_repo(str(raw), key="feedback_github_repo")


def resolve_feedback_open_browser(cfg: AppConfig | None) -> bool:
    env_value = env_get("SYLLIPTOR_FEEDBACK_OPEN_BROWSER")
    if env_value is not None:
        return _coerce_bool(str(env_value), key="SYLLIPTOR_FEEDBACK_OPEN_BROWSER")
    return bool(getattr(cfg, "feedback_open_browser", True))


def resolve_model_metadata_policy(cfg: AppConfig | None) -> str:
    env_policy = str(env_get("SYLLIPTOR_MODEL_METADATA_POLICY") or "").strip().lower()
    if env_policy:
        if env_policy not in _MODEL_METADATA_POLICIES:
            allowed = ", ".join(sorted(_MODEL_METADATA_POLICIES))
            raise ConfigError(f"SYLLIPTOR_MODEL_METADATA_POLICY must be one of: {allowed}")
        return env_policy

    cfg_policy = str(getattr(cfg, "model_metadata_policy", "warn") or "").strip().lower()
    if not cfg_policy:
        return "warn"
    if cfg_policy not in _MODEL_METADATA_POLICIES:
        allowed = ", ".join(sorted(_MODEL_METADATA_POLICIES))
        raise ConfigError(f"model_metadata_policy must be one of: {allowed}")
    return cfg_policy


def resolve_prompt_cache_key(cfg: AppConfig | None) -> str | None:
    env_key = str(env_get("SYLLIPTOR_PROMPT_CACHE_KEY") or "").strip()
    if env_key:
        return env_key
    cfg_key = str(getattr(cfg, "prompt_cache_key", "") or "").strip()
    return cfg_key or None


def resolve_prompt_cache_retention(cfg: AppConfig | None) -> str | None:
    env_retention = str(env_get("SYLLIPTOR_PROMPT_CACHE_RETENTION") or "").strip()
    if env_retention:
        return env_retention
    cfg_retention = str(getattr(cfg, "prompt_cache_retention", "") or "").strip()
    return cfg_retention or None


def resolve_role_temperature(cfg: AppConfig, *, role: str) -> float:
    default = _ROLE_TEMPERATURE_DEFAULTS.get(role, 0.2)
    legacy_raw = getattr(cfg, "temperature", default)
    try:
        legacy_temperature = float(legacy_raw)
    except (TypeError, ValueError):
        legacy_temperature = default
    if legacy_temperature < 0:
        legacy_temperature = default

    field = _ROLE_TEMPERATURE_FIELDS.get(role)
    if field is None:
        return legacy_temperature

    role_raw = getattr(cfg, field, legacy_temperature)
    try:
        role_temperature = float(role_raw)
    except (TypeError, ValueError):
        return legacy_temperature
    if role_temperature < 0:
        return legacy_temperature
    return role_temperature


def set_config_value(cfg: AppConfig, key: str, value: str) -> AppConfig:
    if key not in _SETTABLE_KEYS:
        raise ConfigError(
            f"Unknown/unsupported key: {key}. Supported keys: {', '.join(sorted(_SETTABLE_KEYS))}"
        )

    if key == "base_url":
        from .profiles import update_active_profile_defaults, validate_base_url

        normalized = validate_base_url(value, key="base_url", allow_empty=True)
        if normalized:
            from .profile_presets import find_preset_for_base_url

            preset = find_preset_for_base_url(normalized)
            if preset is not None:
                current_model = str(getattr(cfg, "model", "") or "").strip()
                default_model = _canonicalize_model_for_preset(current_model, preset)
                if default_model is None and preset.suggested_models:
                    default_model = str(preset.suggested_models[0] or "").strip()
                _align_active_profile_to_preset(
                    cfg,
                    preset,
                    base_url=normalized,
                    default_model=default_model,
                )
                return cfg
        cfg.base_url = normalized
        update_active_profile_defaults(cfg, base_url=normalized)
        return cfg

    if key == "model":
        from .profiles import update_active_profile_defaults

        normalized = _canonicalize_model_for_config(cfg, str(value or "").strip())
        cfg.model = normalized
        update_active_profile_defaults(cfg, default_model=normalized)
        return cfg

    if key == "llm_timeout_s":
        cfg.llm_timeout_s = _coerce_positive_float(value, key=key)
        return cfg

    if key == "llm_enable_thinking":
        cfg.llm_enable_thinking = _coerce_optional_bool(value, key=key)
        return cfg

    if key == "llm_reasoning_effort":
        cfg.llm_reasoning_effort = _coerce_reasoning_effort(value, key=key)
        return cfg

    if key == "provider_concurrency_caps":
        cfg.provider_concurrency_caps = _parse_provider_concurrency_caps(value, key=key)
        return cfg

    if key == "provider_retry_max_retries":
        cfg.provider_retry_max_retries = _coerce_non_negative_int(value, key=key)
        return cfg

    if key in {
        "provider_retry_base_delay_seconds",
        "provider_retry_max_delay_seconds",
    }:
        setattr(cfg, key, _coerce_positive_float(value, key=key))
        return cfg

    if key == "model_metadata_policy":
        normalized = value.strip().lower()
        if normalized not in _MODEL_METADATA_POLICIES:
            raise ConfigError("model_metadata_policy must be one of: strict, warn")
        cfg.model_metadata_policy = normalized
        return cfg

    if key == "default_mode":
        if value not in {"review", "auto", "readonly", "fullaccess"}:
            raise ConfigError("default_mode must be one of: review, auto, readonly, fullaccess")
        cfg.default_mode = value
        return cfg

    if key in {"max_steps", "task_max_steps", "subagent_max_steps"}:
        setattr(cfg, key, _coerce_positive_int(value, key=key))
        return cfg

    if key == "temperature":
        _apply_legacy_temperature_override(cfg, _coerce_non_negative_float(value, key=key))
        return cfg

    if key in _ROLE_TEMPERATURE_FIELDS.values():
        setattr(cfg, key, _coerce_non_negative_float(value, key=key))
        return cfg

    if key == "stream":
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            cfg.stream = True
            return cfg
        if v in {"0", "false", "no", "off"}:
            cfg.stream = False
            return cfg
        raise ConfigError("stream must be true/false")

    if key == "routing_mode":
        v = value.strip().lower()
        if v not in {"auto", "code_only"}:
            raise ConfigError("routing_mode must be one of: auto, code_only")
        cfg.routing_mode = v
        return cfg

    if key == "step_budget_policy":
        normalized = value.strip().lower()
        if normalized not in {"adaptive", "fixed"}:
            raise ConfigError("step_budget_policy must be one of: adaptive, fixed")
        cfg.step_budget_policy = normalized
        return cfg

    if key == "subagents_enabled":
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            cfg.subagents_enabled = True
            return cfg
        if v in {"0", "false", "no", "off"}:
            cfg.subagents_enabled = False
            return cfg
        raise ConfigError("subagents_enabled must be true/false")

    if key == "skills_enabled":
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            cfg.skills_enabled = True
            return cfg
        if v in {"0", "false", "no", "off"}:
            cfg.skills_enabled = False
            return cfg
        raise ConfigError("skills_enabled must be true/false")

    if key == "skills_auto_invoke":
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            cfg.skills_auto_invoke = True
            return cfg
        if v in {"0", "false", "no", "off"}:
            cfg.skills_auto_invoke = False
            return cfg
        raise ConfigError("skills_auto_invoke must be true/false")

    if key == "custom_tools_enabled":
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            cfg.custom_tools_enabled = True
            return cfg
        if v in {"0", "false", "no", "off"}:
            cfg.custom_tools_enabled = False
            return cfg
        raise ConfigError("custom_tools_enabled must be true/false")

    if key == "web_search_mode":
        cfg.web_search_mode = _normalize_web_search_mode(value)
        return cfg

    if key == "web_search_enabled":
        cfg.web_search_mode = _coerce_legacy_web_search_mode(value)
        return cfg

    if key == "web_search_adapter":
        cfg.web_search_adapter = _normalize_web_search_adapter(value)
        return cfg

    if key == "web_search_base_url":
        cfg.web_search_base_url = value.strip() or None
        return cfg

    if key == "web_search_model":
        cfg.web_search_model = value.strip() or None
        return cfg

    if key == "web_search_timeout_s":
        cfg.web_search_timeout_s = _coerce_positive_float(value, key=key)
        return cfg

    if key == "update_check_enabled":
        cfg.update_check_enabled = _coerce_bool(value, key=key)
        return cfg

    if key == "update_check_interval_hours":
        cfg.update_check_interval_hours = _coerce_positive_int(value, key=key)
        return cfg

    if key == "update_check_timeout_s":
        cfg.update_check_timeout_s = _coerce_positive_float(value, key=key)
        return cfg

    if key == "feedback_github_enabled":
        cfg.feedback_github_enabled = _coerce_bool(value, key=key)
        return cfg

    if key == "feedback_github_repo":
        cfg.feedback_github_repo = _coerce_github_repo(value, key=key)
        return cfg

    if key == "feedback_open_browser":
        cfg.feedback_open_browser = _coerce_bool(value, key=key)
        return cfg

    if key == "session_log_dir":
        cfg.session_log_dir = value if value.strip() else None
        return cfg

    if key == "prompt_cache_key":
        cfg.prompt_cache_key = value.strip() or None
        return cfg

    if key == "prompt_cache_retention":
        cfg.prompt_cache_retention = value.strip() or None
        return cfg

    if key == "verify_commands":
        cfg.verify_commands = _parse_command_list(
            value,
            key="verify_commands",
            allow_empty=False,
        )
        return cfg

    if key == "integration_verify_mode":
        normalized = value.strip().lower()
        if normalized not in _VERIFY_LIKE_MODES:
            raise ConfigError("integration_verify_mode must be one of: off, warn, strict")
        cfg.integration_verify_mode = normalized
        return cfg

    if key == "integration_verify_commands":
        cfg.integration_verify_commands = _parse_command_list(
            value,
            key="integration_verify_commands",
            allow_empty=True,
        )
        return cfg

    if key == "replanning_mode":
        normalized = value.strip().lower()
        if normalized not in {"off", "suggest", "apply"}:
            raise ConfigError("replanning_mode must be one of: off, suggest, apply")
        cfg.replanning_mode = normalized
        return cfg

    if key == "assets.enabled":
        cfg.assets.enabled = _coerce_bool(value, key=key)
        return cfg

    if key == "assets.comprehension.role":
        role = value.strip().lower()
        if not role:
            raise ConfigError("assets.comprehension.role must be non-empty")
        cfg.assets.comprehension.role = role
        return cfg

    if key == "assets.comprehension.vision_fallback_profile":
        cfg.assets.comprehension.vision_fallback_profile = value.strip().lower() or None
        return cfg

    if key == "assets.comprehension.vision_with_ocr_when_available":
        cfg.assets.comprehension.vision_with_ocr_when_available = _coerce_bool(value, key=key)
        return cfg

    if key == "assets.comprehension.ocr_enabled":
        normalized = value.strip().lower()
        if normalized not in {"auto", "always", "never"}:
            raise ConfigError(
                "assets.comprehension.ocr_enabled must be one of: auto, always, never"
            )
        cfg.assets.comprehension.ocr_enabled = normalized  # type: ignore[assignment]
        return cfg

    if key == "assets.comprehension.ocr_provider":
        provider = value.strip().lower()
        if not provider:
            raise ConfigError("assets.comprehension.ocr_provider must be non-empty")
        cfg.assets.comprehension.ocr_provider = provider
        return cfg

    if key == "assets.comprehension.ocr_timeout_seconds":
        cfg.assets.comprehension.ocr_timeout_seconds = _coerce_positive_int(value, key=key)
        return cfg

    if key == "assets.comprehension.image_max_edge_pixels":
        cfg.assets.comprehension.image_max_edge_pixels = _coerce_positive_int(value, key=key)
        return cfg

    if key == "assets.comprehension.questioning_mode":
        normalized = value.strip().lower()
        if normalized not in {"assertive", "balanced", "assumption_friendly"}:
            raise ConfigError(
                "assets.comprehension.questioning_mode must be one of: "
                "assertive, balanced, assumption_friendly"
            )
        cfg.assets.comprehension.questioning_mode = normalized  # type: ignore[assignment]
        return cfg

    if key == "assets.comprehension.schema_version":
        cfg.assets.comprehension.schema_version = _coerce_positive_int(value, key=key)
        return cfg

    if key == "assets.planner.inline_images":
        cfg.assets.planner.inline_images = _coerce_bool(value, key=key)
        return cfg

    if key == "assets.planner.max_inline_images":
        cfg.assets.planner.max_inline_images = _coerce_positive_int(value, key=key)
        return cfg

    if key == "assets.planner.readiness_policy":
        normalized = value.strip().lower()
        if normalized not in {"soft", "block", "partial"}:
            raise ConfigError(
                "assets.planner.readiness_policy must be one of: soft, block, partial"
            )
        cfg.assets.planner.readiness_policy = normalized  # type: ignore[assignment]
        return cfg

    if key == "assets.planner.readiness_timeout_seconds":
        cfg.assets.planner.readiness_timeout_seconds = _coerce_positive_float(value, key=key)
        return cfg

    if key == "assets.planner.max_chars_per_asset":
        cfg.assets.planner.max_chars_per_asset = _coerce_positive_int(value, key=key)
        return cfg

    if key == "assets.planner.max_primary_per_task":
        cfg.assets.planner.max_primary_per_task = _coerce_positive_int(value, key=key)
        return cfg

    if key == "assets.worker.inline_images":
        cfg.assets.worker.inline_images = _coerce_bool(value, key=key)
        return cfg

    if key == "assets.worker.max_inline_images":
        cfg.assets.worker.max_inline_images = _coerce_positive_int(value, key=key)
        return cfg

    if key == "assets.worker.fail_on_mirror_error":
        cfg.assets.worker.fail_on_mirror_error = _coerce_bool(value, key=key)
        return cfg

    if key == "assets.worker.allocator_role":
        role = value.strip().lower()
        if not role:
            raise ConfigError("assets.worker.allocator_role must be non-empty")
        cfg.assets.worker.allocator_role = role
        return cfg

    if key == "assets.worker.allocator_timeout_seconds":
        cfg.assets.worker.allocator_timeout_seconds = _coerce_positive_int(value, key=key)
        return cfg

    if key == "assets.worker.max_chars_per_asset_block":
        cfg.assets.worker.max_chars_per_asset_block = _coerce_positive_int(value, key=key)
        return cfg

    if key == "assets.worker.max_focused_extract_chars":
        cfg.assets.worker.max_focused_extract_chars = _coerce_positive_int(value, key=key)
        return cfg

    if key == "assets.worker.schema_version":
        cfg.assets.worker.schema_version = _coerce_positive_int(value, key=key)
        return cfg

    if key == "toolbar_items":
        raw = value.strip()
        if not raw:
            raise ConfigError("toolbar_items must be a JSON array of toolbar item names")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ConfigError("toolbar_items must be valid JSON array") from e
        if not isinstance(parsed, list):
            raise ConfigError("toolbar_items must be a JSON array")
        items: list[str] = []
        seen: set[str] = set()
        valid_items = ", ".join(sorted(_VALID_TOOLBAR_ITEMS))
        for item in parsed:
            if not isinstance(item, str):
                raise ConfigError("toolbar_items must be a JSON array of strings")
            name = item.strip().lower()
            if not name:
                raise ConfigError("toolbar_items cannot contain empty values")
            if name not in _VALID_TOOLBAR_ITEMS:
                raise ConfigError(f"Unknown toolbar item: {name}. Valid items: {valid_items}")
            if name in seen:
                continue
            seen.add(name)
            items.append(name)
        cfg.toolbar_items = items
        return cfg

    raise ConfigError(f"Unhandled key: {key}")
