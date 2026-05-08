from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from .web_search_adapters import normalize_web_search_adapter

if TYPE_CHECKING:
    from .config import AppConfig

SUPPORTED_PROTOCOLS: frozenset[str] = frozenset({"openai_compat"})
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_PROFILE_NAME_RE = re.compile(r"^[a-z0-9_-]+$")


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    protocol: str = "openai_compat"
    base_url: str = ""
    api_key_env: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    default_model: str = ""
    web_search_adapter: str = "auto"
    web_search_model: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        _validate_profile_name(self.name)
        _validate_protocol(self.protocol)
        _validate_web_search_adapter(self.web_search_adapter)
        if self.base_url:
            _validate_base_url(self.base_url)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "protocol": self.protocol,
            "base_url": self.base_url,
            "extra_headers": dict(sorted(self.extra_headers.items())),
            "default_model": self.default_model,
            "web_search_adapter": normalize_web_search_adapter(self.web_search_adapter),
            "web_search_model": self.web_search_model,
            "notes": self.notes,
        }
        if self.api_key_env:
            data["api_key_env"] = self.api_key_env
        return data

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> ProfileSpec:
        if not isinstance(data, dict):
            _raise_config_error(f"Profile {name!r} must be a JSON object.")
        return cls(
            name=name,
            protocol=str(data.get("protocol") or "openai_compat").strip(),
            base_url=str(data.get("base_url") or "").strip(),
            api_key_env=_optional_string(data.get("api_key_env")),
            extra_headers=_coerce_headers(data.get("extra_headers")),
            default_model=str(data.get("default_model") or "").strip(),
            web_search_adapter=str(data.get("web_search_adapter") or "auto").strip(),
            web_search_model=str(data.get("web_search_model") or "").strip(),
            notes=str(data.get("notes") or ""),
        )


def list_profiles(cfg: AppConfig) -> list[ProfileSpec]:
    migrate_legacy_to_profiles(cfg)
    profiles = _profile_dict(cfg)
    return [ProfileSpec.from_dict(name, profiles[name]) for name in sorted(profiles)]


def get_profile(cfg: AppConfig, name: str) -> ProfileSpec | None:
    migrate_legacy_to_profiles(cfg)
    profile_name = _normalize_profile_name(name)
    profiles = _profile_dict(cfg)
    data = profiles.get(profile_name)
    if not isinstance(data, dict):
        return None
    return ProfileSpec.from_dict(profile_name, data)


def get_active_profile(cfg: AppConfig) -> ProfileSpec:
    migrate_legacy_to_profiles(cfg)
    active = str((cfg.extra_fields or {}).get("active_profile") or "").strip()
    if not active:
        _raise_config_error("No active profile configured.")
    profile = get_profile(cfg, active)
    if profile is None:
        _raise_config_error(f"Active profile not found: {active}")
    return profile


def sync_active_profile_to_config(cfg: AppConfig) -> bool:
    """Mirror the active profile connection defaults onto legacy top-level fields."""
    active = str((cfg.extra_fields or {}).get("active_profile") or "").strip()
    if not active:
        return False
    profile = get_profile(cfg, active)
    if profile is None:
        return False

    changed = False
    if profile.base_url and not _same_base_url(getattr(cfg, "base_url", ""), profile.base_url):
        cfg.base_url = profile.base_url
        changed = True
    if profile.default_model and str(getattr(cfg, "model", "") or "") != profile.default_model:
        cfg.model = profile.default_model
        changed = True
    return changed


def add_profile(cfg: AppConfig, profile: ProfileSpec) -> None:
    migrate_legacy_to_profiles(cfg)
    profiles = dict(_profile_dict(cfg))
    profiles[profile.name] = profile.to_dict()
    _set_profile_dict(cfg, profiles)


def remove_profile(cfg: AppConfig, name: str) -> None:
    migrate_legacy_to_profiles(cfg)
    profile_name = _normalize_profile_name(name)
    profiles = dict(_profile_dict(cfg))
    if profile_name not in profiles:
        _raise_config_error(f"Profile not found: {profile_name}")
    profiles.pop(profile_name, None)
    _set_profile_dict(cfg, profiles)
    active = str((cfg.extra_fields or {}).get("active_profile") or "").strip()
    if active != profile_name:
        return
    if profiles:
        set_active_profile(cfg, sorted(profiles)[0])
    else:
        cfg.extra_fields.pop("active_profile", None)


def set_active_profile(cfg: AppConfig, name: str) -> None:
    profile_name = _normalize_profile_name(name)
    profile = get_profile(cfg, profile_name)
    if profile is None:
        _raise_config_error(f"Profile not found: {profile_name}")
    cfg.extra_fields["active_profile"] = profile.name
    if profile.base_url:
        cfg.base_url = profile.base_url
    if profile.default_model:
        cfg.model = profile.default_model


def update_active_profile_defaults(
    cfg: AppConfig,
    *,
    base_url: str | None = None,
    default_model: str | None = None,
) -> bool:
    if not isinstance((cfg.extra_fields or {}).get("profiles"), dict):
        return False
    active = str((cfg.extra_fields or {}).get("active_profile") or "").strip()
    if not active:
        return False
    current = get_profile(cfg, active)
    if current is None:
        _raise_config_error(f"Active profile not found: {active}")

    next_base_url = current.base_url
    if base_url is not None:
        next_base_url = validate_base_url(base_url, key="base_url", allow_empty=True)

    next_default_model = current.default_model
    if default_model is not None:
        next_default_model = str(default_model or "").strip()

    updated = ProfileSpec(
        name=current.name,
        protocol=current.protocol,
        base_url=next_base_url,
        api_key_env=current.api_key_env,
        extra_headers=dict(current.extra_headers),
        default_model=next_default_model,
        web_search_adapter=current.web_search_adapter,
        web_search_model=current.web_search_model,
        notes=current.notes,
    )
    if updated.to_dict() == current.to_dict():
        return False
    add_profile(cfg, updated)
    if base_url is not None:
        cfg.base_url = updated.base_url
    if default_model is not None:
        cfg.model = updated.default_model
    return True


def update_profile(cfg: AppConfig, name: str, **fields: Any) -> None:
    profile_name = _normalize_profile_name(name)
    current = get_profile(cfg, profile_name)
    if current is None:
        _raise_config_error(f"Profile not found: {profile_name}")
    values = {
        "name": current.name,
        "protocol": current.protocol,
        "base_url": current.base_url,
        "api_key_env": current.api_key_env,
        "extra_headers": dict(current.extra_headers),
        "default_model": current.default_model,
        "web_search_adapter": current.web_search_adapter,
        "web_search_model": current.web_search_model,
        "notes": current.notes,
    }
    values.update(fields)
    updated = ProfileSpec(**values)
    add_profile(cfg, updated)
    if str((cfg.extra_fields or {}).get("active_profile") or "") == profile_name:
        set_active_profile(cfg, profile_name)


def rename_profile(cfg: AppConfig, old: str, new: str) -> None:
    old_name = _normalize_profile_name(old)
    new_name = _normalize_profile_name(new)
    if old_name == new_name:
        return
    current = get_profile(cfg, old_name)
    if current is None:
        _raise_config_error(f"Profile not found: {old_name}")
    if get_profile(cfg, new_name) is not None:
        _raise_config_error(f"Profile already exists: {new_name}")
    profiles = dict(_profile_dict(cfg))
    profiles.pop(old_name, None)
    profiles[new_name] = ProfileSpec(
        name=new_name,
        protocol=current.protocol,
        base_url=current.base_url,
        api_key_env=current.api_key_env,
        extra_headers=dict(current.extra_headers),
        default_model=current.default_model,
        web_search_adapter=current.web_search_adapter,
        web_search_model=current.web_search_model,
        notes=current.notes,
    ).to_dict()
    _set_profile_dict(cfg, profiles)
    if str((cfg.extra_fields or {}).get("active_profile") or "") == old_name:
        set_active_profile(cfg, new_name)


def migrate_legacy_to_profiles(cfg: AppConfig) -> bool:
    extra_fields = dict(getattr(cfg, "extra_fields", {}) or {})
    if isinstance(extra_fields.get("profiles"), dict):
        cfg.extra_fields = extra_fields
        return False

    base_url = str(getattr(cfg, "base_url", "") or "").strip()
    if not base_url or _same_base_url(base_url, DEFAULT_OPENAI_BASE_URL):
        base_url = DEFAULT_OPENAI_BASE_URL
        api_key_env = None if _legacy_api_key_present() else "OPENAI_API_KEY"
    else:
        api_key_env = None
    profile = ProfileSpec(
        name="default",
        protocol="openai_compat",
        base_url=base_url,
        api_key_env=api_key_env,
        default_model=str(getattr(cfg, "model", "") or "").strip(),
        notes="Migrated from legacy base_url/model config.",
    )
    extra_fields["profiles"] = {"default": profile.to_dict()}
    extra_fields["active_profile"] = "default"
    cfg.extra_fields = extra_fields
    return True


def _profile_dict(cfg: AppConfig) -> dict[str, dict[str, Any]]:
    raw = (cfg.extra_fields or {}).get("profiles")
    if not isinstance(raw, dict):
        return {}
    profiles: dict[str, dict[str, Any]] = {}
    for name, data in raw.items():
        profile_name = _normalize_profile_name(str(name))
        if isinstance(data, dict):
            profiles[profile_name] = dict(data)
    return profiles


def _set_profile_dict(cfg: AppConfig, profiles: dict[str, dict[str, Any]]) -> None:
    extra_fields = dict(cfg.extra_fields or {})
    extra_fields["profiles"] = dict(sorted(profiles.items()))
    cfg.extra_fields = extra_fields


def _normalize_profile_name(name: str) -> str:
    normalized = str(name or "").strip().lower()
    _validate_profile_name(normalized)
    return normalized


def _validate_profile_name(name: str) -> None:
    normalized = str(name or "").strip()
    if not normalized:
        _raise_config_error("Profile name must be non-empty.")
    if normalized != normalized.lower() or not _PROFILE_NAME_RE.fullmatch(normalized):
        _raise_config_error(
            "Profile name must use lowercase letters, digits, hyphens, or underscores."
        )


def _validate_protocol(protocol: str) -> None:
    if str(protocol or "").strip() not in SUPPORTED_PROTOCOLS:
        allowed = ", ".join(sorted(SUPPORTED_PROTOCOLS))
        _raise_config_error(f"Profile protocol must be one of: {allowed}")


def _validate_web_search_adapter(adapter: str) -> None:
    try:
        normalize_web_search_adapter(adapter)
    except ValueError as exc:
        _raise_config_error(str(exc))


def _validate_base_url(base_url: str) -> None:
    validate_base_url(base_url, key="Profile base_url")


def validate_base_url(
    base_url: str,
    *,
    key: str = "base_url",
    allow_empty: bool = False,
) -> str:
    normalized = str(base_url or "").strip()
    if not normalized:
        if allow_empty:
            return ""
        _raise_config_error(f"{key} must be non-empty.")
    if any(ch.isspace() for ch in normalized):
        _raise_config_error(f"{key} must not contain whitespace.")
    try:
        parsed = urlsplit(normalized)
    except ValueError:
        _raise_config_error(f"{key} must be a valid http:// or https:// URL.")
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        _raise_config_error(f"{key} must be a valid http:// or https:// URL.")
    if parsed.query or parsed.fragment:
        _raise_config_error(f"{key} must not include query strings or fragments.")
    return normalized


def resolve_effective_base_url(*, cfg: AppConfig, profile: ProfileSpec | None = None) -> str:
    resolved_profile = profile or get_active_profile(cfg)
    profile_base_url = validate_base_url(
        resolved_profile.base_url,
        key=f"Profile {resolved_profile.name} base_url",
        allow_empty=True,
    ).rstrip("/")
    if profile_base_url:
        return profile_base_url
    cfg_base_url = validate_base_url(
        str(getattr(cfg, "base_url", "") or ""),
        key="base_url",
        allow_empty=True,
    ).rstrip("/")
    default_base_url = DEFAULT_OPENAI_BASE_URL.rstrip("/")
    return cfg_base_url or default_base_url


def _coerce_headers(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        _raise_config_error("Profile extra_headers must be a JSON object.")
    headers: dict[str, str] = {}
    for key, value in raw.items():
        header_name = str(key or "").strip()
        header_value = str(value or "").strip()
        if header_name and header_value:
            headers[header_name] = header_value
    return headers


def _optional_string(raw: Any) -> str | None:
    value = str(raw or "").strip()
    return value or None


def _same_base_url(left: str, right: str) -> bool:
    return str(left or "").strip().rstrip("/") == str(right or "").strip().rstrip("/")


def _legacy_api_key_present() -> bool:
    try:
        from .config import load_persisted_api_key

        return bool(load_persisted_api_key())
    except Exception:
        return False


def _raise_config_error(message: str) -> None:
    from .config import ConfigError

    raise ConfigError(message)
