from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from .branding import env_get
from .config import AppConfig
from .litellm_static_provider import (
    BUNDLED_MODEL_CATALOG_SOURCE,
    resolve_litellm_static_metadata,
)
from .model_metadata_utils import (
    model_name_variants,
    normalize_base_url,
    parse_bool,
    parse_non_negative_float,
    parse_positive_int,
)

_INT_FIELDS = ("context_window_tokens", "max_output_tokens")
_FLOAT_FIELDS = ("input_cost_per_token", "output_cost_per_token")
_BOOL_FIELDS = ("supports_vision",)
_TRACKED_FIELDS = (
    "context_window_tokens",
    "max_output_tokens",
    "supports_vision",
    "input_cost_per_token",
    "output_cost_per_token",
)
_FALLBACKS: dict[str, Any] = {
    "context_window_tokens": 8192,
    "max_output_tokens": 2048,
    "supports_vision": False,
    "input_cost_per_token": None,
    "output_cost_per_token": None,
}
_ENV_FIELD_MAP: dict[str, str] = {
    "context_window_tokens": "SYLLIPTOR_CONTEXT_WINDOW",
    "max_output_tokens": "SYLLIPTOR_MAX_OUTPUT_TOKENS",
    "supports_vision": "SYLLIPTOR_SUPPORTS_VISION",
    "input_cost_per_token": "SYLLIPTOR_INPUT_COST_PER_TOKEN",
    "output_cost_per_token": "SYLLIPTOR_OUTPUT_COST_PER_TOKEN",
}
_DEPRECATED_MODEL_CAPABILITIES_WARNING = (
    "Config key `model_capabilities` is deprecated and ignored; use `model_metadata_overrides`."
)
_FALLBACK_WARNING = (
    "Using fallback context/max_output; set model_metadata_overrides for best performance."
)
_BUILT_IN_MODEL_METADATA: dict[str, dict[str, Any]] = {
    "deepseek-v4-flash": {
        "context_window_tokens": 1_000_000,
        "max_output_tokens": 384_000,
        "supports_vision": False,
        "input_cost_per_token": 0.00000014,
        "output_cost_per_token": 0.00000028,
    },
    "deepseek-v4-pro": {
        "context_window_tokens": 1_000_000,
        "max_output_tokens": 384_000,
        "supports_vision": False,
        "input_cost_per_token": 0.000000435,
        "output_cost_per_token": 0.00000087,
    },
    "deepseek-chat": {
        "context_window_tokens": 1_000_000,
        "max_output_tokens": 384_000,
        "supports_vision": False,
        "input_cost_per_token": 0.000000435,
        "output_cost_per_token": 0.00000087,
    },
    "deepseek-reasoner": {
        "context_window_tokens": 1_000_000,
        "max_output_tokens": 384_000,
        "supports_vision": False,
        "input_cost_per_token": 0.000000435,
        "output_cost_per_token": 0.00000087,
    },
    # Hosted Xiaomi MiMo trial models. The proxy serves these ids when allowlisted,
    # and falls back server-side to its canonical MiMo model otherwise.
    "mimo-v2.5-pro": {
        "context_window_tokens": 1_048_576,
        "max_output_tokens": 131_072,
        "supports_vision": False,
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000003,
    },
    "mimo-v2-flash": {
        "context_window_tokens": 262_144,
        "max_output_tokens": 65_536,
        "supports_vision": False,
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0000003,
    },
    "mimo-v2.5": {
        "context_window_tokens": 1_048_576,
        "max_output_tokens": 131_072,
        "supports_vision": True,
        "input_cost_per_token": 0.0000004,
        "output_cost_per_token": 0.000002,
    },
    # Legacy friendly id kept for sessions that logged in before model choice.
    # Hosted Xiaomi MiMo trial: the CLI sends the friendly id "mimo" (the proxy
    # pins it to the real upstream id server-side). Without an entry here the id
    # resolves nowhere and falls back to 8192/2048 — emitting a metadata warning
    # on every run and silently collapsing the usable context window ~32x.
    # Values mirror the bundled `openrouter/xiaomi/mimo-v2-flash` catalog entry
    # (262144 input / 16384 output); costs use the mimo-v2.5-pro list price.
    "mimo": {
        "context_window_tokens": 262_144,
        "max_output_tokens": 16_384,
        "supports_vision": False,
        "input_cost_per_token": 0.000000435,
        "output_cost_per_token": 0.00000087,
    },
}


@dataclass(frozen=True)
class ModelMeta:
    model_name: str
    context_window_tokens: int
    max_output_tokens: int
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "fallback"
    supports_vision: bool = False
    field_sources: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass
class _LayerData:
    name: str
    values: dict[str, Any] = field(default_factory=dict)
    field_sources: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    model_name: str | None = None


def _parse_field(field: str, raw_value: Any) -> Any | None:
    if field in _INT_FIELDS:
        return parse_positive_int(raw_value)
    if field in _FLOAT_FIELDS:
        return parse_non_negative_float(raw_value)
    if field in _BOOL_FIELDS:
        return parse_bool(raw_value)
    return None


def _dedupe_warnings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = value.strip()
        if not clean:
            continue
        if clean in seen:
            continue
        seen.add(clean)
        deduped.append(clean)
    return deduped


def _build_model_alias_index(mapping: dict[str, Any]) -> dict[str, str]:
    index: dict[str, str] = {}
    for key in mapping:
        if not isinstance(key, str):
            continue
        for variant in model_name_variants(key):
            index.setdefault(variant.casefold(), key)
    return index


def _lookup_model_override(
    *,
    mapping: Any,
    requested_model: str,
    label: str,
    warnings: list[str],
) -> tuple[str, dict[str, Any]] | None:
    if mapping is None:
        return None
    if not isinstance(mapping, dict):
        warnings.append(f"Ignoring invalid {label}: expected object.")
        return None

    alias_index = _build_model_alias_index(mapping)
    for variant in model_name_variants(requested_model):
        key = alias_index.get(variant.casefold())
        if key is None:
            continue
        raw = mapping.get(key)
        if isinstance(raw, dict):
            return key, raw
        warnings.append(f"Ignoring invalid {label}[{key!r}]: expected object.")
        return None
    return None


def _lookup_endpoint_override(
    *,
    endpoints: Any,
    base_url: str,
    warnings: list[str],
) -> tuple[str, dict[str, Any]] | None:
    if endpoints is None:
        return None
    if not isinstance(endpoints, dict):
        warnings.append("Ignoring invalid model_metadata_overrides.endpoints: expected object.")
        return None

    if base_url in endpoints:
        exact = endpoints.get(base_url)
        if isinstance(exact, dict):
            return base_url, exact
        warnings.append(
            f"Ignoring invalid model_metadata_overrides.endpoints[{base_url!r}]: expected object."
        )
        return None

    normalized_base_url = normalize_base_url(base_url)
    if not normalized_base_url:
        return None
    for endpoint_key, endpoint_value in endpoints.items():
        if not isinstance(endpoint_key, str):
            continue
        if normalize_base_url(endpoint_key) != normalized_base_url:
            continue
        if isinstance(endpoint_value, dict):
            return endpoint_key, endpoint_value
        warnings.append(
            "Ignoring invalid model_metadata_overrides.endpoints"
            f"[{endpoint_key!r}]: expected object."
        )
        return None
    return None


class ModelRegistry:
    def __init__(self, *, cfg: AppConfig, api_key: str | None = None) -> None:
        _ = api_key
        self._cfg = cfg
        self.last_error: str | None = None
        self.last_warnings: list[str] = []
        self.last_source: str | None = None

    def _resolve_env_layer(self) -> _LayerData:
        layer = _LayerData(name="env")
        for field_name, env_name in _ENV_FIELD_MAP.items():
            raw_value = env_get(env_name)
            if raw_value is None:
                continue
            parsed = _parse_field(field_name, raw_value)
            if parsed is None:
                layer.warnings.append(f"Ignoring invalid {env_name}: {raw_value!r}")
                continue
            layer.values[field_name] = parsed
            layer.field_sources[field_name] = f"env:{env_name}"
        return layer

    def _apply_user_scope(
        self,
        *,
        layer: _LayerData,
        source: str,
        payload: dict[str, Any],
    ) -> None:
        for field_name in _TRACKED_FIELDS:
            if field_name in layer.values:
                continue
            if field_name not in payload:
                continue
            raw_value = payload.get(field_name)
            if raw_value is None:
                continue
            parsed = _parse_field(field_name, raw_value)
            if parsed is None:
                layer.warnings.append(f"Ignoring invalid {field_name} in {source}: {raw_value!r}")
                continue
            layer.values[field_name] = parsed
            layer.field_sources[field_name] = source

    def _resolve_user_layer(self, requested_model: str) -> _LayerData:
        layer = _LayerData(name="user")
        raw_overrides = self._cfg.extra_fields.get("model_metadata_overrides")
        if raw_overrides is None:
            return layer
        if not isinstance(raw_overrides, dict):
            layer.warnings.append("Ignoring invalid model_metadata_overrides: expected object.")
            return layer

        scoped_payloads: list[tuple[str, dict[str, Any]]] = []
        endpoint_match = _lookup_endpoint_override(
            endpoints=raw_overrides.get("endpoints"),
            base_url=self._cfg.base_url,
            warnings=layer.warnings,
        )
        if endpoint_match is not None:
            endpoint_key, endpoint_payload = endpoint_match
            endpoint_model_match = _lookup_model_override(
                mapping=endpoint_payload.get("models"),
                requested_model=requested_model,
                label=f"model_metadata_overrides.endpoints[{endpoint_key!r}].models",
                warnings=layer.warnings,
            )
            if endpoint_model_match is not None:
                model_key, model_payload = endpoint_model_match
                scoped_payloads.append(
                    (
                        f"user:endpoints[{endpoint_key!r}].models[{model_key!r}]",
                        model_payload,
                    )
                )

            endpoint_default = endpoint_payload.get("default")
            if endpoint_default is not None:
                if isinstance(endpoint_default, dict):
                    scoped_payloads.append(
                        (f"user:endpoints[{endpoint_key!r}].default", endpoint_default)
                    )
                else:
                    layer.warnings.append(
                        "Ignoring invalid model_metadata_overrides.endpoints"
                        f"[{endpoint_key!r}].default: expected object."
                    )

        model_match = _lookup_model_override(
            mapping=raw_overrides.get("models"),
            requested_model=requested_model,
            label="model_metadata_overrides.models",
            warnings=layer.warnings,
        )
        if model_match is not None:
            model_key, model_payload = model_match
            scoped_payloads.append((f"user:models[{model_key!r}]", model_payload))

        default_payload = raw_overrides.get("default")
        if default_payload is not None:
            if isinstance(default_payload, dict):
                scoped_payloads.append(("user:default", default_payload))
            else:
                layer.warnings.append(
                    "Ignoring invalid model_metadata_overrides.default: expected object."
                )

        for source, payload in scoped_payloads:
            self._apply_user_scope(layer=layer, source=source, payload=payload)

        return layer

    def _resolve_bundled_model_catalog_layer(self, requested_model: str) -> _LayerData:
        layer = _LayerData(name=BUNDLED_MODEL_CATALOG_SOURCE)
        static_meta = resolve_litellm_static_metadata(
            requested_model,
            base_url=self._cfg.base_url,
        )
        layer.error = static_meta.error
        layer.raw_metadata = static_meta.raw_metadata
        layer.model_name = static_meta.model_key

        values = {
            "context_window_tokens": static_meta.context_window_tokens,
            "max_output_tokens": static_meta.max_output_tokens,
            "supports_vision": static_meta.supports_vision,
            "input_cost_per_token": static_meta.input_cost_per_token,
            "output_cost_per_token": static_meta.output_cost_per_token,
        }
        for field_name, value in values.items():
            if value is None:
                continue
            layer.values[field_name] = value
            layer.field_sources[field_name] = BUNDLED_MODEL_CATALOG_SOURCE
        return layer

    def _resolve_builtin_layer(self, requested_model: str) -> _LayerData:
        layer = _LayerData(name="built_in")
        model_match = _lookup_model_override(
            mapping=_BUILT_IN_MODEL_METADATA,
            requested_model=requested_model,
            label="built_in_model_catalog",
            warnings=layer.warnings,
        )
        if model_match is None:
            return layer

        model_key, payload = model_match
        layer.model_name = model_key
        layer.raw_metadata = dict(payload)
        self._apply_user_scope(layer=layer, source="built_in", payload=payload)
        return layer

    def _resolve_learned_layer(self, requested_model: str) -> _LayerData:
        _ = requested_model
        # Placeholder layer for deterministic precedence; learned cache is not implemented yet.
        return _LayerData(name="learned")

    def _resolve_field_value(self, field_name: str, layers: list[_LayerData]) -> tuple[Any, str]:
        for layer in layers:
            if field_name not in layer.values:
                continue
            value = layer.values[field_name]
            if value is None:
                continue
            return value, layer.field_sources.get(field_name, layer.name)
        return _FALLBACKS[field_name], "fallback"

    def get(self, model_name: str) -> ModelMeta:
        requested = model_name.strip() or self._cfg.model.strip() or "unknown-model"
        warnings: list[str] = []
        if "model_capabilities" in self._cfg.extra_fields:
            warnings.append(_DEPRECATED_MODEL_CAPABILITIES_WARNING)

        env_layer = self._resolve_env_layer()
        user_layer = self._resolve_user_layer(requested)
        bundled_catalog_layer = self._resolve_bundled_model_catalog_layer(requested)
        built_in_layer = self._resolve_builtin_layer(requested)
        learned_layer = self._resolve_learned_layer(requested)
        layers = [env_layer, user_layer, bundled_catalog_layer, built_in_layer, learned_layer]

        for layer in layers:
            warnings.extend(layer.warnings)

        field_sources: dict[str, str] = {}
        resolved_fields: dict[str, Any] = {}
        for field_name in _TRACKED_FIELDS:
            value, source = self._resolve_field_value(field_name, layers)
            resolved_fields[field_name] = value
            field_sources[field_name] = source

        context_window_tokens = (
            parse_positive_int(resolved_fields["context_window_tokens"])
            or _FALLBACKS["context_window_tokens"]
        )
        max_output_tokens = (
            parse_positive_int(resolved_fields["max_output_tokens"])
            or _FALLBACKS["max_output_tokens"]
        )
        supports_vision = (
            parse_bool(resolved_fields["supports_vision"])
            if resolved_fields["supports_vision"] is not None
            else _FALLBACKS["supports_vision"]
        )
        if supports_vision is None:
            supports_vision = _FALLBACKS["supports_vision"]
        input_cost_per_token = (
            parse_non_negative_float(resolved_fields["input_cost_per_token"])
            if resolved_fields["input_cost_per_token"] is not None
            else None
        )
        output_cost_per_token = (
            parse_non_negative_float(resolved_fields["output_cost_per_token"])
            if resolved_fields["output_cost_per_token"] is not None
            else None
        )

        if max_output_tokens >= context_window_tokens:
            clamped = max(1, context_window_tokens - 1)
            warnings.append(
                "max_output_tokens >= context_window_tokens; "
                f"clamped max_output_tokens to {clamped}."
            )
            max_output_tokens = clamped

        key_fallback = (
            field_sources.get("context_window_tokens") == "fallback"
            or field_sources.get("max_output_tokens") == "fallback"
        )
        if key_fallback:
            warnings.append(_FALLBACK_WARNING)

        all_sources = {field_sources.get(field_name, "fallback") for field_name in _TRACKED_FIELDS}
        overall_source = next(iter(all_sources)) if len(all_sources) == 1 else "mixed"

        resolved_model_name = next(
            (
                layer.model_name
                for layer in layers
                if layer.values and isinstance(layer.model_name, str) and layer.model_name.strip()
            ),
            requested,
        )

        final_warnings = tuple(_dedupe_warnings(warnings))
        self.last_error = bundled_catalog_layer.error if key_fallback else None
        self.last_warnings = list(final_warnings)
        self.last_source = overall_source

        return ModelMeta(
            model_name=resolved_model_name,
            context_window_tokens=context_window_tokens,
            max_output_tokens=max_output_tokens,
            input_cost_per_token=input_cost_per_token,
            output_cost_per_token=output_cost_per_token,
            raw_metadata=bundled_catalog_layer.raw_metadata,
            source=overall_source,
            supports_vision=bool(supports_vision),
            field_sources=field_sources,
            warnings=final_warnings,
        )


def resolve_model_provider_key(
    *,
    cfg: AppConfig,
    model_name: str,
    base_url: str | None = None,
    profile_name: str | None = None,
) -> str | None:
    requested_model = str(model_name or "").strip()
    url_provider = _provider_key_from_base_url(base_url or getattr(cfg, "base_url", None))
    if url_provider == "openrouter":
        return url_provider

    if requested_model:
        meta = ModelRegistry(cfg=cfg).get(requested_model)
        provider = str(meta.raw_metadata.get("litellm_provider") or "").strip()
        if provider:
            return provider
        resolved_model = str(meta.model_name or "").strip()
        if "/" in resolved_model:
            return resolved_model.split("/", 1)[0].strip() or None

    if url_provider:
        return url_provider

    profile_provider = str(profile_name or "").strip()
    if profile_provider and profile_provider.lower() != "default":
        return profile_provider

    if "/" in requested_model:
        return requested_model.split("/", 1)[0].strip() or None
    return requested_model or None


def _provider_key_from_base_url(base_url: str | None) -> str | None:
    raw = str(base_url or "").strip()
    if not raw:
        return None
    try:
        hostname = (urlsplit(raw).hostname or "").rstrip(".").lower()
    except ValueError:
        return None
    if not hostname:
        return None
    if "dashscope" in hostname:
        return "qwen"
    if hostname == "openrouter.ai" or hostname.endswith(".openrouter.ai"):
        return "openrouter"
    if hostname == "api.openai.com":
        return "openai"
    if (
        hostname.endswith(".openai.azure.com")
        or hostname.endswith(".cognitiveservices.azure.com")
        or hostname.endswith(".services.ai.azure.com")
    ):
        return "azure"
    if hostname == "api.deepseek.com" or hostname.endswith(".deepseek.com"):
        return "deepseek"
    if hostname == "generativelanguage.googleapis.com":
        return "gemini"
    if hostname == "api.mistral.ai" or hostname.endswith(".mistral.ai"):
        return "mistral"
    if hostname == "api.x.ai" or hostname == "x.ai" or hostname.endswith(".x.ai"):
        return "xai"
    parts = [
        part
        for part in hostname.split(".")
        if part and part not in {"api", "ai", "www", "com", "v1", "v1beta"}
    ]
    return parts[-1] if parts else hostname
