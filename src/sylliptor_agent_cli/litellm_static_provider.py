from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from importlib import resources
from typing import Any

from .model_metadata_utils import (
    model_name_variants,
    normalize_base_url,
    parse_bool,
    parse_non_negative_float,
    parse_positive_int,
)

_PROVIDER_TOKEN_RE = re.compile(r"[a-z0-9]+")
_GENERIC_PROVIDER_TOKENS = {
    "api",
    "apis",
    "app",
    "coding",
    "compat",
    "compatible",
    "com",
    "cn",
    "http",
    "https",
    "intl",
    "net",
    "org",
    "paas",
    "v1",
    "v2",
    "v3",
    "v4",
    "www",
}
_BUNDLED_CATALOG_PACKAGE = "sylliptor_agent_cli.model_catalog"
_BUNDLED_CATALOG_FILENAME = "litellm_model_prices_snapshot.json"
_BUNDLED_CATALOG_META_FILENAME = "litellm_model_prices_snapshot.meta.json"
BUNDLED_MODEL_CATALOG_SOURCE = "bundled_litellm_snapshot"
_BUNDLED_MODEL_CATALOG_REFRESH_POLICY = "manual_reviewed_only"
_NON_MODEL_TOP_LEVEL_KEYS = {"sample_spec"}
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_FULL_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STRICT_UTC_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


@dataclass(frozen=True)
class LiteLLMStaticMetadata:
    model_key: str | None
    context_window_tokens: int | None
    max_output_tokens: int | None
    supports_vision: bool | None
    input_cost_per_token: float | None
    output_cost_per_token: float | None
    raw_metadata: dict[str, Any]
    error: str | None


@dataclass(frozen=True)
class BundledModelCatalogProvenance:
    schema_version: int | None
    source: str | None
    refresh_policy: str | None
    fetched_at_utc: str | None
    upstream_commit_sha: str | None
    upstream_blob_url: str | None
    bundled_json_sha256: str | None
    bundled_json_size_bytes: int | None
    top_level_entry_count: int | None
    raw_metadata: dict[str, Any]
    error: str | None


def _empty_metadata(error: str) -> LiteLLMStaticMetadata:
    return LiteLLMStaticMetadata(
        model_key=None,
        context_window_tokens=None,
        max_output_tokens=None,
        supports_vision=None,
        input_cost_per_token=None,
        output_cost_per_token=None,
        raw_metadata={},
        error=error,
    )


def _empty_provenance(error: str) -> BundledModelCatalogProvenance:
    return BundledModelCatalogProvenance(
        schema_version=None,
        source=None,
        refresh_policy=None,
        fetched_at_utc=None,
        upstream_commit_sha=None,
        upstream_blob_url=None,
        bundled_json_sha256=None,
        bundled_json_size_bytes=None,
        top_level_entry_count=None,
        raw_metadata={},
        error=error,
    )


def _provider_tokens(*values: str) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        for token in _PROVIDER_TOKEN_RE.findall(value.casefold()):
            if token in _GENERIC_PROVIDER_TOKENS:
                continue
            tokens.add(token)
    return tokens


def _preferred_provider_tokens(base_url: str | None) -> set[str]:
    if base_url is None:
        return set()
    normalized = normalize_base_url(base_url)
    if not normalized:
        return set()
    return _provider_tokens(normalized)


def _is_model_catalog_entry(key: Any, value: Any) -> bool:
    if not isinstance(key, str) or not isinstance(value, dict):
        return False
    return key.casefold() not in _NON_MODEL_TOP_LEVEL_KEYS


def _candidate_rank(
    *,
    key: str,
    raw_metadata: dict[str, Any],
    direct_match: bool,
    preferred_provider_tokens: set[str],
) -> tuple[int, int, int, int]:
    provider_tokens = _provider_tokens(
        key,
        str(raw_metadata.get("litellm_provider") or ""),
    )
    provider_score = len(provider_tokens & preferred_provider_tokens)
    return (
        1 if direct_match else 0,
        provider_score,
        -key.count("/"),
        -len(key),
    )


def _lookup_model_info(
    model_cost: dict[str, Any], requested_model: str, *, base_url: str | None = None
) -> tuple[str, dict[str, Any]] | None:
    lower_index: dict[str, list[str]] = {}
    alias_index: dict[str, list[str]] = {}
    for key, value in model_cost.items():
        if not _is_model_catalog_entry(key, value):
            continue
        lower_index.setdefault(key.casefold(), []).append(key)
        for alias in model_name_variants(key):
            alias_index.setdefault(alias.casefold(), []).append(key)

    preferred_provider_tokens = _preferred_provider_tokens(base_url)
    candidates: list[tuple[tuple[int, int, int, int], str, dict[str, Any]]] = []
    seen: set[str] = set()
    for variant in model_name_variants(requested_model):
        direct = model_cost.get(variant)
        if _is_model_catalog_entry(variant, direct):
            seen.add(variant)
            candidates.append(
                (
                    _candidate_rank(
                        key=variant,
                        raw_metadata=direct,
                        direct_match=True,
                        preferred_provider_tokens=preferred_provider_tokens,
                    ),
                    variant,
                    direct,
                )
            )
        for key in alias_index.get(variant.casefold(), []):
            if key in seen:
                continue
            aliased = model_cost.get(key)
            if not isinstance(aliased, dict):
                continue
            seen.add(key)
            candidates.append(
                (
                    _candidate_rank(
                        key=key,
                        raw_metadata=aliased,
                        direct_match=False,
                        preferred_provider_tokens=preferred_provider_tokens,
                    ),
                    key,
                    aliased,
                )
            )
        for key in lower_index.get(variant.casefold(), []):
            if key in seen:
                continue
            aliased = model_cost.get(key)
            if not isinstance(aliased, dict):
                continue
            seen.add(key)
            candidates.append(
                (
                    _candidate_rank(
                        key=key,
                        raw_metadata=aliased,
                        direct_match=False,
                        preferred_provider_tokens=preferred_provider_tokens,
                    ),
                    key,
                    aliased,
                )
            )
    if candidates:
        _rank, key, metadata = max(candidates, key=lambda item: item[0])
        return key, metadata
    return None


def _resolve_capacity_metadata(
    raw_metadata: dict[str, Any],
) -> tuple[int | None, int | None]:
    raw_total = parse_positive_int(raw_metadata.get("max_tokens"))
    raw_input = parse_positive_int(raw_metadata.get("max_input_tokens"))
    raw_output = parse_positive_int(raw_metadata.get("max_output_tokens"))

    if raw_input is not None and raw_output is not None:
        return raw_input + raw_output, raw_output

    if raw_input is not None and raw_total is not None:
        if raw_total > raw_input:
            return raw_total, raw_total - raw_input
        return raw_input + raw_total, raw_total

    if raw_total is not None:
        return raw_total, raw_output

    if raw_input is not None:
        return raw_input, raw_output

    return None, raw_output


def _parse_full_hex_digest(value: Any, *, digits: int) -> str | None:
    if not isinstance(value, str):
        return None
    clean = value.strip().lower()
    matcher = _FULL_SHA_RE if digits == 40 else _FULL_SHA256_RE
    if not matcher.fullmatch(clean):
        return None
    return clean


def _parse_strict_utc_timestamp(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    clean = value.strip()
    if not _STRICT_UTC_TIMESTAMP_RE.fullmatch(clean):
        return None
    try:
        datetime.strptime(clean, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None
    return clean


@lru_cache(maxsize=1)
def _load_bundled_model_catalog() -> dict[str, Any]:
    try:
        text = (
            resources.files(_BUNDLED_CATALOG_PACKAGE)
            .joinpath(_BUNDLED_CATALOG_FILENAME)
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError) as e:
        raise FileNotFoundError(_BUNDLED_CATALOG_FILENAME) from e
    except (OSError, UnicodeDecodeError) as e:
        raise ValueError("bundled model catalog invalid") from e

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError("bundled model catalog invalid") from e

    if not isinstance(raw, dict):
        raise ValueError("bundled model catalog invalid")
    return raw


@lru_cache(maxsize=1)
def _load_bundled_model_catalog_meta() -> dict[str, Any]:
    try:
        text = (
            resources.files(_BUNDLED_CATALOG_PACKAGE)
            .joinpath(_BUNDLED_CATALOG_META_FILENAME)
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError) as e:
        raise FileNotFoundError(_BUNDLED_CATALOG_META_FILENAME) from e
    except (OSError, UnicodeDecodeError) as e:
        raise ValueError("bundled model catalog provenance invalid") from e

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError("bundled model catalog provenance invalid") from e

    if not isinstance(raw, dict):
        raise ValueError("bundled model catalog provenance invalid")
    return raw


def get_bundled_model_catalog_provenance() -> BundledModelCatalogProvenance:
    try:
        raw = _load_bundled_model_catalog_meta()
    except FileNotFoundError:
        return _empty_provenance("bundled model catalog provenance missing")
    except ValueError:
        return _empty_provenance("bundled model catalog provenance invalid")

    schema_version = parse_positive_int(raw.get("schema_version"))
    source = raw.get("source") if isinstance(raw.get("source"), str) else None
    refresh_policy = (
        raw.get("refresh_policy") if isinstance(raw.get("refresh_policy"), str) else None
    )
    upstream = raw.get("upstream")
    snapshot = raw.get("snapshot")
    if (
        schema_version != 1
        or source != BUNDLED_MODEL_CATALOG_SOURCE
        or refresh_policy != _BUNDLED_MODEL_CATALOG_REFRESH_POLICY
        or not isinstance(upstream, dict)
        or not isinstance(snapshot, dict)
    ):
        return _empty_provenance("bundled model catalog provenance invalid")

    upstream_commit_sha = _parse_full_hex_digest(upstream.get("commit_sha"), digits=40)
    upstream_repo = upstream.get("repo") if isinstance(upstream.get("repo"), str) else None
    upstream_path = upstream.get("path") if isinstance(upstream.get("path"), str) else None
    upstream_blob_url = (
        upstream.get("blob_url") if isinstance(upstream.get("blob_url"), str) else None
    )
    fetched_at_utc = _parse_strict_utc_timestamp(snapshot.get("fetched_at_utc"))
    bundled_json_sha256 = _parse_full_hex_digest(snapshot.get("bundled_json_sha256"), digits=64)
    bundled_json_size_bytes = parse_positive_int(snapshot.get("bundled_json_size_bytes"))
    top_level_entry_count = parse_positive_int(snapshot.get("top_level_entry_count"))
    if (
        upstream_commit_sha is None
        or upstream_repo is None
        or upstream_path is None
        or fetched_at_utc is None
        or bundled_json_sha256 is None
        or bundled_json_size_bytes is None
        or top_level_entry_count is None
    ):
        return _empty_provenance("bundled model catalog provenance invalid")

    return BundledModelCatalogProvenance(
        schema_version=schema_version,
        source=source,
        refresh_policy=refresh_policy,
        fetched_at_utc=fetched_at_utc,
        upstream_commit_sha=upstream_commit_sha,
        upstream_blob_url=upstream_blob_url,
        bundled_json_sha256=bundled_json_sha256,
        bundled_json_size_bytes=bundled_json_size_bytes,
        top_level_entry_count=top_level_entry_count,
        raw_metadata=dict(raw),
        error=None,
    )


def resolve_litellm_static_metadata(
    model_name: str, *, base_url: str | None = None
) -> LiteLLMStaticMetadata:
    try:
        model_cost = _load_bundled_model_catalog()
    except FileNotFoundError:
        return _empty_metadata("bundled model catalog missing")
    except ValueError:
        return _empty_metadata("bundled model catalog invalid")

    match = _lookup_model_info(model_cost, model_name, base_url=base_url)
    if match is None:
        return _empty_metadata("model not found in bundled model catalog")

    model_key, raw_metadata = match
    context_window_tokens, max_output_tokens = _resolve_capacity_metadata(raw_metadata)
    supports_vision = parse_bool(raw_metadata.get("supports_vision"))
    input_cost_per_token = parse_non_negative_float(raw_metadata.get("input_cost_per_token"))
    output_cost_per_token = parse_non_negative_float(raw_metadata.get("output_cost_per_token"))

    return LiteLLMStaticMetadata(
        model_key=model_key,
        context_window_tokens=context_window_tokens,
        max_output_tokens=max_output_tokens,
        supports_vision=supports_vision,
        input_cost_per_token=input_cost_per_token,
        output_cost_per_token=output_cost_per_token,
        raw_metadata=dict(raw_metadata),
        error=None,
    )
