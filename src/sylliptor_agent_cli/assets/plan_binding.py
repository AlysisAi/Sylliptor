from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import AssetError


@dataclass(frozen=True)
class AssetBriefingEntry:
    asset_id: str
    rationale: str
    expected_use: str


@dataclass(frozen=True)
class TaskAssetBriefing:
    primary: list[AssetBriefingEntry]
    may_need: list[AssetBriefingEntry]


def parse_task_asset_briefing(payload: Any | None) -> TaskAssetBriefing | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise AssetError("asset_briefing must be an object")
    _reject_unknown_keys(payload, allowed={"primary", "may_need"}, field="asset_briefing")
    return TaskAssetBriefing(
        primary=_parse_entries(payload.get("primary"), field="asset_briefing.primary"),
        may_need=_parse_entries(payload.get("may_need"), field="asset_briefing.may_need"),
    )


def serialize_task_asset_briefing(briefing: TaskAssetBriefing | None) -> dict[str, Any] | None:
    if briefing is None:
        return None
    return {
        "primary": [_entry_to_dict(entry) for entry in briefing.primary],
        "may_need": [_entry_to_dict(entry) for entry in briefing.may_need],
    }


def collect_referenced_asset_ids(plan: dict[str, Any]) -> set[str]:
    referenced: set[str] = set()
    for task in plan.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        briefing = task_asset_briefing(task)
        if briefing is None:
            continue
        referenced.update(entry.asset_id for entry in briefing.primary)
        referenced.update(entry.asset_id for entry in briefing.may_need)
    return referenced


def task_asset_briefing(task: dict[str, Any]) -> TaskAssetBriefing | None:
    return parse_task_asset_briefing(task.get("asset_briefing"))


def _parse_entries(payload: Any, *, field: str) -> list[AssetBriefingEntry]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise AssetError(f"{field} must be an array")
    out: list[AssetBriefingEntry] = []
    for index, raw_entry in enumerate(payload):
        entry_field = f"{field}[{index}]"
        if not isinstance(raw_entry, dict):
            raise AssetError(f"{entry_field} must be an object")
        _reject_unknown_keys(
            raw_entry,
            allowed={"asset_id", "rationale", "expected_use"},
            field=entry_field,
        )
        asset_id = _required_string(raw_entry.get("asset_id"), field=f"{entry_field}.asset_id")
        rationale = _required_string(raw_entry.get("rationale"), field=f"{entry_field}.rationale")
        expected_use = _required_string(
            raw_entry.get("expected_use"),
            field=f"{entry_field}.expected_use",
        )
        out.append(
            AssetBriefingEntry(
                asset_id=asset_id,
                rationale=rationale,
                expected_use=expected_use,
            )
        )
    return out


def _entry_to_dict(entry: AssetBriefingEntry) -> dict[str, str]:
    return {
        "asset_id": entry.asset_id,
        "rationale": entry.rationale,
        "expected_use": entry.expected_use,
    }


def _required_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise AssetError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise AssetError(f"{field} must be non-empty")
    return text


def _reject_unknown_keys(payload: dict[str, Any], *, allowed: set[str], field: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise AssetError(f"{field} contains unsupported keys: {', '.join(unknown)}")
