from __future__ import annotations

import pytest

from sylliptor_agent_cli.assets import (
    AssetBriefingEntry,
    AssetError,
    TaskAssetBriefing,
    collect_referenced_asset_ids,
    parse_task_asset_briefing,
    serialize_task_asset_briefing,
    task_asset_briefing,
)


def test_parse_none_and_empty_asset_briefing() -> None:
    assert parse_task_asset_briefing(None) is None
    assert parse_task_asset_briefing({}) == TaskAssetBriefing(primary=[], may_need=[])


def test_asset_briefing_round_trips() -> None:
    briefing = TaskAssetBriefing(
        primary=[
            AssetBriefingEntry(
                asset_id="ast_aaaaaaaa",
                rationale="Primary rationale",
                expected_use="Use directly",
            )
        ],
        may_need=[
            AssetBriefingEntry(
                asset_id="ast_bbbbbbbb",
                rationale="May help",
                expected_use="Read if needed",
            )
        ],
    )

    assert parse_task_asset_briefing(serialize_task_asset_briefing(briefing)) == briefing


def test_asset_briefing_rejects_unknown_keys() -> None:
    with pytest.raises(AssetError, match="unsupported keys"):
        parse_task_asset_briefing({"primary": [], "extra": []})


def test_collect_referenced_asset_ids_handles_mixed_tasks() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "asset_briefing": {
                    "primary": [
                        {
                            "asset_id": "ast_aaaaaaaa",
                            "rationale": "A",
                            "expected_use": "Use A",
                        }
                    ],
                    "may_need": [
                        {
                            "asset_id": "ast_bbbbbbbb",
                            "rationale": "B",
                            "expected_use": "Use B",
                        }
                    ],
                },
            },
            {"id": "T02"},
        ]
    }

    assert collect_referenced_asset_ids(plan) == {"ast_aaaaaaaa", "ast_bbbbbbbb"}
    assert task_asset_briefing(plan["tasks"][1]) is None
