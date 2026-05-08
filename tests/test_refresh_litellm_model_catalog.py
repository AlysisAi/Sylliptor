from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_VALID_SHA = "10a48f7655225b0dc765d5521839a8bf621805d9"


def _load_refresh_script() -> object:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "refresh_litellm_model_catalog.py"
    spec = importlib.util.spec_from_file_location("refresh_litellm_model_catalog", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_catalog(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _base_catalog() -> dict[str, object]:
    return {
        "sample_spec": {
            "max_tokens": "example only",
        },
        "openai/gpt-5-nano": {
            "input_cost_per_token": 0.1,
            "max_tokens": 128000,
            "output_cost_per_token": 0.2,
            "supports_vision": False,
        },
    }


def _patch_snapshot_paths(monkeypatch, module: object, tmp_path: Path) -> tuple[Path, Path]:
    snapshot_path = tmp_path / "litellm_model_prices_snapshot.json"
    meta_path = tmp_path / "litellm_model_prices_snapshot.meta.json"
    monkeypatch.setattr(module, "_snapshot_paths", lambda: (snapshot_path, meta_path))
    return snapshot_path, meta_path


def test_refresh_script_rejects_invalid_commit_sha(tmp_path: Path, monkeypatch) -> None:
    module = _load_refresh_script()
    _patch_snapshot_paths(monkeypatch, module, tmp_path)
    input_path = _write_catalog(tmp_path / "catalog.json", _base_catalog())

    with pytest.raises(SystemExit, match="full 40-character hex SHA"):
        module.main(["--input", str(input_path), "--upstream-commit-sha", "deadbeef"])


def test_refresh_script_rejects_invalid_fetched_at_timestamp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_refresh_script()
    _patch_snapshot_paths(monkeypatch, module, tmp_path)
    input_path = _write_catalog(tmp_path / "catalog.json", _base_catalog())

    with pytest.raises(SystemExit, match="strict UTC ISO format"):
        module.main(
            [
                "--input",
                str(input_path),
                "--upstream-commit-sha",
                _VALID_SHA,
                "--fetched-at",
                "2026-03-25T13:46:29+00:00",
            ]
        )


def test_refresh_script_rejects_invalid_top_level_json(tmp_path: Path, monkeypatch) -> None:
    module = _load_refresh_script()
    _patch_snapshot_paths(monkeypatch, module, tmp_path)
    input_path = _write_catalog(tmp_path / "catalog.json", ["not", "an", "object"])

    with pytest.raises(SystemExit, match="top-level object"):
        module.main(["--input", str(input_path), "--upstream-commit-sha", _VALID_SHA])


def test_refresh_script_rejects_invalid_model_entry_fields(tmp_path: Path, monkeypatch) -> None:
    module = _load_refresh_script()
    _patch_snapshot_paths(monkeypatch, module, tmp_path)
    input_path = _write_catalog(
        tmp_path / "catalog.json",
        {
            "openai/gpt-5-nano": {
                "max_tokens": "not-an-int",
            }
        },
    )

    with pytest.raises(SystemExit, match="invalid 'max_tokens'"):
        module.main(["--input", str(input_path), "--upstream-commit-sha", _VALID_SHA])


def test_refresh_script_accepts_upstream_zero_and_integral_float_capacity_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_refresh_script()
    snapshot_path, _meta_path = _patch_snapshot_paths(monkeypatch, module, tmp_path)
    input_path = _write_catalog(
        tmp_path / "catalog.json",
        {
            "embedding-model": {
                "max_tokens": 0,
                "max_input_tokens": 0,
                "max_output_tokens": 0,
            },
            "large-chat-model": {
                "max_tokens": 2000000.0,
                "max_input_tokens": 2000000.0,
                "max_output_tokens": 2000000.0,
            },
        },
    )

    result = module.main(["--input", str(input_path), "--upstream-commit-sha", _VALID_SHA])

    assert result == 0
    assert snapshot_path.read_text(encoding="utf-8") == input_path.read_text(encoding="utf-8")


def test_refresh_script_rejects_floating_refs_without_explicit_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_refresh_script()
    _patch_snapshot_paths(monkeypatch, module, tmp_path)
    input_path = _write_catalog(tmp_path / "catalog.json", _base_catalog())

    with pytest.raises(SystemExit, match="commit-specific unless --allow-floating-ref"):
        module.main(
            [
                "--input",
                str(input_path),
                "--upstream-commit-sha",
                _VALID_SHA,
                "--upstream-ref",
                "main",
            ]
        )


def test_refresh_script_writes_stable_commit_oriented_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_refresh_script()
    snapshot_path, meta_path = _patch_snapshot_paths(monkeypatch, module, tmp_path)
    monkeypatch.setattr(module, "_iso_now", lambda: "2026-03-25T13:46:29Z")
    input_path = _write_catalog(tmp_path / "catalog.json", _base_catalog())

    result = module.main(
        [
            "--input",
            str(input_path),
            "--upstream-commit-sha",
            _VALID_SHA,
        ]
    )

    assert result == 0
    assert snapshot_path.read_text(encoding="utf-8") == input_path.read_text(encoding="utf-8")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["schema_version"] == 1
    assert meta["refresh_policy"] == "manual_reviewed_only"
    assert meta["source"] == "bundled_litellm_snapshot"
    assert meta["snapshot"]["fetched_at_utc"] == "2026-03-25T13:46:29Z"
    assert meta["upstream"]["commit_sha"] == _VALID_SHA
    assert meta["upstream"]["blob_url"].endswith(
        f"/blob/{_VALID_SHA}/model_prices_and_context_window.json"
    )
    assert "ref" not in meta["upstream"]


def test_refresh_script_allows_floating_ref_only_when_explicitly_requested(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_refresh_script()
    _patch_snapshot_paths(monkeypatch, module, tmp_path)
    monkeypatch.setattr(module, "_iso_now", lambda: "2026-03-25T13:46:29Z")
    input_path = _write_catalog(tmp_path / "catalog.json", _base_catalog())

    module.main(
        [
            "--input",
            str(input_path),
            "--upstream-commit-sha",
            _VALID_SHA,
            "--upstream-ref",
            "main",
            "--allow-floating-ref",
        ]
    )

    meta_path = tmp_path / "litellm_model_prices_snapshot.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["upstream"]["ref"] == "main"
