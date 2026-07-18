from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (_repo_root() / path).read_text(encoding="utf-8")


def test_tracked_paths_are_case_unique_for_cross_platform_checkout() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    tracked_paths = [path for path in result.stdout.split("\0") if path]
    by_casefold: dict[str, str] = {}
    collisions: list[str] = []
    for path in tracked_paths:
        folded = path.casefold()
        existing = by_casefold.setdefault(folded, path)
        if existing != path:
            collisions.append(f"{existing} <> {path}")

    assert not collisions, "Tracked paths differ only by case: " + ", ".join(collisions)


def test_python_runtime_baseline_is_documented_and_ci_aligned() -> None:
    repo_root = _repo_root()
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    readme = _read("README.md")
    server_doc = _read("docs/server.md")
    contributing = _read("CONTRIBUTING.md")
    ci = _read(".github/workflows/ci.yml")

    assert pyproject["project"]["requires-python"] == ">=3.11"
    assert pyproject["tool"]["ruff"]["target-version"] == "py311"
    assert '"python-version":"3.11"' in ci
    assert '"python-version":"3.12"' in ci
    assert "Sylliptor requires Python 3.11 or newer" in readme
    assert "Use Python 3.11+" in contributing
    assert "Python 3.11 or newer" in server_doc


def test_readme_keeps_public_launch_surface() -> None:
    readme = _read("README.md")

    assert "SYLLIPTOR" in readme
    assert (
        "raw.githubusercontent.com/AlysisAi/Sylliptor/main/docs/assets/sylliptor-demo.gif" in readme
    )
    assert "https://sylliptor.alysisai.com/" in readme
    assert 'href="https://github.com/AlysisAi/Sylliptor/tree/main/docs">Docs</a>' in readme
    assert (
        'href="https://github.com/AlysisAi/Sylliptor/blob/main/CHANGELOG.md">Changelog</a>'
        in readme
    )
    assert 'href="https://github.com/sponsors/AlysisAi"' in readme
    assert "pipx install sylliptor-agent-cli" in readme
    assert "Apache License 2.0" in readme


def test_governance_and_packaging_metadata_are_public_ready() -> None:
    repo_root = _repo_root()
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(pyproject["project"]["dependencies"])

    assert pyproject["project"]["license"]["text"] == "Apache-2.0"
    assert (
        "License :: OSI Approved :: Apache Software License" in pyproject["project"]["classifiers"]
    )
    assert "click>=8.0.0" in dependencies
    assert "packaging>=23.0" in dependencies
    assert (repo_root / "LICENSE").read_text(encoding="utf-8").startswith("Apache License")
    assert "Copyright 2026 AlysisAi" in _read("NOTICE")
    assert "sylliptor-contributor-pack" not in _read("CONTRIBUTING.md")
    assert "products@alysisai.com" in _read("SECURITY.md")
    assert pyproject["project"]["urls"]["Documentation"].endswith("/tree/main/docs")


def test_docs_index_links_only_existing_public_docs() -> None:
    repo_root = _repo_root()
    docs_index = _read("docs/README.md")
    linked_paths = re.findall(r"\]\(([^)]+\.md)\)", docs_index)

    assert "reference.md" in docs_index
    assert "architecture.md" in docs_index
    assert "quickstart.md" in docs_index
    assert "security_model.md" in docs_index
    assert "subagents.md" in docs_index
    for raw_path in linked_paths:
        target = (repo_root / "docs" / raw_path).resolve()
        if raw_path.startswith("../"):
            target = (repo_root / "docs" / raw_path).resolve()
        assert target.exists(), raw_path


def test_public_docs_cover_core_user_and_contributor_topics() -> None:
    reference = _read("docs/reference.md")
    architecture = _read("docs/architecture.md")
    subagents = _read("docs/subagents.md")
    mcp = _read("docs/mcp.md")
    skills = _read("docs/skills.md")
    forge = _read("docs/forge.md")
    security = _read("docs/security_model.md")

    assert "readonly" in reference
    assert "review" in reference
    assert "fullaccess" in reference
    assert "runtime_kind" in reference
    assert "High-level flow" in architecture
    assert "workspace_root" in architecture
    assert "active_workdir" in architecture
    assert "Subagents" in subagents
    for name in (
        "explorer",
        "implementer",
        "frontend-engineer",
        "debugger",
        "code-reviewer",
        "test-strategist",
        "visual-designer",
    ):
        assert name in subagents
    assert "image_generation.enabled" in subagents
    assert "Visual QA" in subagents
    assert "Streamable HTTP" in mcp
    assert "OAuth" in mcp
    assert "Skills" in skills
    assert "skill_read" in skills
    assert "Forge" in forge
    assert "forge exec" in forge
    assert "verification" in forge.lower()
    assert "safe HTTP" in security
    assert "MCP" in security


def test_internal_cleanup_artifacts_stay_absent() -> None:
    repo_root = _repo_root()
    absent_paths = [
        "RE" + "FACTOR_NOTES.md",
        "RE" + "FACTOR_PLAN.md",
        "qa_" + "reports",
        "scripts/skills_" + "do" + "gfood_campaign.py",
        "scripts/skills_" + "do" + "gfood_validation_round.py",
        "docs/SANDBOX.md",
        "docs/FORGE.md",
        "docs/skills_" + "evals.md",
        "docs/internal/ui_smoke_checklist.md",
        "docs/forge_plan_phase.md",
        "docs/forge_execution_phase.md",
        "docs/forge_swarm.md",
    ]

    existing_paths = {path.relative_to(repo_root).as_posix() for path in repo_root.rglob("*")}
    for path in absent_paths:
        assert path not in existing_paths, path

    terminals = _read("docs/terminals.md")
    assert "shell_background" in terminals
    assert "shell_output" in terminals
    assert "shell_kill" in terminals
    assert "shell_list" in terminals


def test_source_urls_and_container_docs_point_to_alysisai() -> None:
    branding = _read("src/sylliptor_agent_cli/branding.py")
    sandbox = _read("docs/shell_sandbox.md")

    assert 'PROJECT_SOURCE_URL = "https://github.com/AlysisAi/Sylliptor"' in branding
    assert "ghcr.io/alysisai/sylliptor-sandbox" in sandbox
    assert "ap" + "fivos/sylliptor" not in branding
    assert "ghcr.io/ap" + "fivos" not in sandbox
