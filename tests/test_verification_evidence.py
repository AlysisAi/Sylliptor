from __future__ import annotations

from pathlib import Path

import pytest

from sylliptor_agent_cli.agent.acceptance_contract import (
    EvidenceOrigin,
    build_acceptance_contract,
    classify_evidence_origin,
)
from sylliptor_agent_cli.agent_loop import (
    VerificationEvidenceCategory,
    _shell_command_is_verification_attempt,
    classify_verification_evidence,
)


def test_authoritative_matching_command_is_authoritative_evidence() -> None:
    evidence = classify_verification_evidence(
        "PYTHONPATH=src pytest tests/test_cli.py -q",
        known_verification_commands=["pytest -q"],
        authoritative=True,
        exit_code=0,
        output="1 passed\n",
        real_execution=True,
    )

    assert evidence.category == VerificationEvidenceCategory.AUTHORITATIVE
    assert evidence.allowed_to_satisfy_contract is True
    assert evidence.covered_verification_commands == ("pytest -q",)
    assert evidence.reason == "matched_authoritative_contract"


def test_trusted_pipeline_matching_command_is_contract_evidence() -> None:
    command = "printf 'a\\nb\\n' | tail -n 1"

    evidence = classify_verification_evidence(
        command,
        known_verification_commands=[command],
        authoritative=False,
        exit_code=0,
        output="b\n",
        real_execution=True,
    )

    assert evidence.category == VerificationEvidenceCategory.REPO_NATIVE
    assert evidence.allowed_to_satisfy_contract is True
    assert evidence.covered_verification_commands == (command,)
    assert evidence.reason == "matched_contract"


def test_trusted_command_that_mutates_source_is_supplemental_until_clean_rerun() -> None:
    command = "python check.py"

    mutating = classify_verification_evidence(
        command,
        known_verification_commands=[command],
        exit_code=0,
        output="ok\n",
        real_execution=True,
        material_touched_paths={"src/app.py"},
    )
    clean = classify_verification_evidence(
        command,
        known_verification_commands=[command],
        exit_code=0,
        output="ok\n",
        real_execution=True,
        material_touched_paths=set(),
    )

    assert mutating.allowed_to_satisfy_contract is False
    assert mutating.reason == "mutated_material_paths"
    assert clean.allowed_to_satisfy_contract is True
    assert clean.covered_verification_commands == (command,)


def test_authoritative_contract_rejects_unrelated_task_specific_check(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text("print('ok')\n", encoding="utf-8")

    evidence = classify_verification_evidence(
        "python check.py",
        known_verification_commands=["pytest -q"],
        authoritative=True,
        changed_paths={"check.py"},
        exit_code=0,
        output="ok\n",
        root=tmp_path,
    )

    assert evidence.category == VerificationEvidenceCategory.TASK_ACCEPTANCE
    assert evidence.allowed_to_satisfy_contract is False
    assert evidence.supplemental_only is True
    assert evidence.covered_verification_commands == ()


def test_partial_authoritative_coverage_remains_partial() -> None:
    evidence = classify_verification_evidence(
        "pytest tests/test_cli.py -q",
        known_verification_commands=["pytest -q", "ruff check ."],
        authoritative=True,
        exit_code=0,
        output="1 passed\n",
        real_execution=True,
    )

    assert evidence.category == VerificationEvidenceCategory.AUTHORITATIVE
    assert evidence.allowed_to_satisfy_contract is True
    assert evidence.covered_verification_commands == ("pytest -q",)


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q",
        "python -m pytest tests/test_cli.py -q",
        "python -m unittest -v",
        "go test ./...",
        "cargo test",
        "npm test",
        "pnpm test",
        "yarn test",
        "make verify",
        "just check",
        "ruff check src",
    ],
)
def test_repo_native_commands_are_classified_without_known_contract(command: str) -> None:
    evidence = classify_verification_evidence(
        command,
        exit_code=0,
        output="ok\n",
        real_execution=True,
    )

    assert evidence.category == VerificationEvidenceCategory.REPO_NATIVE
    assert evidence.allowed_to_satisfy_contract is True


@pytest.mark.parametrize(
    "command",
    [
        "pytest --collect-only",
        "pytest -q --co",
        "python --version",
        "pytest --help",
        "go test -run '^$' ./...",
    ],
)
def test_non_executing_repo_native_forms_do_not_count(command: str) -> None:
    evidence = classify_verification_evidence(
        command,
        exit_code=0,
        output="collected 0 items\n",
    )

    assert evidence.category == VerificationEvidenceCategory.NOT_VERIFICATION
    assert evidence.allowed_to_satisfy_contract is False


def test_changed_python_script_counts_as_task_acceptance(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text("print('ok')\n", encoding="utf-8")

    evidence = classify_verification_evidence(
        "python check.py",
        changed_paths={"check.py"},
        exit_code=0,
        output="ok\n",
        root=tmp_path,
    )

    assert evidence.category == VerificationEvidenceCategory.TASK_ACCEPTANCE
    assert evidence.allowed_to_satisfy_contract is True
    assert evidence.reason == "changed_repo_local_script_execution"


def test_changed_r_script_counts_as_task_acceptance(tmp_path: Path) -> None:
    (tmp_path / "solution.R").write_text("print('ok')\n", encoding="utf-8")

    evidence = classify_verification_evidence(
        "Rscript solution.R",
        changed_paths={"solution.R"},
        exit_code=0,
        output="ok\n",
        root=tmp_path,
    )

    assert evidence.category == VerificationEvidenceCategory.TASK_ACCEPTANCE
    assert evidence.allowed_to_satisfy_contract is True


def test_validation_script_path_counts_as_task_acceptance(tmp_path: Path) -> None:
    (tmp_path / "checks" / "validate_output.py").parent.mkdir()
    (tmp_path / "checks" / "validate_output.py").write_text("print('ok')\n", encoding="utf-8")

    evidence = classify_verification_evidence(
        "python checks/validate_output.py",
        exit_code=0,
        output="ok\n",
        root=tmp_path,
    )

    assert evidence.category == VerificationEvidenceCategory.TASK_ACCEPTANCE
    assert evidence.reason == "repo_local_validation_script"


@pytest.mark.parametrize("command", ["diff expected.txt actual.txt", "cmp expected.bin actual.bin"])
def test_diff_and_cmp_count_as_task_acceptance(command: str) -> None:
    evidence = classify_verification_evidence(command, exit_code=0, output="")

    assert evidence.category == VerificationEvidenceCategory.TASK_ACCEPTANCE
    assert evidence.allowed_to_satisfy_contract is True
    assert evidence.reason == "real_output_comparison"


def test_arbitrary_interpreter_invocation_without_context_does_not_count(tmp_path: Path) -> None:
    (tmp_path / "unrelated.py").write_text("print('ok')\n", encoding="utf-8")

    evidence = classify_verification_evidence(
        "python unrelated.py",
        exit_code=0,
        output="ok\n",
        root=tmp_path,
    )

    assert evidence.category == VerificationEvidenceCategory.NOT_VERIFICATION
    assert evidence.allowed_to_satisfy_contract is False


@pytest.mark.parametrize(
    "command",
    [
        "echo success",
        "cat output.txt",
        "ls",
        "pwd",
        "python --version",
        "pytest --collect-only",
        "bash -lc 'pytest -q || true'",
    ],
)
def test_false_positive_commands_do_not_count(command: str) -> None:
    evidence = classify_verification_evidence(command, exit_code=0, output="success\n")

    assert evidence.allowed_to_satisfy_contract is False


def test_material_mutating_verifier_is_not_allowed_to_satisfy_contract() -> None:
    evidence = classify_verification_evidence(
        "pytest -q",
        known_verification_commands=["pytest -q"],
        exit_code=0,
        output="ok\n",
        real_execution=True,
        material_touched_paths={"src/app.py"},
    )

    assert evidence.category == VerificationEvidenceCategory.REPO_NATIVE
    assert evidence.allowed_to_satisfy_contract is False
    assert evidence.reason == "mutated_material_paths"


def test_boolean_shell_helper_preserves_contract_matching_behavior() -> None:
    assert _shell_command_is_verification_attempt(
        "pytest tests/test_cli.py -q",
        known_verification_commands=["pytest -q"],
    )
    assert not _shell_command_is_verification_attempt(
        "python check.py",
        known_verification_commands=["pytest -q"],
    )
    assert _shell_command_is_verification_attempt(
        "python check.py",
        known_verification_commands=None,
    )


def test_acceptance_provenance_identifies_preexisting_checker(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text("print('ok')\n", encoding="utf-8")
    contract = build_acceptance_contract(
        root=tmp_path,
        instruction="Implement the requested behavior.",
    )

    origin = classify_evidence_origin(
        contract=contract,
        command="python check.py",
        touched_paths=set(),
    )

    assert origin == EvidenceOrigin.PREEXISTING_TASK_CHECKER
