from __future__ import annotations

import hashlib
import shlex
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class VerificationCommandExecutionMode(StrEnum):
    ARGV = "ARGV"
    TRUSTED_SHELL_EXPRESSION = "TRUSTED_SHELL_EXPRESSION"
    INTERPRETER_SNIPPET = "INTERPRETER_SNIPPET"
    INVALID = "INVALID"


class VerificationCommandProvenance(StrEnum):
    HOST_AUTHORITATIVE = "HOST_AUTHORITATIVE"
    EXPLICIT_USER_COMMAND = "EXPLICIT_USER_COMMAND"
    PREEXISTING_REPO_NATIVE = "PREEXISTING_REPO_NATIVE"
    PREEXISTING_TASK_CHECKER = "PREEXISTING_TASK_CHECKER"
    INFERRED_HEURISTIC = "INFERRED_HEURISTIC"


class VerificationCommandTrustLevel(StrEnum):
    TRUSTED = "TRUSTED"
    UNTRUSTED = "UNTRUSTED"


class VerificationCommandRequirement(StrEnum):
    REQUIRED = "REQUIRED"
    ADVISORY = "ADVISORY"


class VerificationCommandValidationStatus(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"


@dataclass(frozen=True)
class VerificationCommandSpec:
    command_id: str
    original_text: str
    display_text: str
    execution_mode: VerificationCommandExecutionMode
    provenance: VerificationCommandProvenance
    trust_level: VerificationCommandTrustLevel
    requirement: VerificationCommandRequirement
    working_directory: str = "."
    timeout_policy: str = "session_verification_timeout"
    acceptance_criterion_ids: tuple[str, ...] = tuple()
    validation_status: VerificationCommandValidationStatus = (
        VerificationCommandValidationStatus.VALID
    )
    rejection_reason: str = ""
    argv: tuple[str, ...] = field(default_factory=tuple)

    def as_payload(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "original_text": self.original_text,
            "display_text": self.display_text,
            "execution_mode": self.execution_mode.value,
            "provenance": self.provenance.value,
            "trust_level": self.trust_level.value,
            "requirement": self.requirement.value,
            "working_directory": self.working_directory,
            "timeout_policy": self.timeout_policy,
            "acceptance_criterion_ids": list(self.acceptance_criterion_ids),
            "validation_status": self.validation_status.value,
            "rejection_reason": self.rejection_reason,
            "argv": list(self.argv),
        }


_TRUSTED_SHELL_PROVENANCE = {
    VerificationCommandProvenance.HOST_AUTHORITATIVE,
    VerificationCommandProvenance.EXPLICIT_USER_COMMAND,
    VerificationCommandProvenance.PREEXISTING_REPO_NATIVE,
    VerificationCommandProvenance.PREEXISTING_TASK_CHECKER,
}
_DISALLOWED_VERIFICATION_SHELL_TOKENS = {"||", "&&", ";", "|", "&"}
_SUPPORTED_VERIFY_HEADS = {
    "pytest",
    "py.test",
    "unittest",
    "mypy",
    "ruff",
    "go",
    "cargo",
    "npm",
    "pnpm",
    "yarn",
    "make",
    "just",
}
_PYTHON_EXECUTABLES = {"python", "python3", "py"}


def build_verification_command_specs(
    commands: tuple[str, ...] | list[str],
    *,
    source: str,
    contract_type: str = "",
    acceptance_criterion_ids: tuple[str, ...] = tuple(),
) -> tuple[VerificationCommandSpec, ...]:
    return tuple(
        build_verification_command_spec(
            command,
            source=source,
            contract_type=contract_type,
            acceptance_criterion_ids=acceptance_criterion_ids,
            index=index,
        )
        for index, command in enumerate(commands, start=1)
        if str(command).strip()
    )


def build_verification_command_spec(
    command: str,
    *,
    source: str,
    contract_type: str = "",
    acceptance_criterion_ids: tuple[str, ...] = tuple(),
    index: int = 1,
) -> VerificationCommandSpec:
    original = str(command or "").strip()
    provenance = _provenance_for_source(source=source, contract_type=contract_type)
    trust_level = (
        VerificationCommandTrustLevel.TRUSTED
        if provenance in _TRUSTED_SHELL_PROVENANCE
        else VerificationCommandTrustLevel.UNTRUSTED
    )
    requirement = (
        VerificationCommandRequirement.ADVISORY
        if provenance == VerificationCommandProvenance.INFERRED_HEURISTIC
        and contract_type in {"generic_fallback", "unavailable"}
        else VerificationCommandRequirement.REQUIRED
    )
    parsed_parts: tuple[str, ...] = tuple()
    validation_status = VerificationCommandValidationStatus.VALID
    rejection_reason = ""
    execution_mode = VerificationCommandExecutionMode.ARGV
    if not original:
        validation_status = VerificationCommandValidationStatus.INVALID
        execution_mode = VerificationCommandExecutionMode.INVALID
        rejection_reason = "empty_command"
    else:
        try:
            parsed_parts = tuple(shlex.split(original, posix=True))
        except ValueError as exc:
            validation_status = VerificationCommandValidationStatus.INVALID
            execution_mode = VerificationCommandExecutionMode.INVALID
            rejection_reason = f"parse_error: {exc}"
        else:
            if not parsed_parts:
                validation_status = VerificationCommandValidationStatus.INVALID
                execution_mode = VerificationCommandExecutionMode.INVALID
                rejection_reason = "empty_command"
            elif _has_disallowed_shell_control_flow(original):
                if provenance in _TRUSTED_SHELL_PROVENANCE:
                    execution_mode = VerificationCommandExecutionMode.TRUSTED_SHELL_EXPRESSION
                else:
                    validation_status = VerificationCommandValidationStatus.INVALID
                    execution_mode = VerificationCommandExecutionMode.INVALID
                    rejection_reason = "untrusted_shell_expression"
            elif _is_python_interpreter_snippet(parsed_parts):
                execution_mode = VerificationCommandExecutionMode.INTERPRETER_SNIPPET
            elif _parse_verification_command_shape(original) is None and not _source_is_trusted(
                source=source,
                contract_type=contract_type,
            ):
                validation_status = VerificationCommandValidationStatus.INVALID
                execution_mode = VerificationCommandExecutionMode.INVALID
                rejection_reason = "unrecognized_inferred_command"
    command_id = _stable_command_id(
        source=source,
        contract_type=contract_type,
        command=original,
        index=index,
    )
    return VerificationCommandSpec(
        command_id=command_id,
        original_text=original,
        display_text=original,
        execution_mode=execution_mode,
        provenance=provenance,
        trust_level=trust_level,
        requirement=requirement,
        acceptance_criterion_ids=acceptance_criterion_ids,
        validation_status=validation_status,
        rejection_reason=rejection_reason,
        argv=parsed_parts if execution_mode == VerificationCommandExecutionMode.ARGV else tuple(),
    )


def trusted_shell_expression_commands(
    commands: tuple[str, ...] | list[str],
    *,
    source: str,
    contract_type: str = "",
) -> set[str]:
    return {
        _normalize_exact_command(spec.original_text)
        for spec in build_verification_command_specs(
            commands,
            source=source,
            contract_type=contract_type,
        )
        if spec.execution_mode == VerificationCommandExecutionMode.TRUSTED_SHELL_EXPRESSION
        and spec.validation_status == VerificationCommandValidationStatus.VALID
    }


def command_matches_trusted_shell_expression(
    observed_command: str,
    known_commands: tuple[str, ...] | list[str],
    *,
    source: str,
    contract_type: str = "",
) -> set[str]:
    observed = _normalize_exact_command(observed_command)
    if not observed:
        return set()
    trusted = trusted_shell_expression_commands(
        known_commands,
        source=source,
        contract_type=contract_type,
    )
    return {command for command in trusted if observed == command}


def command_specs_payload(specs: tuple[VerificationCommandSpec, ...]) -> list[dict[str, Any]]:
    return [spec.as_payload() for spec in specs]


def _source_is_trusted(*, source: str, contract_type: str) -> bool:
    provenance = _provenance_for_source(source=source, contract_type=contract_type)
    return provenance in _TRUSTED_SHELL_PROVENANCE


def _provenance_for_source(
    *,
    source: str,
    contract_type: str,
) -> VerificationCommandProvenance:
    source = str(source or "")
    contract_type = str(contract_type or "")
    if source == "environment.authoritative_verification_commands" or contract_type in {
        "authoritative_override",
        "explicit_override",
    }:
        return (
            VerificationCommandProvenance.HOST_AUTHORITATIVE
            if source == "environment.authoritative_verification_commands"
            else VerificationCommandProvenance.EXPLICIT_USER_COMMAND
        )
    if source == "cli.verify_cmd" or source.startswith("task_refinement.explicit"):
        return VerificationCommandProvenance.EXPLICIT_USER_COMMAND
    if source == "task_refinement.explicit_user_command" or contract_type == "task_acceptance":
        return VerificationCommandProvenance.EXPLICIT_USER_COMMAND
    if source == "repo_scan.likely_test_commands" or contract_type == "repo_native":
        return VerificationCommandProvenance.PREEXISTING_REPO_NATIVE
    if source.startswith("task_refinement.") and contract_type == "task_inferred":
        return VerificationCommandProvenance.INFERRED_HEURISTIC
    return VerificationCommandProvenance.INFERRED_HEURISTIC


def _normalize_exact_command(command: str) -> str:
    return " ".join(str(command or "").strip().split())


def _has_disallowed_shell_control_flow(raw: str) -> bool:
    if "\n" in str(raw) or "\r" in str(raw):
        return True
    normalized = _normalize_exact_command(raw)
    if not normalized:
        return True
    try:
        lexer = shlex.shlex(normalized, posix=True, punctuation_chars="|&;")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return True
    if not tokens:
        return True
    return any(token in _DISALLOWED_VERIFICATION_SHELL_TOKENS for token in tokens)


def _parse_verification_command_shape(raw: str) -> tuple[str, ...] | None:
    try:
        parts = shlex.split(raw, posix=True)
    except ValueError:
        return None
    if not parts:
        return None
    while parts and _looks_like_env_assignment(parts[0]):
        parts = parts[1:]
    if not parts:
        return None
    if parts[0] == "env":
        parts = parts[1:]
        while parts and _looks_like_env_assignment(parts[0]):
            parts = parts[1:]
    if (
        len(parts) >= 3
        and Path(parts[0]).name.casefold() in _PYTHON_EXECUTABLES
        and parts[1] == "-m"
    ):
        parts = [parts[2], *parts[3:]]
    if not parts:
        return None
    head = Path(parts[0]).name.casefold()
    if head not in _SUPPORTED_VERIFY_HEADS:
        return None
    if head == "ruff" and (len(parts) < 2 or parts[1] != "check"):
        return None
    if head in {"go", "cargo", "npm", "pnpm", "yarn"} and len(parts) < 2:
        return None
    if head == "go" and parts[1] != "test":
        return None
    if head == "cargo" and parts[1] not in {"test", "check"}:
        return None
    if head in {"npm", "pnpm", "yarn"} and parts[1] != "test":
        return None
    return tuple(parts)


def _looks_like_env_assignment(part: str) -> bool:
    if "=" not in part or not part:
        return False
    name = part.split("=", 1)[0]
    return name.replace("_", "a").isalnum() and not name[0].isdigit()


def _is_python_interpreter_snippet(parts: tuple[str, ...]) -> bool:
    return (
        len(parts) >= 3
        and Path(parts[0]).name.casefold() in _PYTHON_EXECUTABLES
        and parts[1] == "-c"
    )


def _stable_command_id(
    *,
    source: str,
    contract_type: str,
    command: str,
    index: int,
) -> str:
    digest = hashlib.sha256(f"{source}\0{contract_type}\0{index}\0{command}".encode()).hexdigest()
    return f"vc_{digest[:16]}"
