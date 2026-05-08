from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from ..config import (
    normalize_verify_module_invocation,
    split_verify_command_parts,
    strip_verify_runner_prefix,
)
from .prompt_context import _normalized_verify_commands


def _normalize_shell_command_for_match(raw: str) -> str:
    return " ".join(str(raw or "").casefold().split())


_VERIFICATION_ENV_ASSIGNMENT_RE = re.compile(r"^[a-z_][a-z0-9_]*=.*$")


_DISALLOWED_VERIFICATION_SHELL_TOKENS = {"||", "&&", ";", "|", "&"}


_NON_VERIFICATION_META_OPTIONS = {"--help", "-h", "--version"}


_PYTEST_NON_VERIFICATION_OPTIONS = {"--fixtures", "--markers", "--collect-only", "--co"}


_PYTEST_NON_EXECUTING_OPTIONS = {"--setup-plan"}


_CARGO_TEST_NON_EXECUTING_OPTIONS = {"--no-run", "--list"}


_GO_TEST_NON_EXECUTING_OPTIONS = {"-c", "-list"}


_RUFF_CHECK_NON_EXECUTING_OPTIONS = {"--fix", "--fix-only"}


_MYPY_NON_EXECUTING_OPTIONS = {"--install-types"}


_PYTEST_REPORTER_OPTIONS = {"-q", "--quiet", "--verbose"}


@dataclass(frozen=True)
class VerificationCommandShape:
    family: str
    args: tuple[str, ...]
    options: tuple[str, ...]
    positionals: tuple[str, ...]


def _command_options_include(options: tuple[str, ...], flag: str) -> bool:
    return any(option == flag or option.startswith(f"{flag}=") for option in options)


def _verification_shape_is_real_execution_mode(shape: VerificationCommandShape) -> bool:
    if shape.family == "pytest":
        return not any(
            _command_options_include(shape.options, flag) for flag in _PYTEST_NON_EXECUTING_OPTIONS
        )
    if shape.family == "cargo:test":
        return not any(
            _command_options_include(shape.options, flag)
            for flag in _CARGO_TEST_NON_EXECUTING_OPTIONS
        )
    if shape.family == "go:test":
        if any(
            _command_options_include(shape.options, flag) for flag in _GO_TEST_NON_EXECUTING_OPTIONS
        ):
            return False
        for idx, arg in enumerate(shape.args):
            if arg == "-run":
                if idx + 1 >= len(shape.args):
                    return False
                if shape.args[idx + 1] == "^$":
                    return False
            elif arg.startswith("-run=") and arg.partition("=")[2] == "^$":
                return False
        return True
    if shape.family == "ruff:check":
        return not any(
            _command_options_include(shape.options, flag)
            for flag in _RUFF_CHECK_NON_EXECUTING_OPTIONS
        )
    if shape.family == "mypy":
        return not any(
            _command_options_include(shape.options, flag) for flag in _MYPY_NON_EXECUTING_OPTIONS
        )
    return True


def _looks_like_env_assignment_token(part: str) -> bool:
    return bool(_VERIFICATION_ENV_ASSIGNMENT_RE.match(part))


def _strip_verification_env_prefix(parts: list[str]) -> list[str] | None:
    if not parts:
        return None
    out = list(parts)
    if out[0] == "env":
        out = out[1:]
        if not out or out[0].startswith("-"):
            return None
    while out and _looks_like_env_assignment_token(out[0]):
        out = out[1:]
    return out or None


def _strip_verification_runner_prefix(parts: list[str]) -> list[str] | None:
    return strip_verify_runner_prefix(parts)


def _normalize_and_unwrap_verification_command(raw: str) -> str | None:
    normalized = _normalize_shell_command_for_match(raw)
    if not normalized:
        return None

    while True:
        wrapped = _unwrap_shell_wrapper_command(normalized)
        if not wrapped or wrapped == normalized:
            break
        normalized = wrapped
    return normalized


def _has_disallowed_shell_control_flow(raw: str) -> bool:
    if "\n" in str(raw) or "\r" in str(raw):
        return True
    normalized = _normalize_and_unwrap_verification_command(raw)
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
    if len(tokens) >= 4 and tokens[0] == "cd" and tokens[2] == "&&":
        tokens = tokens[3:]
        if not tokens:
            return True
    return any(token in _DISALLOWED_VERIFICATION_SHELL_TOKENS for token in tokens)


def _canonicalize_verification_command_for_match(raw: str) -> str | None:
    normalized = _normalize_and_unwrap_verification_command(raw)
    if not normalized or _has_disallowed_shell_control_flow(normalized):
        return None

    parts = split_verify_command_parts(normalized)
    if not parts:
        return None
    if len(parts) >= 4 and parts[0] == "cd" and parts[2] == "&&":
        parts = parts[3:]
        if not parts:
            return None

    while True:
        changed = False
        stripped_env = _strip_verification_env_prefix(parts)
        if stripped_env is None:
            return None
        if stripped_env != parts:
            parts = stripped_env
            changed = True

        stripped_runner = _strip_verification_runner_prefix(parts)
        if stripped_runner is None:
            return None
        if stripped_runner != parts:
            parts = stripped_runner
            changed = True

        if not changed:
            break

    parts = normalize_verify_module_invocation(parts)

    return " ".join(parts) if parts else None


def _split_verification_shape_args(args: list[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    options: list[str] = []
    positionals: list[str] = []
    for arg in args:
        if arg == "--":
            continue
        if arg.startswith("-") and arg != "-":
            options.append(arg)
            continue
        positionals.append(arg)
    return tuple(options), tuple(positionals)


def _parse_verification_command_shape(raw: str) -> VerificationCommandShape | None:
    canonical = _canonicalize_verification_command_for_match(raw)
    if not canonical:
        return None
    try:
        parts = shlex.split(canonical, posix=True)
    except ValueError:
        return None
    if not parts:
        return None

    head = parts[0]
    tail = parts[1:]
    family: str
    args: list[str]

    if head in {"pytest", "py.test"}:
        family = "pytest"
        args = tail
    elif head == "unittest":
        family = "unittest"
        args = tail
    elif head == "mypy":
        family = "mypy"
        args = tail
    elif head == "go" and tail and tail[0] == "test":
        family = "go:test"
        args = tail[1:]
    elif head == "cargo" and tail and tail[0] in {"test", "check"}:
        family = f"cargo:{tail[0]}"
        args = tail[1:]
    elif head in {"npm", "pnpm", "yarn"} and tail and tail[0] == "test":
        family = f"{head}:test"
        args = tail[1:]
    elif head in {"make", "just"} and tail and tail[0] in {"test", "check", "verify"}:
        family = f"{head}:{tail[0]}"
        args = tail[1:]
    elif head == "ruff" and tail and tail[0] == "check":
        family = "ruff:check"
        args = tail[1:]
    else:
        return None

    options, positionals = _split_verification_shape_args(args)
    if _NON_VERIFICATION_META_OPTIONS & set(options):
        return None
    if family == "pytest" and _PYTEST_NON_VERIFICATION_OPTIONS & set(options):
        return None
    shape = VerificationCommandShape(
        family=family,
        args=tuple(args),
        options=options,
        positionals=positionals,
    )
    if not _verification_shape_is_real_execution_mode(shape):
        return None
    return shape


def _verification_command_shapes_match(
    *,
    observed: VerificationCommandShape,
    expected: VerificationCommandShape,
) -> bool:
    if observed.family != expected.family:
        return False
    observed_options = set(observed.options)
    expected_options = set(expected.options)
    if observed.family == "pytest":
        observed_options = {
            option for option in observed_options if not _pytest_option_is_reporter_variant(option)
        }
        expected_options = {
            option for option in expected_options if not _pytest_option_is_reporter_variant(option)
        }
    return expected_options.issubset(observed_options)


def _pytest_option_is_reporter_variant(option: str) -> bool:
    return option in _PYTEST_REPORTER_OPTIONS or bool(re.fullmatch(r"-v+", option))


def _effective_verification_command_matches(
    *,
    normalized_cmd: str,
    known_verification_commands: list[str],
) -> bool:
    observed_canonical = _canonicalize_verification_command_for_match(normalized_cmd)
    if observed_canonical:
        for configured in known_verification_commands:
            expected_canonical = _canonicalize_verification_command_for_match(configured)
            if expected_canonical and observed_canonical == expected_canonical:
                return True

    observed = _parse_verification_command_shape(normalized_cmd)
    if observed is None:
        return False
    return any(
        _verification_command_shapes_match(observed=observed, expected=expected)
        for expected in (
            _parse_verification_command_shape(configured)
            for configured in known_verification_commands
        )
        if expected is not None
    )


def _verify_run_commands_match_effective_contract(
    *,
    requested_commands: list[str],
    effective_verification_commands: list[str],
) -> list[str]:
    known = _normalized_verify_commands(effective_verification_commands)
    if not known:
        return []
    incompatible: list[str] = []
    for command in requested_commands:
        normalized_command = _normalize_shell_command_for_match(command)
        if not _effective_verification_command_matches(
            normalized_cmd=normalized_command,
            known_verification_commands=known,
        ):
            incompatible.append(command)
    return incompatible


def _matching_effective_verification_commands(
    *,
    observed_command: str,
    effective_verification_commands: list[str] | None,
) -> set[str]:
    known = _normalized_verify_commands(effective_verification_commands or [])
    if not known:
        return set()
    normalized_observed = _normalize_shell_command_for_match(observed_command)
    observed_canonical = _canonicalize_verification_command_for_match(normalized_observed)
    if observed_canonical:
        exact_matches: set[str] = set()
        for configured in known:
            configured_canonical = _canonicalize_verification_command_for_match(configured)
            if configured_canonical and configured_canonical == observed_canonical:
                exact_matches.add(configured)
        if exact_matches:
            return exact_matches

    observed = _parse_verification_command_shape(normalized_observed)
    if observed is None:
        return set()
    matches: set[str] = set()
    for configured in known:
        expected = _parse_verification_command_shape(configured)
        if expected is None:
            continue
        if _verification_command_shapes_match(observed=observed, expected=expected):
            matches.add(configured)
    return matches


def _unwrap_shell_wrapper_command(normalized_cmd: str) -> str | None:
    if not normalized_cmd:
        return None
    try:
        parts = shlex.split(normalized_cmd, posix=True)
    except ValueError:
        return None
    if not parts:
        return None

    head = parts[0]
    if head in {"bash", "sh", "zsh"}:
        if len(parts) == 3 and parts[1] == "-lc":
            return _normalize_shell_command_for_match(parts[2])
        return None
    if head == "fish":
        if len(parts) == 3 and parts[1] == "-c":
            return _normalize_shell_command_for_match(parts[2])
        return None
    if head == "cmd":
        if len(parts) == 3 and parts[1] == "/c":
            return _normalize_shell_command_for_match(parts[2])
        return None
    if head in {"powershell", "pwsh"}:
        if len(parts) == 3 and parts[1] == "-command":
            return _normalize_shell_command_for_match(parts[2])
        return None
    return None


def _looks_like_verification_entrypoint(parts: list[str]) -> bool:
    if not parts:
        return False
    head = Path(parts[0]).name
    head_lower = head.lower()
    if head_lower in {
        "pytest",
        "py.test",
        "tox",
        "nox",
        "vitest",
        "jest",
        "mypy",
        "flake8",
        "pylint",
        "ruff",
        "rscript",
        "mix",
        "sbt",
        "swift",
        "composer",
    }:
        return True
    if len(parts) >= 2 and head_lower in {
        "make",
        "just",
        "npm",
        "pnpm",
        "yarn",
        "cargo",
        "go",
        "mvn",
        "mvnw",
        "gradle",
        "gradlew",
        "dotnet",
    }:
        return parts[1] in {"test", "check", "verify"}
    if len(parts) >= 2 and head_lower in {"python", "python3"} and parts[1] == "-m":
        return len(parts) >= 3 and parts[2] in {"pytest", "unittest"}
    if len(parts) >= 2 and head_lower == "ruby":
        return any(part.endswith("_test.rb") for part in parts[1:])
    if len(parts) >= 2 and head_lower in {"sh", "bash"}:
        return any("test" in Path(part).name for part in parts[1:])
    return False


def _marker_fallback_is_verification_attempt(normalized_cmd: str) -> bool:
    canonical = _canonicalize_verification_command_for_match(normalized_cmd)
    if not canonical:
        return False
    try:
        parts = shlex.split(canonical, posix=True)
    except ValueError:
        return False
    if _looks_like_verification_entrypoint(parts):
        return True
    return False


def _shell_command_is_verification_attempt(
    cmd: str,
    *,
    known_verification_commands: list[str] | None,
) -> bool:
    if _has_disallowed_shell_control_flow(cmd):
        return False
    normalized_cmd = _normalize_shell_command_for_match(cmd)
    if not normalized_cmd:
        return False
    known = _normalized_verify_commands(known_verification_commands or [])
    if known:
        return _effective_verification_command_matches(
            normalized_cmd=normalized_cmd,
            known_verification_commands=known,
        )
    return _marker_fallback_is_verification_attempt(normalized_cmd)
