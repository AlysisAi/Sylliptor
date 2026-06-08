from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .file_classification import INFERRED_FILE_EXTENSIONS
from .git_safe import build_git_process_env
from .runtime_artifacts import ROOT_RUNTIME_ARTIFACT_DIR_NAMES, is_runtime_artifact_path

_AGENT_INTERNAL_SCOPE_PREFIXES = tuple(sorted(ROOT_RUNTIME_ARTIFACT_DIR_NAMES | {".forge"}))
_IGNORED_INTERNAL_PREFIXES = _AGENT_INTERNAL_SCOPE_PREFIXES
_SPECIAL_FILENAMES = {
    "README",
    "README.md",
    "Dockerfile",
    "Makefile",
    "LICENSE",
    "NOTICE",
    "CHANGELOG",
}
_README_ALIAS_FILENAMES = frozenset({"README", "README.md"})
_BROAD_NON_PACKAGE_DIR_NAMES = {
    "app",
    "apps",
    "bin",
    "doc",
    "docs",
    "example",
    "examples",
    "lib",
    "scripts",
    "src",
    "tool",
    "tools",
}
_INFERRED_FILE_EXTENSIONS = INFERRED_FILE_EXTENSIONS
_PATH_HINT_RE = re.compile(
    r"(?<![\w/-])("
    r"(?:\.[A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)*)"
    r"|(?:[A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)+)"
    r"|(?:[A-Za-z0-9_.-]+\.[A-Za-z0-9_*?\[\]-]+)"
    r"|(?:README(?:\.md)?)"
    r"|(?:Dockerfile|Makefile|LICENSE|NOTICE|CHANGELOG)"
    r")(?![\w/-])"
)
_PATH_HINT_CONTEXT_SPLIT_RE = re.compile(r"[\n;]+|(?<=[.!?])\s+")
_FORBIDDEN_PATH_DIRECT_PREFIX_RE = re.compile(
    r"(?:^|[\s,(])(?:(?:do\s+not|don't|must\s+not|never)\s+"
    r"(?:edit|modify|change|touch|write(?:\s+to)?|update|delete|remove|create)"
    r"|without\s+(?:touching|modifying|changing|writing\s+to)"
    r"|not\s+(?:touch|read|write|include)"
    r"|no\s+longer\s+read"
    r"|(?:exclude|ignore|skip)"
    r"|(?:leave|keep)\s+(?:the\s+)?(?:file\s+)?"
    r"|(?:preserve|retain|maintain)\s+(?:the\s+)?(?:file\s+)?)\s+"
    r"(?:the\s+)?(?:(?:untracked|tracked|existing|current|local|generated|root|"
    r"workspace|repo|repository)\s+){0,3}(?:file\s+)?$",
    re.IGNORECASE,
)
_FORBIDDEN_PATH_DIRECT_OBJECT_RE = re.compile(
    r"\b(?:preserve|retain|maintain|leave|keep)\s+(?:the\s+)?"
    r"(?:(?:untracked|tracked|existing|current|local|generated|root|workspace|repo|repository)\s+){0,3}"
    r"(?:file\s+)?$",
    re.IGNORECASE,
)
_FORBIDDEN_PATH_SUFFIX_MARKER_RE = re.compile(
    r"^\s*(?:file\s+)?(?:is|are|must\s+remain|should\s+remain|stays?|stay|left)?\s*"
    r"(?:not\s+)?(?:untouched|unchanged|unmodified|preserved|not\s+modified|"
    r"not\s+changed|not\s+touched)\b"
    r"|^\s*(?:file\s+)?not\s+(?:modified|changed|touched|written)\b",
    re.IGNORECASE,
)
_CONDITIONAL_FORBIDDEN_PATH_EXCEPTION_RE = re.compile(
    r"\bunless\s+(?:a\s+|the\s+)?(?:genuine\s+|real\s+|actual\s+)?"
    r"(?:bug|defect|implementation\s+bug|product\s+bug)\s+"
    r"(?:is\s+)?(?:found|discovered|identified|confirmed)\b"
    r"|\bunless\s+(?:it\s+is\s+)?(?:necessary|required|needed)\s+"
    r"(?:to\s+fix|for\s+the\s+fix|to\s+make\s+verification\s+pass)\b"
    r"|\bexcept\s+(?:when|if)\s+(?:it\s+is\s+)?(?:necessary|required|needed)\b",
    re.IGNORECASE,
)
_FORBIDDEN_PATH_MARKER_WINDOW_CHARS = 100
_NON_MATERIAL_ROOT_SCRATCH_FILENAMES = frozenset(
    {
        "command_output.txt",
        "output.txt",
        "pip_err.txt",
        "pip_install_out.txt",
        "pip_log.txt",
        "pip_out.txt",
        "pip_output.txt",
        "pytest_output.txt",
        "pytest_results.txt",
        "shell_output.txt",
        "stderr.txt",
        "stdout.txt",
        "test_output.txt",
        "test_results.txt",
        "wheel_log.txt",
        "wheel_output.txt",
    }
)
_NON_MATERIAL_ROOT_SCRATCH_SUFFIXES = (
    "-output.txt",
    "-results.txt",
    "_output.txt",
    "_results.txt",
    ".log",
)
_NON_MATERIAL_ROOT_SCRATCH_RE = re.compile(
    r"^(?:pytest|test|shell|command|run|stdout|stderr|debug|diagnostic|tmp|temp|"
    r"pip|uv|poetry|python|wheel|build|install|package|npm|pnpm|yarn|node|cargo|go)"
    r"[_-](?:out|output|stdout|stderr|err|errors?|log|logs|full|results?|dump|"
    r"lines[_-]dump|install[_-]out)"
    r"(?:\d+)?\.txt$"
    r"|^_(?:(?:tmp|temp|debug|diag|diagnostic|out|output|content|pytest|test)"
    r"(?:\d+|[_-][a-z0-9][a-z0-9_-]*)?|d\d+)\.txt$"
)
_NON_MATERIAL_ROOT_STATE_FILE_RE = re.compile(
    r"^\.(?:[a-z0-9_-]*"
    r"(?:state|data|db|store|storage|cache|habits?|todos?|tasks?|notes?|items?)"
    r"[a-z0-9_-]*)\.json$"
)
_EXTENSIONLESS_FILE_RE = re.compile(r"[A-Za-z0-9_-]+")
_CARGO_MANIFEST_FILENAME = "Cargo.toml"
_CARGO_LOCK_FILENAME = "Cargo.lock"
_CARGO_WORKSPACE_RE = re.compile(r"(?m)^\s*\[workspace\]\s*$")
_CARGO_PACKAGE_RE = re.compile(r"(?m)^\s*\[package\]\s*$")
_RUST_SCOPE_DIR_NAMES = frozenset({"src", "tests", "test", "benches", "examples"})
_RUST_ENTRYPOINT_FILENAMES = ("lib.rs", "main.rs")
_READ_ONLY_GIT_TIMEOUT_S = 5.0
SCRATCH_ARTIFACT_ENV = "SYLLIPTOR_ARTIFACT_SCRATCH_DIR"

SCOPE_CLASS_IN_SCOPE = "in_scope"
SCOPE_CLASS_EXPECTED_COMPANION = "expected_companion_file"
SCOPE_CLASS_SCRATCH_ARTIFACT = "scratch_diagnostic_artifact"
SCOPE_CLASS_LIKELY_MISSING_SCOPE = "likely_legitimate_missing_scope"
SCOPE_CLASS_DANGEROUS_UNRELATED = "dangerous_unrelated_path"
SCOPE_CLASS_FORBIDDEN = "forbidden_path_violation"


@dataclass(frozen=True)
class ForbiddenPathHint:
    path: str
    reason_code: str
    evidence: str

    def to_payload(self) -> dict[str, str]:
        return {
            "path": self.path,
            "reason_code": self.reason_code,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class ScopeCompanionExpansion:
    source_path: str
    companion_path: str
    reason_code: str
    evidence: str

    def to_payload(self) -> dict[str, str]:
        return {
            "source_path": self.source_path,
            "companion_path": self.companion_path,
            "reason_code": self.reason_code,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class ScopeViolationDiagnostic:
    path: str
    classification: str
    reason_code: str
    evidence: str
    recommended_action: str
    allowed: bool = False
    source_path: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "classification": self.classification,
            "reason_code": self.reason_code,
            "evidence": self.evidence,
            "recommended_action": self.recommended_action,
            "allowed": self.allowed,
        }
        if self.source_path is not None:
            payload["source_path"] = self.source_path
        return payload


@dataclass(frozen=True)
class ScopeAssessment:
    ok: bool
    blocking_paths: list[str]
    diagnostics: list[ScopeViolationDiagnostic]
    effective_changed_files: list[str]
    expanded_allowed_scope: list[str]
    companion_expansions: list[ScopeCompanionExpansion]

    def to_payload(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "blocking_paths": self.blocking_paths,
            "diagnostics": [item.to_payload() for item in self.diagnostics],
            "effective_changed_files": self.effective_changed_files,
            "expanded_allowed_scope": self.expanded_allowed_scope,
            "companion_expansions": [item.to_payload() for item in self.companion_expansions],
        }


def _normalize_path(value: str) -> str:
    cleaned = value.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def _run_git_capture(
    root: Path, args: list[str], *, text: bool = False
) -> subprocess.CompletedProcess[Any] | None:
    try:
        return subprocess.run(
            ["git", "-C", os.fspath(root), *args],
            check=False,
            capture_output=True,
            text=text,
            env=build_git_process_env(),
            timeout=_READ_ONLY_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _has_glob(value: str) -> bool:
    return any(ch in value for ch in ["*", "?", "["])


def _strip_wrapping_punctuation(value: str) -> str:
    cleaned = value.strip().strip("`'\"")
    while cleaned and cleaned[-1] in {",", ";", ":", ".", ")", "]", "}"}:
        cleaned = cleaned[:-1].rstrip()
    while cleaned and cleaned[0] in {"(", "[", "{", ":"}:
        cleaned = cleaned[1:].lstrip()
    return cleaned


def normalize_repo_path_entry(
    value: str,
    *,
    allow_extensionless_file: bool = False,
) -> str | None:
    cleaned = _normalize_path(_strip_wrapping_punctuation(value))
    if not cleaned:
        return None
    if cleaned.startswith(("/", "../")):
        return None
    if "://" in cleaned or "\n" in cleaned or "\t" in cleaned:
        return None
    if " " in cleaned:
        return None
    if cleaned in {".", ".."}:
        return None
    if cleaned.endswith("/"):
        cleaned = cleaned.rstrip("/") + "/**"
    if cleaned in _SPECIAL_FILENAMES:
        return cleaned
    if _has_glob(cleaned):
        return cleaned
    if "/" in cleaned or "." in cleaned:
        return cleaned
    if (
        allow_extensionless_file
        and cleaned not in _BROAD_NON_PACKAGE_DIR_NAMES
        and _EXTENSIONLESS_FILE_RE.fullmatch(cleaned)
    ):
        return cleaned
    return None


def split_normalized_repo_path_list(value: Any) -> tuple[list[str], list[str]]:
    if not isinstance(value, list):
        return [], []
    seen: set[str] = set()
    out: list[str] = []
    dropped: list[str] = []
    for item in value:
        raw = str(item).strip()
        if not raw:
            continue
        normalized = normalize_repo_path_entry(raw, allow_extensionless_file=True)
        identity_key = _scope_pattern_identity_key(normalized) if normalized else ""
        if not normalized or identity_key in seen:
            if not normalized:
                dropped.append(raw)
            continue
        seen.add(identity_key)
        out.append(normalized)
    return out, dropped


def normalize_repo_path_list(value: Any) -> list[str]:
    out, _ = split_normalized_repo_path_list(value)
    return out


def is_explicit_repo_path_pattern(value: str) -> bool:
    normalized = normalize_repo_path_entry(value, allow_extensionless_file=True)
    if not normalized:
        return False
    return not _has_glob(normalized)


def extract_repo_path_hints(text: str) -> list[str]:
    seen: set[str] = set()
    hints: list[str] = []
    for match in _PATH_HINT_RE.findall(text or ""):
        normalized = normalize_repo_path_entry(match)
        if not normalized:
            continue
        tail = normalized.rstrip("/").split("/")[-1]
        if tail not in _SPECIAL_FILENAMES:
            if "." not in tail:
                continue
            ext = tail.rsplit(".", 1)[1].lower()
            if ext not in _INFERRED_FILE_EXTENSIONS:
                continue
        head_segment = normalized.split("/", 1)[0]
        if head_segment.isdigit():
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        hints.append(normalized)
    if "README.md" in seen and "README" in seen:
        hints = [hint for hint in hints if hint != "README"]
    return hints


def _evidence_fragment(context: str) -> str:
    cleaned = " ".join(str(context or "").strip().split())
    if len(cleaned) <= 180:
        return cleaned
    return cleaned[:177].rstrip() + "..."


def _forbidden_reason_for_path_context(
    *,
    context: str,
    hint_start: int,
    hint_end: int,
) -> str | None:
    before_hint = context[max(0, hint_start - _FORBIDDEN_PATH_MARKER_WINDOW_CHARS) : hint_start]
    after_hint = context[hint_end : hint_end + _FORBIDDEN_PATH_MARKER_WINDOW_CHARS]
    if _CONDITIONAL_FORBIDDEN_PATH_EXCEPTION_RE.search(after_hint) is not None:
        return None
    if (
        _FORBIDDEN_PATH_DIRECT_PREFIX_RE.search(before_hint) is not None
        or _FORBIDDEN_PATH_DIRECT_OBJECT_RE.search(before_hint) is not None
    ):
        return "direct_path_forbidden_instruction"
    if _FORBIDDEN_PATH_SUFFIX_MARKER_RE.search(after_hint) is not None:
        return "path_must_remain_unchanged"
    return None


def extract_forbidden_repo_path_hint_records(text: str) -> list[ForbiddenPathHint]:
    hints = extract_repo_path_hints(text)
    if not hints:
        return []
    forbidden: list[ForbiddenPathHint] = []
    seen: set[str] = set()
    contexts = [fragment.strip() for fragment in _PATH_HINT_CONTEXT_SPLIT_RE.split(text or "")]
    for hint in hints:
        hint_key = hint.casefold()
        for context in contexts:
            context_cf = context.casefold()
            hint_start = context_cf.find(hint_key)
            if hint_start < 0:
                continue
            hint_end = hint_start + len(hint_key)
            reason_code = _forbidden_reason_for_path_context(
                context=context_cf,
                hint_start=hint_start,
                hint_end=hint_end,
            )
            if reason_code is None:
                continue
            identity_key = _scope_pattern_identity_key(hint).casefold()
            if identity_key in seen:
                break
            seen.add(identity_key)
            forbidden.append(
                ForbiddenPathHint(
                    path=hint,
                    reason_code=reason_code,
                    evidence=_evidence_fragment(context),
                )
            )
            break
    paths = {item.path for item in forbidden}
    if "README.md" in paths and "README" in paths:
        forbidden = [item for item in forbidden if item.path != "README"]
    return forbidden


def extract_forbidden_repo_path_hints(text: str) -> list[str]:
    return [item.path for item in extract_forbidden_repo_path_hint_records(text)]


def _forbidden_path_identity_keys_for_task(task: dict[str, Any]) -> set[str]:
    acceptance = task.get("acceptance_criteria") or []
    acceptance_items = acceptance if isinstance(acceptance, list) else []
    text = "\n".join(
        [
            str(task.get("title") or ""),
            str(task.get("description") or ""),
            *(str(item or "") for item in acceptance_items),
        ]
    )
    return {
        _scope_pattern_identity_key(path).casefold()
        for path in extract_forbidden_repo_path_hints(text)
    }


def normalize_scope_patterns(task: dict[str, Any], *, root: Path | None = None) -> list[str]:
    expanded = _expand_support_file_patterns(normalize_claimed_scope_patterns(task), root=root)
    forbidden = _forbidden_path_identity_keys_for_task(task)
    if not forbidden:
        return expanded
    return [
        item for item in expanded if _scope_pattern_identity_key(item).casefold() not in forbidden
    ]


def normalize_claimed_scope_patterns(task: dict[str, Any]) -> list[str]:
    write_scope = normalize_repo_path_list(task.get("write_scope"))
    estimated_files = normalize_repo_path_list(task.get("estimated_files"))
    seen: set[str] = set()
    combined: list[str] = []
    for item in [*write_scope, *estimated_files]:
        identity_key = _scope_pattern_identity_key(item)
        if identity_key in seen:
            continue
        seen.add(identity_key)
        combined.append(item)
    return combined


def _scope_glob_variants(pattern: str) -> tuple[str, ...]:
    variants = [pattern]
    if "/**/" in pattern:
        direct_child_variant = pattern.replace("/**/", "/")
        if direct_child_variant != pattern:
            variants.append(direct_child_variant)
    return tuple(variants)


def _is_existing_directory_scope(*, pattern: str, root: Path | None) -> bool:
    if root is None or _has_glob(pattern) or _is_root_readme_alias(pattern):
        return False
    try:
        return (root.resolve() / pattern).is_dir()
    except OSError:
        return False


def _normalize_scope_match_pattern(value: str) -> str | None:
    normalized = normalize_repo_path_entry(value, allow_extensionless_file=True)
    if normalized:
        return normalized
    cleaned = _normalize_path(_strip_wrapping_punctuation(value))
    if not cleaned:
        return None
    if cleaned.startswith(("/", "../")):
        return None
    if "://" in cleaned or "\n" in cleaned or "\t" in cleaned:
        return None
    if cleaned in {".", ".."}:
        return None
    if cleaned.endswith("/"):
        cleaned = cleaned.rstrip("/") + "/**"
    return cleaned


def scope_path_matches_pattern(
    path: str,
    pattern: str,
    *,
    root: Path | None = None,
) -> bool:
    normalized_path = _normalize_path(path).rstrip("/")
    normalized_pattern = _normalize_scope_match_pattern(pattern)
    if not normalized_path or not normalized_pattern:
        return False
    if normalized_path.startswith(("/", "../")):
        return False
    if _is_root_readme_alias_match(path=normalized_path, pattern=normalized_pattern):
        return True
    if _has_glob(normalized_pattern):
        return any(
            fnmatch.fnmatchcase(normalized_path, variant)
            for variant in _scope_glob_variants(normalized_pattern)
        )
    if normalized_path == normalized_pattern:
        return True
    if _is_existing_directory_scope(pattern=normalized_pattern, root=root):
        return normalized_path.startswith(normalized_pattern.rstrip("/") + "/")
    return False


def _is_ignored_internal_path(path: str) -> bool:
    for prefix in _IGNORED_INTERNAL_PREFIXES:
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


def is_agent_internal_scope_path(path: str) -> bool:
    cleaned = _normalize_path(path)
    return _is_ignored_internal_path(cleaned) or is_runtime_artifact_path(cleaned)


def is_internal_sylliptor_path(path: str) -> bool:
    return is_agent_internal_scope_path(path)


def _is_python_file_path(path: str) -> bool:
    return path.endswith(".py") and "/" in path


def _is_python_test_file(path: PurePosixPath) -> bool:
    filename = path.name
    return "tests" in path.parts[:-1] and (
        filename.startswith("test_") or filename.endswith("_test.py")
    )


def _supports_python_package_init(path: PurePosixPath) -> bool:
    parent_name = path.parent.name
    if parent_name in {"", "."}:
        return False
    if parent_name == "tests":
        return True
    return parent_name not in _BROAD_NON_PACKAGE_DIR_NAMES


def _is_rust_source_file_path(path: str) -> bool:
    return path.endswith(".rs")


def _cargo_manifest_declares_workspace(manifest_path: Path) -> bool:
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return _CARGO_WORKSPACE_RE.search(text) is not None


def _cargo_manifest_declares_package(manifest_path: Path) -> bool:
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return _CARGO_PACKAGE_RE.search(text) is not None


def _nearest_cargo_manifest_dir(*, root: Path, path: PurePosixPath) -> Path | None:
    root_resolved = root.resolve()
    current = root_resolved.joinpath(
        *[segment for segment in path.parent.parts if segment not in {"", "."}]
    )
    while True:
        if (current / _CARGO_MANIFEST_FILENAME).is_file():
            return current
        if current == root_resolved:
            return None
        current = current.parent


def _workspace_cargo_manifest_dir(*, root: Path, manifest_dir: Path) -> Path | None:
    root_resolved = root.resolve()
    current = manifest_dir
    while True:
        manifest_path = current / _CARGO_MANIFEST_FILENAME
        if manifest_path.is_file() and _cargo_manifest_declares_workspace(manifest_path):
            return current
        if current == root_resolved:
            return None
        current = current.parent


def _scope_concrete_lookup_path(path: str) -> PurePosixPath | None:
    normalized = normalize_repo_path_entry(path, allow_extensionless_file=True)
    if not normalized:
        return None
    pure_path = PurePosixPath(normalized)
    if _is_rust_source_file_path(normalized):
        return pure_path.parent

    concrete_parts: list[str] = []
    for segment in pure_path.parts:
        if _has_glob(segment):
            break
        concrete_parts.append(segment)
    if not concrete_parts:
        return None

    return PurePosixPath(*concrete_parts)


def _path_declares_package_manifest(*, root: Path, path: PurePosixPath) -> bool:
    manifest_dir = root.resolve().joinpath(
        *[segment for segment in path.parts if segment not in {"", "."}]
    )
    return _cargo_manifest_declares_package(manifest_dir / _CARGO_MANIFEST_FILENAME)


def _manifest_relative_scope_path(
    *, root: Path, manifest_dir: Path, scope_path: PurePosixPath
) -> PurePosixPath | None:
    try:
        manifest_rel = manifest_dir.relative_to(root.resolve()).as_posix()
    except ValueError:
        return None
    if manifest_rel in {"", "."}:
        return scope_path
    try:
        return scope_path.relative_to(PurePosixPath(manifest_rel))
    except ValueError:
        return None


def _rust_scope_lookup_dir(*, root: Path, path: str) -> PurePosixPath | None:
    concrete_path = _scope_concrete_lookup_path(path)
    if concrete_path is None:
        return None
    if _path_declares_package_manifest(root=root, path=concrete_path):
        return concrete_path

    manifest_dir = _nearest_cargo_manifest_dir(root=root, path=concrete_path / "_scope.rs")
    if manifest_dir is None:
        return None
    manifest_path = manifest_dir / _CARGO_MANIFEST_FILENAME
    if not _cargo_manifest_declares_package(manifest_path):
        return None

    relative_scope_path = _manifest_relative_scope_path(
        root=root,
        manifest_dir=manifest_dir,
        scope_path=concrete_path,
    )
    if relative_scope_path is None or not relative_scope_path.parts:
        return None
    if relative_scope_path.parts[0] in _RUST_SCOPE_DIR_NAMES:
        return concrete_path
    return None


def _cargo_lock_support_patterns_for_path(*, root: Path, path: str) -> list[str]:
    normalized = normalize_repo_path_entry(path, allow_extensionless_file=True)
    if not normalized:
        return []

    lookup_dir = _rust_scope_lookup_dir(root=root, path=normalized)
    if lookup_dir is None:
        return []

    manifest_dir = _nearest_cargo_manifest_dir(root=root, path=lookup_dir / "_scope.rs")
    if manifest_dir is None:
        return []

    workspace_manifest_dir = _workspace_cargo_manifest_dir(root=root, manifest_dir=manifest_dir)
    lock_parent = workspace_manifest_dir or manifest_dir
    try:
        relative_lock = lock_parent.relative_to(root.resolve()).as_posix()
    except ValueError:
        return []
    if relative_lock in {"", "."}:
        return [_CARGO_LOCK_FILENAME]
    return [f"{relative_lock}/{_CARGO_LOCK_FILENAME}"]


def _cargo_lock_support_patterns_for_manifest(*, root: Path, path: str) -> list[str]:
    normalized = normalize_repo_path_entry(path, allow_extensionless_file=True)
    if not normalized:
        return []
    pure_path = PurePosixPath(normalized)
    if pure_path.name != _CARGO_MANIFEST_FILENAME:
        return []
    manifest_dir = root.resolve().joinpath(
        *[segment for segment in pure_path.parent.parts if segment not in {"", "."}]
    )
    manifest_path = manifest_dir / _CARGO_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return []
    if not (
        _cargo_manifest_declares_package(manifest_path)
        or _cargo_manifest_declares_workspace(manifest_path)
    ):
        return []
    workspace_manifest_dir = _workspace_cargo_manifest_dir(root=root, manifest_dir=manifest_dir)
    lock_parent = workspace_manifest_dir or manifest_dir
    try:
        relative_lock = lock_parent.relative_to(root.resolve()).as_posix()
    except ValueError:
        return []
    if relative_lock in {"", "."}:
        return [_CARGO_LOCK_FILENAME]
    return [f"{relative_lock}/{_CARGO_LOCK_FILENAME}"]


_NODE_LOCKFILES_BY_PACKAGE_MANAGER = {
    "npm": "package-lock.json",
    "pnpm": "pnpm-lock.yaml",
    "yarn": "yarn.lock",
    "bun": "bun.lockb",
}
_NODE_LOCKFILE_NAMES = frozenset(_NODE_LOCKFILES_BY_PACKAGE_MANAGER.values())
_PYPROJECT_LOCKFILE_RULES = (
    ("poetry.lock", "[tool.poetry]", "poetry"),
    ("uv.lock", "[tool.uv]", "uv"),
    ("pdm.lock", "[tool.pdm]", "pdm"),
)


def _package_manager_from_package_json(package_json: Path) -> str | None:
    try:
        import json

        payload = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    raw = str(payload.get("packageManager") or "").strip().casefold()
    if not raw:
        return None
    return raw.split("@", 1)[0].strip() or None


def _node_lock_support_patterns_for_path(*, root: Path, path: str) -> list[str]:
    normalized = normalize_repo_path_entry(path, allow_extensionless_file=True)
    if not normalized:
        return []
    pure_path = PurePosixPath(normalized)
    if pure_path.name != "package.json":
        return []
    package_dir = root.resolve().joinpath(
        *[segment for segment in pure_path.parent.parts if segment not in {"", "."}]
    )
    package_json = package_dir / "package.json"
    if not package_json.is_file():
        return []
    lock_name: str | None = None
    manager = _package_manager_from_package_json(package_json)
    if manager in _NODE_LOCKFILES_BY_PACKAGE_MANAGER:
        lock_name = _NODE_LOCKFILES_BY_PACKAGE_MANAGER[manager]
    else:
        existing = sorted(name for name in _NODE_LOCKFILE_NAMES if (package_dir / name).is_file())
        if len(existing) == 1:
            lock_name = existing[0]
    if lock_name is None:
        return []
    lock_path = pure_path.parent / lock_name
    return [lock_path.as_posix()]


def _pyproject_lock_support_patterns_for_path(*, root: Path, path: str) -> list[str]:
    normalized = normalize_repo_path_entry(path, allow_extensionless_file=True)
    if not normalized:
        return []
    pure_path = PurePosixPath(normalized)
    if pure_path.name != "pyproject.toml":
        return []
    project_dir = root.resolve().joinpath(
        *[segment for segment in pure_path.parent.parts if segment not in {"", "."}]
    )
    pyproject = project_dir / "pyproject.toml"
    if not pyproject.is_file():
        return []
    try:
        pyproject_text = pyproject.read_text(encoding="utf-8").casefold()
    except OSError:
        pyproject_text = ""
    support: list[str] = []
    for lock_name, marker, _tool_name in _PYPROJECT_LOCKFILE_RULES:
        lock_path = project_dir / lock_name
        marker_present = marker.casefold() in pyproject_text
        if not lock_path.is_file() and not marker_present:
            continue
        support.append((pure_path.parent / lock_name).as_posix())
    return support


def _requirements_lock_support_patterns_for_path(*, root: Path, path: str) -> list[str]:
    normalized = normalize_repo_path_entry(path, allow_extensionless_file=True)
    if not normalized:
        return []
    pure_path = PurePosixPath(normalized)
    name = pure_path.name.casefold()
    if not (name.startswith("requirements") and name.endswith(".txt")):
        return []
    requirements_dir = root.resolve().joinpath(
        *[segment for segment in pure_path.parent.parts if segment not in {"", "."}]
    )
    support: list[str] = []
    for candidate in ("constraints.txt", "requirements.lock", "requirements.lock.txt"):
        if (requirements_dir / candidate).is_file():
            support.append((pure_path.parent / candidate).as_posix())
    return support


def _companion_patterns_for_path(path: str, *, root: Path | None = None) -> list[str]:
    if root is None:
        return []
    normalized = normalize_repo_path_entry(path, allow_extensionless_file=True)
    if not normalized:
        return []
    companions: list[str] = []
    if _has_glob(normalized):
        companions.extend(_cargo_lock_support_patterns_for_path(root=root, path=normalized))
        return list(dict.fromkeys(companions))
    companions.extend(_cargo_lock_support_patterns_for_manifest(root=root, path=normalized))
    companions.extend(_cargo_lock_support_patterns_for_path(root=root, path=normalized))
    companions.extend(_node_lock_support_patterns_for_path(root=root, path=normalized))
    companions.extend(_pyproject_lock_support_patterns_for_path(root=root, path=normalized))
    companions.extend(_requirements_lock_support_patterns_for_path(root=root, path=normalized))
    return list(dict.fromkeys(companions))


def companion_generated_paths_for(path: str, *, root: Path | None = None) -> list[str]:
    return _companion_patterns_for_path(path, root=root)


def companion_expansions_for_patterns(
    patterns: list[str],
    *,
    root: Path | None = None,
) -> list[ScopeCompanionExpansion]:
    expansions: list[ScopeCompanionExpansion] = []
    seen: set[str] = set()
    for pattern in patterns:
        normalized = normalize_repo_path_entry(pattern, allow_extensionless_file=True)
        if not normalized:
            continue
        for companion in _companion_patterns_for_path(normalized, root=root):
            key = _scope_pattern_identity_key(companion).casefold()
            if key in seen:
                continue
            seen.add(key)
            expansions.append(
                ScopeCompanionExpansion(
                    source_path=normalized,
                    companion_path=companion,
                    reason_code="ecosystem_companion",
                    evidence=f"{companion} is a repo-aware companion for {normalized}",
                )
            )
    return expansions


def _rust_entrypoint_support_patterns_for_path(*, root: Path, path: str) -> list[str]:
    normalized = normalize_repo_path_entry(path, allow_extensionless_file=True)
    if not normalized or not _is_rust_source_file_path(normalized):
        return []
    pure_path = PurePosixPath(normalized)
    if pure_path.name in _RUST_ENTRYPOINT_FILENAMES:
        return []

    lookup_dir = _rust_scope_lookup_dir(root=root, path=normalized)
    if lookup_dir is None:
        return []

    manifest_dir = _nearest_cargo_manifest_dir(root=root, path=lookup_dir / "_scope.rs")
    if manifest_dir is None:
        return []
    try:
        manifest_rel = manifest_dir.relative_to(root.resolve()).as_posix()
    except ValueError:
        return []

    support_patterns: list[str] = []
    for filename in _RUST_ENTRYPOINT_FILENAMES:
        host_path = manifest_dir / "src" / filename
        if not host_path.is_file():
            continue
        rel_path = PurePosixPath("src", filename)
        if manifest_rel not in {"", "."}:
            rel_path = PurePosixPath(manifest_rel) / rel_path
        support_patterns.append(rel_path.as_posix())
    return support_patterns


def _support_file_patterns_for_path(path: str, *, root: Path | None = None) -> list[str]:
    normalized = normalize_repo_path_entry(path, allow_extensionless_file=True)
    if not normalized:
        return []

    support_patterns: list[str] = []
    if not _has_glob(normalized) and _is_python_file_path(normalized):
        pure_path = PurePosixPath(normalized)
        parent = pure_path.parent.as_posix()
        if parent not in {"", "."}:
            if pure_path.name != "__init__.py" and _supports_python_package_init(pure_path):
                support_patterns.append(f"{parent}/__init__.py")
            if _is_python_test_file(pure_path):
                support_patterns.append(f"{parent}/conftest.py")

    if root is not None:
        support_patterns.extend(_companion_patterns_for_path(normalized, root=root))
        support_patterns.extend(
            _rust_entrypoint_support_patterns_for_path(root=root, path=normalized)
        )
    return support_patterns


def _expand_support_file_patterns(patterns: list[str], *, root: Path | None = None) -> list[str]:
    seen: set[str] = set()
    expanded: list[str] = []
    for pattern in patterns:
        normalized = normalize_repo_path_entry(pattern, allow_extensionless_file=True)
        if not normalized:
            continue
        identity_key = _scope_pattern_identity_key(normalized)
        if identity_key not in seen:
            seen.add(identity_key)
            expanded.append(normalized)
        for support_pattern in _support_file_patterns_for_path(normalized, root=root):
            support_identity_key = _scope_pattern_identity_key(support_pattern)
            if support_identity_key in seen:
                continue
            seen.add(support_identity_key)
            expanded.append(support_pattern)
    return expanded


def _is_root_readme_alias(value: str) -> bool:
    normalized = _normalize_path(value).rstrip("/")
    return "/" not in normalized and normalized in _README_ALIAS_FILENAMES


def _is_root_readme_alias_match(*, path: str, pattern: str) -> bool:
    return _is_root_readme_alias(path) and _is_root_readme_alias(pattern)


def _scope_pattern_identity_key(value: str) -> str:
    normalized = _normalize_path(value).rstrip("/")
    if _is_root_readme_alias(normalized):
        return "__readme_alias__"
    return normalized


def ancestor_directory_scope_patterns(
    patterns: list[str], *, root: Path | None = None
) -> list[str]:
    seen: set[str] = set()
    ancestors: list[str] = []
    for pattern in _expand_support_file_patterns(patterns, root=root):
        normalized = normalize_repo_path_entry(pattern, allow_extensionless_file=True)
        if not normalized or _has_glob(normalized):
            continue
        parent = PurePosixPath(normalized).parent.as_posix()
        while parent not in {"", "."}:
            if parent not in seen:
                seen.add(parent)
                ancestors.append(parent)
            parent = PurePosixPath(parent).parent.as_posix()
    return ancestors


def is_non_material_untracked_path(path: str) -> bool:
    normalized = _normalize_path(path).rstrip("/")
    if not normalized:
        return False
    pure = PurePosixPath(normalized)
    if any(part.endswith(".egg-info") for part in pure.parts):
        return True
    if len(pure.parts) != 1:
        return False
    name = pure.name.casefold()
    return (
        name in _NON_MATERIAL_ROOT_SCRATCH_FILENAMES
        or any(name.endswith(suffix) for suffix in _NON_MATERIAL_ROOT_SCRATCH_SUFFIXES)
        or _NON_MATERIAL_ROOT_SCRATCH_RE.fullmatch(name) is not None
        or _NON_MATERIAL_ROOT_STATE_FILE_RE.fullmatch(name) is not None
    )


def _is_known_sylliptor_scratch_path(path: str) -> bool:
    normalized = _normalize_path(path).rstrip("/")
    if not normalized:
        return False
    pure = PurePosixPath(normalized)
    if len(pure.parts) != 1:
        return False
    name = pure.name.casefold()
    return (
        name in _NON_MATERIAL_ROOT_SCRATCH_FILENAMES
        or any(name.endswith(suffix) for suffix in _NON_MATERIAL_ROOT_SCRATCH_SUFFIXES)
        or _NON_MATERIAL_ROOT_SCRATCH_RE.fullmatch(name) is not None
    )


def _unique_artifact_destination(artifact_dir: Path, filename: str) -> Path:
    candidate = artifact_dir / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    for index in range(1, 1000):
        indexed = artifact_dir / f"{stem}.{index}{suffix}"
        if not indexed.exists():
            return indexed
    return artifact_dir / f"{stem}.overflow{suffix}"


def relocate_known_scratch_artifacts(
    *,
    root: Path,
    artifact_dir: Path,
) -> list[ScopeViolationDiagnostic]:
    diagnostics: list[ScopeViolationDiagnostic] = []
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for rel in _git_untracked_leaf_files(root, include_non_material=True):
        normalized = _normalize_path(rel).rstrip("/")
        if not _is_known_sylliptor_scratch_path(normalized):
            continue
        source = (root.resolve() / normalized).resolve()
        try:
            source.relative_to(root.resolve())
        except ValueError:
            continue
        if not source.is_file():
            continue
        destination = _unique_artifact_destination(artifact_dir, PurePosixPath(normalized).name)
        try:
            shutil.move(os.fspath(source), os.fspath(destination))
        except OSError:
            continue
        diagnostics.append(
            ScopeViolationDiagnostic(
                path=normalized,
                classification=SCOPE_CLASS_SCRATCH_ARTIFACT,
                reason_code="moved_known_root_scratch_file",
                evidence=(
                    f"Moved untracked diagnostic scratch file {normalized} to {destination.name}"
                ),
                recommended_action="recorded_in_artifacts",
                allowed=True,
            )
        )
    return diagnostics


def _forbidden_path_identity_keys_for_text(text: str) -> set[str]:
    return {
        _scope_pattern_identity_key(item.path).casefold()
        for item in extract_forbidden_repo_path_hint_records(text)
    }


def _forbidden_path_records_for_task(task: dict[str, Any] | None) -> list[ForbiddenPathHint]:
    if not isinstance(task, dict):
        return []
    acceptance = task.get("acceptance_criteria") or []
    acceptance_items = acceptance if isinstance(acceptance, list) else []
    text = "\n".join(
        [
            str(task.get("title") or ""),
            str(task.get("description") or ""),
            *(str(item or "") for item in acceptance_items),
        ]
    )
    return extract_forbidden_repo_path_hint_records(text)


def _path_matches_any_pattern(
    path: str,
    patterns: list[str],
    *,
    root: Path | None = None,
) -> bool:
    return any(scope_path_matches_pattern(path, pattern, root=root) for pattern in patterns)


def _companion_source_for_path(
    path: str,
    expansions: list[ScopeCompanionExpansion],
    *,
    root: Path | None = None,
) -> ScopeCompanionExpansion | None:
    for expansion in expansions:
        if scope_path_matches_pattern(path, expansion.companion_path, root=root):
            return expansion
    return None


def _likely_legitimate_missing_scope(path: str, *, allowed_patterns: list[str]) -> bool:
    normalized = _normalize_path(path).rstrip("/")
    if not normalized:
        return False
    if normalized.startswith(".") or normalized.startswith("../"):
        return False
    pure = PurePosixPath(normalized)
    if any(part in {"node_modules", "vendor", "dist", "build", "target"} for part in pure.parts):
        return False
    leaf = pure.name
    if "." not in leaf:
        return False
    extension = leaf.rsplit(".", 1)[1].casefold()
    if extension not in _INFERRED_FILE_EXTENSIONS:
        return False
    allowed_dirs = {
        PurePosixPath(pattern).parent.as_posix()
        for pattern in allowed_patterns
        if pattern and not _has_glob(pattern)
    }
    parent = pure.parent.as_posix()
    return parent in allowed_dirs or any(parent.startswith(item + "/") for item in allowed_dirs)


def assess_scope_changes(
    changed_files: list[str],
    allowed_patterns: list[str],
    *,
    task: dict[str, Any] | None = None,
    root: Path | None = None,
    extra_diagnostics: list[ScopeViolationDiagnostic] | None = None,
) -> ScopeAssessment:
    expanded_patterns = _expand_support_file_patterns(allowed_patterns, root=root)
    ancestor_dirs = set(ancestor_directory_scope_patterns(allowed_patterns, root=root))
    companion_expansions = companion_expansions_for_patterns(allowed_patterns, root=root)
    companion_patterns = [item.companion_path for item in companion_expansions]
    forbidden_records = _forbidden_path_records_for_task(task)
    forbidden_by_identity = {
        _scope_pattern_identity_key(item.path).casefold(): item for item in forbidden_records
    }

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in changed_files:
        rel = _normalize_path(raw).rstrip("/")
        if not rel or rel in seen or is_runtime_artifact_path(rel, root=root):
            continue
        seen.add(rel)
        normalized.append(rel)

    diagnostics: list[ScopeViolationDiagnostic] = list(extra_diagnostics or [])
    blocking_paths: list[str] = []
    effective_changed_files: list[str] = []
    for rel in normalized:
        identity_key = _scope_pattern_identity_key(rel).casefold()
        forbidden_record = forbidden_by_identity.get(identity_key)
        if forbidden_record is not None:
            diagnostics.append(
                ScopeViolationDiagnostic(
                    path=rel,
                    classification=SCOPE_CLASS_FORBIDDEN,
                    reason_code=forbidden_record.reason_code,
                    evidence=forbidden_record.evidence,
                    recommended_action="reject_hard",
                    allowed=False,
                )
            )
            blocking_paths.append(rel)
            effective_changed_files.append(rel)
            continue
        if rel in ancestor_dirs:
            effective_changed_files.append(rel)
            continue
        companion = _companion_source_for_path(rel, companion_expansions, root=root)
        if companion is not None or _path_matches_any_pattern(rel, companion_patterns, root=root):
            source_path = companion.source_path if companion is not None else None
            diagnostics.append(
                ScopeViolationDiagnostic(
                    path=rel,
                    classification=SCOPE_CLASS_EXPECTED_COMPANION,
                    reason_code="allowed_ecosystem_companion",
                    evidence=(
                        companion.evidence
                        if companion is not None
                        else f"{rel} matches an ecosystem companion rule"
                    ),
                    recommended_action="allow_and_record",
                    allowed=True,
                    source_path=source_path,
                )
            )
            effective_changed_files.append(rel)
            continue
        if _path_matches_any_pattern(rel, expanded_patterns, root=root):
            effective_changed_files.append(rel)
            continue
        if _likely_legitimate_missing_scope(rel, allowed_patterns=allowed_patterns):
            classification = SCOPE_CLASS_LIKELY_MISSING_SCOPE
            reason_code = "same_area_code_path_outside_scope"
            recommended_action = "create_scope_delta_proposal"
        else:
            classification = SCOPE_CLASS_DANGEROUS_UNRELATED
            reason_code = "path_not_covered_by_task_scope"
            recommended_action = "reject_and_replan"
        diagnostics.append(
            ScopeViolationDiagnostic(
                path=rel,
                classification=classification,
                reason_code=reason_code,
                evidence=f"{rel} is not covered by allowed scope {allowed_patterns or ['(none)']}",
                recommended_action=recommended_action,
                allowed=False,
            )
        )
        blocking_paths.append(rel)
        effective_changed_files.append(rel)

    return ScopeAssessment(
        ok=not blocking_paths,
        blocking_paths=blocking_paths,
        diagnostics=diagnostics,
        effective_changed_files=effective_changed_files,
        expanded_allowed_scope=expanded_patterns,
        companion_expansions=companion_expansions,
    )


def check_scope(
    changed_files: list[str],
    allowed_patterns: list[str],
    *,
    root: Path | None = None,
) -> tuple[bool, list[str]]:
    assessment = assess_scope_changes(changed_files, allowed_patterns, root=root)
    return assessment.ok, assessment.blocking_paths


def _git_untracked_leaf_files(root: Path, *, include_non_material: bool = False) -> list[str]:
    if shutil.which("git") is None:
        return []
    proc = _run_git_capture(
        root,
        ["ls-files", "--others", "--exclude-standard", "-z"],
    )
    if proc is None:
        return []
    if proc.returncode != 0:
        return []

    stdout = proc.stdout
    raw_stdout = (
        stdout.encode("utf-8", errors="surrogateescape")
        if isinstance(stdout, str)
        else (stdout or b"")
    )
    seen: set[str] = set()
    changed: list[str] = []
    for item in raw_stdout.split(b"\0"):
        if not item:
            continue
        rel = _normalize_path(item.decode("utf-8", errors="surrogateescape").strip('"'))
        if not rel or rel in seen:
            continue
        if not include_non_material and is_non_material_untracked_path(rel):
            continue
        seen.add(rel)
        changed.append(rel)
    return changed


def list_untracked_packaging_metadata_paths(root: Path) -> list[str]:
    return [
        rel
        for rel in _git_untracked_leaf_files(root, include_non_material=True)
        if is_non_material_untracked_path(rel)
    ]


def list_changed_files_including_untracked(root: Path) -> list[str]:
    if shutil.which("git") is None:
        return []
    proc = _run_git_capture(root, ["status", "--porcelain"], text=True)
    if proc is None:
        return []
    if proc.returncode != 0:
        return []
    seen: set[str] = set()
    changed: list[str] = []
    for line in proc.stdout.splitlines():
        if len(line) < 4:
            continue
        if line.startswith("?? "):
            continue
        rel = line[3:].strip()
        if " -> " in rel:
            rel = rel.split(" -> ", 1)[1].strip()
        rel = _normalize_path(rel.strip('"'))
        if not rel or rel in seen:
            continue
        seen.add(rel)
        changed.append(rel)
    for rel in _git_untracked_leaf_files(root):
        if rel in seen:
            continue
        seen.add(rel)
        changed.append(rel)
    return changed
