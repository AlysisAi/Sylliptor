from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

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
_INFERRED_FILE_EXTENSIONS = {
    "bash",
    "c",
    "cfg",
    "conf",
    "cpp",
    "css",
    "csv",
    "env",
    "go",
    "h",
    "hpp",
    "html",
    "ini",
    "java",
    "js",
    "json",
    "jsx",
    "kt",
    "md",
    "mjs",
    "php",
    "py",
    "rb",
    "rs",
    "scss",
    "sh",
    "sql",
    "svg",
    "swift",
    "toml",
    "ts",
    "tsx",
    "txt",
    "xml",
    "yaml",
    "yml",
}
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
_FORBIDDEN_PATH_PREFIX_MARKER_RE = re.compile(
    r"\b(?:do\s+not|must\s+not|never|without\s+(?:touching|modifying|changing)|"
    r"leave\s+(?:untouched|unchanged)|preserve|not\s+(?:touch|read|write|include)|"
    r"no\s+longer\s+read|exclude|ignore|skip)\b|don't",
    re.IGNORECASE,
)
_FORBIDDEN_PATH_SUFFIX_MARKER_RE = re.compile(
    r"\b(?:is|are|must\s+remain|should\s+remain|stays?|stay|leave|left)\s+"
    r"(?:not\s+)?(?:untouched|unchanged|unmodified|preserved|not\s+modified|not\s+changed|not\s+touched)\b"
    r"|\bnot\s+(?:modified|changed|touched|written)\b",
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
    r"[_-](?:out|output|stdout|stderr|err|errors?|log|logs|full|results?|install[_-]out)"
    r"(?:\d+)?\.txt$"
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


def extract_forbidden_repo_path_hints(text: str) -> list[str]:
    hints = extract_repo_path_hints(text)
    if not hints:
        return []
    forbidden: list[str] = []
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
            before_hint = context_cf[
                max(0, hint_start - _FORBIDDEN_PATH_MARKER_WINDOW_CHARS) : hint_start
            ]
            after_hint = context_cf[hint_end : hint_end + _FORBIDDEN_PATH_MARKER_WINDOW_CHARS]
            if _CONDITIONAL_FORBIDDEN_PATH_EXCEPTION_RE.search(after_hint) is not None:
                continue
            if (
                _FORBIDDEN_PATH_PREFIX_MARKER_RE.search(before_hint) is None
                and _FORBIDDEN_PATH_SUFFIX_MARKER_RE.search(after_hint) is None
            ):
                continue
            identity_key = _scope_pattern_identity_key(hint).casefold()
            if identity_key in seen:
                break
            seen.add(identity_key)
            forbidden.append(hint)
            break
    if "README.md" in forbidden and "README" in forbidden:
        forbidden = [hint for hint in forbidden if hint != "README"]
    return forbidden


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
        support_patterns.extend(_cargo_lock_support_patterns_for_path(root=root, path=normalized))
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


def check_scope(
    changed_files: list[str],
    allowed_patterns: list[str],
    *,
    root: Path | None = None,
) -> tuple[bool, list[str]]:
    expanded_patterns = _expand_support_file_patterns(allowed_patterns, root=root)
    ancestor_dirs = set(ancestor_directory_scope_patterns(allowed_patterns, root=root))
    normalized = [
        rel
        for p in changed_files
        if (rel := _normalize_path(p)) and not is_runtime_artifact_path(rel, root=root)
    ]
    violations: list[str] = []
    for rel in normalized:
        rel_dir = rel.rstrip("/")
        if rel_dir in ancestor_dirs:
            continue
        if any(scope_path_matches_pattern(rel, pat, root=root) for pat in expanded_patterns):
            continue
        violations.append(rel)
    return not violations, violations


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
