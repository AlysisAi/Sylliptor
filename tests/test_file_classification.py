from __future__ import annotations

import pytest

from sylliptor_agent_cli.file_classification import (
    describe_path_kinds,
    is_code_implementation_path,
    is_symbol_scannable_path,
    is_test_path,
    language_for_path,
    source_extensions_for_languages,
)


@pytest.mark.parametrize(
    "path",
    [
        "src/main/java/com/example/ConfigLoader.java",
        "src/ConfigLoader.java",
        "src/**.java",
    ],
)
def test_java_implementation_paths_are_first_class(path: str) -> None:
    assert language_for_path(path) == "java"
    assert is_code_implementation_path(path)
    assert is_symbol_scannable_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "src/test/java/com/example/ConfigLoaderTest.java",
        "src/test/java/**/*.java",
        "tests/ConfigLoaderTest.java",
    ],
)
def test_java_test_paths_are_not_implementation_paths(path: str) -> None:
    assert language_for_path(path) == "java"
    assert is_test_path(path)
    assert not is_code_implementation_path(path)
    assert is_symbol_scannable_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "build/generated/src/main/java/com/example/ConfigLoader.java",
        "vendor/src/ConfigLoader.java",
        "src/generated/java/com/example/ConfigLoader.java",
    ],
)
def test_generated_vendor_and_build_paths_are_not_normal_implementation_scope(
    path: str,
) -> None:
    assert language_for_path(path) == "java"
    assert not is_code_implementation_path(path)
    assert not is_symbol_scannable_path(path)


@pytest.mark.parametrize(
    "path, language",
    [
        ("src/app.py", "python"),
        ("src/app.ts", "typescript"),
        ("src/app.js", "javascript"),
        ("src/lib.rs", "rust"),
        ("cmd/server.go", "go"),
    ],
)
def test_existing_implementation_languages_stay_supported(path: str, language: str) -> None:
    assert language_for_path(path) == language
    assert is_code_implementation_path(path)
    assert is_symbol_scannable_path(path)


def test_java_repo_scan_language_extensions_preserve_kotlin_hinting() -> None:
    assert {".java", ".kt"} <= source_extensions_for_languages(["java"])


def test_existing_test_patterns_stay_excluded_from_implementation() -> None:
    for path in [
        "tests/test_app.py",
        "src/app.test.ts",
        "src/app.spec.js",
        "src/lib_test.go",
    ]:
        assert is_test_path(path)
        assert not is_code_implementation_path(path)


def test_path_kind_descriptions_include_known_and_unknown_kinds() -> None:
    assert describe_path_kinds(["src/test/java/com/example/ConfigLoaderTest.java"]) == (
        "Java test files only"
    )
    assert describe_path_kinds(["README.md", "settings.json"]) == "docs/config only"
    assert describe_path_kinds(["notes/data.unknownext"]) == "docs only"
    assert describe_path_kinds(["scripts/task.weird"]) == "unknown .weird files only"
