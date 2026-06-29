from __future__ import annotations

import pytest

from sylliptor_agent_cli.turn_intent import (
    classify_local_materialization_requirement,
    classify_repo_execution_intent,
)


@pytest.mark.parametrize(
    "instruction",
    [
        "can you inspect the repo we are working?",
        "locate and inspect the repository before answering",
        "please list the workspace files",
        "show me the current git status",
    ],
)
def test_repo_inspection_requests_are_read_only(instruction: str) -> None:
    assert classify_repo_execution_intent(instruction) == "advisory_non_execution"


@pytest.mark.parametrize(
    "instruction",
    [
        "inspect the repo and fix the parser bug",
        "list the relevant files, then update the handler",
    ],
)
def test_repo_inspection_with_explicit_change_stays_execution(instruction: str) -> None:
    assert classify_repo_execution_intent(instruction) == "execute"


def test_local_materialization_overrides_tell_me_wording() -> None:
    requirement = classify_local_materialization_requirement(
        "Tell me the count and save it to /workspace/answer.txt"
    )

    assert requirement.required is True
    assert requirement.confidence >= 0.8
    assert "/workspace/answer.txt" in requirement.output_paths
    assert (
        classify_repo_execution_intent("Tell me the count and save it to /workspace/answer.txt")
        == "execute"
    )


def test_explain_only_counting_remains_advisory() -> None:
    requirement = classify_local_materialization_requirement("Explain how to count it.")

    assert requirement.required is False
    assert classify_repo_execution_intent("Explain how to count it.") == "advisory_non_execution"
