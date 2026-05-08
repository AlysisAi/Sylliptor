from __future__ import annotations

import re
import textwrap

from _terminal_matrix_harness import render_scenario, scenario_ids, to_visible


def _normalize_visible_content(value: str) -> str:
    normalized_lines = []
    for line in textwrap.dedent(value).splitlines():
        stripped = re.sub(r" +", " ", line.strip())
        if stripped:
            normalized_lines.append(stripped)
    return "\n".join(normalized_lines)


def test_visible_text_is_identical_across_terminals() -> None:
    visibles = [to_visible(render_scenario(scenario_id)) for scenario_id in scenario_ids()]
    normalized = [_normalize_visible_content(visible) for visible in visibles]
    canonical = normalized[0]
    for scenario_id, visible in zip(scenario_ids()[1:], normalized[1:], strict=True):
        assert visible == canonical, (
            f"Scenario {scenario_id} visible text differs from scenario 1.\n"
            f"=== scenario 1 ===\n{canonical}\n"
            f"=== scenario {scenario_id} ===\n{visible}\n"
        )
