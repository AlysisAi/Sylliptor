from __future__ import annotations

import pytest
from _terminal_matrix_harness import (
    assert_consistency,
    load_scenario,
    render,
    scenario_ids,
    simulate,
    to_visible,
    write_outputs,
)


@pytest.mark.parametrize("scenario_id", scenario_ids())
def test_terminal_matrix_consistency(scenario_id: int) -> None:
    scenario = load_scenario(scenario_id)
    with simulate(scenario) as simulation:
        ansi = render(scenario)
        actual_theme = assert_consistency(scenario, ansi)
        write_outputs(
            scenario,
            ansi=ansi,
            visible=to_visible(ansi),
            debug=simulation.debug_text(),
            actual_theme=actual_theme,
        )
