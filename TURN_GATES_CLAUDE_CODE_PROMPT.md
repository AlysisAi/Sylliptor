# Task: execute BEHAVIORAL_TURN_GATES_SPEC.md

Work on branch `feat/behavioral-turn-gates` (it already exists and contains the spec — check it
out; do NOT work on `main`). The spec file `BEHAVIORAL_TURN_GATES_SPEC.md` at the repo root is the
source of truth for this change. Read it fully first, then implement it exactly. This prompt adds
execution context the spec doesn't carry.

## Goal in one sentence

The turn/completion gates must stop classifying assistant prose with hardcoded phrase lists
(English + Greek marker vocabularies in `src/sylliptor_agent_cli/agent/turn/exploration.py`) and
instead use the three structured signals the codebase already has: the provider-reported
`AssistantResponsePhase`, observed tool evidence, and the structured `blocked:`/question formats.

## Order of work

1. Read `BEHAVIORAL_TURN_GATES_SPEC.md`, then read `agent/turn/exploration.py` and every call
   site listed in the spec (`agent/turn/core.py` ~2678, ~2689, ~2734, ~4832, ~5401;
   `agent_loop.py` import block ~370; `agent/turn/__init__.py` lazy re-export map).
2. Rewire the `core.py` gates first (per spec), while the old functions still exist — keep the
   tree importable at every step.
3. Audit and fix `agent_loop.py`'s own uses of the three deleted functions the same way
   (behavior/phase/structured replacements — find them with grep, don't assume the import block
   is the only coupling).
4. Delete the three vocabularies and their three functions from `exploration.py`; update the
   `__init__.py` lazy map. `_assistant_text_has_structured_blocker_marker` and
   `_assistant_text_has_well_formed_blocker` stay; `_BLOCKER_OBSTACLE_RE` survives only inside
   the well-formed (already-structured) check as a detail-quality test.
5. Update `tests/test_agent_loop_one_shot_follow_through.py`: its mock responses must carry
   `assistant_phase` (non-final for "still working" stubs, FINAL_ANSWER for answers) instead of
   relying on trigger phrases. Assertions about nudges/gates keep their intent; only the fixture
   trigger mechanism changes.
6. Verify (below), commit in logical slices with clear messages as you go.

## Hard constraints

- NEVER re-add a phrase/vocabulary list to make a test pass. If a test can only pass via prose
  matching, change the test's fixtures to set `assistant_phase` — that is the point of the change.
- Do not weaken the structured-blocker contract or the question-shape checks — they are protocol
  parsing and stay.
- Do not touch `main`, do not `git checkout`/`switch` with uncommitted work (this repo has had
  work destroyed that way), and commit frequently.
- Behavior deltas must be deliberate: where the spec says a gate becomes stricter (completion
  claims without tool action no longer bypass the read-only gate), that is intended — note it in
  the commit message rather than "fixing" it.

## Verification gates (all must pass before you call it done)

```bash
# 1. No survivor references anywhere (must print nothing):
grep -rn "_ONE_SHOT_COMPLETION_MARKERS\|_ONE_SHOT_NON_FINAL_PROGRESS_MARKERS\|_ONE_SHOT_BLOCKER_MARKERS\|_assistant_text_has_completion_marker\|_assistant_text_contains_progress_intent\|_assistant_text_has_blocker_marker" src tests --include=*.py

# 2. Focused suites:
python -m pytest tests/test_agent_loop_one_shot_follow_through.py tests/test_agent_loop_review_approval.py -q --timeout=120

# 3. The behavioral regression net (asserts outcomes, not markers — run unchanged):
python -m pytest tests/test_power_*.py -q --timeout=180 -p no:cacheprovider

# 4. Full suite as the final gate (slow; the per-test timeout is mandatory —
#    without it one interactive-prompt test class can block forever):
python -m pytest tests/ -q -p no:cacheprovider --timeout=120 --timeout-method=thread
```

Environment notes: the dev venv is `~/.venv-syl` (WSL, editable install of this repo). If runtime
behavior looks stale versus the source, delete `src/**/__pycache__` — stale bytecode on the
`/mnt/c` mount is a known trap here.

## Done means

All four verification gates green; a short summary in the final message listing (a) each gate
site and the structured signal that replaced its prose check, (b) any deliberate behavior deltas,
(c) test fixture changes. Leave the branch pushed-ready; do not merge to `main`.
