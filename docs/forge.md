# Forge

Forge is Sylliptor's plan-driven workflow for larger coding tasks. It turns a
broad request into scoped work, runs ready tasks, and keeps verification and
review evidence together.

Use it for multi-file changes, staged refactors, or release work where you want
to review the plan before implementation starts.

## Start Forge

Open Sylliptor in the repository and enter `/forge`. Describe the outcome you
want, then use the on-screen actions to refine the goal, inspect tasks, and
approve execution.

`/back` returns to the normal session without losing the current run. Use
`/forge resume` when you want to reopen the active run for that workspace.

## The Plan

A good Forge task has:

- one clear objective
- a small, explicit write scope
- useful acceptance criteria or verification commands
- dependencies only where another task must finish first

Forge keeps the structured plan, a readable summary, task logs, verification
results, and reviews under the workspace's local Sylliptor runtime directory.
This makes interrupted work resumable and completed work easier to inspect.

## Execution And Review

Ready tasks can run independently and in parallel. Forge respects their
dependencies, keeps worker changes scoped, and integrates successful work in
batches.

Verification can be advisory or strict. In strict mode, a task does not pass
without successful verification. Review results and failed checks stay with the
run so they can guide a retry instead of being lost.

Forge can also wrap a task in a local PR-style flow with its own branch, commit,
verification, review, and merge gates. Nothing is pushed to a remote by that
local flow.

## Modes And Safety

Forge follows the same execution modes as the rest of Sylliptor:

- `readonly` inspects and plans only.
- `review` asks before writes and shell commands.
- `auto` applies routine changes with fewer prompts.
- `fullaccess` removes mode-level prompts in a trusted workspace.

Start in `review` when the plan or write scope still needs inspection. Move to
`auto` only when the tasks and verification are clear.

## Practical Guidance

- Keep unrelated work in separate tasks.
- Review write scopes before execution.
- Use smaller batches when failures would be hard to untangle.
- Treat failed verification as useful evidence.
- Review the final diff before committing or merging.

Forge also has commands for automation, including `forge exec` and
`forge swarm`. They are listed in the [Reference](reference.md); the interactive
workflow is the normal place to start.

See [Execution modes](../README.md#execution-modes),
[Shell sandbox](shell_sandbox.md), and [Security model](security_model.md) for
the controls Forge builds on.
