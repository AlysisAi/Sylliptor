# Sandbox

Canonical sandbox documentation lives in [Shell sandbox](shell_sandbox.md). This
short entry remains for historical links that reference `docs/sandbox.md`.

Sylliptor shell and verification runs default to strict sandboxing too. Verification sandbox mode is
(default `strict`) and does not fall back to host shell when the selected sandbox runtime cannot
enforce the requested network policy.

Default shell/verification constraints include:

- `network=off`
- strict mode with `bwrap` on supported Linux hosts
- warn mode with `docker` when explicitly configured

Server workers use `SYLLIPTOR_SERVER_WORKER_SANDBOX_MODE`; supported operator choices include
`strict` and `warn`, with `bwrap` or `docker` selected by the resolved sandbox backend. Deployment
policy decides the effective server worker mode.
