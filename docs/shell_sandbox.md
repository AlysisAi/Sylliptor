# Shell Sandbox

Sylliptor can run shell and verification commands inside a sandboxed environment. The sandbox is designed to reduce the blast radius of local command execution while keeping normal repository workflows usable.

The shell sandbox applies to `shell_run` tool executions. Verification commands and the `verify_run` tool use the verification sandbox path, which reuses the same backend settings unless configured otherwise.

Sandboxing is not a substitute for reviewing code, dependencies, or commands before running them.

## Quick Start

Diagnose the current machine:

```bash
sylliptor sandbox doctor --smoke
```

Run guided setup:

```bash
sylliptor sandbox setup
```

Pull default Docker images:

```bash
sylliptor sandbox pull
```

Useful commands:

| Goal | Command |
| --- | --- |
| Diagnose readiness | `sylliptor sandbox doctor [--smoke] [--env]` |
| Guided setup | `sylliptor sandbox setup [--no-pull]` |
| Pull images | `sylliptor sandbox pull [--no-server] [--image NAME]` |

## Backends

Sylliptor supports two sandbox backends:

- `bwrap`: Bubblewrap-based isolation, recommended on Linux when available.
- `docker`: cross-platform container isolation, recommended on macOS and Windows.

Backend selection defaults to `auto`.

### Bubblewrap

The Bubblewrap backend mounts the workspace at `/workspace`, provides isolated temporary and process namespaces where supported, and can use a hardened profile that avoids broad host mounts.

Install Bubblewrap with your operating-system package manager, for example:

```bash
sudo apt install bubblewrap
```

### Docker

The Docker backend runs each command in an ephemeral container, mounts the workspace at `/workspace`, disables network by default, drops Linux capabilities, and avoids forwarding the host environment unless explicitly configured.

Install Docker Desktop on macOS or Windows. On Linux, make sure the current user can access the Docker daemon.

## Default Settings

Current defaults:

| Setting | Default |
| --- | --- |
| `shell_sandbox.mode` | `strict` |
| `shell_sandbox.backend` | `auto` |
| `shell_sandbox.network` | `off` |
| `shell_sandbox.bwrap_profile` | `hardened` |
| `shell_sandbox.clear_env` | `true` |
| `shell_sandbox.protect_repo_meta` | `true` |

In environment shorthand, the production default is:

- `network=off`

Mode behavior:

- `strict`: require a usable sandbox backend.
- `warn`: attempt sandbox execution and warn on setup problems; it does not fall back to host shell.
- `off`: run on the host shell. This is an explicit unsafe opt-in.

Host shell execution requires:

```bash
export SYLLIPTOR_SHELL_SANDBOX_MODE=off
```

or equivalent config:

```json
{
  "shell_sandbox": {
    "mode": "off"
  }
}
```

Use `off` only for trusted local work where you would run the same commands directly.

## Docker Images

Published GHCR images:

- `ghcr.io/alysisai/sylliptor-sandbox:base`: minimal shell sandbox image.
- `ghcr.io/alysisai/sylliptor-sandbox:dev`: default development image for shell and verification commands.
- `ghcr.io/alysisai/sylliptor-sandbox:server`: worker image for server mode.

Pull the default image:

```bash
docker pull ghcr.io/alysisai/sylliptor-sandbox:dev
```

Build locally:

```bash
docker build --build-arg VARIANT=dev -t sylliptor-sandbox:dev -f sandbox/Dockerfile sandbox/
```

Pin a production image by digest:

```bash
docker buildx imagetools inspect ghcr.io/alysisai/sylliptor-sandbox:dev
export SYLLIPTOR_SHELL_SANDBOX_DOCKER_IMAGE=ghcr.io/alysisai/sylliptor-sandbox@sha256:<digest>
```

When available, verify signature and provenance with your release process before pinning a digest.

## Environment Configuration

Environment variables override config values:

```bash
export SYLLIPTOR_SHELL_SANDBOX_MODE=strict
export SYLLIPTOR_SHELL_SANDBOX_BACKEND=auto
export SYLLIPTOR_SHELL_SANDBOX_NETWORK=off
export SYLLIPTOR_SHELL_SANDBOX_DOCKER_IMAGE=ghcr.io/alysisai/sylliptor-sandbox:dev
export SYLLIPTOR_SHELL_SANDBOX_CLEAR_ENV=1
```

Supported shell sandbox variables:

- `SYLLIPTOR_SHELL_SANDBOX_MODE=off|warn|strict`
- `SYLLIPTOR_SHELL_SANDBOX_BACKEND=auto|bwrap|docker`
- `SYLLIPTOR_SHELL_SANDBOX_NETWORK=off|on`
- `SYLLIPTOR_SHELL_SANDBOX_BWRAP_PROFILE=compat|hardened`
- `SYLLIPTOR_SHELL_SANDBOX_DOCKER_IMAGE=<image>`
- `SYLLIPTOR_SHELL_SANDBOX_CLEAR_ENV=0|1`
- `SYLLIPTOR_SHELL_SANDBOX_DOCKER_PIDS_LIMIT=<int>`
- `SYLLIPTOR_SHELL_SANDBOX_DOCKER_MEMORY=<value>`
- `SYLLIPTOR_SHELL_SANDBOX_DOCKER_CPUS=<value>`
- `SYLLIPTOR_SHELL_SANDBOX_DOCKER_READ_ONLY=0|1`
- `SYLLIPTOR_SHELL_SANDBOX_PROTECT_REPO_META=0|1`
- `SYLLIPTOR_SHELL_SANDBOX_DOCKER_ENV_ALLOWLIST=VAR1,VAR2`

Equivalent `config.json` shape:

```json
{
  "shell_sandbox": {
    "mode": "strict",
    "backend": "auto",
    "network": "off",
    "bwrap_profile": "hardened",
    "docker_image": "ghcr.io/alysisai/sylliptor-sandbox:dev",
    "clear_env": true,
    "docker_pids_limit": 256,
    "docker_memory": "1g",
    "docker_cpus": "1.5",
    "docker_read_only": true,
    "protect_repo_meta": true,
    "docker_env_allowlist": ["LANG"]
  }
}
```

## Verification Sandbox

Verification commands use `SYLLIPTOR_VERIFY_SANDBOX_MODE` or `verify_sandbox.mode`:

```bash
export SYLLIPTOR_VERIFY_SANDBOX_MODE=strict
```

Supported values are `off`, `warn`, and `strict` (default `strict`).

Example config:

```json
{
  "verify_sandbox": {
    "mode": "strict"
  }
}
```

Verification commands default to strict sandboxing too. Verification reuses the
shell sandbox backend, image, network policy, and environment settings. Keep
network disabled unless a verification command genuinely needs outbound access.

## Production Profile

For production-style or shared environments:

```bash
export SYLLIPTOR_SHELL_SANDBOX_MODE=strict
export SYLLIPTOR_SHELL_SANDBOX_BACKEND=docker
export SYLLIPTOR_SHELL_SANDBOX_NETWORK=off
export SYLLIPTOR_SHELL_SANDBOX_DOCKER_READ_ONLY=1
export SYLLIPTOR_SHELL_SANDBOX_PROTECT_REPO_META=1
export SYLLIPTOR_SHELL_SANDBOX_DOCKER_PIDS_LIMIT=256
export SYLLIPTOR_SHELL_SANDBOX_DOCKER_MEMORY=1g
export SYLLIPTOR_SHELL_SANDBOX_DOCKER_CPUS=1.5
export SYLLIPTOR_SHELL_SANDBOX_DOCKER_ENV_ALLOWLIST=LANG
```

Recommended practices:

- pin Docker images by digest
- keep network disabled by default
- use a custom image with preinstalled dependencies instead of downloading packages during verification
- keep repository metadata protection enabled
- use narrow environment allowlists

## Troubleshooting

Run:

```bash
sylliptor sandbox doctor --smoke --env
```

Common cases:

| Symptom | Likely cause | Next step |
| --- | --- | --- |
| No usable backend | Docker or Bubblewrap is missing or unavailable | Install/start the backend, then rerun doctor. |
| Docker daemon error | Docker Desktop is closed or permissions are missing | Start Docker or fix daemon access. |
| Image missing | Docker works but the sandbox image is not local | Run `sylliptor sandbox pull`. |
| Pull timeout | Registry, proxy, DNS, or auth issue | Retry with a longer timeout and inspect pull output. |
| `pytest` missing in `:base` | The base image is intentionally minimal | Use `:dev` or a custom image with project tools. |
| Slow WSL2 mounts | Repository is under `/mnt/c/...` | Prefer a Linux filesystem checkout under WSL. |
| `.git/` write denied | Repository metadata protection is active | Avoid writing repo metadata, or disable only for trusted local work. |

Disable repository metadata protection only when the task is trusted and needs it:

```bash
export SYLLIPTOR_SHELL_SANDBOX_PROTECT_REPO_META=0
```

## Limitations

- The shell sandbox isolates shell and verification command execution, not every host-side orchestration step.
- Sandboxed commands can still modify the mounted workspace unless the Docker read-only option or task policy prevents it.
- A sandbox reduces impact; it does not make untrusted code safe.
- Custom tools have their own subprocess execution and capability checks; they are not automatically routed through the shell sandbox.
