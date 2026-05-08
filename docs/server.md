# Server Mode

## Overview

`sylliptor server start` exposes an HTTP API for creating run workspaces and executing jobs in an
outer worker sandbox.

- Run workspace mount inside worker: `/workspace`
- Job artifact mount inside worker: `/sylliptor_job`
- Worker config/data paths inside sandbox:
  - `SYLLIPTOR_CONFIG_DIR=/sylliptor_job/config`
  - `SYLLIPTOR_DATA_DIR=/sylliptor_job/data`

## API Mode Support

Server endpoints expose modes as follows:

- `POST /v1/runs/{run_id}/jobs/run`: `readonly|review|auto|fullaccess`
- `POST /v1/runs/{run_id}/jobs/forge_exec`: `readonly|review|auto|fullaccess`
- `POST /v1/runs/{run_id}/jobs/forge_swarm`: `auto` only

Swarm remains auto-only in server mode because non-dry-run swarm orchestration enforces
`--mode auto` at runtime.

`fullaccess` disables mode-level approval/guard prompts in the inner agent runtime, but jobs still
execute inside the configured outer worker sandbox backend (`bwrap`/`docker`) and its policy.

## Authentication And Locality Policy

Protected routes use `SYLLIPTOR_SERVER_TOKEN` as follows:

- If `SYLLIPTOR_SERVER_TOKEN` is set, requests must include
  `Authorization: Bearer <token>`.
  - Missing Bearer token: `401`
  - Wrong token: `403`
- If `SYLLIPTOR_SERVER_TOKEN` is unset, protected routes only allow localhost clients
  (`127.0.0.1`, `::1`, `localhost`).
  - Non-localhost clients are rejected with `403`.

`/health` remains public.

## Start The Server

Server mode uses the same package runtime baseline as the CLI: Python 3.11 or newer.

Install server dependencies:

```bash
python -m pip install "sylliptor-agent-cli[server]"
```

Start:

```bash
sylliptor server start --host 127.0.0.1 --port 7070
```

Minimal authenticated API example:

```bash
export SYLLIPTOR_SERVER_TOKEN="your-token"

RUN_ID=$(curl -sS -X POST \
  -H "Authorization: Bearer $SYLLIPTOR_SERVER_TOKEN" \
  http://127.0.0.1:7070/v1/runs/empty | python -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')

curl -sS -X POST \
  -H "Authorization: Bearer $SYLLIPTOR_SERVER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"instruction":"Implement the task","mode":"fullaccess"}' \
  "http://127.0.0.1:7070/v1/runs/${RUN_ID}/jobs/run"
```

`forge_exec` also supports `mode=fullaccess`; `forge_swarm` remains `mode=auto` only.

Optional data directory override:

```bash
sylliptor server start --data-dir /var/lib/sylliptor-server
```

`SYLLIPTOR_SERVER_MAX_JOBS` controls the server worker-pool size. It bounds both:

- how many jobs may execute in worker subprocesses at the same time
- how many server worker threads are created to service queued jobs

## Upload Limits

`POST /v1/runs` stages uploaded ZIP archives to a temporary file before import.

- `SYLLIPTOR_SERVER_MAX_UPLOAD_BYTES` is enforced while the multipart body is being read in chunks.
- Oversized uploads are rejected as soon as the configured limit is crossed; the server does not
  buffer the entire archive into a single Python `bytes` object first.
- The staged temporary ZIP is removed after the request completes.
- Existing ZIP validation still applies after staging succeeds, including bad-ZIP rejection, path
  sanitization, and the uncompressed-size guard during extraction.

## Job Queue And Cancellation

Jobs move through `queued -> running -> succeeded|failed|cancelled`.

- `start_job()` enqueues work onto a fixed worker pool instead of creating one new thread per job.
- When all worker slots are busy, additional jobs remain `queued` without spawning extra worker
  threads or subprocesses.
- Queued jobs cancelled via `POST /v1/jobs/{job_id}/cancel` become terminal `cancelled`
  immediately and are skipped by workers.
- Cancelling a running job requests process termination when possible; once the worker finishes
  teardown, the job becomes terminal `cancelled`.
- Job metadata stays under the per-job directory while queued/running, and `result.json` is
  written when the job reaches a terminal state.

## Worker Backends

`SYLLIPTOR_SERVER_WORKER_BACKEND` selects the outer worker sandbox backend:

- `bwrap` on Linux by default
- `docker` on macOS/Windows by default

Worker sandbox mode default depends on backend when
`SYLLIPTOR_SERVER_WORKER_SANDBOX_MODE` is unset:

- `strict` for `bwrap`
- `warn` for `docker`

Inside workers, nested tool sandbox defaults are hardened:

- `SYLLIPTOR_SHELL_SANDBOX_BACKEND=bwrap`
- `SYLLIPTOR_SHELL_SANDBOX_BWRAP_PROFILE=hardened`
- `SYLLIPTOR_SHELL_SANDBOX_NETWORK=off`
- `SYLLIPTOR_SHELL_SANDBOX_CLEAR_ENV=1`
- `SYLLIPTOR_SHELL_SANDBOX_PROTECT_REPO_META=1`

## Model And Base URL Policy

Security defaults are server-operator first:

- Client `base_url` override is disabled by default
  (`SYLLIPTOR_SERVER_ALLOW_CLIENT_BASE_URL=0`).
- If client `base_url` override is enabled, client-provided URLs must use
  `http://` or `https://`.
- Client model override is enabled by default
  (`SYLLIPTOR_SERVER_ALLOW_CLIENT_MODEL=1`).
- Set fixed defaults with:
  - `SYLLIPTOR_SERVER_MODEL`
  - `SYLLIPTOR_SERVER_BASE_URL`

`SYLLIPTOR_SERVER_BASE_URL` must be an `http://` or `https://` URL.

## Docker Worker Image

By default, docker server workers use `ghcr.io/alysisai/sylliptor-sandbox:server`.

Build the server worker image:

```bash
docker build --build-arg VARIANT=server -t sylliptor-sandbox:server -f sandbox/Dockerfile .
```

Override image tag used by server workers:

```bash
export SYLLIPTOR_SERVER_DOCKER_IMAGE=sylliptor-sandbox:server
```

The image should include:

- `sylliptor-agent-cli` installed as a Python package
- `git`
- `ripgrep`
- `ca-certificates`
- `bubblewrap` (recommended; best-effort inner sandbox support)
