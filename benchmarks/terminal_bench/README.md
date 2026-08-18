# Benchmarking Sylliptor with Terminal-Bench

This directory contains the public adapters needed to run Sylliptor against
[Terminal-Bench](https://www.tbench.ai/) through
[Harbor](https://www.harborframework.com/). The goal is reproducibility: the
agent, model, package, dataset, timeout, and raw outputs should all be explicit.

The benchmark integration is intentionally separate from Sylliptor's normal
runtime. It does not include private datasets, unpublished results, or hidden
execution modes.

## What is included

| Adapter | Use |
| --- | --- |
| `benchmarks.terminal_bench.harbor_agent:SylliptorAgent` | Recommended adapter for current Harbor releases |
| `benchmarks.terminal_bench.sylliptor_agent:SylliptorSimpleAgent` | Compatibility adapter for the legacy `terminal-bench` runner |

Both adapters install Sylliptor inside the task container and invoke
`sylliptor run` non-interactively. The current Harbor adapter installs an exact
wheel supplied by the benchmark operator. The legacy adapter copies a clean
snapshot of the local source tree.

## Prerequisites

- Docker with enough resources for the selected Terminal-Bench tasks.
- Harbor installed according to its official documentation.
- A Sylliptor wheel built from the exact commit being evaluated.
- An OpenAI-compatible model endpoint and API key.

Build the wheel from the repository root:

```bash
uv build
```

Set benchmark configuration in your shell. Keep real credentials outside the
repository and shell command arguments:

```bash
export SYLLIPTOR_WHEEL="/absolute/path/to/dist/sylliptor_agent_cli-<version>-py3-none-any.whl"
export SYLLIPTOR_BENCH_VERSION="<git-commit-or-release>"
export SYLLIPTOR_API_KEY="<provider-api-key>"
export SYLLIPTOR_BASE_URL="https://provider.example/v1"
export SYLLIPTOR_MODEL="<provider-model-id>"
```

No provider, endpoint, or model is selected implicitly. This prevents a local
environment from silently changing the configuration being measured.

## Quick start with Harbor

From the repository root, run the public Terminal-Bench 2.1 dataset with the
custom installed-agent adapter:

```bash
harbor run \
  --dataset terminal-bench/terminal-bench-2-1 \
  --agent benchmarks.terminal_bench.harbor_agent:SylliptorAgent \
  --model "$SYLLIPTOR_MODEL"
```

Harbor CLI flags can change between releases. If your installed version uses a
different dataset or custom-agent flag, follow Harbor's
[agent documentation](https://www.harborframework.com/docs/agents) while
keeping the same adapter import path.

Start with a small task subset before launching the full dataset. Terminal
benchmarks can be expensive and may run for many hours.

## Configuration

Required environment variables:

| Variable | Purpose |
| --- | --- |
| `SYLLIPTOR_WHEEL` | Absolute path to the wheel evaluated by Harbor |
| `SYLLIPTOR_API_KEY` | Provider credential passed to the container at run time |
| `SYLLIPTOR_BASE_URL` | OpenAI-compatible provider endpoint |
| `SYLLIPTOR_MODEL` | Exact provider model identifier |

Useful optional variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SYLLIPTOR_BENCH_VERSION` | Wheel filename | Version recorded by the adapter |
| `SYLLIPTOR_RUN_PROFILE` | `auto` | Sylliptor execution profile |
| `SYLLIPTOR_MAX_STEPS` | Sylliptor default | Maximum agent steps |
| `SYLLIPTOR_DEADLINE_SECONDS` | Harbor-controlled | Inner Sylliptor deadline |
| `SYLLIPTOR_LLM_TIMEOUT_S` | Sylliptor default | Per-request model timeout |
| `SYLLIPTOR_WEB_SEARCH_MODE` | `off` | Web-search policy for the run |
| `SYLLIPTOR_EXTRA_ARGS` | Empty | Additional CLI arguments parsed with shell-style quoting |

The legacy adapter additionally requires the final effective host timeout as
`managed_host_agent_timeout_sec`. It subtracts setup time and a shutdown
reserve before passing a required deadline to Sylliptor. This lets Sylliptor
finalize and flush diagnostics before the outer runner terminates the task.

## How a run works

1. Harbor creates an isolated task container.
2. The adapter uploads the selected wheel and the public `setup.sh`.
3. `setup.sh` creates a dedicated Python environment and installs that wheel.
4. The adapter runs one headless Sylliptor task with the supplied instruction.
5. Harbor runs the dataset's verifier and stores the trial result and artifacts.

Sylliptor's shell sandbox is disabled inside the benchmark container because
the container is the isolation boundary. Do not run the benchmark adapter
directly against a host workspace.

## Reproducible reporting

When publishing a result, include:

- Sylliptor commit, release, and wheel SHA-256.
- Harbor version and exact dataset name/version.
- Model identifier, endpoint provider, and reasoning settings.
- Task subset, number of independent trials, concurrency, and timeout policy.
- Whether web search or any additional tools were enabled.
- Aggregate score together with per-task results and failures.
- Raw Harbor job artifacts, with secrets and user data removed.

Run multiple isolated trials when making comparative claims. A single run is
useful for development but not enough to characterize agent performance.

## Security and privacy

- Credentials are read from environment variables and are never embedded in
  the generated `sylliptor run` command.
- The setup phase does not read provider credentials.
- Setup and agent output are written to Harbor's task artifact directories.
- Review artifacts before publishing them; task output can contain repository
  content or model-generated sensitive text.
- Use dedicated, rate-limited benchmark credentials whenever possible.

## Development

Run the adapter contract tests without installing Harbor:

```bash
pytest -q tests/test_terminal_bench_deadline_adapter.py
```

The module provides small compatibility shims when the optional benchmark
packages are absent, so the public safety and command-construction contracts
remain covered by the normal repository test suite.
