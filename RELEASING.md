# Releasing

This is the maintainer checklist for publishing Sylliptor packages and sandbox images.

## Prepare The Release

1. Set the same semantic version in `pyproject.toml`,
   `src/sylliptor_agent_cli/__init__.py`, and the Sylliptor project entry in
   `uv.lock`.
2. Move the completed public changes from `[Unreleased]` into a dated section in
   `CHANGELOG.md`.
3. Confirm that the lockfile is current and the release metadata agrees:

```bash
uv lock --check
uv run pytest -q tests/test_release_smoke.py
```

4. Run the release-quality checks locally:

```bash
uv sync --frozen --no-editable --extra dev
uv run python scripts/release/audit_locked_dependencies.py
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

5. Build into an empty directory and validate both distributions:

```bash
uv build --no-create-gitignore --no-build-isolation --no-sources --out-dir dist/python
uv run python scripts/release/validate_python_distributions.py dist/python
```

## Commit And Tag

1. Put the complete public release in one reviewable commit named
   `release: v0.x.y`. If changes arrived through another branch, squash or
   fast-forward them so the public release does not introduce a merge commit.
2. Make sure that exact commit is on the default branch before tagging it.
3. Create and push the matching tag:

```bash
git tag v0.x.y
git push origin v0.x.y
```

Pushing the tag starts the release workflow. It verifies the tag, version, default
branch ancestry, tests, distributions, dependency audit, SBOM, and provenance before
publishing.

## PyPI

Release tags build the wheel and source distribution. The publish job expects PyPI trusted
publishing to be configured for this repository and release workflow.

After the workflow finishes:

- Confirm the package page shows the expected version.
- Install the package in a clean environment.
- Run `sylliptor --help`.

## Sandbox Images

Sandbox images are published under:

```text
ghcr.io/alysisai/sylliptor-sandbox
```

Each variant is published as:

- `:<variant>` for the moving variant tag, for example `:dev`
- `:<variant>-<sha12>` for the immutable per-commit tag
- `:<variant>-<git-tag>` for release tags

The default variant is `dev`.

## Verify A Release Image

Pull the image:

```bash
docker pull ghcr.io/alysisai/sylliptor-sandbox:dev
```

For production use, prefer a digest-pinned image:

```bash
docker buildx imagetools inspect ghcr.io/alysisai/sylliptor-sandbox:dev
export SYLLIPTOR_SHELL_SANDBOX_DOCKER_IMAGE=ghcr.io/alysisai/sylliptor-sandbox@sha256:<digest>
```

Verify signature and provenance when release signing is enabled:

```bash
cosign verify ghcr.io/alysisai/sylliptor-sandbox@<digest> \
  --certificate-identity-regexp 'https://github\.com/AlysisAi/Sylliptor/.*' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com

gh attestation verify oci://ghcr.io/alysisai/sylliptor-sandbox@<digest> \
  --owner AlysisAi
```

## Troubleshooting

- GHCR rate limits: authenticate before repeated pulls.
- Package visibility: confirm the GHCR package is public before public launch.
- Vulnerability findings: review the advisory, decide whether it is exploitable, then patch or
  document an explicit temporary exception.
