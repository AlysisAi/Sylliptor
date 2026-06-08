# Releasing

This is the maintainer checklist for publishing Sylliptor packages and sandbox images.

## Version And Tag

1. Bump the package version in `pyproject.toml` and `src/sylliptor_agent_cli/__init__.py`.
2. Update `CHANGELOG.md` with user-facing changes and known limitations.
3. Commit the release changes.
4. Create and push the release tag:

```bash
git tag v0.x.y
git push origin v0.x.y
```

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
