from __future__ import annotations

import json
import subprocess
from datetime import date, datetime, timezone
from typing import Any


class LockedDependencyAuditError(RuntimeError):
    """The frozen dependency set failed the release security policy."""


# socksio 1.0.0 is an archived but vulnerability-free transitive dependency of ddgs via
# httpx[socks]. It remains allowlisted only as an adverse maintenance status; any vulnerability,
# any additional adverse package, any status change, or passing the mandatory review date still
# blocks the release.
_ALLOWED_ADVERSE_STATUSES = {("socksio", "archived"): date(2026, 10, 31)}


def validate_audit_report(
    payload: object,
    *,
    today: date | None = None,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(payload, dict):
        raise LockedDependencyAuditError("uv audit did not return a JSON object.")
    vulnerabilities = payload.get("vulnerabilities")
    adverse_statuses = payload.get("adverse_statuses")
    summary = payload.get("summary")
    if (
        not isinstance(vulnerabilities, list)
        or not isinstance(adverse_statuses, list)
        or not isinstance(summary, dict)
    ):
        raise LockedDependencyAuditError("uv audit JSON schema is unsupported.")
    if vulnerabilities or summary.get("vulnerabilities") != 0:
        identifiers = sorted(
            str(item.get("display_id") or item.get("id") or "unknown")
            for item in vulnerabilities
            if isinstance(item, dict)
        )
        raise LockedDependencyAuditError(
            f"Frozen dependencies contain known vulnerabilities: {identifiers or ['unknown']}"
        )

    observed: set[tuple[str, str]] = set()
    for item in adverse_statuses:
        if not isinstance(item, dict):
            raise LockedDependencyAuditError("uv audit adverse status is malformed.")
        name = item.get("name")
        status = item.get("status")
        if not isinstance(name, str) or not isinstance(status, str):
            raise LockedDependencyAuditError("uv audit adverse status identity is malformed.")
        observed.add((name, status))
    if len(observed) != len(adverse_statuses):
        raise LockedDependencyAuditError("uv audit returned duplicate adverse statuses.")
    unexpected = observed - _ALLOWED_ADVERSE_STATUSES.keys()
    if unexpected:
        raise LockedDependencyAuditError(
            f"Frozen dependencies contain unapproved adverse statuses: {sorted(unexpected)}"
        )
    if summary.get("adverse_statuses") != len(observed):
        raise LockedDependencyAuditError("uv audit adverse-status summary is inconsistent.")
    review_date = today or datetime.now(timezone.utc).date()
    expired = sorted(
        (name, status, expiry.isoformat())
        for (name, status), expiry in _ALLOWED_ADVERSE_STATUSES.items()
        if (name, status) in observed and review_date > expiry
    )
    if expired:
        raise LockedDependencyAuditError(
            f"Frozen dependency adverse-status waivers require re-review: {expired}"
        )
    return tuple(sorted(observed))


def audit_locked_dependencies() -> tuple[tuple[str, str], ...]:
    completed = subprocess.run(
        [
            "uv",
            "audit",
            "--locked",
            "--output-format",
            "json",
            "--preview-features",
            "audit-command",
            "--preview-features",
            "json-output",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in {0, 1}:
        raise LockedDependencyAuditError(
            f"uv audit failed before producing a report (exit {completed.returncode})."
        )
    try:
        payload: Any = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise LockedDependencyAuditError("uv audit did not produce valid JSON.") from exc
    return validate_audit_report(payload)


def main() -> int:
    allowlisted = audit_locked_dependencies()
    if allowlisted:
        print(
            "Dependency audit passed with the reviewed maintenance-status allowlist: "
            + ", ".join(f"{name} ({status})" for name, status in allowlisted)
        )
    else:
        print("Dependency audit passed with no vulnerabilities or adverse statuses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
