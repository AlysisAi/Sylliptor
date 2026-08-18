from __future__ import annotations

from datetime import date

import pytest

from scripts.release.audit_locked_dependencies import (
    LockedDependencyAuditError,
    validate_audit_report,
)


def _report(*, vulnerabilities: list[object], adverse: list[object]) -> dict[str, object]:
    return {
        "schema": {"version": "preview"},
        "summary": {
            "audited_packages": 10,
            "vulnerabilities": len(vulnerabilities),
            "adverse_statuses": len(adverse),
        },
        "vulnerabilities": vulnerabilities,
        "adverse_statuses": adverse,
    }


def test_allows_only_reviewed_socksio_archival_status() -> None:
    report = _report(
        vulnerabilities=[],
        adverse=[{"name": "socksio", "status": "archived", "reason": None}],
    )

    assert validate_audit_report(report, today=date(2026, 10, 31)) == (("socksio", "archived"),)


def test_rejects_expired_socksio_archival_waiver() -> None:
    report = _report(
        vulnerabilities=[],
        adverse=[{"name": "socksio", "status": "archived", "reason": None}],
    )

    with pytest.raises(LockedDependencyAuditError, match="re-review"):
        validate_audit_report(report, today=date(2026, 11, 1))


@pytest.mark.parametrize(
    "report",
    [
        _report(vulnerabilities=[{"id": "CVE-TEST", "display_id": "CVE-TEST"}], adverse=[]),
        _report(vulnerabilities=[], adverse=[{"name": "other", "status": "archived"}]),
        _report(vulnerabilities=[], adverse=[{"name": "socksio", "status": "deprecated"}]),
        {"summary": {}, "vulnerabilities": [], "adverse_statuses": "invalid"},
    ],
)
def test_rejects_vulnerabilities_unapproved_statuses_and_schema_drift(
    report: dict[str, object],
) -> None:
    with pytest.raises(LockedDependencyAuditError):
        validate_audit_report(report)


def test_rejects_inconsistent_summary() -> None:
    report = _report(vulnerabilities=[], adverse=[])
    summary = report["summary"]
    assert isinstance(summary, dict)
    summary["adverse_statuses"] = 1

    with pytest.raises(LockedDependencyAuditError, match="inconsistent"):
        validate_audit_report(report)
