from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


class AttestationReceiptValidationError(ValueError):
    """Raised when verified GitHub attestation output is not release-bound."""


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttestationReceiptValidationError(f"{label} is not valid JSON: {path}") from exc


def _sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise AttestationReceiptValidationError(f"artifact must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_receipt(
    receipt_path: Path,
    artifact_path: Path,
    *,
    predicate_type: str,
    expected_predicate_path: Path | None = None,
) -> None:
    receipt = _load_json(receipt_path, "attestation verification receipt")
    if not isinstance(receipt, list) or not receipt:
        raise AttestationReceiptValidationError(
            "attestation verification receipt must be a non-empty JSON array"
        )
    artifact_digest = _sha256(artifact_path)
    expected_predicate = (
        _load_json(expected_predicate_path, "expected predicate")
        if expected_predicate_path is not None
        else None
    )

    for entry in receipt:
        if not isinstance(entry, dict):
            continue
        result = entry.get("verificationResult")
        if not isinstance(result, dict):
            continue
        statement = result.get("statement")
        if not isinstance(statement, dict) or statement.get("predicateType") != predicate_type:
            continue
        subjects = statement.get("subject")
        if not isinstance(subjects, list) or not any(
            isinstance(subject, dict)
            and isinstance(subject.get("digest"), dict)
            and subject["digest"].get("sha256") == artifact_digest
            for subject in subjects
        ):
            continue
        if expected_predicate is not None and statement.get("predicate") != expected_predicate:
            continue
        return

    detail = " with the exact expected predicate" if expected_predicate_path is not None else ""
    raise AttestationReceiptValidationError(
        f"no verified {predicate_type!r} attestation binds the exact artifact SHA-256{detail}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate JSON emitted by `gh attestation verify --format json`."
    )
    parser.add_argument("receipt", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--predicate-type", required=True)
    parser.add_argument("--expected-predicate", type=Path)
    args = parser.parse_args(argv)
    try:
        validate_receipt(
            args.receipt,
            args.artifact,
            predicate_type=args.predicate_type,
            expected_predicate_path=args.expected_predicate,
        )
    except AttestationReceiptValidationError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
