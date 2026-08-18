from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.release.validate_github_attestation_receipt import (
    AttestationReceiptValidationError,
    validate_receipt,
)

PROVENANCE = "https://slsa.dev/provenance/v1"
CYCLONEDX = "https://cyclonedx.org/bom"


def _receipt(
    path: Path,
    artifact: Path,
    *,
    predicate_type: str,
    predicate: object,
) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "verificationResult": {
                        "statement": {
                            "predicateType": predicate_type,
                            "subject": [
                                {
                                    "name": artifact.name,
                                    "digest": {
                                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()
                                    },
                                }
                            ],
                            "predicate": predicate,
                        }
                    }
                }
            ]
        ),
        encoding="utf-8",
    )


def test_accepts_exact_artifact_and_semantically_equal_sbom(tmp_path: Path) -> None:
    artifact = tmp_path / "candidate.whl"
    artifact.write_bytes(b"candidate")
    sbom = tmp_path / "candidate.cdx.json"
    sbom.write_text('{"components":[],"bomFormat":"CycloneDX"}', encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    _receipt(
        receipt,
        artifact,
        predicate_type=CYCLONEDX,
        predicate={"bomFormat": "CycloneDX", "components": []},
    )

    validate_receipt(
        receipt,
        artifact,
        predicate_type=CYCLONEDX,
        expected_predicate_path=sbom,
    )


@pytest.mark.parametrize("mutation", ["digest", "predicate_type", "predicate"])
def test_rejects_substituted_attestation_claim(tmp_path: Path, mutation: str) -> None:
    artifact = tmp_path / "candidate.whl"
    artifact.write_bytes(b"candidate")
    sbom = tmp_path / "candidate.cdx.json"
    sbom.write_text('{"bomFormat":"CycloneDX"}', encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    _receipt(
        receipt,
        artifact,
        predicate_type=CYCLONEDX,
        predicate={"bomFormat": "CycloneDX"},
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    statement = payload[0]["verificationResult"]["statement"]
    if mutation == "digest":
        statement["subject"][0]["digest"]["sha256"] = "0" * 64
    elif mutation == "predicate_type":
        statement["predicateType"] = PROVENANCE
    else:
        statement["predicate"] = {"bomFormat": "substituted"}
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AttestationReceiptValidationError, match="no verified"):
        validate_receipt(
            receipt,
            artifact,
            predicate_type=CYCLONEDX,
            expected_predicate_path=sbom,
        )


def test_rejects_empty_receipt(tmp_path: Path) -> None:
    artifact = tmp_path / "candidate.whl"
    artifact.write_bytes(b"candidate")
    receipt = tmp_path / "receipt.json"
    receipt.write_text("[]", encoding="utf-8")

    with pytest.raises(AttestationReceiptValidationError, match="non-empty"):
        validate_receipt(receipt, artifact, predicate_type=PROVENANCE)
