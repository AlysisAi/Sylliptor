"""Turn-contract v2: apply-don't-advise + spec-literalism enforcement (step 4).

This is the pure, side-effect-free core of turn-contract v2. It builds on the
same machinery as steps 1-3: the acceptance-contract derivation records concrete
*expectations* from the task text (semantic understanding), and the completion
gate enforces them **mechanically** — fact matching only, never NL heuristics.

Two problems this step targets (evidence: SWE-bench Verified subsets 1-5):

* "described the fix, didn't apply it" — an execute-intent turn ends in prose
  with zero edits. The gate refuses to finalize such a turn silently; it demands
  a recorded ``advisory_completion`` disposition (reason + explanation) that is
  surfaced in the final summary.
* "alternative-mechanism fixes that contradict text the agent read" — the task
  names an expected output literal, a faulty locus, or a behavioral contract, and
  the agent ships something else. Each such *expectation* must reach an explicit
  disposition (confirmed / superseded / not_applicable) at finalization, else the
  turn finalizes honestly with an ``UNCONFIRMED EXPECTATIONS`` marker.

Design invariants (identical in spirit to steps 1-3):

* Semantic extraction of expectations is done in the contract-derivation step
  (see ``acceptance_contract.extract_task_expectations``); this module never
  applies NL/keyword heuristics to user or assistant text. The only text matching
  here is a *contract-literal substring lookup against observed command output*.
* Every function is pure and unit-testable without an LLM call.
* Dispositions and the advisory-completion reason are enum-validated. In this
  release production populates them mechanically (an ``expected_output`` literal
  observed in a post-edit run, or a named ``locus`` that was edited, confirms the
  expectation; a zero-edit execute finalization synthesizes an advisory reason).
  ``superseded`` / ``not_applicable`` remain first-class enum members so a future
  agent-provided disposition channel can populate them without touching the gate.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..branding import env_get

# ---------------------------------------------------------------------------
# Kill-switch (mirrors the route-arbitration / evidence-v2 / regression idiom)
# ---------------------------------------------------------------------------


def _turn_contract_v2_enabled(cfg: Any | None) -> bool:
    """Kill-switch for turn-contract v2 gate enforcement (step 4).

    ``SYLLIPTOR_TURN_CONTRACT_V2`` (off/0/false/no/disabled) wins over the config
    value; default is on. When off, expectation extraction may still run and log
    (contract derivation is unconditional), but the completion-gate policy for
    ``expectations_unaddressed`` and advisory-completion reverts to legacy.
    """
    env_value = env_get("SYLLIPTOR_TURN_CONTRACT_V2")
    if env_value is not None:
        normalized = str(env_value).strip().lower()
        if normalized in {"off", "0", "false", "no", "disabled"}:
            return False
        if normalized in {"on", "1", "true", "yes", "enabled"}:
            return True
    return bool(getattr(cfg, "turn_contract_v2_enabled", True))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ExpectationKind(StrEnum):
    """A concrete, checkable expectation extracted from the task text."""

    #: A literal string the task shows as the desired output/behavior.
    EXPECTED_OUTPUT = "expected_output"
    #: A file / function / commit / PR the task identifies as faulty or the fix site.
    NAMED_LOCUS = "named_locus"
    #: A one-sentence behavioral contract stated in the task ("X returns None, not {}").
    NAMED_BEHAVIOR = "named_behavior"


class ExpectationDisposition(StrEnum):
    """How an execute turn resolved a single expectation at finalization."""

    #: Backed by observed evidence (a linked run output) or by editing the locus.
    CONFIRMED = "confirmed"
    #: The agent states the expectation is obsolete/wrong (never blocks; counted).
    SUPERSEDED = "superseded"
    #: The expectation does not apply to the delivered work (with a stated reason).
    NOT_APPLICABLE = "not_applicable"


class AdvisoryCompletionReason(StrEnum):
    """Why an execute-intent turn finalized with zero verification-relevant edits."""

    NO_CHANGE_NEEDED = "no_change_needed"
    CANNOT_REPRODUCE = "cannot_reproduce"
    BLOCKED_MISSING_INFORMATION = "blocked_missing_information"
    OUT_OF_SCOPE_REQUEST = "out_of_scope_request"
    OTHER = "other"


#: Cap on extracted expectations. Precision over recall — a short, high-signal list
#: keeps the gate's disposition demand tractable and its markers legible.
MAX_EXPECTATIONS = 8

#: Minimum non-space length for an ``expected_output`` literal to be usable for
#: evidence matching. Not a fuzzy heuristic — a precision floor that stops a
#: 1-2 char literal (``{}``, ``[]``, a lone digit) from matching arbitrary output.
MIN_EXPECTED_OUTPUT_LITERAL_LEN = 3


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Expectation:
    """One concrete expectation extracted from the task text (contract v2)."""

    expectation_id: str
    kind: ExpectationKind
    #: The verbatim quote of the expected output / locus / behavior.
    text: str
    #: The surrounding source clause the expectation was extracted from.
    source_quote: str = ""

    def as_payload(self) -> dict[str, Any]:
        return {
            "id": self.expectation_id,
            "kind": self.kind.value,
            "text": self.text,
            "source_quote": self.source_quote,
        }


@dataclass(frozen=True)
class ExpectationEvidence:
    """A post-edit run whose output contains an ``expected_output`` literal."""

    expectation_id: str
    normalized_command: str
    generation: int

    def as_payload(self) -> dict[str, Any]:
        return {
            "expectation_id": self.expectation_id,
            "normalized_command": self.normalized_command,
            "generation": self.generation,
        }


@dataclass(frozen=True)
class DispositionRecord:
    """A recorded disposition for one expectation."""

    expectation_id: str
    disposition: ExpectationDisposition
    rationale: str = ""
    #: For ``confirmed``: the normalized command of the evidencing run, if any.
    evidence_command: str = ""

    def as_payload(self) -> dict[str, Any]:
        return {
            "expectation_id": self.expectation_id,
            "disposition": self.disposition.value,
            "rationale": self.rationale,
            "evidence_command": self.evidence_command,
        }


@dataclass(frozen=True)
class AdvisoryCompletion:
    """A recorded reason for finalizing an execute turn with no material edits."""

    reason: AdvisoryCompletionReason
    explanation: str = ""

    def as_payload(self) -> dict[str, Any]:
        return {
            "reason": self.reason.value,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class ExpectationAssessment:
    """The mechanical disposition of every expectation at finalization."""

    confirmed: tuple[str, ...] = ()
    superseded: tuple[str, ...] = ()
    not_applicable: tuple[str, ...] = ()
    unaddressed: tuple[str, ...] = ()

    @property
    def has_unaddressed(self) -> bool:
        return bool(self.unaddressed)

    def as_payload(self) -> dict[str, Any]:
        return {
            "confirmed": list(self.confirmed),
            "superseded": list(self.superseded),
            "not_applicable": list(self.not_applicable),
            "unaddressed": list(self.unaddressed),
            "superseded_count": len(self.superseded),
        }


# ---------------------------------------------------------------------------
# Enum coercion / validation (pure)
# ---------------------------------------------------------------------------


def coerce_expectation_disposition(value: Any) -> ExpectationDisposition | None:
    """Return the enum member for ``value`` or ``None`` (never raises)."""
    if isinstance(value, ExpectationDisposition):
        return value
    try:
        return ExpectationDisposition(str(value).strip().lower())
    except (ValueError, AttributeError):
        return None


def coerce_advisory_reason(value: Any) -> AdvisoryCompletionReason | None:
    """Return the enum member for ``value`` or ``None`` (never raises)."""
    if isinstance(value, AdvisoryCompletionReason):
        return value
    try:
        return AdvisoryCompletionReason(str(value).strip().lower())
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Evidence linker (pure) — the ONLY text matching in this module
# ---------------------------------------------------------------------------


def _usable_expected_output_literal(expectation: Expectation) -> str:
    if expectation.kind != ExpectationKind.EXPECTED_OUTPUT:
        return ""
    literal = str(expectation.text or "")
    if len(literal.strip()) < MIN_EXPECTED_OUTPUT_LITERAL_LEN:
        return ""
    return literal


def match_expectation_evidence(
    expectations: Iterable[Expectation],
    run_outputs: Iterable[Mapping[str, Any]],
) -> list[ExpectationEvidence]:
    """Substring-match each ``expected_output`` literal against run outputs.

    ``run_outputs`` are the recorded post-edit qualifying runs, each a mapping
    with ``normalized_command``, ``output`` (the observed stdout/stderr text) and
    ``generation``. A hit records supporting evidence for the expectation. The
    match is a plain ``in`` substring test on the contract literal against each
    run's output **individually** (never a concatenation), so a literal cannot
    spuriously match across two runs' outputs or across a capture boundary.
    Multiline literals match verbatim. Deterministic: for each expectation the
    first matching run (in the given order) is the evidence.
    """
    runs = [
        (
            str(run.get("normalized_command") or ""),
            str(run.get("output") or ""),
            int(run.get("generation") or 0),
        )
        for run in run_outputs
    ]
    evidence: list[ExpectationEvidence] = []
    for expectation in expectations:
        literal = _usable_expected_output_literal(expectation)
        if not literal:
            continue
        for command, output, generation in runs:
            if literal in output:
                evidence.append(
                    ExpectationEvidence(
                        expectation_id=expectation.expectation_id,
                        normalized_command=command,
                        generation=generation,
                    )
                )
                break
    return evidence


# ---------------------------------------------------------------------------
# Locus path normalization (pure) — mechanical named_locus confirmation
# ---------------------------------------------------------------------------


def normalize_locus_path(text: str) -> str:
    """Best-effort normalization of a named-locus path for edited-path matching.

    Mirrors the repo-relative normalization used for touched paths (strip quotes,
    normalize separators, drop leading ``./``), casefolded. A locus that is a bare
    symbol (no path shape) simply will not match any edited path — which is the
    safe direction (it stays unaddressed rather than being wrongly confirmed).
    """
    cleaned = str(text or "").strip().strip("`'\"").replace("\\", "/")
    cleaned = cleaned.rstrip(".,;:!?)]}").strip()
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.casefold()


def _named_locus_confirmed_by_edit(expectation: Expectation, edited_loci: set[str]) -> bool:
    if expectation.kind != ExpectationKind.NAMED_LOCUS:
        return False
    target = normalize_locus_path(expectation.text)
    if not target:
        return False
    return target in edited_loci


# ---------------------------------------------------------------------------
# Assessment (pure)
# ---------------------------------------------------------------------------


def assess_expectations(
    *,
    expectations: Iterable[Expectation],
    evidence: Iterable[ExpectationEvidence],
    edited_loci: Iterable[str] = (),
    dispositions: Mapping[str, DispositionRecord] | None = None,
) -> ExpectationAssessment:
    """Bucket every expectation into confirmed / superseded / not_applicable / unaddressed.

    Precedence per expectation:

    1. an explicit recorded ``superseded`` / ``not_applicable`` disposition
       (agent-declared; never blocks) wins;
    2. otherwise the expectation is *confirmed* when it has linked evidence (an
       ``expected_output`` literal observed in a post-edit run), OR its
       ``named_locus`` path was edited this turn, OR an explicit ``confirmed``
       disposition was recorded;
    3. otherwise it is *unaddressed* — the gate demands it be resolved.

    Pure and deterministic; no LLM, no NL heuristics.
    """
    disposition_map = dict(dispositions or {})
    evidenced_ids = {item.expectation_id for item in evidence}
    normalized_loci = {normalize_locus_path(path) for path in edited_loci}
    normalized_loci.discard("")

    confirmed: list[str] = []
    superseded: list[str] = []
    not_applicable: list[str] = []
    unaddressed: list[str] = []
    for expectation in expectations:
        recorded = disposition_map.get(expectation.expectation_id)
        if recorded is not None and recorded.disposition == ExpectationDisposition.SUPERSEDED:
            superseded.append(expectation.expectation_id)
            continue
        if recorded is not None and recorded.disposition == ExpectationDisposition.NOT_APPLICABLE:
            not_applicable.append(expectation.expectation_id)
            continue
        if (
            expectation.expectation_id in evidenced_ids
            or _named_locus_confirmed_by_edit(expectation, normalized_loci)
            or (recorded is not None and recorded.disposition == ExpectationDisposition.CONFIRMED)
        ):
            confirmed.append(expectation.expectation_id)
            continue
        unaddressed.append(expectation.expectation_id)

    return ExpectationAssessment(
        confirmed=tuple(confirmed),
        superseded=tuple(superseded),
        not_applicable=tuple(not_applicable),
        unaddressed=tuple(unaddressed),
    )


# ---------------------------------------------------------------------------
# Finalization markers / summary lines (fail honest, never silent)
# ---------------------------------------------------------------------------


_UNCONFIRMED_EXPECTATIONS_MARKER_PREFIX = (
    "\n\n---\n"
    "⚠️ UNCONFIRMED EXPECTATIONS: {ids}. The task named these concrete "
    "expectations, and I could neither confirm them by observed evidence nor "
    "record why they no longer apply. This result is finalized with these "
    "expectations UNCONFIRMED."
)


def build_unconfirmed_expectations_marker(ids: list[str] | tuple[str, ...]) -> str:
    """Visible marker appended when a turn finalizes with unaddressed expectations."""
    joined = ", ".join(str(item) for item in ids if str(item).strip())
    return _UNCONFIRMED_EXPECTATIONS_MARKER_PREFIX.format(ids=joined)


DEFAULT_ADVISORY_COMPLETION_EXPLANATION = (
    "This execute-intent turn finalized with no material edits, and no explicit "
    "reason was recorded."
)


def resolve_advisory_completion(
    recorded: AdvisoryCompletion | None,
) -> AdvisoryCompletion:
    """Return the recorded advisory completion, or a synthesized ``other`` default.

    The gate never lets an execute turn finalize with zero material edits *silently*
    — it always resolves an advisory-completion disposition. When the agent recorded
    one it is used verbatim; otherwise a factual ``other`` reason is synthesized so
    the finalization is honest and non-silent (never omission).
    """
    if recorded is not None:
        return recorded
    return AdvisoryCompletion(
        reason=AdvisoryCompletionReason.OTHER,
        explanation=DEFAULT_ADVISORY_COMPLETION_EXPLANATION,
    )


_ADVISORY_COMPLETION_SUMMARY_PREFIX = "\n\n---\nNo changes made: {reason} — {explanation}"


def build_advisory_completion_summary(
    reason: AdvisoryCompletionReason | str,
    explanation: str,
) -> str:
    """Visible line appended when an execute turn finalizes with no material edits."""
    reason_value = reason.value if isinstance(reason, AdvisoryCompletionReason) else str(reason)
    clean_explanation = str(explanation or "").strip() or "no further detail provided"
    return _ADVISORY_COMPLETION_SUMMARY_PREFIX.format(
        reason=reason_value,
        explanation=clean_explanation,
    )
