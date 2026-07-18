from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

RepoExecutionIntent = Literal["execute", "plan_or_analysis_only", "advisory_non_execution"]
SkillLifecycleIntent = Literal["create", "install", "enable", "disable", "remove", "validate"]


@dataclass(frozen=True)
class LocalMaterializationRequirement:
    required: bool
    confidence: float = 0.0
    output_paths: tuple[str, ...] = tuple()
    action_verb: str = ""
    evidence_span: str = ""
    reason: str = ""


_PLAN_OR_ANALYSIS_ONLY_MARKERS = (
    "plan only",
    "just a plan",
    "only a plan",
    "analysis only",
    "analyze only",
    "analyse only",
    "just analyze",
    "just analyse",
    "δωσε μονο πλανο",
    "μονο πλανο",
    "κανε μονο αναλυση",
    "μονο αναλυση",
)
_ADVISORY_NON_EXECUTION_MARKERS = (
    "review only",
    "just review",
    "just explain",
    "explain only",
    "answer only",
    "reply only",
    "read only",
    "read-only",
    "readonly",
    "suggest improvements without modifying anything",
    "suggest improvements without changing anything",
    "advice only",
    "μονο review",
    "μονο εξηγηση",
    "εξηγηση μονο",
    "μονο συμβουλες",
    "συμβουλες μονο",
)
_NON_EXECUTION_OPT_OUT_MARKERS = (
    "do not modify",
    "don't modify",
    "without modifying",
    "without modifying anything",
    "without making changes",
    "without changing anything",
    "no code changes",
    "no changes",
    "μην αλλαξεις",
    "μην αλλαξεις αρχεια",
    "μην τροποποιησεις",
    "μην κανεις αλλαγες",
    "χωρις αλλαγες",
    "χωρις αλλαγες στον κωδικα",
    "χωρις να τροποποιησεις",
    "χωρις να τροποποιησεις τιποτα",
    "χωρις να αλλαξεις τιποτα",
    "οχι αλλαγες",
)
_EXPLANATORY_NON_EXECUTION_PATTERNS = (
    re.compile(r"^(?:how does|what does|why is|why does)\b"),
    re.compile(r"^(?:what(?:'s| is)\s+(?:the\s+)?)current\s+(?:behavior|state|semantics)\b"),
    re.compile(r"^(?:what(?:'s| is)\s+(?:the\s+)?)behavior\b"),
    re.compile(r"^(?:what(?:'s| is)\s+(?:the\s+)?)current\b"),
    re.compile(r"^(?:explain|walk me through|help me understand)\b"),
    re.compile(r"^(?:just\s+)?explain\s+(?:whether|if)\b"),
    re.compile(r"^(?:can you|could you)\s+(?:explain|walk me through|help me understand)\b"),
    re.compile(r"^(?:tell me|show me)\s+how\b"),
    re.compile(r"^(?:read[\s-]?only|readonly)\s*:"),
    re.compile(r"^(?:review this|review the)\s+(?:design|architecture|approach)\b"),
    re.compile(
        r"^(?:can you|could you)\s+review\s+(?:this|the)\s+(?:design|architecture|approach)\b"
    ),
    re.compile(r"^(?:πως\s+λειτουργει|τι\s+κανει|γιατι\s+(?:ειναι|συμβαινει|αποτυγχανει))\b"),
    re.compile(r"^(?:εξηγησε|βοηθησε\s+με\s+να\s+καταλαβω|περιγραψε)\b"),
    re.compile(
        r"^(?:μπορεις(?:\s+να)?|θα\s+μπορουσες(?:\s+να)?)\s+"
        r"(?:εξηγησεις|εξηγησε|βοηθησεις\s+με\s+να\s+καταλαβω)\b"
    ),
)
_READ_ONLY_REPO_INSPECTION_PATTERNS = (
    re.compile(
        r"^(?:please\s+)?(?:(?:can|could)\s+you\s+|"
        r"(?:use|ask)\s+(?:(?:a|the)\s+)?(?:[\w-]+\s+)?subagent\s+to\s+)?"
        r"(?:locate\s+(?:and\s+)?inspect|inspect|look\s+(?:at|over|through)|"
        r"take\s+a\s+look\s+at|list|show|map|summari[sz]e|orient(?:\s+me)?|"
        r"check\s+(?:the\s+)?status|read)\b"
        r".*\b(?:repo|repository|workspace|codebase|project|tree|files?|status|"
        r"branch|remote|diff|changes?)\b"
    ),
    re.compile(
        r"^(?:(?:can|could)\s+you\s+|please\s+)?"
        r"(?:where\s+is|where\s+are|what\s+files?|which\s+files?)\b"
    ),
)
_REPO_CHANGE_SUMMARY_FOLLOW_UP_PATTERNS = (
    re.compile(r"^what\s+did\s+you\s+(?:change|modify|update|fix)\b"),
    re.compile(r"^what\s+changed\b"),
    re.compile(r"^(?:please\s+)?summari[sz]e\s+(?:what\s+)?(?:changed|the\s+changes)\b"),
    re.compile(
        r"^(?:give\s+me|provide|write)\s+(?:a\s+)?(?:short\s+|brief\s+)?"
        r"summary\s+of\s+(?:the\s+)?changes\b"
    ),
    re.compile(
        r"^(?:give\s+me|provide|write)\s+(?:a\s+)?(?:short\s+|brief\s+)?"
        r"summary\s+of\s+what\s+changed\b"
    ),
)
_REPO_CHANGE_EXPLANATION_FOLLOW_UP_PATTERNS = (
    re.compile(
        r"^(?:please\s+)?(?:explain|walk\s+me\s+through|help\s+me\s+understand)\s+"
        r"(?:the\s+)?(?:fix|change|changes|patch|update|implementation)\b"
    ),
    re.compile(
        r"^(?:can\s+you|could\s+you)\s+"
        r"(?:explain|walk\s+me\s+through|help\s+me\s+understand)\s+"
        r"(?:the\s+)?(?:fix|change|changes|patch|update|implementation)\b"
    ),
    re.compile(r"^(?:how\s+does|how\s+do)\s+(?:it|that|this)\s+work\b"),
    re.compile(r"^(?:can\s+you|could\s+you)\s+explain\s+(?:it|that|this|that\s+part)\b"),
)
_EXPLICIT_CHANGE_VERB_PATTERN = (
    r"(?:fix(?:ing)?|patch(?:ing)?|implement(?:ing)?|add(?:ing)?|update(?:ing)?|"
    r"modify(?:ing)?|change(?:ing)?|create(?:ing)?|refactor(?:ing)?|improve(?:ing)?|"
    r"resolve(?:ing)?|repair(?:ing)?|edit(?:ing)?|correct(?:ing)?|remove(?:ing)?|"
    r"delete(?:ing)?|drop(?:ping)?|rename(?:ing)?|"
    r"(?:διορθω|υλοποιη|προσθε|ενημερω|τροποποιη|αλλαξ|φτιαξ|βελτιω|ανανεω|"
    r"αφαιρε|διαγραψ|μετονομασ)\w*)"
)
_HYPOTHETICAL_CHANGE_QUESTION_PATTERNS = (
    re.compile(r"^what\s+change\s+would\s+you\s+make\b"),
    re.compile(
        r"^(?:what|which)\s+(?:is|would be)\s+(?:the\s+)?"
        r"(?:right|best|safest|recommended)\s+"
        r"(?:fix|change|patch|update|approach)\b"
    ),
    re.compile(
        r"^what\s+is\s+the\s+(?:right|best|safest|recommended)\s+way\s+to\s+"
        r"(?:fix|patch|implement|update|modify|change|resolve|repair)\b"
    ),
    re.compile(
        r"^how\s+should\s+(?:we|i)\s+"
        r"(?:fix|patch|implement|update|modify|change|resolve|repair)\b"
    ),
    re.compile(
        r"^how\s+would\s+you\s+"
        r"(?:fix|patch|implement|update|modify|change|resolve|repair)\b"
    ),
    re.compile(r"^τι\s+αλλαγη\s+θα\s+εκαν(?:ες|ατε)\b"),
    re.compile(
        r"^(?:ποια|ποιο)\s+(?:ειναι|θα\s+ηταν)\s+(?:η\s+)?"
        r"(?:σωστη|καλυτερη|ασφαλεστερη|προτεινομενη)\s+"
        r"(?:διορθωση|αλλαγη|προσεγγιση)\b"
    ),
    re.compile(
        r"^ποιος\s+ειναι\s+ο\s+(?:σωστος|καλυτερος|ασφαλεστερος|προτεινομενος)\s+"
        r"τροπος\s+να\s+(?:διορθωσ|υλοποιησ|ενημερωσ|τροποποιησ|αλλαξ|φτιαξ)\w*\b"
    ),
    re.compile(
        r"^πως\s+θα\s+(?:πρεπει\s+να\s+)?"
        r"(?:διορθωσ|υλοποιησ|ενημερωσ|τροποποιησ|αλλαξ|φτιαξ)\w*\b"
    ),
)
_EXPLICIT_CHANGE_INTENT_PATTERNS = (
    re.compile(
        r"^(?:(?:can you|could you|please|μπορεις(?:\s+να)?|θα\s+μπορουσες(?:\s+να)?|παρακαλω)\s+)?"
        rf"{_EXPLICIT_CHANGE_VERB_PATTERN}\b"
    ),
    re.compile(rf"[.?!;:]\s*{_EXPLICIT_CHANGE_VERB_PATTERN}\b"),
    re.compile(
        r"\b(?:and|and then|then|και|και\s+μετα|μετα|επειτα)\s+"
        rf"{_EXPLICIT_CHANGE_VERB_PATTERN}\b"
    ),
    re.compile(
        r"\b(?:and|and then|then|before|after|και|και\s+μετα|μετα|πριν|επειτα)\s+"
        r"(?:you\s+)?"
        rf"{_EXPLICIT_CHANGE_VERB_PATTERN}\b"
    ),
    re.compile(
        r"\b(?:make|making|apply|applying|do|doing)\s+"
        r"(?:(?:the|a|an|this|that)\s+)?"
        r"(?:(?:chosen|required|missing|needed)\s+)?"
        r"(?:change|fix|patch|edit|update)\b"
    ),
    re.compile(r"\b(?:clean|cleaning)\s+(?:it|this|that)\s+up\b"),
    re.compile(
        r"\b(?:κανε|κανω|κανεις|κανει|κανουμε|κανετε|κανουν|"
        r"εφαρμοσε|εφαρμοζω|εφαρμοζεις|εφαρμοζει|εφαρμοζουμε|εφαρμοζετε|εφαρμοζουν)\s+"
        r"(?:(?:την|τη|το|αυτη|αυτο)\s+)?"
        r"(?:αλλαγη|διορθωση|ενημερωση|τροποποιηση)\b"
    ),
)
_IMPLICIT_REPO_EXECUTION_PROBLEM_PATTERNS = (
    re.compile(
        r"\b(?:bug|issue|problem|failure|failing|fails|broken|broke|regression|wrong|"
        r"annoying|duplicate|duplicated|repeated|blank|empty|missing|unexpected|"
        r"ordering|unstable|cleanup|clean\s+up|off-?by-?one)\b"
    ),
    re.compile(r"\bnot\s+acceptable\b"),
)
_IMPLICIT_REPO_EXECUTION_ARTIFACT_PATTERNS = (
    re.compile(
        r"\b(?:formatter|parser|module|handler|startup|flow|render(?:er|ing)?|"
        r"output|command|cli|test|tests|helper|filter|endpoint|api|schema)\b"
    ),
    re.compile(
        r"\b[\w./\\-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|kt|c|cpp|h|hpp|cs|rb|php|"
        r"md|json|yaml|yml|toml|ini|sh)\b"
    ),
)
_IMPLICIT_REPO_IMPROVEMENT_LIMITATION_PATTERNS = (
    re.compile(r"\btoo\s+(?:limited|basic|rigid)\b"),
    re.compile(r"\bnot\s+(?:flexible|powerful|capable)\s+enough\b"),
    re.compile(r"\b(?:doesn't|does\s+not|can't|cannot)\s+(?:handle|support|fit)\b"),
    re.compile(r"\b(?:missing|lacks?|lacking)\b"),
)
_IMPLICIT_REPO_IMPROVEMENT_LOCAL_ARTIFACT_PATTERNS = (
    re.compile(
        r"\b(?:this|current|existing|our|my)\s+"
        r"(?:(?:[\w-]+\s+){0,3})?"
        r"(?:cli|tool|script|app|service|command|workflow|parser|formatter|builder|generator)\b"
    ),
)
_IMPLICIT_REPO_IMPROVEMENT_WORKFLOW_PATTERNS = (
    re.compile(r"\bfor\s+the\s+way\s+i\s+(?:actually\s+)?work\b"),
    re.compile(r"\bfor\s+our\s+workflow\b"),
    re.compile(r"\bday\s+to\s+day\b"),
    re.compile(r"\bin\s+practice\b"),
)
_REPO_CHANGE_CONSTRAINT_PATTERNS = (
    re.compile(r"\bpreserv\w*\b"),
    re.compile(r"\b(?:keep|leave)\b.*\b(?:unchanged|stable|same|as\s+is)\b"),
    re.compile(r"\b(?:do\s+not|don't)\s+change\b"),
    re.compile(r"\bwithout\s+changing\b"),
    re.compile(r"\b(?:διατηρ|κρατ)\w*"),
    re.compile(r"\b(?:μην|μη)\s+αλλαξ\w*\b"),
    re.compile(r"\bχωρις\s+να\s+αλλαξ\w*\b"),
)
_LOW_SIGNAL_META_FOLLOW_UP_PATTERNS = (
    re.compile(r"^(?:sounds?\s+good(?:\s+thanks?)?)$"),
    re.compile(r"^(?:please\s+)?continue\b"),
    re.compile(r"^(?:go\s+ahead)\b"),
    re.compile(r"^(?:please\s+)?keep\s+going\b"),
    re.compile(r"\b(?:elaborat|explain)\w*\b.*\b(?:more|bit\s+more|little\s+more)\b"),
)
_TASK_BRIEF_CONSTRAINT_DIRECTIVE_PATTERNS = (
    re.compile(r"^(?:also\s+)?(?:handle|support|allow|accept|treat|cover|retain)\b"),
    re.compile(r"^only\s+(?:touch|edit|modify|update)\b"),
)
_TASK_BRIEF_CONSTRAINT_ARTIFACT_HINT_PATTERNS = (
    re.compile(
        r"\b(?:line|lines|row|rows|input|inputs|value|values|case|cases|entry|entries|"
        r"field|fields|record|records)\b"
    ),
)
_LOCAL_MATERIALIZATION_ACTION_VERB_RE = re.compile(
    r"\b(save|write|create|produce|output|generate|emit|export|store|move|"
    r"put|place|materiali[sz]e|persist)\b",
    re.IGNORECASE,
)
_LOCAL_MATERIALIZATION_TARGET_RE = re.compile(
    r"\b(?:answer|result|results|count|counts|output|artifact|file|report|summary)\b",
    re.IGNORECASE,
)
_LOCAL_MATERIALIZATION_PATH_RE = re.compile(
    r"`([^`\n]+)`|"
    r"(?<![\w@])((?:\.{1,2}/|/|~\/|[A-Za-z]:[\\/])?[A-Za-z0-9_.@:+-]+"
    r"(?:[\\/][A-Za-z0-9_.@:+-]+)*"
    r"(?:\.[A-Za-z0-9][A-Za-z0-9_-]{0,12}|[\\/]))"
)
_LOCAL_MATERIALIZATION_DIR_RE = re.compile(
    r"\b(?:file|path|dir|directory|folder|output\s+(?:file|dir|directory|folder))\s+"
    r"(?:named\s+|called\s+|at\s+|to\s+|in\s+|into\s+|as\s+)?"
    r"(`[^`\n]+`|(?:\.{1,2}/|/|~\/|[A-Za-z]:[\\/])?[A-Za-z0-9_.@:+-]+"
    r"(?:[\\/][A-Za-z0-9_.@:+-]+)*(?:\.[A-Za-z0-9][A-Za-z0-9_-]{0,12}|[\\/])?)",
    re.IGNORECASE,
)
_LOCAL_MATERIALIZATION_SERVICE_RE = re.compile(
    r"\b(?:create|build|implement|add|generate|write|start|run|serve|launch)\b"
    r".{0,80}\b(?:service|server|daemon|executable|binary|package|cli|command)\b",
    re.IGNORECASE | re.DOTALL,
)
_SKILL_LIFECYCLE_OBJECT_PATTERN = (
    r"(?:skill(?:s)?|skill\s+bundle(?:s)?|δεξιοτ(?:ητα|ητες|ητων|ητας)?)"
)
_SKILL_LIFECYCLE_INTENT_PATTERNS: tuple[
    tuple[SkillLifecycleIntent, tuple[re.Pattern[str], ...]],
    ...,
] = (
    (
        "create",
        (
            re.compile(
                rf"\b(?:create|scaffold|initialize|initialise|init|bootstrap|author|"
                rf"make|generate|start)\b(?:\s+[\w./-]+){{0,6}}\s+{_SKILL_LIFECYCLE_OBJECT_PATTERN}\b"
            ),
            re.compile(
                rf"\b(?:δημιουργ\w*|φτιαξ\w*|στησ\w*|αρχικοποι\w*|σκαφολ\w*)\b"
                rf"(?:\s+[\w./-]+){{0,6}}\s+{_SKILL_LIFECYCLE_OBJECT_PATTERN}\b"
            ),
            re.compile(r"\bsylliptor\s+skill\s+(?:init|create)\b"),
        ),
    ),
    (
        "install",
        (
            re.compile(
                rf"\b(?:install)\b(?:\s+[\w./-]+){{0,6}}\s+{_SKILL_LIFECYCLE_OBJECT_PATTERN}\b"
            ),
            re.compile(
                rf"\b(?:εγκατ\w*)\b(?:\s+[\w./-]+){{0,6}}\s+{_SKILL_LIFECYCLE_OBJECT_PATTERN}\b"
            ),
            re.compile(r"\bsylliptor\s+skill\s+install\b"),
        ),
    ),
    (
        "disable",
        (
            re.compile(
                rf"\b(?:disable|deactivate|turn\s+off)\b(?:\s+[\w./-]+){{0,6}}\s+{_SKILL_LIFECYCLE_OBJECT_PATTERN}\b"
            ),
            re.compile(
                rf"\b(?:απενεργοποι\w*)\b(?:\s+[\w./-]+){{0,6}}\s+{_SKILL_LIFECYCLE_OBJECT_PATTERN}\b"
            ),
            re.compile(r"\bsylliptor\s+skill\s+disable\b"),
        ),
    ),
    (
        "enable",
        (
            re.compile(
                rf"\b(?:enable|activate|re-enable|turn\s+on)\b(?:\s+[\w./-]+){{0,6}}\s+{_SKILL_LIFECYCLE_OBJECT_PATTERN}\b"
            ),
            re.compile(
                rf"\b(?:ενεργοποι\w*)\b(?:\s+[\w./-]+){{0,6}}\s+{_SKILL_LIFECYCLE_OBJECT_PATTERN}\b"
            ),
            re.compile(r"\bsylliptor\s+skill\s+enable\b"),
        ),
    ),
    (
        "remove",
        (
            re.compile(
                rf"\b(?:remove|uninstall|delete)\b(?:\s+[\w./-]+){{0,6}}\s+{_SKILL_LIFECYCLE_OBJECT_PATTERN}\b"
            ),
            re.compile(
                rf"\b(?:αφαιρε\w*|διαγραψ\w*|απεγκαταστ\w*|ξεεγκαταστ\w*)\b"
                rf"(?:\s+[\w./-]+){{0,6}}\s+{_SKILL_LIFECYCLE_OBJECT_PATTERN}\b"
            ),
            re.compile(r"\bsylliptor\s+skill\s+(?:remove|uninstall)\b"),
        ),
    ),
    (
        "validate",
        (
            re.compile(
                rf"\b(?:validate|verify|check)\b(?:\s+[\w./-]+){{0,6}}\s+{_SKILL_LIFECYCLE_OBJECT_PATTERN}\b"
            ),
            re.compile(
                rf"\b(?:επικυρω\w*|ελεγξ\w*|επαληθευσ\w*)\b"
                rf"(?:\s+[\w./-]+){{0,6}}\s+{_SKILL_LIFECYCLE_OBJECT_PATTERN}\b"
            ),
            re.compile(r"\bsylliptor\s+skill\s+validate\b"),
        ),
    ),
)


def normalize_turn_intent_text(text: str) -> str:
    lowered = str(text or "").casefold()
    normalized = unicodedata.normalize("NFKD", lowered)
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    without_marks = re.sub(r"σ\b", "ς", without_marks)
    return " ".join(without_marks.split())


def contains_any_normalized_marker(normalized_text: str, markers: tuple[str, ...]) -> bool:
    for marker in markers:
        normalized_marker = normalize_turn_intent_text(marker)
        if normalized_marker and normalized_marker in normalized_text:
            return True
    return False


def _clean_materialization_path(raw: str) -> str:
    cleaned = str(raw or "").strip().strip("`'\"").replace("\\", "/")
    cleaned = cleaned.rstrip(".,;:!?)]}")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def _looks_like_materialization_path(raw: str) -> bool:
    cleaned = _clean_materialization_path(raw)
    if not cleaned or cleaned.startswith("-"):
        return False
    lowered = cleaned.casefold()
    if lowered.startswith(("http://", "https://", "mailto:")):
        return False
    if cleaned in {".", "..", "/"}:
        return False
    if ".." in cleaned.split("/"):
        return False
    if "/" in cleaned or cleaned.startswith(("~", "/")):
        return True
    if re.search(r"\.[A-Za-z0-9][A-Za-z0-9_-]{0,12}$", cleaned):
        return True
    return cleaned in {"Dockerfile", "Gemfile", "Makefile", "Procfile", "Rakefile"}


def _materialization_paths_in_text(text: str) -> tuple[str, ...]:
    paths: list[str] = []
    seen: set[str] = set()
    for match in _LOCAL_MATERIALIZATION_PATH_RE.finditer(text):
        raw = match.group(1) or match.group(2) or ""
        cleaned = _clean_materialization_path(raw)
        if not _looks_like_materialization_path(cleaned):
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        paths.append(cleaned)
        if len(paths) >= 8:
            break
    return tuple(paths)


def _first_materialization_action_verb(text: str) -> str:
    match = _LOCAL_MATERIALIZATION_ACTION_VERB_RE.search(text)
    return str(match.group(1) or "").casefold() if match else ""


def classify_local_materialization_requirement(
    instruction: str,
) -> LocalMaterializationRequirement:
    text = str(instruction or "").strip()
    if not text:
        return LocalMaterializationRequirement(
            required=False,
            reason="empty_instruction",
        )
    normalized = normalize_turn_intent_text(text)
    if contains_any_normalized_marker(normalized, _PLAN_OR_ANALYSIS_ONLY_MARKERS):
        return LocalMaterializationRequirement(
            required=False,
            reason="plan_or_analysis_only_opt_out",
        )
    if contains_any_normalized_marker(normalized, _ADVISORY_NON_EXECUTION_MARKERS):
        return LocalMaterializationRequirement(
            required=False,
            reason="advisory_non_execution_opt_out",
        )
    if contains_any_normalized_marker(normalized, _NON_EXECUTION_OPT_OUT_MARKERS):
        return LocalMaterializationRequirement(
            required=False,
            reason="non_execution_opt_out",
        )

    paths = _materialization_paths_in_text(text)
    action = _first_materialization_action_verb(text)
    output_target = bool(_LOCAL_MATERIALIZATION_TARGET_RE.search(text))
    action_match = _LOCAL_MATERIALIZATION_ACTION_VERB_RE.search(text)
    evidence_span = ""
    if action_match is not None:
        evidence_span = text[max(0, action_match.start() - 40) : action_match.end() + 120]
    elif paths:
        evidence_span = text[:180]

    if paths and action:
        return LocalMaterializationRequirement(
            required=True,
            confidence=0.95,
            output_paths=paths,
            action_verb=action,
            evidence_span=" ".join(evidence_span.split()),
            reason="output_path_with_materialization_action",
        )

    dir_match = _LOCAL_MATERIALIZATION_DIR_RE.search(text)
    if dir_match is not None and action:
        raw_path = str(dir_match.group(1) or "")
        path = _clean_materialization_path(raw_path)
        output_paths = (path,) if _looks_like_materialization_path(path) else paths
        return LocalMaterializationRequirement(
            required=True,
            confidence=0.9,
            output_paths=output_paths,
            action_verb=action,
            evidence_span=" ".join(str(dir_match.group(0) or "").split()),
            reason="explicit_output_file_or_directory",
        )

    if paths and output_target and re.search(r"\b(?:to|into|in|at|as)\b", normalized):
        return LocalMaterializationRequirement(
            required=True,
            confidence=0.86,
            output_paths=paths,
            action_verb=action or "write",
            evidence_span=" ".join(text[:180].split()),
            reason="answer_or_result_targeted_to_local_path",
        )

    service_match = _LOCAL_MATERIALIZATION_SERVICE_RE.search(text)
    if service_match is not None and (
        action or instruction_explicitly_requests_repo_changes(normalized)
    ):
        return LocalMaterializationRequirement(
            required=True,
            confidence=0.82,
            output_paths=paths,
            action_verb=action or "create",
            evidence_span=" ".join(str(service_match.group(0) or "").split()),
            reason="local_executable_or_service_materialization",
        )

    return LocalMaterializationRequirement(
        required=False,
        confidence=0.0,
        output_paths=paths,
        action_verb=action,
        evidence_span="",
        reason="no_high_confidence_materialization",
    )


def looks_like_explanatory_repo_question(normalized_instruction: str) -> bool:
    return any(
        pattern.search(normalized_instruction) is not None
        for pattern in _EXPLANATORY_NON_EXECUTION_PATTERNS
    )


def looks_like_read_only_repo_inspection_request(normalized_instruction: str) -> bool:
    return any(
        pattern.search(normalized_instruction) is not None
        for pattern in _READ_ONLY_REPO_INSPECTION_PATTERNS
    )


def looks_like_repo_change_summary_follow_up(normalized_instruction: str) -> bool:
    return any(
        pattern.search(normalized_instruction) is not None
        for pattern in _REPO_CHANGE_SUMMARY_FOLLOW_UP_PATTERNS
    )


def looks_like_repo_change_explanation_follow_up(normalized_instruction: str) -> bool:
    return any(
        pattern.search(normalized_instruction) is not None
        for pattern in _REPO_CHANGE_EXPLANATION_FOLLOW_UP_PATTERNS
    )


def instruction_explicitly_requests_repo_changes(normalized_instruction: str) -> bool:
    return any(
        pattern.search(normalized_instruction) is not None
        for pattern in _EXPLICIT_CHANGE_INTENT_PATTERNS
    )


def looks_like_hypothetical_or_recommendation_change_question(
    normalized_instruction: str,
) -> bool:
    return any(
        pattern.search(normalized_instruction) is not None
        for pattern in _HYPOTHETICAL_CHANGE_QUESTION_PATTERNS
    )


def looks_like_implicit_repo_bugfix_request(instruction: str) -> bool:
    normalized = normalize_turn_intent_text(instruction)
    if not normalized:
        return False
    if classify_repo_execution_intent(instruction) != "execute":
        return False
    problem = any(
        pattern.search(normalized) is not None
        for pattern in _IMPLICIT_REPO_EXECUTION_PROBLEM_PATTERNS
    )
    artifact = any(
        pattern.search(normalized) is not None
        for pattern in _IMPLICIT_REPO_EXECUTION_ARTIFACT_PATTERNS
    )
    return problem and artifact


def looks_like_implicit_repo_improvement_request(instruction: str) -> bool:
    normalized = normalize_turn_intent_text(instruction)
    if not normalized:
        return False
    if classify_repo_execution_intent(instruction) != "execute":
        return False
    artifact = any(
        pattern.search(normalized) is not None
        for pattern in _IMPLICIT_REPO_EXECUTION_ARTIFACT_PATTERNS
    )
    if not artifact:
        return False
    limitation = any(
        pattern.search(normalized) is not None
        for pattern in _IMPLICIT_REPO_IMPROVEMENT_LIMITATION_PATTERNS
    )
    if not limitation:
        return False
    local_artifact = any(
        pattern.search(normalized) is not None
        for pattern in _IMPLICIT_REPO_IMPROVEMENT_LOCAL_ARTIFACT_PATTERNS
    )
    workflow = any(
        pattern.search(normalized) is not None
        for pattern in _IMPLICIT_REPO_IMPROVEMENT_WORKFLOW_PATTERNS
    )
    return local_artifact or workflow


def has_task_brief_constraint_signal(instruction: str) -> bool:
    normalized = normalize_turn_intent_text(instruction)
    if not normalized:
        return False
    if any(pattern.search(normalized) is not None for pattern in _REPO_CHANGE_CONSTRAINT_PATTERNS):
        return True
    directive = any(
        pattern.search(normalized) is not None
        for pattern in _TASK_BRIEF_CONSTRAINT_DIRECTIVE_PATTERNS
    )
    if not directive:
        return False
    if any(
        pattern.search(normalized) is not None
        for pattern in _IMPLICIT_REPO_EXECUTION_PROBLEM_PATTERNS
    ):
        return True
    if any(
        pattern.search(normalized) is not None
        for pattern in _IMPLICIT_REPO_EXECUTION_ARTIFACT_PATTERNS
    ):
        return True
    return any(
        pattern.search(normalized) is not None
        for pattern in _TASK_BRIEF_CONSTRAINT_ARTIFACT_HINT_PATTERNS
    )


def has_task_brief_positive_signal(instruction: str) -> bool:
    normalized = normalize_turn_intent_text(instruction)
    if not normalized:
        return False
    if instruction_explicitly_requests_repo_changes(normalized):
        return True
    if has_task_brief_constraint_signal(instruction):
        return True
    if looks_like_implicit_repo_improvement_request(instruction):
        return True
    problem = any(
        pattern.search(normalized) is not None
        for pattern in _IMPLICIT_REPO_EXECUTION_PROBLEM_PATTERNS
    )
    artifact = any(
        pattern.search(normalized) is not None
        for pattern in _IMPLICIT_REPO_EXECUTION_ARTIFACT_PATTERNS
    )
    return problem and artifact


def looks_like_low_signal_meta_follow_up(normalized_instruction: str) -> bool:
    return any(
        pattern.search(normalized_instruction) is not None
        for pattern in _LOW_SIGNAL_META_FOLLOW_UP_PATTERNS
    )


def classify_repo_execution_intent(instruction: str) -> RepoExecutionIntent:
    normalized = normalize_turn_intent_text(instruction)
    if contains_any_normalized_marker(normalized, _PLAN_OR_ANALYSIS_ONLY_MARKERS):
        return "plan_or_analysis_only"
    if contains_any_normalized_marker(normalized, _ADVISORY_NON_EXECUTION_MARKERS):
        return "advisory_non_execution"
    if contains_any_normalized_marker(normalized, _NON_EXECUTION_OPT_OUT_MARKERS):
        return "advisory_non_execution"
    if classify_local_materialization_requirement(instruction).required:
        return "execute"
    direct_change_requested = instruction_explicitly_requests_repo_changes(normalized)
    if looks_like_repo_change_explanation_follow_up(normalized) and not direct_change_requested:
        return "advisory_non_execution"
    if looks_like_repo_change_summary_follow_up(normalized) and not direct_change_requested:
        return "advisory_non_execution"
    if (
        looks_like_hypothetical_or_recommendation_change_question(normalized)
        and not direct_change_requested
    ):
        return "advisory_non_execution"
    if direct_change_requested:
        return "execute"
    if looks_like_read_only_repo_inspection_request(normalized):
        return "advisory_non_execution"
    if looks_like_explanatory_repo_question(normalized):
        return "advisory_non_execution"
    return "execute"


def detect_skill_lifecycle_intent(instruction: str) -> SkillLifecycleIntent | None:
    normalized = normalize_turn_intent_text(instruction)
    if not normalized:
        return None
    for intent, patterns in _SKILL_LIFECYCLE_INTENT_PATTERNS:
        if any(pattern.search(normalized) is not None for pattern in patterns):
            return intent
    return None


def is_skill_lifecycle_request(instruction: str) -> bool:
    return detect_skill_lifecycle_intent(instruction) is not None
