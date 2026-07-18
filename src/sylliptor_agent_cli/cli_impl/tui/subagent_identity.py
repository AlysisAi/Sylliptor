"""Per-subagent visual identity — accent colour and activity tagline.

One tiny authority (like ``forge_status``) so the transcript spawn line, the
live status, and the footer badge can never disagree about who a given subagent
is or what it is doing. The /subagent picker rows and the result attribution
print the registry name alone and deliberately do not consult this module —
a subagent wears no per-agent symbol, so they have nothing to keep in sync.

The name is the identity: the shared ``↪``/``↩`` marks say only that a nested
run started or ended, never which agent it was. Colours are distinct from the
fixed mode accents (green chat, violet forge, and cyan brand) so a
subagent badge never impersonates a mode. Custom subagents get a colour picked
deterministically from the name — the same agent looks the same every run — and
an empty tagline (callers fall back to the description).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass


@dataclass(frozen=True)
class SubagentIdentity:
    color: str
    tagline: str  # empty for custom subagents — callers fall back to the description


_BUILTIN_IDENTITIES: dict[str, SubagentIdentity] = {
    "frontend-engineer": SubagentIdentity("#a371f7", "crafting the interface"),
    "visual-designer": SubagentIdentity("#c9d1d9", "composing the asset"),
    "explorer": SubagentIdentity("#58a6ff", "charting the codebase"),
    "implementer": SubagentIdentity("#f0883e", "shaping the change"),
    "debugger": SubagentIdentity("#f47067", "hunting the root cause"),
    "code-reviewer": SubagentIdentity("#db61a2", "combing the diff"),
    "test-strategist": SubagentIdentity("#39c5cf", "plotting the test net"),
}

_FALLBACK_COLORS: tuple[str, ...] = (
    "#58a6ff",
    "#f0883e",
    "#db61a2",
    "#39c5cf",
    "#f47067",
    "#d2a8ff",
)


def subagent_identity(name: str) -> SubagentIdentity:
    """Identity for ``name`` (built-in aliases resolved, e.g. explore→explorer)."""
    clean = str(name or "").strip().lower()
    try:
        from ...subagents import canonical_subagent_name

        clean = canonical_subagent_name(clean) or clean
    except Exception:  # noqa: BLE001 - identity lookup must never raise
        pass
    known = _BUILTIN_IDENTITIES.get(clean)
    if known is not None:
        return known
    digest = zlib.crc32(clean.encode("utf-8", "replace"))
    return SubagentIdentity(_FALLBACK_COLORS[digest % len(_FALLBACK_COLORS)], "")


def subagent_tagline(name: str, description: str = "") -> str:
    """The built-in activity tagline for ``name`` — unless ``description`` shows
    the definition is a CUSTOM one shadowing a built-in name, in which case the
    agent keeps its own story (empty return; callers fall back to the
    description). Pass the definition's description whenever it is at hand.
    """
    ident = subagent_identity(name)
    if not ident.tagline:
        return ""
    desc = str(description or "").strip()
    if not desc:
        return ident.tagline
    try:
        from ...subagents import built_in_subagents, canonical_subagent_name

        canonical = canonical_subagent_name(name) or str(name or "").strip().lower()
        builtin = built_in_subagents().get(canonical)
    except Exception:  # noqa: BLE001 - identity lookup must never raise
        return ident.tagline
    if builtin is not None and desc != str(builtin.description or "").strip():
        return ""
    return ident.tagline


__all__ = ["SubagentIdentity", "subagent_identity", "subagent_tagline"]
