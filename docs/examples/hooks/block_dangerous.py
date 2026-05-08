#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys

_DANGEROUS_PATTERNS = (
    r"\brm\s+-rf\b",
    r"\bgit\s+push\b.*\s--force(?:-with-lease)?\b",
    r"\bcurl\b.*\|\s*(?:sh|bash)\b",
)


def main() -> int:
    payload = json.load(sys.stdin)
    command = str(payload.get("tool_input", {}).get("cmd") or "")
    for pattern in _DANGEROUS_PATTERNS:
        if re.search(pattern, command, flags=re.IGNORECASE):
            print(
                f"blocked by docs/examples/hooks/block_dangerous.py: matched {pattern!r}",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
