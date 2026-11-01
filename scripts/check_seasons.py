"""Check that a change keeps past seasons as they were (docs/seasons.md).

Seasons may only be added at the end. An existing season's rules never change, because
published results were judged under them; only its ``summary`` and ``details`` wording may.

Usage: python scripts/check_seasons.py <base git ref>   (CI passes the PR's base branch)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PATH = "src/memjudge/judge/seasons.json"
EDITABLE = {"summary", "details"}


def main(base: str) -> int:
    shown = subprocess.run(["git", "show", f"{base}:{PATH}"], capture_output=True, text=True)
    if shown.returncode != 0:
        print(f"{PATH} is not on {base} yet: nothing to compare.")
        return 0
    before = json.loads(shown.stdout)
    after = json.loads(Path(PATH).read_text(encoding="utf-8"))
    errors = []
    if len(after) < len(before):
        errors.append("a season was removed")
    for old, new in zip(before, after, strict=False):
        changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
        if set(changed) - EDITABLE:
            errors.append(f"season {old['id']!r} changed: {', '.join(changed)}")
    for message in errors:
        print(f"error: {message}. Past seasons are fixed: add a new season instead.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
