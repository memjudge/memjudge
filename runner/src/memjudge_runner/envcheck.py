"""Image build check for ``/opt/core``, the environment the API and the finish job run from.

No connector SDK may be installed there (ADR 0009): a wheel can ship a ``.pth`` file or a
``sitecustomize`` module, which Python runs at startup in every process, with no import.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

LOCAL_PACKAGES = frozenset({"memjudge", "memjudge-runner"})
"""Built from this repository, so they are not in the lock."""
STARTUP_HOOKS = ("*.pth", "sitecustomize.py", "usercustomize.py")


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_pins(text: str) -> dict[str, str]:
    """Name → version for every ``name==version`` line of a pip lock file."""
    pins = {}
    for line in text.splitlines():
        if match := re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\;]+)", line):
            pins[canonical_name(match[1])] = match[2]
    return pins


def core_env_problems(
    lock_text: str, installed: Iterable[tuple[str, str]], site_dirs: Iterable[Path]
) -> list[str]:
    """Why the environment differs from the lock (``installed`` is name, version pairs) or
    can run code at startup. Empty when it holds exactly the lock and the local packages."""
    pins = lock_pins(lock_text)
    found = {canonical_name(name): version for name, version in installed}
    problems = [
        f"installed but not in the lock: {name}=={version}"
        for name, version in sorted(found.items())
        if name not in pins and name not in LOCAL_PACKAGES
    ]
    for name, version in sorted(pins.items()):
        if name not in found:
            problems.append(f"in the lock but not installed: {name}=={version}")
        elif found[name] != version:
            problems.append(f"{name} is {found[name]}, the lock pins {version}")
    problems += [
        f"startup hook: {path}"
        for site in site_dirs
        for pattern in STARTUP_HOOKS
        for path in sorted(site.glob(pattern))
    ]
    return problems
