"""RFC 8785 (JCS) canonical JSON and hashing helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import rfc8785
from pydantic import BaseModel


def canonical_bytes(value: Any) -> bytes:
    """Serialise a JSON-compatible value (or pydantic model) with RFC 8785."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return bytes(rfc8785.dumps(value))


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value: Any) -> str:
    """SHA-256 of the canonical JSON form, prefixed with the algorithm."""
    return "sha256:" + sha256_hex(canonical_bytes(value))


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def directory_hash(path: Path) -> str:
    """Hash every file under ``path`` (relative path + contents), ignoring caches."""
    h = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = file.relative_to(path).as_posix()
        if "__pycache__" in rel or rel.endswith(".pyc"):
            continue
        h.update(rel.encode() + b"\0")
        # Normalise line endings so the hash is identical on Windows and Linux checkouts.
        h.update(file.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return "sha256:" + h.hexdigest()
