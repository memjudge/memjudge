"""Sealed key bundles: the only way users' keys reach a job.

The API encrypts a bundle with Cloud KMS (bound to the run id as additional authenticated
data) and stores the ciphertext under a random, unguessable name. A job reads it once,
deletes it, and decrypts it in memory. Job service accounts can get and delete objects in
the bucket but not list them, so a job can open only the bundle it was given.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any, Protocol


class Cipher(Protocol):
    def encrypt(self, key: str, plaintext: bytes, aad: bytes) -> bytes: ...
    def decrypt(self, key: str, ciphertext: bytes, aad: bytes) -> bytes: ...


class Blobs(Protocol):
    def put(self, bucket: str, name: str, data: bytes) -> None: ...
    def get(self, bucket: str, name: str) -> bytes: ...
    def delete(self, bucket: str, name: str) -> None: ...
    def download(self, bucket: str, name: str, dest: Path) -> None: ...
    def exists(self, bucket: str, name: str) -> bool: ...
    def signed_put_url(self, bucket: str, name: str, minutes: int) -> str: ...
    def signed_get_url(self, bucket: str, name: str, minutes: int) -> str: ...


def seal(
    cipher: Cipher, blobs: Blobs, bucket: str, kms_key: str, run_id: str, payload: dict[str, Any]
) -> str:
    """Encrypt and store ``payload``; returns the object name (a capability: keep it private)."""
    name = secrets.token_urlsafe(32)
    data = cipher.encrypt(kms_key, json.dumps(payload).encode(), run_id.encode())
    blobs.put(bucket, name, data)
    return name


def open_once(
    cipher: Cipher, blobs: Blobs, bucket: str, kms_key: str, run_id: str, name: str
) -> dict[str, Any]:
    """Read, delete, then decrypt a bundle. A second open fails: the object is gone."""
    data = blobs.get(bucket, name)
    blobs.delete(bucket, name)
    payload: dict[str, Any] = json.loads(cipher.decrypt(kms_key, data, run_id.encode()))
    return payload
