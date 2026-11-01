"""Ed25519 signatures over RFC 8785 canonical JSON (ADR 0006).

A signed result is an envelope::

    {"payload": {...RunResult with key_id...},
     "signature": {"alg": "Ed25519", "key_id": "<hex>", "value": "<base64>"}}

The signature covers ``canonical_bytes(payload)`` only.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from memjudge.canonical import canonical_bytes

ALG = "Ed25519"


def key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:16]


def generate_keypair(directory: Path) -> tuple[Path, Path, str]:
    """Write ``<key_id>.key`` (PEM, keep secret) and ``<key_id>.pub`` (PEM) into ``directory``."""
    private = Ed25519PrivateKey.generate()
    kid = key_id(private.public_key())
    directory.mkdir(parents=True, exist_ok=True)
    priv_path, pub_path = directory / f"{kid}.key", directory / f"{kid}.pub"
    priv_path.write_bytes(
        private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    pub_path.write_bytes(
        private.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    return priv_path, pub_path, kid


def load_private_key(pem: bytes) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("not an Ed25519 private key")
    return key


def load_public_key(pem: bytes) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(pem)
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError("not an Ed25519 public key")
    return key


def sign_payload(payload: dict[str, Any], private_key: Ed25519PrivateKey) -> dict[str, Any]:
    kid = key_id(private_key.public_key())
    payload = {**payload, "key_id": kid}
    signature = private_key.sign(canonical_bytes(payload))
    return {
        "payload": payload,
        "signature": {"alg": ALG, "key_id": kid, "value": base64.b64encode(signature).decode()},
    }


class VerificationError(Exception):
    pass


def sign_result(
    payload: dict[str, Any], private_key: Ed25519PrivateKey, artifact_url_prefix: str | None = None
) -> dict[str, Any]:
    """Sign a result payload. With a prefix, artifact URLs become ``<prefix><run_id>/<name>``."""
    if artifact_url_prefix:
        payload = {
            **payload,
            "artifacts": [
                {**a, "url": f"{artifact_url_prefix}{payload['run_id']}/{a['name']}"}
                for a in payload["artifacts"]
            ],
        }
    return sign_payload(payload, private_key)


def verify_envelope(envelope: dict[str, Any], public_keys: dict[str, Ed25519PublicKey]) -> str:
    """Verify a signed envelope against trusted keys. Returns the key id or raises."""
    try:
        payload = envelope["payload"]
        sig = envelope["signature"]
        kid = sig["key_id"]
        value = base64.b64decode(sig["value"], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise VerificationError(f"malformed envelope: {exc}") from None
    if sig.get("alg") != ALG:
        raise VerificationError(f"unsupported algorithm {sig.get('alg')!r}")
    if payload.get("key_id") != kid:
        raise VerificationError("payload key_id does not match signature key_id")
    key = public_keys.get(kid)
    if key is None:
        raise VerificationError(f"unknown signing key {kid}")
    try:
        key.verify(value, canonical_bytes(payload))
    except InvalidSignature:
        raise VerificationError("signature does not match payload") from None
    return str(kid)


def load_public_keys(directory: Path) -> dict[str, Ed25519PublicKey]:
    keys: dict[str, Ed25519PublicKey] = {}
    for path in sorted(directory.glob("*.pub")):
        key = load_public_key(path.read_bytes())
        keys[key_id(key)] = key
    return keys
