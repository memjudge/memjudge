from __future__ import annotations

import copy
from pathlib import Path

import pytest

from memjudge.canonical import canonical_bytes, directory_hash
from memjudge.signing import (
    VerificationError,
    generate_keypair,
    load_private_key,
    load_public_keys,
    sign_payload,
    verify_envelope,
)


def test_rfc8785_canonical_form() -> None:
    # Keys sorted by UTF-16 code units, no whitespace, numbers in shortest ES6 form.
    assert (
        canonical_bytes({"b": 1, "a": [1.0, "é", None, True]})
        == '{"a":[1,"é",null,true],"b":1}'.encode()
    )
    assert canonical_bytes({"x": 1e21, "y": 0.1}) == b'{"x":1e+21,"y":0.1}'


def test_sign_and_verify_round_trip(tmp_path: Path) -> None:
    priv, _, kid = generate_keypair(tmp_path)
    envelope = sign_payload(
        {"run_id": "r1", "metrics": {"accuracy": 0.5}}, load_private_key(priv.read_bytes())
    )
    assert envelope["payload"]["key_id"] == kid
    assert verify_envelope(envelope, load_public_keys(tmp_path)) == kid


def test_tampering_is_detected(tmp_path: Path) -> None:
    priv, _, _ = generate_keypair(tmp_path)
    envelope = sign_payload({"metrics": {"accuracy": 0.5}}, load_private_key(priv.read_bytes()))
    keys = load_public_keys(tmp_path)

    tampered = copy.deepcopy(envelope)
    tampered["payload"]["metrics"]["accuracy"] = 0.9
    with pytest.raises(VerificationError, match="does not match"):
        verify_envelope(tampered, keys)

    other = copy.deepcopy(envelope)
    other["payload"]["key_id"] = "0" * 16
    with pytest.raises(VerificationError, match="key_id"):
        verify_envelope(other, keys)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    priv, _, _ = generate_keypair(tmp_path / "a")
    generate_keypair(tmp_path / "b")
    envelope = sign_payload({"x": 1}, load_private_key(priv.read_bytes()))
    with pytest.raises(VerificationError, match="unknown signing key"):
        verify_envelope(envelope, load_public_keys(tmp_path / "b"))


def test_directory_hash_ignores_line_endings_and_caches(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "connector.py").write_bytes(b"x = 1\n")
    (tmp_path / "b" / "connector.py").write_bytes(b"x = 1\r\n")
    (tmp_path / "b" / "__pycache__").mkdir()
    (tmp_path / "b" / "__pycache__" / "c.pyc").write_bytes(b"junk")
    assert directory_hash(tmp_path / "a") == directory_hash(tmp_path / "b")
