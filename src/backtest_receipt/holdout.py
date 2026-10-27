"""Encrypt a holdout frame and refuse a second successful unlock.

The seal file stores ciphertext plus a SHA-256 of the plaintext. The key lives
in a separate file created with mode ``0600``. A successful unlock is an event
the caller must log. This module does the cryptography; :class:`Study` writes
the append-only event and decides whether a second unlock is refused or flagged.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from cryptography.fernet import Fernet, InvalidToken

from backtest_receipt.canonical import content_hash
from backtest_receipt.errors import HoldoutError, HoldoutKeyError
from backtest_receipt.frames import frame_from_payload, frame_to_payload, payload_bytes

_ALGORITHM = "fernet"


@dataclass(frozen=True)
class SealInfo:
    """Public description of a seal. It does not include the key or the rows."""

    seal_id: str
    plaintext_sha256: str
    n_rows: int
    columns: tuple[str, ...]


def write_seal(
    frame: pd.DataFrame,
    *,
    seal_path: str | Path,
    key_path: str | Path,
    metadata: dict[str, Any],
) -> SealInfo:
    """Encrypt ``frame`` and write the key. Refuses to overwrite either file."""
    seal_path = Path(seal_path)
    key_path = Path(key_path)
    if seal_path.exists():
        raise HoldoutError(f"seal already exists: {seal_path}")
    if key_path.exists():
        raise HoldoutError(f"key already exists: {key_path}")
    payload = {"metadata": metadata, "frame": frame_to_payload(frame)}
    plaintext = payload_bytes(payload)
    key = Fernet.generate_key()
    ciphertext = Fernet(key).encrypt(plaintext)
    ciphertext_text = ciphertext.decode("ascii")
    info = SealInfo(
        seal_id=hashlib.sha256(ciphertext).hexdigest(),
        plaintext_sha256=hashlib.sha256(plaintext).hexdigest(),
        n_rows=int(len(frame)),
        columns=tuple(str(column) for column in frame.columns),
    )
    outer = {
        "version": 1,
        "algorithm": _ALGORITHM,
        "seal_id": info.seal_id,
        "plaintext_sha256": info.plaintext_sha256,
        "metadata": metadata,
        "ciphertext": ciphertext_text,
    }
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    _write_key(key_path, key, info.seal_id)
    seal_path.write_text(json.dumps(outer, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return info


def read_seal(
    seal_path: str | Path, key_path: str | Path
) -> tuple[pd.DataFrame, dict[str, Any], SealInfo]:
    """Decrypt a seal and check that the outer wrapper matches the plaintext."""
    seal_path = Path(seal_path)
    key_path = Path(key_path)
    try:
        outer = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HoldoutError(f"unreadable seal: {seal_path}") from exc
    if outer.get("version") != 1 or outer.get("algorithm") != _ALGORITHM:
        raise HoldoutError("unsupported seal format")
    ciphertext_text = outer.get("ciphertext")
    if not isinstance(ciphertext_text, str):
        raise HoldoutError("seal is missing ciphertext")
    ciphertext = ciphertext_text.encode("ascii")
    digest = hashlib.sha256(ciphertext).hexdigest()
    if outer.get("seal_id") != digest:
        raise HoldoutError("seal id does not match the ciphertext")
    key = _read_key(key_path, digest)
    try:
        plaintext = Fernet(key).decrypt(ciphertext)
    except InvalidToken as exc:
        raise HoldoutKeyError("holdout key did not decrypt the seal") from exc
    plaintext_sha = hashlib.sha256(plaintext).hexdigest()
    if outer.get("plaintext_sha256") != plaintext_sha:
        raise HoldoutError("plaintext hash does not match the seal")
    try:
        payload = json.loads(plaintext.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HoldoutError("decrypted payload is not JSON") from exc
    inner_meta = payload.get("metadata")
    if inner_meta != outer.get("metadata"):
        raise HoldoutError("seal metadata does not match the encrypted payload")
    frame = frame_from_payload(payload["frame"])
    info = SealInfo(
        seal_id=digest,
        plaintext_sha256=plaintext_sha,
        n_rows=int(len(frame)),
        columns=tuple(str(column) for column in frame.columns),
    )
    return frame, inner_meta, info


def inspect_seal(seal_path: str | Path) -> SealInfo:
    """Read the public wrapper and check the ciphertext digest. Does not decrypt."""
    seal_path = Path(seal_path)
    try:
        outer = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HoldoutError(f"unreadable seal: {seal_path}") from exc
    ciphertext_text = outer.get("ciphertext")
    if not isinstance(ciphertext_text, str):
        raise HoldoutError("seal is missing ciphertext")
    ciphertext = ciphertext_text.encode("ascii")
    digest = hashlib.sha256(ciphertext).hexdigest()
    if outer.get("seal_id") != digest:
        raise HoldoutError("seal id does not match the ciphertext")
    plaintext_sha = outer.get("plaintext_sha256")
    if not isinstance(plaintext_sha, str):
        raise HoldoutError("seal is missing the plaintext hash")
    columns = tuple(str(column) for column in outer.get("metadata", {}).get("columns", []))
    n_rows = outer.get("metadata", {}).get("n_rows")
    return SealInfo(
        seal_id=digest,
        plaintext_sha256=plaintext_sha,
        n_rows=0 if not isinstance(n_rows, int) else n_rows,
        columns=columns,
    )


def metadata_hash(metadata: dict[str, Any]) -> str:
    """Hash the metadata stored beside the ciphertext."""
    return content_hash(metadata)


def _write_key(path: Path, key: bytes, seal_id: str) -> None:
    payload = json.dumps(
        {"version": 1, "algorithm": _ALGORITHM, "seal_id": seal_id, "key": key.decode("ascii")}
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, (payload + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o600)


def _read_key(path: Path, seal_id: str) -> bytes:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HoldoutKeyError("unreadable holdout key") from exc
    if document.get("seal_id") != seal_id or document.get("algorithm") != _ALGORITHM:
        raise HoldoutKeyError("holdout key does not match this seal")
    key = document.get("key")
    if not isinstance(key, str):
        raise HoldoutKeyError("holdout key is missing")
    return key.encode("ascii")
