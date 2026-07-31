"""Content hashing helpers.

Hashes are the backbone of the lineage guarantee: the artifact hash ties a
served model to the exact bytes that were gated, so a tampered or drifting
artifact can be detected at promotion time.
"""

import hashlib
from pathlib import Path

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of a byte string."""
    return hashlib.sha256(data).hexdigest()
