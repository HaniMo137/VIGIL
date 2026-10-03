"""Portable graph/event identities for score export (not model inputs)."""

import hashlib
from pathlib import Path


def source_graph_id(path):
    """Fingerprint the graph filename and exact artifact bytes.

    Independent of the absolute artifact root, so copying artifacts preserves
    identity. Replacing a graph invalidates its old scores. Identical copies
    with the same filename are the same source, not extra windows.
    """
    path = Path(path)
    digest = hashlib.sha256(path.name.encode("utf-8") + b"\0")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_uuid_hash(value):
    """Hash original UUID text without assuming a particular UUID format."""
    if value is None or str(value) == "":
        return ""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()
