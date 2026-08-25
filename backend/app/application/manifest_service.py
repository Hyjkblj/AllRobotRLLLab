"""Manifest freeze service."""

from __future__ import annotations

from backend.app.domain.contracts import RunManifest


def freeze_manifest(manifest: RunManifest) -> RunManifest:
    """Return an immutable-by-contract manifest with its canonical hash pinned."""

    return manifest.freeze()


__all__ = ["freeze_manifest"]

