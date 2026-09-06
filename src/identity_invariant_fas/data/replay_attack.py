"""Replay-Attack manifest adapter.

The dataset is license-restricted and is therefore not bundled. Once official
access is granted and frames are prepared, use the common manifest schema from
``identity_invariant_fas.data.manifest``.
"""

from __future__ import annotations

from pathlib import Path

from .manifest import FASSample, read_manifest


def load_replay_attack_manifest(path: str | Path) -> list[FASSample]:
    samples = read_manifest(path)
    wrong = [sample for sample in samples if sample.dataset not in {"Replay-Attack", "ReplayAttack"}]
    if wrong:
        raise ValueError("Replay-Attack manifest contains records for a different dataset.")
    return samples
